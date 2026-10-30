r"""Quantitative runtime verification (Section 4 / Algorithm 1).

Takes a probabilistic initial state (a ProbStar), the per-vehicle plant
matrices, and a collision half-space, propagates the reachable ProbStar forward,
and returns the per-step collision probability, following Algorithm 1.

The loop:
* Outer loop over reachability steps k = 0 .. k_max, each of duration dt_r.
* Inner loop of dt_r / dt_m modeling substeps: affine-map the ProbStar by the
  plant matrix built at the small step dt_m. This keeps linearization error low
  while recomputing the probstar only every dt_r -- the paper's efficiency trick.
* At each reachable set Theta_k, intersect with the collision half-space H
  (Prop. 1) and compute the probability of the intersection (Def. 2). That's the
  collision probability at step k.

Per-step vs cumulative. The paper reports the per-step probability p_k. The
course brief also wants the probability of at least one collision over the whole
horizon, which has to account for temporal dependence. We return two
aggregations:

* cumulative_independent: 1 - prod_k (1 - p_k). Only valid if the per-step
  events were independent, which they are NOT (consecutive reachable sets are
  highly correlated), so it over-estimates. Kept as a loose upper reference.
* cumulative_max: max_k p_k. A valid lower bound on P(at least one) -- the most
  likely single step can't exceed the union.

Because the perception initial condition is shared across all steps, the per-step
events are strongly positively correlated, so the true union sits between these
two and near max_k p_k. For the exact union we use Monte Carlo over the
reachable-set trajectory (monte_carlo.py); the ProbStar per-step probabilities
bound and track it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .dynamics import VehicleModel, build_system_matrix, STATE_DIM
from .probstar import ProbStar
from .collision import collision_half_space


@dataclass
class VerificationResult:
    """Container for a verification run's outputs."""
    per_step_prob: np.ndarray               # p_k for k=1..n_steps
    times: np.ndarray                       # elapsed time at each step (s)
    cumulative_independent: np.ndarray      # 1-prod(1-p) upper reference
    cumulative_max: np.ndarray              # running max_k p_k (lower bound)
    runtime_s: float = 0.0                  # wall-clock for the reachability loop
    reachable_centers: Optional[np.ndarray] = None   # (n_steps+1, dim) centers
    meta: dict = field(default_factory=dict)

    @property
    def union_lower_bound(self) -> float:
        """Lower bound on P(at least one collision over horizon)."""
        return float(self.cumulative_max[-1]) if len(self.cumulative_max) else 0.0

    @property
    def union_independent(self) -> float:
        """Independence-assumption reference for P(at least one collision)."""
        return float(self.cumulative_independent[-1]) if len(self.cumulative_independent) else 0.0


class QuantitativeVerifier:
    """Perception-based quantitative runtime verifier (Algorithm 1)."""

    def __init__(self, models: Sequence[VehicleModel], dt_r: float, dt_m: float,
                 length_half: float, width_half: float,
                 prob_backend: str = "auto", genz_samples: int = 20000,
                 scipy_maxdim: int = 6,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            models: vehicle models (ego first), fixed over the horizon.
            dt_r: reachability time step.
            dt_m: modeling (linearization) time step; dt_r must be a multiple.
            length_half, width_half: vehicle half-extents for collision geometry.
            prob_backend, genz_samples, scipy_maxdim: probability settings.
            rng: optional RNG for the stochastic probability backend.
        """
        self.models = list(models)
        self.n_vehicles = len(self.models)
        self.dt_r = dt_r
        self.dt_m = dt_m
        self.length_half = length_half
        self.width_half = width_half
        self.prob_backend = prob_backend
        self.genz_samples = genz_samples
        self.scipy_maxdim = scipy_maxdim
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.substeps = int(round(dt_r / dt_m))
        if self.substeps < 1:
            self.substeps = 1
        # Plant matrix for one modeling substep (Euler, as in the paper).
        self.A_sub = build_system_matrix(self.models, dt_m, method="euler")

    # ------------------------------------------------------------------ #
    def verify(self, initial: ProbStar, collision_H: np.ndarray,
               collision_d: np.ndarray, k_max: int,
               record_centers: bool = True,
               recompute_collision: bool = False,
               ego_index: int = 0, obs_index: int = 1,
               substep_matrices: Optional[Sequence[np.ndarray]] = None
               ) -> VerificationResult:
        """Run Algorithm 1 for ``k_max`` reachability steps.

        Args:
            initial: initial ProbStar Theta_0.
            collision_H, collision_d: collision half-space ``H X <= d`` used when
                ``recompute_collision`` is False.
            k_max: number of reachability steps.
            record_centers: store reachable-set centers for plotting.
            recompute_collision: if True, re-derive the collision half-space at
                each step from the current reachable-set-center headings of the
                ego and obstacle, matching Algorithm 1's ``collisionPredicate(Theta_k)``
                (line 27).  This matters when vehicles turn (their bounding boxes
                rotate); for straight-line motion the fixed ``collision_H`` is
                already exact, so the default is False (exact and faster there).
            ego_index, obs_index: vehicle indices for the collision pair, used
                only when ``recompute_collision`` is True.
            substep_matrices: optional list of length ``k_max`` of modeling-substep
                plant matrices (one per reach step), for a TIME-VARYING plant such
                as a braking ego whose speed changes over the horizon (see
                ``dynamics.braking_substep_matrices``).  If None, the fixed
                ``self.A_sub`` is used for every step.

        Returns:
            A VerificationResult.
        """
        if substep_matrices is not None and len(substep_matrices) != k_max:
            raise ValueError("substep_matrices must have length k_max")

        t0 = time.perf_counter()
        theta = initial.clone()
        per_step = []
        centers = [theta.center.copy()] if record_centers else None

        for _k in range(k_max):
            A_sub = (substep_matrices[_k] if substep_matrices is not None
                     else self.A_sub)
            # --- inner modeling substeps (propagate dt_r worth of dt_m steps) ---
            for _ in range(self.substeps):
                theta = theta.affine_map(A=A_sub)
            if record_centers:
                centers.append(theta.center.copy())

            # --- collision predicate (optionally heading-aware, as in Alg. 1) ---
            if recompute_collision:
                H, d = self._collision_at(theta, ego_index, obs_index)
            else:
                H, d = collision_H, collision_d

            # --- intersect with collision half-space and get probability ---
            unsafe = theta.intersect_halfspace(H, d)
            p = unsafe.estimate_probability(
                backend=self.prob_backend, genz_samples=self.genz_samples,
                scipy_maxdim=self.scipy_maxdim, rng=self.rng)
            per_step.append(float(np.clip(p, 0.0, 1.0)))

        runtime = time.perf_counter() - t0

        per_step = np.array(per_step)
        times = (np.arange(1, k_max + 1)) * self.dt_r
        # cumulative aggregations (see module docstring)
        cum_indep = 1.0 - np.cumprod(1.0 - per_step)
        cum_max = np.maximum.accumulate(per_step) if len(per_step) else per_step

        return VerificationResult(
            per_step_prob=per_step,
            times=times,
            cumulative_independent=cum_indep,
            cumulative_max=cum_max,
            runtime_s=runtime,
            reachable_centers=(np.array(centers) if record_centers else None),
            meta={"substeps": self.substeps, "k_max": k_max,
                  "dt_r": self.dt_r, "dt_m": self.dt_m},
        )

    # ------------------------------------------------------------------ #
    def _collision_at(self, theta: ProbStar, ego_index: int, obs_index: int):
        """Collision half-space from the current reachable-set-center headings.

        Reads the (phi_x, phi_y) components of the ego and obstacle from the
        reachable set's center, converts them to headings, and rebuilds the
        Minkowski-difference collision half-space for the rotated bounding
        boxes (Algorithm 1, ``collisionPredicate(Theta_k)``).
        """
        c = theta.center
        ex = STATE_DIM * ego_index
        ox = STATE_DIM * obs_index
        head_pi = float(np.arctan2(c[ex + 3], c[ex + 2]))
        head_om = float(np.arctan2(c[ox + 3], c[ox + 2]))
        return collision_half_space(
            self.n_vehicles, ego_index, obs_index,
            self.length_half, self.width_half,
            heading_pi=head_pi, heading_omega=head_om)

    # ------------------------------------------------------------------ #
    def reachable_sets(self, initial: ProbStar, k_max: int) -> List[ProbStar]:
        """Return the list of reachable ProbStars Theta_0 .. Theta_{k_max}."""
        theta = initial.clone()
        out = [theta.clone()]
        for _k in range(k_max):
            for _ in range(self.substeps):
                theta = theta.affine_map(A=self.A_sub)
            out.append(theta.clone())
        return out
