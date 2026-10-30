r"""Monte-Carlo collision-probability estimator (Section 5.4).

A check that shares no code with ProbStar: sample the initial state from
N(mu, Sigma), roll each sample forward with the exact discrete dynamics, and test
the collision condition on the concrete sampled states. This gives a ground-truth
reference for:

* per-step probability p_k (fraction of samples colliding at step k), and
* the true union probability P(at least one collision over the horizon) (fraction
  colliding at any step), which handles the temporal dependence that the per-step
  probabilities can't be combined to give on their own.

Table 4 of the paper compares ProbStar against this kind of Monte Carlo on two
hand-computable scenarios.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .dynamics import VehicleModel, build_system_matrix, STATE_DIM


@dataclass
class MonteCarloResult:
    per_step_prob: np.ndarray       # p_k, k=1..n_steps
    union_prob: float               # P(at least one collision over horizon)
    times: np.ndarray
    n_samples: int
    runtime_s: float
    per_step_stderr: np.ndarray     # binomial standard error per step
    union_stderr: float


class MonteCarloValidator:
    """Samples the initial Gaussian and rolls out exact dynamics."""

    def __init__(self, models: Sequence[VehicleModel], dt_r: float, dt_m: float,
                 length_half: float, width_half: float,
                 rng: Optional[np.random.Generator] = None):
        self.models = list(models)
        self.n_vehicles = len(self.models)
        self.dt_r = dt_r
        self.dt_m = dt_m
        self.length_half = length_half
        self.width_half = width_half
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.substeps = max(1, int(round(dt_r / dt_m)))
        self.A_sub = build_system_matrix(self.models, dt_m, method="euler")
        # One reachability-step transition = substeps modeling steps.
        self.A_step = np.linalg.matrix_power(self.A_sub, self.substeps)

    def run(self, mu: np.ndarray, Sigma: np.ndarray, k_max: int,
            n_samples: int, ego_index: int = 0, obs_index: int = 1,
            collision_C: Optional[np.ndarray] = None,
            collision_d: Optional[np.ndarray] = None,
            recompute_collision: bool = False,
            step_matrices: Optional[Sequence[np.ndarray]] = None
            ) -> MonteCarloResult:
        """Estimate per-step and union collision probabilities by simulation.

        Args:
            mu, Sigma: initial-state Gaussian (stacked 4n-dim).
            k_max: number of reachability steps to simulate.
            n_samples: number of Monte-Carlo samples.
            ego_index, obs_index: vehicle indices for the collision pair.
            collision_C, collision_d: optional precomputed xy-plane collision
                polytope ``C r <= d`` (from ``minkowski_difference_box``).  If
                None, an axis-aligned box of summed half-extents is used.
            recompute_collision: if True, recompute the collision polygon at each
                step from the per-step mean headings of the two vehicles (to
                match the verifier's heading-aware ``collisionPredicate(Theta_k)``).
            step_matrices: optional list of length ``k_max`` of modeling-SUBSTEP
                plant matrices (one per reach step), for a time-varying plant
                (e.g. braking).  Each is raised to the ``substeps`` power to form
                that reach step's full transition, matching the verifier's
                ``substep_matrices`` so the two stay directly comparable.  If
                None, the fixed ``self.A_step`` is used for every step.

        Returns:
            A MonteCarloResult.
        """
        from .collision import minkowski_difference_box

        if step_matrices is not None and len(step_matrices) != k_max:
            raise ValueError("step_matrices must have length k_max")

        t0 = time.perf_counter()
        mu = np.asarray(mu, dtype=float).ravel()
        Sigma = np.asarray(Sigma, dtype=float)

        if collision_C is None:
            # axis-aligned: |rx| <= 2*lh, |ry| <= 2*wh
            collision_C = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
            collision_d = np.array([2 * self.length_half, 2 * self.length_half,
                                    2 * self.width_half, 2 * self.width_half])

        X = self.rng.multivariate_normal(mu, Sigma, size=n_samples)  # (N, dim)
        mean_state = mu.copy()  # tracks the per-step mean for heading recompute

        ex = STATE_DIM * ego_index
        ox = STATE_DIM * obs_index

        collided_any = np.zeros(n_samples, dtype=bool)
        per_step_counts = np.zeros(k_max)

        for k in range(k_max):
            if step_matrices is not None:
                A_step_k = np.linalg.matrix_power(step_matrices[k], self.substeps)
            else:
                A_step_k = self.A_step
            X = X @ A_step_k.T                          # advance one reach step
            if recompute_collision:
                mean_state = A_step_k @ mean_state
                head_pi = float(np.arctan2(mean_state[ex + 3], mean_state[ex + 2]))
                head_om = float(np.arctan2(mean_state[ox + 3], mean_state[ox + 2]))
                collision_C, collision_d = minkowski_difference_box(
                    self.length_half, self.width_half,
                    heading_pi=head_pi, heading_omega=head_om)
            rel = X[:, ox:ox + 2] - X[:, ex:ex + 2]    # (N, 2) relative offset
            inside = np.all(rel @ collision_C.T <= collision_d + 1e-12, axis=1)
            per_step_counts[k] = inside.sum()
            collided_any |= inside

        runtime = time.perf_counter() - t0
        per_step = per_step_counts / n_samples
        union = collided_any.mean()
        times = np.arange(1, k_max + 1) * self.dt_r
        per_step_se = np.sqrt(per_step * (1 - per_step) / n_samples)
        union_se = np.sqrt(union * (1 - union) / n_samples)

        return MonteCarloResult(
            per_step_prob=per_step, union_prob=float(union), times=times,
            n_samples=n_samples, runtime_s=runtime,
            per_step_stderr=per_step_se, union_stderr=float(union_se),
        )
