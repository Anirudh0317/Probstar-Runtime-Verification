r"""Probabilistic Star (ProbStar) set.

Self-contained reimplementation of the ProbStar representation from the paper
(Definition 1, Definition 2, Propositions 1 and 2), checked against the official
StarV code (``StarV/set/probstar.py`` in https://github.com/V2A2/StarV).

A ProbStar Theta = <c, V, N, P, l, u> is the set

    [[Theta]] = { x = c + sum_i alpha_i v_i
                  | alpha ~ N(mu, Sigma),  C alpha <= d,  l <= alpha <= u }.

Stored as the same tuple of arrays StarV uses:

    V        : (n, m+1) basis matrix; column 0 is the center c, columns 1..m are
               the basis vectors v_1..v_m. (Augmented layout from StarV, so an
               affine map only touches V.)
    C, d     : half-space predicate  C alpha <= d   (may be empty).
    mu, Sig  : mean and covariance of the predicate variables alpha.
    pred_lb, : box bounds on the predicate variables (l, u).
    pred_ub

Main operations:
* affine_map(A, b)        -- Prop. 2: V <- [A c + b | A V_basis].
* intersect_halfspace(H,g)-- Prop. 1: append H alpha <= g to (C, d) and tighten
                             the predicate box.
* estimate_probability()  -- Def. 2: truncated-MVN probability of the predicate,
                             via probstar_rv.truncated_mvn.

Where this differs from StarV:
1. Probability integral. StarV calls scipy.stats.mvn.mvnun (removed in SciPy
   >= 1.13). We call truncated_mvn.probability, which uses
   scipy.stats.multivariate_normal.cdf at small dimension and a Genz SOV Monte
   Carlo estimator (variable reordering + antithetic variates, not Botev
   tilting) otherwise. The constraint handling -- stacking the box as two-sided
   constraints and turning C alpha <= d into a half-space normal integral,
   including the singular C Sig C^T SVD path -- follows StarV's
   estimateProbability.
2. LP solver. StarV defaults to gurobipy (commercial) for exact predicate
   ranges; we use scipy.optimize.linprog (HiGHS) where needed. The reachability
   pipeline needs no LP -- it uses the same analytic bound tightening StarV has
   (updatePredicateRanges).
3. We expose a small typed API, only the operations Algorithm 1 needs, not
   StarV's full class.
"""

from __future__ import annotations

import copy
from typing import Optional, Tuple

import numpy as np

from .truncated_mvn import probability as _tmvn_probability


class ProbStar:
    """A probabilistic star set (see module docstring)."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, V: np.ndarray, C, d, mu: np.ndarray, Sig: np.ndarray,
                 pred_lb: np.ndarray, pred_ub: np.ndarray, copy_: bool = True):
        """Build a ProbStar from its full tuple (StarV 7-argument form).

        Args:
            V: (n, m+1) augmented basis matrix (column 0 = center).
            C: (k, m) constraint matrix, or empty array / None.
            d: (k,) constraint vector, or empty array / None.
            mu: (m,) predicate mean.
            Sig: (m, m) predicate covariance (SPD).
            pred_lb, pred_ub: (m,) predicate box bounds.
            copy_: deep-copy inputs if True.
        """
        if copy_:
            V, C, d, mu, Sig, pred_lb, pred_ub = copy.deepcopy(
                (V, C, d, mu, Sig, pred_lb, pred_ub))

        V = np.asarray(V, dtype=float)
        mu = np.asarray(mu, dtype=float).ravel()
        Sig = np.asarray(Sig, dtype=float)
        pred_lb = np.asarray(pred_lb, dtype=float).ravel()
        pred_ub = np.asarray(pred_ub, dtype=float).ravel()

        if C is None or (hasattr(C, "__len__") and len(C) == 0):
            C = np.zeros((0, V.shape[1] - 1))
            d = np.zeros((0,))
        else:
            C = np.asarray(C, dtype=float)
            d = np.asarray(d, dtype=float).ravel()

        assert V.ndim == 2, "V must be a 2D array"
        assert Sig.shape[0] == Sig.shape[1] == mu.shape[0], \
            "mu/Sigma dimension mismatch"
        assert pred_lb.shape[0] == pred_ub.shape[0] == mu.shape[0], \
            "predicate bound dimension mismatch"
        assert V.shape[1] == mu.shape[0] + 1, \
            "V must have m+1 columns (center + m basis vectors)"

        self.V = V
        self.C = C
        self.d = d
        self.mu = mu
        self.Sig = Sig
        self.pred_lb = pred_lb
        self.pred_ub = pred_ub

        self.dim = V.shape[0]
        self.nVars = V.shape[1] - 1

    # ------------------------------------------------------------------ #
    # Alternative constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_gaussian(cls, mu: np.ndarray, Sig: np.ndarray,
                      delta: float = 6.0) -> "ProbStar":
        """Build the standardized initial ProbStar for a Gaussian state.

        Paper Section 4.1: from X0 ~ N(mu, Sigma), standardize so the predicate
        variables are alpha ~ N(0, I) and the state is x = mu + L alpha, with
        L L^T = Sigma (Cholesky). The box is the +/- delta-sigma truncation
        l = -delta, u = +delta (delta = 6 in the paper), which holds nearly all
        the probability mass and keeps the truncated-normal integral well behaved.

        Args:
            mu: (n,) state mean.
            Sig: (n, n) state covariance (SPD).
            delta: predicate box half-width in standard deviations.

        Returns:
            A ProbStar whose [[.]] is {mu + L alpha : alpha~N(0,I), -delta<=alpha<=delta}.
        """
        mu = np.asarray(mu, dtype=float).ravel()
        Sig = np.asarray(Sig, dtype=float)
        n = mu.shape[0]
        # Cholesky "A" with A A^T = Sigma  (StarV uses diag(sigma) for the
        # diagonal case; a full Cholesky generalizes it to correlated Sigma).
        L = _safe_cholesky(Sig)
        V = np.hstack([mu.reshape(n, 1), L])         # x = mu + L alpha
        mu_a = np.zeros(n)
        Sig_a = np.eye(n)
        lb = -delta * np.ones(n)
        ub = delta * np.ones(n)
        C = np.zeros((0, n))
        d = np.zeros((0,))
        return cls(V, C, d, mu_a, Sig_a, lb, ub, copy_=False)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    @property
    def center(self) -> np.ndarray:
        """The center vector c (column 0 of V)."""
        return self.V[:, 0].copy()

    @property
    def basis(self) -> np.ndarray:
        """The basis matrix [v_1 ... v_m] (columns 1.. of V)."""
        return self.V[:, 1:].copy()

    def clone(self) -> "ProbStar":
        return copy.deepcopy(self)

    # ------------------------------------------------------------------ #
    # Proposition 2: affine mapping
    # ------------------------------------------------------------------ #
    def affine_map(self, A: Optional[np.ndarray] = None,
                   b: Optional[np.ndarray] = None) -> "ProbStar":
        r"""Return A * self + b as a new ProbStar (Proposition 2).

        Only the basis matrix V changes:  V' = A V, then V'[:,0] += b.  The
        predicate (C, d, mu, Sigma, l, u) is unchanged.
        """
        V = self.V.copy()
        if A is not None:
            A = np.asarray(A, dtype=float)
            assert A.shape[1] == self.dim, "A/ProbStar dimension mismatch"
            V = A @ V
        if b is not None:
            b = np.asarray(b, dtype=float).ravel()
            V[:, 0] = V[:, 0] + b
        return ProbStar(V, self.C, self.d, self.mu, self.Sig,
                        self.pred_lb, self.pred_ub)

    # ------------------------------------------------------------------ #
    # Proposition 1: intersection with a half-space H x <= g
    # ------------------------------------------------------------------ #
    def intersect_halfspace(self, H: np.ndarray, g: np.ndarray) -> "ProbStar":
        r"""Intersect with the half-space { x : H x <= g } (Proposition 1).

        In predicate space, x = c + V_basis alpha, so H x <= g becomes

            (H V_basis) alpha <= g - H c,

        which is appended to the predicate constraints.  The box bounds are
        tightened by the analytic single-constraint bound propagation
        (``_update_predicate_ranges``), mirroring StarV's ``addConstraint``.
        """
        H = np.atleast_2d(np.asarray(H, dtype=float))
        g = np.asarray(g, dtype=float).ravel()
        c = self.center
        Vb = self.basis

        newC = H @ Vb                      # (k, m)
        newd = g - H @ c                   # (k,)

        if self.C.shape[0] == 0:
            C = newC
            d = newd
        else:
            C = np.vstack([self.C, newC])
            d = np.concatenate([self.d, newd])

        pred_lb = self.pred_lb.copy()
        pred_ub = self.pred_ub.copy()
        for i in range(newC.shape[0]):
            pred_lb, pred_ub = _update_predicate_ranges(
                newC[i], newd[i], pred_lb, pred_ub)

        return ProbStar(self.V, C, d, self.mu, self.Sig, pred_lb, pred_ub)

    # ------------------------------------------------------------------ #
    # Definition 2: probability of the probstar
    # ------------------------------------------------------------------ #
    def estimate_probability(self, backend: str = "auto",
                             genz_samples: int = 20000,
                             scipy_maxdim: int = 6,
                             rng: Optional[np.random.Generator] = None) -> float:
        r"""Probability P(Theta) that the predicate variables satisfy the
        constraints and box bounds (Definition 2).

        Same approach as StarV's estimateProbability:

        * No constraints: truncated-normal mass of the box, P(l <= alpha <= u)
          under N(mu, Sigma).
        * With constraints C alpha <= d: stack the box as two-sided constraints
          and form A = C_full Sig C_full^T. If A is SPD, the constraint set is a
          shifted normal rectangle with upper bound d_full - C_full mu and
          covariance A (lower bound -inf). If A is singular, add auxiliary
          variables via the SVD path.
        """
        mu, Sig = self.mu, self.Sig
        lb, ub = self.pred_lb, self.pred_ub
        n = self.nVars

        if self.C.shape[0] == 0:
            return _tmvn_probability(lb, ub, mu, Sig, backend=backend,
                                     genz_samples=genz_samples,
                                     scipy_maxdim=scipy_maxdim, rng=rng)

        # Stack box bounds as explicit constraints: I alpha <= ub, -I alpha <= -lb.
        C1 = np.vstack([np.eye(n), -np.eye(n)])
        d1 = np.concatenate([ub, -lb])
        C = np.vstack([self.C, C1])
        d = np.concatenate([self.d, d1])

        A = C @ Sig @ C.T
        eig = np.linalg.eigvalsh(A)
        if np.all(eig > 1e-12):
            new_lb = np.full(len(d), -np.inf)
            new_ub = d - C @ mu
            new_mu = np.zeros(len(d))
            new_Sig = A
            return _tmvn_probability(new_lb, new_ub, new_mu, new_Sig,
                                     backend=backend, genz_samples=genz_samples,
                                     scipy_maxdim=scipy_maxdim, rng=rng)

        # Singular case: SVD path (mirrors StarV).
        U, Q, Lt = np.linalg.svd(C)
        r = Q.shape[0]
        Q1 = np.diag(Q) @ Lt[:r, :]
        mu1 = Q1 @ mu
        Sig1 = Q1 @ Sig @ Q1.T
        m_aux = U.shape[0] - len(Q)
        mu2 = np.zeros(m_aux)
        Sig2 = 1e-10 * np.eye(m_aux)
        new_mu = np.concatenate([mu1, mu2])
        from scipy.linalg import block_diag
        new_Sig = block_diag(Sig1, Sig2)
        new_ub = d - U @ new_mu
        new_lb = np.full(len(d), -np.inf)
        new_Sig = U @ new_Sig @ U.T
        new_Sig = 0.5 * (new_Sig + new_Sig.T) + 1e-12 * np.eye(len(d))
        return _tmvn_probability(new_lb, new_ub, np.zeros(len(d)), new_Sig,
                                 backend=backend, genz_samples=genz_samples,
                                 scipy_maxdim=scipy_maxdim, rng=rng)

    # ------------------------------------------------------------------ #
    # Sampling (for cross-validation and visualization)
    # ------------------------------------------------------------------ #
    def sample(self, n: int, rng: Optional[np.random.Generator] = None
               ) -> np.ndarray:
        """Sample ``n`` states from the probstar by rejection sampling.

        Draws alpha ~ N(mu, Sigma), keeps those inside the box and satisfying
        C alpha <= d, and maps them through x = c + V_basis alpha.  Returns an
        (n, dim) array (fewer rows if acceptance is very low).
        """
        if rng is None:
            rng = np.random.default_rng(0)
        kept = []
        batch = max(n * 4, 1000)
        tries = 0
        while sum(len(k) for k in kept) < n and tries < 200:
            A = rng.multivariate_normal(self.mu, self.Sig, size=batch)
            inside = np.all((A >= self.pred_lb) & (A <= self.pred_ub), axis=1)
            if self.C.shape[0] > 0:
                inside &= np.all(A @ self.C.T <= self.d + 1e-12, axis=1)
            kept.append(A[inside])
            tries += 1
        if not kept:
            return np.zeros((0, self.dim))
        A = np.vstack(kept)[:n]
        return (self.center[None, :] + A @ self.basis.T)

    def __repr__(self) -> str:
        return (f"ProbStar(dim={self.dim}, nVars={self.nVars}, "
                f"nConstraints={self.C.shape[0]})")


# ---------------------------------------------------------------------- #
# Module-level numerical helpers
# ---------------------------------------------------------------------- #
def _safe_cholesky(Sig: np.ndarray) -> np.ndarray:
    """Lower-triangular factor L with L L^T ~= Sigma, robust to tiny negatives.

    Adds a minimal diagonal jitter if Sigma is only numerically PSD.
    """
    Sig = 0.5 * (Sig + Sig.T)
    try:
        return np.linalg.cholesky(Sig)
    except np.linalg.LinAlgError:
        w, Vv = np.linalg.eigh(Sig)
        w = np.clip(w, 1e-12, None)
        return np.linalg.cholesky(Vv @ np.diag(w) @ Vv.T)


def _update_predicate_ranges(newC_row: np.ndarray, newd_val: float,
                             pred_lb: np.ndarray, pred_ub: np.ndarray
                             ) -> Tuple[np.ndarray, np.ndarray]:
    r"""Tighten box bounds given one new constraint ``newC_row . alpha <= newd``.

    This is the analytic interval-propagation StarV uses in
    ``updatePredicateRanges``: for each variable i with coefficient x = c_i != 0,
    solve the constraint for alpha_i and bound the remaining terms using the
    current box, giving a one-sided tightening of alpha_i's range.
    """
    new_lb = pred_lb.copy()
    new_ub = pred_ub.copy()
    m = newC_row.shape[0]
    for i in range(m):
        x = newC_row[i]
        if x == 0:
            continue
        # Solve the constraint for alpha_i:  alpha_i (<= or >=) d1 - v1 . alpha_rest,
        # where v1 = (newC / x) with the i-th entry removed and d1 = newd / x.
        v1 = np.delete(newC_row / x, i)
        d1 = newd_val / x
        v2 = -v1
        v_pos = np.where(v2 < 0, 0.0, v2)   # positive part of -v1
        v_neg = np.where(v2 > 0, 0.0, v2)   # negative part of -v1
        lb_rest = np.delete(pred_lb, i)
        ub_rest = np.delete(pred_ub, i)
        if x > 0:
            # alpha_i <= d1 + (-v1) . alpha_rest ; maximize RHS over the box.
            xmax = d1 + v_pos @ ub_rest + v_neg @ lb_rest
            new_ub[i] = min(xmax, new_ub[i])
        else:  # x < 0: dividing by x flips the inequality -> alpha_i >= ...
            # minimize RHS over the box.
            xmin = d1 + v_pos @ lb_rest + v_neg @ ub_rest
            new_lb[i] = max(xmin, new_lb[i])
    # Guard against crossed bounds from an infeasible constraint.
    cross = new_lb > new_ub
    if np.any(cross):
        new_ub[cross] = new_lb[cross]
    return new_lb, new_ub
