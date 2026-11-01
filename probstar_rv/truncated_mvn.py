r"""Probability of a truncated multivariate normal, P(l <= X <= u) for
X ~ N(mu, Sigma). ProbStar.estimate_probability relies on this.

StarV computes the same integral with scipy.stats.mvn.mvnun, but SciPy removed
that function in 1.13, so it no longer runs on a current SciPy. This module has
two replacements:

1. scipy_cdf: inclusion-exclusion over scipy.stats.multivariate_normal.cdf.
   Accurate for low dimension (ours is 4 or 8), but gets slow and less accurate
   as dimension grows.

2. genz_sov: Genz's separation-of-variables Monte Carlo estimator. It reorders
   the variables (_cholperm) so the tightest axis is integrated first, uses
   antithetic variates, and reports both an estimate and a relative standard
   error. Scales better than scipy_cdf in high dimension.

   Note: this is Genz SOV with variable reordering, not Botev's minimax
   exponential tilting. _cholperm uses the Gibbs/Genz ordering (truncated-normal
   means to pick pivots); it does not solve for tilting parameters. SOV with
   reordering is accurate enough at our dimensions (4-8), which the tests against
   Monte Carlo confirm. Tilting would mainly help for very rare events in high
   dimension; not implemented.

probability() picks between the two based on config.

References:
* A. Genz (1992), "Numerical computation of multivariate normal probabilities",
  J. Comp. Graph. Stat. 1, 141-149. (The SOV estimator used here.)
* Z. Botev (2017), "The normal law under linear restrictions: simulation and
  estimation via minimax tilting", JRSS-B 79(1), 125-148. (The tilting method,
  cited for context, not implemented here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.stats import norm


# Low-dimension path: scipy's multivariate_normal.cdf
def scipy_cdf(lb: np.ndarray, ub: np.ndarray, mu: np.ndarray,
              Sigma: np.ndarray) -> float:
    """P(lb <= X <= ub) by inclusion-exclusion over the scipy normal CDF.

    Uses the box-probability identity

        P(lb<=X<=ub) = sum_{corners} (-1)^{#lower-bounds-used} * F(corner)

    with F the CDF P(X <= corner). That's 2**n CDF calls, so it's only for small
    n (<= ~6). Dimensions with lb = -inf are left out of the corner enumeration.
    """
    from scipy.stats import multivariate_normal

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    # Dimensions whose lower bound is -inf never contribute a "lower" corner.
    finite_lb = np.isfinite(lb)
    idx_finite = np.where(finite_lb)[0]

    rv = multivariate_normal(mean=mu, cov=Sigma, allow_singular=True)

    total = 0.0
    m = len(idx_finite)
    # Enumerate the 2**m choices of using ub (0) or lb (1) on finite-lb dims.
    for mask in range(1 << m):
        corner = ub.copy()
        sign = 1.0
        for bit in range(m):
            if mask & (1 << bit):
                corner[idx_finite[bit]] = lb[idx_finite[bit]]
                sign = -sign
        total += sign * rv.cdf(corner)
    return float(np.clip(total, 0.0, 1.0))


# General path: Genz separation-of-variables (SOV)
@dataclass
class TMVNResult:
    prob: float
    rel_err: float


def _cholperm(Sigma: np.ndarray, l: np.ndarray, u: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cholesky factor with Gibbs/Genz variable reordering.

    Puts the tightest integration axis first, which cuts the variance of the SOV
    estimator a lot. Returns (L, l, u, perm): L lower-triangular with L @ L.T the
    permuted covariance, and l, u the permuted bounds.
    """
    d = Sigma.shape[0]
    L = np.zeros((d, d))
    z = np.zeros(d)
    perm = np.arange(d)
    l = l.astype(float).copy()
    u = u.astype(float).copy()
    Sig = Sigma.astype(float).copy()

    for j in range(d):
        pr = np.full(d, np.inf)
        I = np.arange(j, d)
        # Diagonal residual variances for candidate pivots.
        D = np.diag(Sig)
        s = D[I] - np.sum(L[I, :j] ** 2, axis=1)
        s[s < 0] = np.finfo(float).eps
        s = np.sqrt(s)
        # Conditional bounds given variables chosen so far.
        tl = (l[I] - L[I, :j] @ z[:j]) / s
        tu = (u[I] - L[I, :j] @ z[:j]) / s
        pr[I] = _lnNpr(tl, tu)
        # Pick the pivot with smallest (most negative) expected probability mass.
        k = int(np.argmin(pr))
        # Swap k into position j across Sig, L, l, u, perm.
        jk = [j, k]
        kj = [k, j]
        Sig[jk, :] = Sig[kj, :]
        Sig[:, jk] = Sig[:, kj]
        L[jk, :] = L[kj, :]
        l[jk] = l[kj]
        u[jk] = u[kj]
        perm[jk] = perm[kj]

        # Compute the j-th Cholesky column.
        s_jj = Sig[j, j] - np.sum(L[j, :j] ** 2)
        s_jj = max(s_jj, np.finfo(float).eps)
        L[j, j] = np.sqrt(s_jj)
        if j < d - 1:
            L[j + 1:, j] = (Sig[j + 1:, j] - L[j + 1:, :j] @ L[j, :j]) / L[j, j]
        # Expected value of the truncated standard normal at this axis (z_j).
        tl = (l[j] - L[j, :j] @ z[:j]) / L[j, j]
        tu = (u[j] - L[j, :j] @ z[:j]) / L[j, j]
        w = _trunc_norm_mean(tl, tu)
        z[j] = float(np.ravel(w)[0])
    return L, l, u, perm


def _lnNpr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """log P(a < Z < b) for standard normal Z, computed stably."""
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    p = np.zeros_like(a)
    # Both bounds positive: use upper tail.
    I = a > 0
    if np.any(I):
        pa = norm.logsf(a[I])
        pb = norm.logsf(b[I])
        p[I] = pa + np.log1p(-np.exp(pb - pa))
    # Both bounds negative: use lower tail.
    idx = b < 0
    if np.any(idx):
        pa = norm.logcdf(a[idx])
        pb = norm.logcdf(b[idx])
        p[idx] = pb + np.log1p(-np.exp(pa - pb))
    # Straddling zero.
    I = (~I) & (~idx)
    if np.any(I):
        pa = norm.cdf(a[I])
        pb = norm.sf(b[I])
        p[I] = np.log1p(-pa - pb)
    return p


def _trunc_norm_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean of a standard normal truncated to [a, b], computed stably."""
    a = np.atleast_1d(a).astype(float)
    b = np.atleast_1d(b).astype(float)
    pl = norm.cdf(a)
    pu = norm.cdf(b)
    denom = pu - pl
    denom = np.where(np.abs(denom) < 1e-300, 1e-300, denom)
    w = (norm.pdf(a) - norm.pdf(b)) / denom
    # Guard against pathological intervals.
    w = np.where(np.isfinite(w), w, 0.5 * (np.nan_to_num(a) + np.nan_to_num(b)))
    return w


def genz_sov(lb: np.ndarray, ub: np.ndarray, mu: np.ndarray,
             Sigma: np.ndarray, n_samples: int = 20000,
             rng: np.random.Generator | None = None) -> TMVNResult:
    """Estimate P(lb <= X <= ub), X~N(mu,Sigma), with Genz SOV.

    Uses the _cholperm reordering and antithetic variates. Returns the estimate
    and its relative standard error; unbiased and consistent, error ~ 1/sqrt(n).
    Not Botev tilting (see the module docstring).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    lb = np.asarray(lb, dtype=float).copy()
    ub = np.asarray(ub, dtype=float).copy()
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    d = len(mu)

    # Shift so we integrate N(0, Sigma) over [lb-mu, ub-mu].
    l = lb - mu
    u = ub - mu

    # Reorder + Cholesky for variance reduction.
    L, l, u, _ = _cholperm(Sigma, l, u)

    # SOV: for each draw, integrate one axis at a time, carrying the per-axis
    # interval through L.
    N = int(n_samples)
    # Plain MC with antithetic variates (one draw and its mirror 1-U).
    half = N // 2
    U = rng.random((max(half, 1), d - 1)) if d > 1 else np.empty((max(half, 1), 0))
    U = np.vstack([U, 1.0 - U])  # antithetic
    Nused = U.shape[0]

    vals = np.zeros(Nused)
    y = np.zeros((Nused, d))
    # First axis.
    a = l[0] / L[0, 0]
    b = u[0] / L[0, 0]
    Pa = norm.cdf(a)
    Pb = norm.cdf(b)
    p = (Pb - Pa)
    vals[:] = p
    for i in range(1, d):
        # Draw y_{i-1} from truncated normal on [a,b] using inverse CDF.
        c = Pa + U[:, i - 1] * (Pb - Pa)
        c = np.clip(c, 1e-15, 1 - 1e-15)
        y[:, i - 1] = norm.ppf(c)
        # Conditional bounds for axis i.
        s = L[i, :i] @ y[:, :i].T
        a = (l[i] - s) / L[i, i]
        b = (u[i] - s) / L[i, i]
        Pa = norm.cdf(a)
        Pb = norm.cdf(b)
        vals *= (Pb - Pa)

    prob = float(np.clip(vals.mean(), 0.0, 1.0))
    if Nused > 1:
        se = vals.std(ddof=1) / np.sqrt(Nused)
    else:
        se = 0.0
    rel_err = float(se / prob) if prob > 0 else float("inf")
    return TMVNResult(prob=prob, rel_err=rel_err)


# Dispatcher
def probability(lb: np.ndarray, ub: np.ndarray, mu: np.ndarray,
                Sigma: np.ndarray, backend: str = "auto",
                genz_samples: int = 20000, scipy_maxdim: int = 6,
                rng: np.random.Generator | None = None) -> float:
    """Compute P(lb <= X <= ub), X ~ N(mu, Sigma).

    Args:
        lb, ub: bound vectors (entries may be +/- inf).
        mu: mean vector.
        Sigma: covariance matrix (SPD).
        backend: "auto" | "scipy" | "genz".
        genz_samples: samples for the Genz SOV estimator.
        scipy_maxdim: in "auto" mode, use the scipy path up to this dimension.
        rng: optional numpy Generator for the stochastic backend.

    Returns:
        Probability in [0, 1].
    """
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)

    # Trivial / empty cases.
    if n == 0:
        return 1.0
    if np.any(ub <= lb + 0):  # empty box in some coordinate (ub < lb)
        if np.any(ub < lb):
            return 0.0

    if backend == "auto":
        backend = "scipy" if n <= scipy_maxdim else "genz"

    if backend == "scipy":
        try:
            return scipy_cdf(lb, ub, mu, Sigma)
        except Exception:
            backend = "genz"  # fall back

    res = genz_sov(lb, ub, mu, Sigma, n_samples=genz_samples, rng=rng)
    return res.prob


# Old name kept as an alias. It implied Botev tilting, which this does not do;
# genz_sov is the accurate name.
genz_botev = genz_sov
