"""Tests for the truncated multivariate-normal probability evaluator.

These check the evaluator that replaces the (removed) scipy.stats.mvn.mvnun
against analytic values and brute-force Monte-Carlo integration.
"""

import numpy as np
import pytest

from probstar_rv.truncated_mvn import probability, scipy_cdf, genz_sov, genz_botev


def test_1d_standard_normal():
    # P(-1 < Z < 1) = 0.682689...
    p = probability(np.array([-1.0]), np.array([1.0]),
                    np.array([0.0]), np.array([[1.0]]))
    assert abs(p - 0.6826894921) < 1e-4


def test_2d_independent_quadrant():
    # P(X1<0, X2<0) for independent standard normals = 0.25
    p = probability(np.array([-np.inf, -np.inf]), np.array([0.0, 0.0]),
                    np.zeros(2), np.eye(2))
    assert abs(p - 0.25) < 1e-4


def test_1d_tail_matches_norm_cdf():
    from scipy.stats import norm
    g = genz_sov(np.array([-np.inf]), np.array([-2.0]),
                   np.array([0.0]), np.array([[1.0]]),
                   n_samples=100000, rng=np.random.default_rng(0))
    assert abs(g.prob - norm.cdf(-2.0)) < 1e-3


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_2d_correlated_box_vs_mc(seed):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=2) * 0.3
    A = rng.normal(size=(2, 2))
    Sig = A @ A.T + np.eye(2) * 0.5
    sd = np.sqrt(np.diag(Sig))
    lb = mu - 1.3 * sd
    ub = mu + 1.1 * sd
    p_scipy = scipy_cdf(lb, ub, mu, Sig)
    g = genz_sov(lb, ub, mu, Sig, n_samples=300000, rng=rng)
    X = rng.multivariate_normal(mu, Sig, size=3_000_000)
    p_mc = np.mean(np.all((X >= lb) & (X <= ub), axis=1))
    assert abs(p_scipy - p_mc) < 3e-3
    assert abs(g.prob - p_mc) < 5e-3


def test_6d_genz_vs_scipy_vs_mc():
    rng = np.random.default_rng(7)
    d = 6
    A = rng.normal(size=(d, d))
    Sig = A @ A.T / d + np.eye(d) * 0.5
    mu = rng.normal(size=d) * 0.2
    sd = np.sqrt(np.diag(Sig))
    lb = mu - 1.5 * sd
    ub = mu + 1.2 * sd
    p_scipy = scipy_cdf(lb, ub, mu, Sig)
    g = genz_sov(lb, ub, mu, Sig, n_samples=400000, rng=rng)
    X = rng.multivariate_normal(mu, Sig, size=3_000_000)
    p_mc = np.mean(np.all((X >= lb) & (X <= ub), axis=1))
    assert abs(p_scipy - p_mc) < 5e-3
    assert abs(g.prob - p_mc) < 7e-3


def test_empty_box_is_zero():
    # ub < lb in one coordinate -> probability 0
    p = probability(np.array([1.0, 0.0]), np.array([0.5, 1.0]),
                    np.zeros(2), np.eye(2))
    assert p == 0.0


def test_genz_botev_alias_points_to_genz_sov():
    # Backward-compatible alias must be the exact same function object.
    assert genz_botev is genz_sov
