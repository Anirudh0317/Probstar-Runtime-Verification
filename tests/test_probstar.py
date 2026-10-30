"""Tests for the ProbStar set operations: construction, affine map,
intersection, and probability -- all cross-checked against Monte Carlo on the
represented set itself.
"""

import numpy as np
import pytest

from probstar_rv import ProbStar


def _mc_prob_on_set(mu, Sig, constraints, n=3_000_000, seed=0):
    """Monte-Carlo probability that N(mu,Sig) satisfies a list of (H, g) rows
    meaning H x <= g."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mu, Sig, size=n)
    mask = np.ones(n, dtype=bool)
    for H, g in constraints:
        mask &= (X @ np.asarray(H) <= g + 1e-12)
    return mask.mean()


def test_from_gaussian_probability_is_one():
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    assert abs(P.estimate_probability() - 1.0) < 1e-3


def test_affine_map_center_and_covariance():
    rng = np.random.default_rng(1)
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    A = np.array([[0.0, -1.0], [1.0, 0.0]])
    b = np.array([2.0, 3.0])
    P2 = P.affine_map(A, b)
    assert np.allclose(P2.center, A @ mu + b, atol=1e-9)
    S = P2.sample(200000, rng)
    assert np.allclose(S.mean(0), A @ mu + b, atol=2e-2)
    assert np.allclose(np.cov(S.T), A @ Sig @ A.T, atol=3e-2)


def test_single_halfspace_probability_vs_mc():
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    Pi = P.intersect_halfspace(np.array([[1.0, 0.0]]), np.array([1.0]))
    p_ps = Pi.estimate_probability()
    p_mc = _mc_prob_on_set(mu, Sig, [([1.0, 0.0], 1.0)])
    assert abs(p_ps - p_mc) < 5e-3


def test_two_halfspaces_probability_vs_mc():
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    Pi = P.intersect_halfspace(np.array([[1.0, 0.0], [0.0, -1.0]]),
                               np.array([1.3, 0.8]))
    p_ps = Pi.estimate_probability()
    p_mc = _mc_prob_on_set(mu, Sig, [([1.0, 0.0], 1.3), ([0.0, -1.0], 0.8)])
    assert abs(p_ps - p_mc) < 5e-3


def test_sequential_equals_batched_intersection():
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    batched = P.intersect_halfspace(np.array([[1.0, 0.0], [0.0, -1.0]]),
                                    np.array([1.3, 0.8]))
    seq = (P.intersect_halfspace(np.array([[1.0, 0.0]]), np.array([1.3]))
             .intersect_halfspace(np.array([[0.0, -1.0]]), np.array([0.8])))
    assert abs(batched.estimate_probability()
               - seq.estimate_probability()) < 2e-3


def test_diagonal_constraint_vs_mc():
    mu = np.array([1.0, -0.5])
    Sig = np.array([[0.4, 0.1], [0.1, 0.3]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    Pi = P.intersect_halfspace(np.array([[1.0, 1.0]]), np.array([0.5]))
    p_ps = Pi.estimate_probability()
    p_mc = _mc_prob_on_set(mu, Sig, [([1.0, 1.0], 0.5)])
    assert abs(p_ps - p_mc) < 5e-3


def test_affine_then_intersect_matches_state_space_constraint():
    # Mapping the set and constraining in mapped space must equal constraining
    # the pushed-forward constraint in original space.
    mu = np.array([0.2, 0.4])
    Sig = np.array([[0.5, -0.2], [-0.2, 0.6]])
    P = ProbStar.from_gaussian(mu, Sig, delta=6.0)
    A = np.array([[1.0, 0.5], [0.0, 1.0]])
    b = np.array([0.1, -0.3])
    Pm = P.affine_map(A, b)
    Pmc = Pm.intersect_halfspace(np.array([[1.0, 0.0]]), np.array([0.8]))
    # equivalent MC: y = A x + b, constraint y0 <= 0.8
    rng = np.random.default_rng(3)
    X = rng.multivariate_normal(mu, Sig, 3_000_000)
    Y = X @ A.T + b
    p_mc = np.mean(Y[:, 0] <= 0.8)
    assert abs(Pmc.estimate_probability() - p_mc) < 5e-3
