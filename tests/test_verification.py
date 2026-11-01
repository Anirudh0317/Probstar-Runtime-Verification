"""End-to-end tests of the verification engine against the Monte-Carlo
validator and the paper's hand-computable Figure-8 trials (Table 4).
"""

import numpy as np
import pytest

from probstar_rv import (
    ProbStar, QuantitativeVerifier, MonteCarloValidator,
    collision_half_space, minkowski_difference_box, scenarios,
)


def _run_pair(scn, n_mc=200000, seed=0):
    H, d = collision_half_space(len(scn.models), scn.ego_index, scn.obs_index,
                                scn.length_half, scn.width_half)
    Cxy, dxy = minkowski_difference_box(scn.length_half, scn.width_half)
    ver = QuantitativeVerifier(scn.models, 0.25, 0.025,
                               scn.length_half, scn.width_half,
                               rng=np.random.default_rng(seed))
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=6.0)
    res = ver.verify(P0, H, d, scn.k_max)
    mc = MonteCarloValidator(scn.models, 0.25, 0.025,
                             scn.length_half, scn.width_half,
                             rng=np.random.default_rng(seed + 1))
    mcres = mc.run(scn.mu, scn.Sigma, scn.k_max, n_mc,
                   collision_C=Cxy, collision_d=dxy)
    return res, mcres


def test_trial1_rear_end_50_percent():
    scn = scenarios.trial1_rear_end()
    res, mc = _run_pair(scn)
    # Final-step probability should be the hand-computed 50%.
    assert abs(res.per_step_prob[-1] - 0.5) < 0.02
    assert abs(mc.per_step_prob[-1] - 0.5) < 0.02
    # ProbStar must agree with MC per-step.
    assert abs(res.per_step_prob[-1] - mc.per_step_prob[-1]) < 0.01


def test_trial2_side_2_28_percent():
    scn = scenarios.trial2_side()
    res, mc = _run_pair(scn, n_mc=1_000_000)
    # First-step probability should be the hand-computed 2.28%.
    assert abs(res.per_step_prob[0] - 0.0228) < 0.005
    assert abs(mc.per_step_prob[0] - 0.0228) < 0.005
    assert abs(res.per_step_prob[0] - mc.per_step_prob[0]) < 0.005


def test_probstar_tracks_mc_whole_curve():
    scn = scenarios.trial1_rear_end()
    res, mc = _run_pair(scn)
    # Every per-step probability within a small tolerance of MC.
    assert np.max(np.abs(res.per_step_prob - mc.per_step_prob)) < 0.02


def test_cumulative_bounds_order():
    # Lower bound (max) <= true union (MC) <= independence reference.
    scn = scenarios.trial1_rear_end()
    res, mc = _run_pair(scn)
    assert res.union_lower_bound <= mc.union_prob + 0.02
    assert mc.union_prob <= res.union_independent + 0.02


def test_rear_end_probability_peaks_and_tracks_mc():
    # In the dynamic rear-end approach the per-step collision probability rises
    # to ~1 as the ego reaches the obstacle, then (in this stop-free kinematic
    # model) falls as it passes through.  Check the peak reaches high probability
    # and that ProbStar tracks MC across the whole curve.
    scn = scenarios.rear_end_dynamic(gap=1.2, v_ego=1.0, k_max=10)
    res, mc = _run_pair(scn, n_mc=100000)
    assert res.per_step_prob.max() > 0.9
    assert np.max(np.abs(res.per_step_prob - mc.per_step_prob)) < 0.05


def test_reachable_set_centers_recorded():
    scn = scenarios.trial1_rear_end()
    H, d = collision_half_space(2, 0, 1, scn.length_half, scn.width_half)
    ver = QuantitativeVerifier(scn.models, 0.25, 0.025,
                               scn.length_half, scn.width_half)
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=6.0)
    res = ver.verify(P0, H, d, scn.k_max, record_centers=True)
    assert res.reachable_centers.shape == (scn.k_max + 1, len(scn.mu))


def test_recompute_collision_matches_fixed_for_straight_motion():
    # For straight-line motion the vehicle headings never change, so the
    # heading-aware collision predicate must give identical results to the
    # fixed collision half-space.
    scn = scenarios.trial1_rear_end()
    H, d = collision_half_space(2, 0, 1, scn.length_half, scn.width_half)
    ver = QuantitativeVerifier(scn.models, 0.25, 0.025,
                               scn.length_half, scn.width_half)
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=6.0)
    res_fixed = ver.verify(P0.clone(), H, d, scn.k_max, recompute_collision=False)
    res_dyn = ver.verify(P0.clone(), H, d, scn.k_max, recompute_collision=True)
    # Same geometry (headings never change); differences are only float/solver
    # noise from rebuilding the identical polygon each step.
    assert np.max(np.abs(res_fixed.per_step_prob - res_dyn.per_step_prob)) < 1e-3


def test_recompute_collision_turning_tracks_mc():
    # A turning ego that grazes the obstacle: the heading-aware collision
    # predicate (verifier) must track the heading-aware Monte-Carlo ground truth.
    from probstar_rv import VehicleModel
    L, lh, wh = 0.33, 0.275, 0.15
    models = [VehicleModel(L, 1.0, 0.15), VehicleModel(L, 0.0, 0.0)]
    mu = np.array([0, 0, 1, 0, 1.3, 0.35, 1, 0], float)
    Sigma = np.diag([0.04**2, 0.04**2, 1e-4, 1e-4,
                     0.04**2, 0.04**2, 1e-4, 1e-4])
    k_max = 12
    H, d = collision_half_space(2, 0, 1, lh, wh)
    ver = QuantitativeVerifier(models, 0.25, 0.025, lh, wh)
    res = ver.verify(ProbStar.from_gaussian(mu, Sigma, 6.0), H, d, k_max,
                     recompute_collision=True)
    mc = MonteCarloValidator(models, 0.25, 0.025, lh, wh,
                             rng=np.random.default_rng(0))
    mcres = mc.run(mu, Sigma, k_max, 200000, recompute_collision=True)
    assert res.per_step_prob.max() > 0.9  # it does graze/collide
    assert np.max(np.abs(res.per_step_prob - mcres.per_step_prob)) < 0.02


def test_time_varying_braking_plant_matches_mc():
    # A braking ego (time-varying plant) verified with substep_matrices must
    # agree with a Monte-Carlo run using the SAME per-step matrices.
    from probstar_rv import braking_substep_matrices
    from probstar_rv.dynamics import VehicleModel
    L, lh, wh = 0.33, 0.275, 0.15
    v0, gap, sp = 1.5, 1.15, 0.06
    k_max = 12
    mu = np.array([0, 0, 1, 0, gap, 0, 1, 0], float)
    Sigma = np.diag([sp**2, sp**2, 1e-4, 1e-4, sp**2, sp**2, 1e-4, 1e-4])
    ego = VehicleModel(L, v0, 0.0)
    other = [VehicleModel(L, 0.0, 0.0)]
    H, d = collision_half_space(2, 0, 1, lh, wh)
    Cxy, dxy = minkowski_difference_box(lh, wh)

    sub = braking_substep_matrices(ego, other, v0=v0, decel=3.0,
                                   dt_r=0.25, dt_m=0.025, k_max=k_max)
    ver = QuantitativeVerifier([ego] + other, 0.25, 0.025, lh, wh)
    res = ver.verify(ProbStar.from_gaussian(mu, Sigma, 6.0), H, d, k_max,
                     substep_matrices=sub)
    mc = MonteCarloValidator([ego] + other, 0.25, 0.025, lh, wh,
                             rng=np.random.default_rng(3))
    mcres = mc.run(mu, Sigma, k_max, 100000, collision_C=Cxy, collision_d=dxy,
                   step_matrices=sub)
    # braking produces a partial (graded) peak here, a meaningful comparison
    assert 0.05 < res.per_step_prob.max() < 0.95
    assert np.max(np.abs(res.per_step_prob - mcres.per_step_prob)) < 0.02


def test_substep_matrices_length_validation():
    scn = scenarios.trial1_rear_end()
    H, d = collision_half_space(2, 0, 1, scn.length_half, scn.width_half)
    ver = QuantitativeVerifier(scn.models, 0.25, 0.025,
                               scn.length_half, scn.width_half)
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=6.0)
    with pytest.raises(ValueError):
        ver.verify(P0, H, d, scn.k_max, substep_matrices=[ver.A_sub])  # wrong len
