"""End-to-end pipeline test: simulated pose detection -> Figure-4 motion
estimation -> Taylor uncertainty propagation -> ProbStar initial set ->
verification, validated against Monte Carlo.

This exercises the whole chain composing together (the thing exp6 demonstrates),
rather than building the initial Gaussian directly.
"""

import numpy as np
from scipy.linalg import block_diag

from probstar_rv import (
    QuantitativeVerifier, MonteCarloValidator, collision_half_space,
    simulate_pose_detection, estimate_motion_from_poses,
    build_initial_distribution, build_initial_probstar,
)
from probstar_rv.dynamics import VehicleModel


def test_full_pipeline_composes_and_matches_mc():
    L, lh, wh = 0.33, 0.275, 0.15
    dt_r, dt_m = 0.25, 0.025
    rng = np.random.default_rng(0)
    noise = {"sigma_x": 0.015, "sigma_y": 0.015,
             "sigma_ti": 0.03, "sigma_tj": 0.03}

    # ego: known pose (no jitter)
    ego_pose = simulate_pose_detection(0.0, 0.0, 0.0, noise=noise, jitter=False)
    ego_init = build_initial_distribution(ego_pose, zeta_mu=0.0,
                                          zeta_var=0.01**2, V_mu=1.0)

    # obstacle: perceived from two poses on a straight oncoming path
    p0 = simulate_pose_detection(2.5, 0.2, np.pi, noise=noise, jitter=True, rng=rng)
    p1 = simulate_pose_detection(2.5 - 0.6 * dt_r, 0.2, np.pi,
                                 noise=noise, jitter=True, rng=rng)
    motion = estimate_motion_from_poses(p0, p1, dt_r, L)
    obs_init = build_initial_distribution(p1, zeta_mu=motion["zeta_mu"],
                                          zeta_var=motion["zeta_var"],
                                          V_mu=motion["V_mu"])

    inits = [ego_init, obs_init]
    P0 = build_initial_probstar(inits, delta=6.0)
    mu = np.concatenate([vi.mean for vi in inits])
    Sigma = block_diag(*[vi.cov for vi in inits])

    # sanity: the estimated obstacle speed is near the true 0.6 m/s
    assert abs(motion["V_mu"] - 0.6) < 0.15

    models = [VehicleModel(L, ego_init.V_mu, ego_init.zeta_mu),
              VehicleModel(L, obs_init.V_mu, obs_init.zeta_mu)]
    k_max = 10
    H, d = collision_half_space(2, 0, 1, lh, wh)

    ver = QuantitativeVerifier(models, dt_r, dt_m, lh, wh)
    res = ver.verify(P0, H, d, k_max, recompute_collision=True)
    mc = MonteCarloValidator(models, dt_r, dt_m, lh, wh,
                             rng=np.random.default_rng(1))
    mcres = mc.run(mu, Sigma, k_max, 100000, recompute_collision=True)

    # the pipeline produced a valid probability curve that MC agrees with
    assert res.per_step_prob.shape == (k_max,)
    assert np.all((res.per_step_prob >= 0) & (res.per_step_prob <= 1))
    assert np.max(np.abs(res.per_step_prob - mcres.per_step_prob)) < 0.03
