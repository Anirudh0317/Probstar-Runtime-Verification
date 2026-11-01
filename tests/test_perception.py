"""Tests for perception simulation, Taylor uncertainty propagation, and the
Figure-4 motion (steering/speed) estimation.
"""

import numpy as np

from probstar_rv.perception import (
    PoseEstimate, simulate_pose_detection, propagate_heading_slip,
    estimate_motion_from_poses, taylor_moments, build_initial_distribution,
)


def test_taylor_moments_quadratic_exact():
    # For f(x)=x^2 with X~N(mu,var): E=mu^2+var, Var=4 mu^2 var + 2 var^2.
    # The paper's 2nd-order formula gives E=mu^2+var exactly and
    # Var=4 mu^2 var + var^2 (misses the +var^2 kurtosis term by var^2).
    mu, var = 2.0, 0.1
    m, v = taylor_moments(lambda x: x**2, lambda x: 2*x, lambda x: 2.0, mu, var)
    assert abs(m - (mu**2 + var)) < 1e-9
    # variance within the 2nd-order truncation error of the exact 4mu^2var+2var^2
    assert abs(v - (4*mu**2*var + 1*var**2)) < 1e-9


def test_heading_slip_propagation_vs_mc():
    rng = np.random.default_rng(5)
    true_theta = np.deg2rad(60)
    pose = PoseEstimate(x=0.5, y=1.0,
                        theta_i=np.cos(true_theta), theta_j=np.sin(true_theta),
                        sigma_x=0.02, sigma_y=0.02, sigma_ti=0.05, sigma_tj=0.05)
    zeta_mu, zeta_var = 0.2, 0.02**2
    hp = propagate_heading_slip(pose, zeta_mu, zeta_var)

    N = 2_000_000
    ti = np.cos(true_theta) + rng.normal(0, 0.05, N)
    tj = np.sin(true_theta) + rng.normal(0, 0.05, N)
    th = np.arctan2(tj, ti)
    z = zeta_mu + rng.normal(0, 0.02, N)
    be = np.arctan(np.tan(z) / 2.0)
    a = th + be
    assert abs(hp["phi_x_mu"] - np.cos(a).mean()) < 3e-3
    assert abs(hp["phi_x_var"] - np.cos(a).var()) < 3e-3
    assert abs(hp["phi_y_mu"] - np.sin(a).mean()) < 3e-3
    assert abs(hp["phi_y_var"] - np.sin(a).var()) < 3e-3


def test_figure4_motion_recovers_circular_arc():
    L, V, zeta_true, dt = 0.33, 0.8, 0.25, 0.25
    beta = np.arctan(np.tan(zeta_true) / 2.0)
    w = V * np.tan(zeta_true) * np.cos(beta) / L

    def pose_at(t):
        x = y = th = 0.0
        n = int(round(t / 5e-4))
        for _ in range(n):
            x += 5e-4 * V * np.cos(th + beta)
            y += 5e-4 * V * np.sin(th + beta)
            th += 5e-4 * w
        return x, y, th

    x0, y0, t0 = pose_at(0.0)
    x1, y1, t1 = pose_at(dt)
    p0 = PoseEstimate(x0, y0, np.cos(t0), np.sin(t0))
    p1 = PoseEstimate(x1, y1, np.cos(t1), np.sin(t1))
    est = estimate_motion_from_poses(p0, p1, dt, L)
    assert abs(est["zeta_mu"] - zeta_true) < 2e-2
    assert abs(est["V_mu"] - V) < 2e-2


def test_figure4_straight_motion():
    # Two poses on a straight line -> zeta ~ 0, V = distance/dt.
    L, dt = 0.33, 0.25
    p0 = PoseEstimate(0.0, 0.0, 1.0, 0.0)
    p1 = PoseEstimate(0.25, 0.0, 1.0, 0.0)
    est = estimate_motion_from_poses(p0, p1, dt, L)
    assert abs(est["zeta_mu"]) < 1e-6
    assert abs(est["V_mu"] - 1.0) < 1e-6


def test_simulated_detection_mean_without_jitter():
    pose = simulate_pose_detection(1.0, 2.0, 0.3, jitter=False)
    assert abs(pose.x - 1.0) < 1e-12 and abs(pose.y - 2.0) < 1e-12
    assert abs(pose.theta - 0.3) < 1e-9


def test_build_initial_distribution_spd():
    pose = simulate_pose_detection(0.5, 1.0, 0.2, jitter=False)
    vi = build_initial_distribution(pose, zeta_mu=0.1, zeta_var=0.01, V_mu=0.8)
    assert vi.mean.shape == (4,)
    assert vi.cov.shape == (4, 4)
    assert np.all(np.linalg.eigvalsh(vi.cov) > 0)
