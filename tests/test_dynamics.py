"""Tests for the vehicle dynamics: straight-line motion, turning vs nonlinear
ground truth, phi unit-vector behavior, discretization, and block assembly.
"""

import numpy as np

from probstar_rv import (VehicleModel, build_system_matrix,
                         braking_speed_profile, braking_substep_matrices)
from probstar_rv.dynamics import rollout, STATE_DIM


def _nonlinear_rollout(V, L, zeta, t_end, dt=5e-4):
    beta = np.arctan(np.tan(zeta) / 2.0)
    w = V * np.tan(zeta) * np.cos(beta) / L
    x = y = th = 0.0
    n = int(round(t_end / dt))
    for _ in range(n):
        x += dt * V * np.cos(th + beta)
        y += dt * V * np.sin(th + beta)
        th += dt * w
    return x, y, th


def test_straight_line():
    m = VehicleModel(wheelbase_L=0.33, speed_V=1.0, steering_zeta=0.0)
    s0 = VehicleModel.state_from_pose(0, 0, 0.0, 0.0)
    traj = rollout([m], s0, 10, 0.1)
    # After 1 s at 1 m/s heading +x: x~1, y~0, phi=[1,0].
    assert abs(traj[-1][0] - 1.0) < 1e-6
    assert abs(traj[-1][1]) < 1e-9
    assert np.allclose(traj[-1][2:], [1.0, 0.0], atol=1e-9)


def test_turning_position_matches_nonlinear():
    V, L, zeta = 1.0, 0.33, 0.25
    m = VehicleModel(L, V, zeta)
    s0 = VehicleModel.state_from_pose(0, 0, 0.0, zeta)
    # 1 second at the paper's modeling step dt_m = 0.025.
    traj = rollout([m], s0, 40, 0.025, method="euler")
    xe, ye, _ = VehicleModel.pose_from_state(traj[-1])
    x, y, _ = _nonlinear_rollout(V, L, zeta, 1.0)
    # position error should be within ~2 cm (paper reports ~2cm over 2.5s)
    assert np.hypot(xe - x, ye - y) < 0.02


def test_expm_matches_nonlinear_position_closely():
    V, L, zeta = 1.0, 0.33, 0.25
    m = VehicleModel(L, V, zeta)
    s0 = VehicleModel.state_from_pose(0, 0, 0.0, zeta)
    traj = rollout([m], s0, 2000, 0.001, method="expm")
    xl, yl, _ = VehicleModel.pose_from_state(traj[-1])
    x, y, _ = _nonlinear_rollout(V, L, zeta, 2.0)
    assert np.hypot(xl - x, yl - y) < 5e-3


def test_phi_norm_growth_small_at_paper_dtm():
    V, L, zeta = 1.0, 0.33, 0.25
    m = VehicleModel(L, V, zeta)
    s0 = VehicleModel.state_from_pose(0, 0, 0.0, zeta)
    traj = rollout([m], s0, 40, 0.025, method="euler")  # 1.0 s
    norms = np.linalg.norm(traj[:, 2:], axis=1)
    assert norms[0] == 1.0 or abs(norms[0] - 1.0) < 1e-9
    assert norms[-1] < 1.02  # < 2% growth over 1 s


def test_block_diagonal_assembly():
    m1 = VehicleModel(0.33, 1.0, 0.1)
    m2 = VehicleModel(0.33, 0.5, 0.0)
    A = build_system_matrix([m1, m2], 0.1)
    assert A.shape == (2 * STATE_DIM, 2 * STATE_DIM)
    # off-diagonal blocks must be zero (vehicles are decoupled)
    assert np.allclose(A[:STATE_DIM, STATE_DIM:], 0.0)
    assert np.allclose(A[STATE_DIM:, :STATE_DIM], 0.0)


def test_pose_state_roundtrip():
    x, y, th = 0.5, -1.2, 0.7
    s = VehicleModel.state_from_pose(x, y, th, zeta=0.0)
    rx, ry, rth = VehicleModel.pose_from_state(s)
    assert abs(rx - x) < 1e-12 and abs(ry - y) < 1e-12
    assert abs(rth - th) < 1e-12


def test_braking_speed_profile():
    # v0=2, decel=1 -> speeds 2.0, 1.75, 1.5, ... clamped at 0.
    sp = braking_speed_profile(v0=2.0, decel=1.0, dt_r=0.25, k_max=12)
    assert sp[0] == 2.0
    assert abs(sp[1] - 1.75) < 1e-12
    assert sp[-1] == 0.0                    # has stopped by the end
    assert np.all(sp >= 0.0)                # never reverses
    assert np.all(np.diff(sp) <= 1e-12)     # non-increasing


def test_braking_substep_matrices_shapes_and_speed_decrease():
    L = 0.33
    ego = VehicleModel(L, 2.0, 0.0)
    other = [VehicleModel(L, 0.0, 0.0)]
    mats = braking_substep_matrices(ego, other, v0=2.0, decel=2.0,
                                    dt_r=0.25, dt_m=0.025, k_max=10)
    assert len(mats) == 10
    for M in mats:
        assert M.shape == (2 * STATE_DIM, 2 * STATE_DIM)
    # The ego's x<-phi_x coupling (A[0,2] = V*dt_m) must shrink as it brakes.
    coupling = [M[0, 2] for M in mats]
    assert coupling[0] > coupling[3] > coupling[-1] - 1e-12
    assert abs(coupling[-1]) < 1e-9        # stopped -> zero speed coupling
