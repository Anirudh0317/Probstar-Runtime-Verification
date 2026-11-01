r"""Vehicle dynamics (Section 3.2 of the paper).

Each vehicle uses the kinematic bicycle model. Linearizing in [x, y, theta]
works poorly, so the paper replaces the heading with two unit-vector components

    phi_x = cos(theta + beta),   phi_y = sin(theta + beta),

giving the 4D per-vehicle state s = [x, y, phi_x, phi_y]^T.

Continuous model (paper Eq. 3): d/dt s = A_c s, with

           [ 0  0   V                    0                 ]
    A_c =  [ 0  0   0                    V                 ]
           [ 0  0   0                   -V tan(zeta)cos(b)/L ]
           [ 0  0   V tan(zeta)cos(b)/L  0                 ]

V is speed, zeta the steering angle, b = beta the slip angle, L the wheelbase.
So x-dot = V*phi_x and y-dot = V*phi_y (position follows the heading vector), and
the bottom-right 2x2 block rotates that vector at yaw rate
omega = V tan(zeta) cos(beta)/L.

Discrete model: s[k+1] = A_d s[k] with A_d ~= I + A_c * dt. The paper uses a
small modeling step dt_m (e.g. dt_r/10) so this Euler step is accurate enough
next to the perception noise. An exact matrix-exponential discretization is also
provided for testing.

Multi-vehicle (paper Eq. 4): the system matrix is block diagonal over the ego
car pi and each detected car omega_i: A = blockdiag(A_pi, A_{omega_0}, ...).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.linalg import block_diag, expm


# Per-vehicle state indices within the 4D block.
IX, IY, IPHIX, IPHIY = 0, 1, 2, 3
STATE_DIM = 4
STATE_NAMES = ("x", "y", "phi_x", "phi_y")


@dataclass
class VehicleModel:
    """Kinematic-bicycle vehicle in the phi-augmented linear form.

    Attributes:
        wheelbase_L: wheelbase ``L`` (metres).
        speed_V: current speed ``V`` (m/s).
        steering_zeta: current steering angle ``zeta`` (rad).
    """

    wheelbase_L: float
    speed_V: float = 0.0
    steering_zeta: float = 0.0

    # ------------------------------------------------------------------ #
    @property
    def beta(self) -> float:
        r"""Slip angle beta = arctan( tan(zeta) / 2 )  (paper Eq. 1)."""
        return np.arctan(np.tan(self.steering_zeta) / 2.0)

    @property
    def yaw_rate(self) -> float:
        r"""Yaw rate omega = V tan(zeta) cos(beta) / L."""
        return (self.speed_V * np.tan(self.steering_zeta)
                * np.cos(self.beta) / self.wheelbase_L)

    # ------------------------------------------------------------------ #
    def continuous_matrix(self) -> np.ndarray:
        """Return the 4x4 continuous-time system matrix ``A_c`` (Eq. 3)."""
        V = self.speed_V
        w = self.yaw_rate  # = V tan(zeta) cos(beta) / L
        A = np.zeros((STATE_DIM, STATE_DIM))
        A[IX, IPHIX] = V
        A[IY, IPHIY] = V
        A[IPHIX, IPHIY] = -w
        A[IPHIY, IPHIX] = w
        return A

    def discrete_matrix(self, dt: float, method: str = "euler") -> np.ndarray:
        """Return the 4x4 discrete-time matrix ``A_d``.

        Args:
            dt: time step.
            method: "euler" for A_d = I + A_c dt (the paper's choice), or
                "expm" for the exact A_d = exp(A_c dt) (reference/testing).
        """
        Ac = self.continuous_matrix()
        if method == "euler":
            return np.eye(STATE_DIM) + Ac * dt
        elif method == "expm":
            return expm(Ac * dt)
        else:
            raise ValueError(f"unknown discretization method: {method!r}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def state_from_pose(x: float, y: float, theta: float,
                        zeta: float = 0.0) -> np.ndarray:
        r"""Build the 4D state ``[x, y, phi_x, phi_y]`` from a pose.

        ``phi_x = cos(theta + beta)``, ``phi_y = sin(theta + beta)`` with
        ``beta = arctan(tan(zeta)/2)`` (paper Eq. 2).
        """
        beta = np.arctan(np.tan(zeta) / 2.0)
        return np.array([x, y, np.cos(theta + beta), np.sin(theta + beta)])

    @staticmethod
    def pose_from_state(s: Sequence[float]) -> tuple[float, float, float]:
        """Recover ``(x, y, heading)`` from a 4D state (heading = atan2(phi_y, phi_x))."""
        s = np.asarray(s, dtype=float)
        return float(s[IX]), float(s[IY]), float(np.arctan2(s[IPHIY], s[IPHIX]))


def build_system_matrix(models: List[VehicleModel], dt: float,
                        method: str = "euler") -> np.ndarray:
    """Assemble the block-diagonal multi-vehicle discrete system matrix (Eq. 4).

    Args:
        models: list of VehicleModel, ego first then obstacles.
        dt: discretization time step.
        method: "euler" or "expm" (passed to each block).

    Returns:
        A (4n x 4n) block-diagonal matrix, n = len(models).
    """
    blocks = [m.discrete_matrix(dt, method=method) for m in models]
    return block_diag(*blocks)


def rollout(models: List[VehicleModel], s0: np.ndarray, n_steps: int,
            dt: float, method: str = "euler") -> np.ndarray:
    """Deterministically roll the stacked state forward ``n_steps`` steps.

    Args:
        models: vehicle models (fixed matrices over the horizon).
        s0: initial stacked state (4n,).
        n_steps: number of steps.
        dt: time step.
        method: discretization method.

    Returns:
        (n_steps+1, 4n) array of states including the initial state.
    """
    A = build_system_matrix(models, dt, method=method)
    s = np.asarray(s0, dtype=float).copy()
    traj = [s.copy()]
    for _ in range(n_steps):
        s = A @ s
        traj.append(s.copy())
    return np.array(traj)


def braking_speed_profile(v0: float, decel: float, dt_r: float,
                          k_max: int) -> np.ndarray:
    r"""Ego speed at the start of each reachability step under constant braking.

    Models a vehicle decelerating at a constant ``decel`` (m/s^2) from initial
    speed ``v0``, clamped at 0 (the car does not reverse).  Speed at the start of
    reach step ``k`` (time ``k*dt_r``) is ``max(v0 - decel*k*dt_r, 0)``.

    Returns an array of length ``k_max`` of per-step speeds.
    """
    t = np.arange(k_max) * dt_r
    return np.maximum(v0 - decel * t, 0.0)


def braking_substep_matrices(ego_template: "VehicleModel",
                             other_models: List["VehicleModel"],
                             v0: float, decel: float, dt_r: float, dt_m: float,
                             k_max: int, ego_index: int = 0,
                             method: str = "euler") -> List[np.ndarray]:
    r"""Per-reach-step modeling-substep matrices for a braking ego.

    Builds, for each reachability step ``k``, the block-diagonal plant matrix for
    ONE modeling substep (``dt_m``), using the ego speed given by
    ``braking_speed_profile`` at step ``k`` (piecewise-constant within the step)
    while the other vehicles keep their models.  Suitable for passing to
    ``QuantitativeVerifier.verify(substep_matrices=...)``.

    Args:
        ego_template: a VehicleModel supplying the ego's wheelbase and steering
            (its speed is overridden per step by the braking profile).
        other_models: the non-ego vehicle models, in stacked order (the ego is
            inserted at ``ego_index``).
        v0, decel: initial speed and constant deceleration.
        dt_r, dt_m, k_max: reachability step, modeling step, horizon.
        ego_index: index of the ego within the stacked vehicle list.
        method: discretization method.

    Returns:
        A list of length ``k_max`` of (4n x 4n) matrices.
    """
    speeds = braking_speed_profile(v0, decel, dt_r, k_max)
    mats = []
    for vk in speeds:
        ego_k = VehicleModel(ego_template.wheelbase_L, float(vk),
                             ego_template.steering_zeta)
        models = list(other_models)
        models.insert(ego_index, ego_k)
        mats.append(build_system_matrix(models, dt_m, method=method))
    return mats
