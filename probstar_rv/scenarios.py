r"""Driving-scenario builders.

Build the paper's validation scenarios (Figure 8 Trials 1 & 2) and a tunable
rear-end scenario (Figure 6 style) in one place, so tests and experiments share
the same definitions.

Each builder returns a Scenario with what the verifier and Monte-Carlo validator
need: the vehicle models, the stacked initial mean and covariance, the collision
pair, and the expected hand-computable probability where there is one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .dynamics import VehicleModel, STATE_DIM


@dataclass
class Scenario:
    name: str
    models: List[VehicleModel]
    mu: np.ndarray
    Sigma: np.ndarray
    k_max: int
    ego_index: int = 0
    obs_index: int = 1
    expected_prob: Optional[float] = None    # hand-computable truth, if known
    expected_kind: str = "per_step_final"    # where expected_prob applies
    note: str = ""
    length_half: float = 0.275
    width_half: float = 0.15


_TINY = 1e-8  # variance floor for "known with certainty" states


def _stack(states_mu, states_var):
    return np.concatenate(states_mu), np.diag(np.concatenate(states_var))


def trial1_rear_end(length_half=0.275, width_half=0.15,
                    sigma_long=0.3, v_ego=1.0) -> Scenario:
    r"""Figure 8 Trial 1: rear-end collision, 50% true probability.

    Ego starts at the origin moving along +x at ``v_ego``.  A stationary
    obstacle sits ahead; its position along the direction of travel is Gaussian
    with std ``sigma_long`` (the paper's 0.3 m -- it labels this axis y_w in its
    own vehicle frame; in our ego-aligned frame the direction of travel is x).
    The obstacle mean is placed so the ego's front just reaches the obstacle
    center at the horizon end, so exactly the nearer half of the obstacle
    distribution collides -> P = 0.5.

    Horizon: 4.0 s (k_max = 16 at dt_r = 0.25).
    """
    L = 0.33
    v_reach_dist = v_ego * 4.0                      # ego travels 4 m in 4 s
    obs_mean_x = v_reach_dist + 2 * length_half     # front touches center at end
    models = [VehicleModel(L, v_ego, 0.0), VehicleModel(L, 0.0, 0.0)]
    mu, Sig = _stack(
        [np.array([0, 0, 1, 0]), np.array([obs_mean_x, 0, 1, 0])],
        [np.array([_TINY, _TINY, _TINY, _TINY]),
         np.array([sigma_long ** 2, _TINY, _TINY, _TINY])],
    )
    return Scenario(
        name="trial1_rear_end", models=models, mu=mu, Sigma=Sig, k_max=16,
        expected_prob=0.5, expected_kind="per_step_final",
        note="Fig 8 Trial 1: rear-end, hand-computed 50% true probability.",
        length_half=length_half, width_half=width_half,
    )


def trial2_side(length_half=0.275, width_half=0.15, v=1.0) -> Scenario:
    r"""Figure 8 Trial 2: side collision, 2.28% true probability.

    Ego and obstacle move along +x at the same speed (so they stay
    longitudinally aligned -> closest approach persists over the horizon).  The
    obstacle sits laterally ``3W`` away with std ``W`` (W = car width = 2*width_half).
    A collision needs lateral separation < ``2*width_half = W``, i.e. the
    obstacle y must fall from mean 3W down past 2W of displacement = 2 std, so
    P = P(Z < -2) = 0.0228.

    Horizon: 2.0 s (k_max = 8).
    """
    L = 0.33
    W = 2 * width_half
    models = [VehicleModel(L, v, 0.0), VehicleModel(L, v, 0.0)]
    mu, Sig = _stack(
        [np.array([0, 0, 1, 0]), np.array([0, 3 * W, 1, 0])],
        [np.array([_TINY, _TINY, _TINY, _TINY]),
         np.array([_TINY, W ** 2, _TINY, _TINY])],
    )
    return Scenario(
        name="trial2_side", models=models, mu=mu, Sigma=Sig, k_max=8,
        expected_prob=0.0228, expected_kind="per_step_first",
        note="Fig 8 Trial 2: side collision, hand-computed 2.28% true probability.",
        length_half=length_half, width_half=width_half,
    )


def rear_end_dynamic(length_half=0.275, width_half=0.15,
                     v_ego=1.0, gap=1.5, sigma_pos=0.05,
                     k_max=12) -> Scenario:
    r"""Figure 6 style rear-end: ego accelerates toward a stationary car.

    A general, tunable rear-end scenario (not a hand-computable trial): ego
    moving at ``v_ego`` toward a stationary obstacle ``gap`` metres ahead, with
    perception position std ``sigma_pos`` on both vehicles.  Produces the
    characteristic rising collision-probability curve of Figure 6.
    """
    L = 0.33
    models = [VehicleModel(L, v_ego, 0.0), VehicleModel(L, 0.0, 0.0)]
    mu, Sig = _stack(
        [np.array([0, 0, 1, 0]), np.array([gap, 0, 1, 0])],
        [np.array([sigma_pos ** 2, sigma_pos ** 2, 1e-4, 1e-4]),
         np.array([sigma_pos ** 2, sigma_pos ** 2, 1e-4, 1e-4])],
    )
    return Scenario(
        name="rear_end_dynamic", models=models, mu=mu, Sigma=Sig, k_max=k_max,
        note="Figure 6 style rear-end approach (tunable).",
        length_half=length_half, width_half=width_half,
    )
