"""
probstar_rv
===========

Perception-based quantitative runtime verification for learning-enabled
cyber-physical systems, following Brown et al., ICCPS '25.

Public API (import from the top-level package):

    from probstar_rv import (
        ProbStar,                 # probabilistic star set
        VehicleModel, build_system_matrix,
        simulate_pose_detection, build_initial_probstar,
        collision_half_space, QuantitativeVerifier, VerificationResult,
        MonteCarloValidator,
        load_config,
    )
"""

from .config import load_config
from .probstar import ProbStar
from .dynamics import (
    VehicleModel, build_system_matrix,
    braking_speed_profile, braking_substep_matrices,
)
from .perception import (
    PoseEstimate,
    simulate_pose_detection,
    estimate_motion_from_poses,
    build_initial_distribution,
    build_initial_probstar,
)
from .collision import collision_half_space, minkowski_difference_box
from .verification import QuantitativeVerifier, VerificationResult
from .monte_carlo import MonteCarloValidator
from . import scenarios

__all__ = [
    "load_config",
    "ProbStar",
    "VehicleModel",
    "build_system_matrix",
    "braking_speed_profile",
    "braking_substep_matrices",
    "PoseEstimate",
    "simulate_pose_detection",
    "estimate_motion_from_poses",
    "build_initial_distribution",
    "build_initial_probstar",
    "collision_half_space",
    "minkowski_difference_box",
    "QuantitativeVerifier",
    "VerificationResult",
    "MonteCarloValidator",
    "scenarios",
]

__version__ = "1.0.0"
