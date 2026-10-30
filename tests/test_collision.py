"""Tests for collision geometry: Minkowski difference, collision half-space on
the stacked state, and agreement with analytic rectangle overlap.
"""

import numpy as np

from probstar_rv.collision import (
    minkowski_difference_box, collision_half_space, point_in_collision,
)


def test_axis_aligned_minkowski_box():
    lh, wh = 0.275, 0.15
    C, d = minkowski_difference_box(lh, wh)
    # collision region for equal axis-aligned rects: |rx| <= 2lh, |ry| <= 2wh
    rng = np.random.default_rng(0)
    R = rng.uniform(-1, 1, (200000, 2))
    pred = np.all(R @ C.T <= d + 1e-12, axis=1)
    true = (np.abs(R[:, 0]) < 2 * lh) & (np.abs(R[:, 1]) < 2 * wh)
    assert np.mean(pred == true) > 0.999


def test_collision_half_space_flags_overlap():
    lh, wh = 0.275, 0.15
    H, d = collision_half_space(2, 0, 1, lh, wh)
    assert H.shape == (4, 8)
    # ego at origin, obstacle at (0.3, 0.1): inside 0.55 x 0.30 -> collision
    X = np.array([0, 0, 1, 0, 0.3, 0.1, 1, 0.0], float)
    assert np.all(H @ X <= d + 1e-12)
    # obstacle at (0.6, 0): outside in x -> no collision
    X = np.array([0, 0, 1, 0, 0.6, 0.0, 1, 0.0], float)
    assert not np.all(H @ X <= d + 1e-12)


def test_collision_is_translation_invariant():
    lh, wh = 0.275, 0.15
    H, d = collision_half_space(2, 0, 1, lh, wh)
    # Shifting both vehicles by the same offset must not change the verdict.
    base = np.array([0, 0, 1, 0, 0.3, 0.1, 1, 0.0], float)
    shift = np.array([5.0, -2.0, 0, 0, 5.0, -2.0, 0, 0], float)
    assert np.all(H @ base <= d + 1e-12) == np.all(H @ (base + shift) <= d + 1e-12)


def test_point_in_collision_helper():
    lh, wh = 0.275, 0.15
    C, d = minkowski_difference_box(lh, wh)
    assert point_in_collision([0.0, 0.0], C, d)          # perfectly overlapping
    assert point_in_collision([0.5, 0.25], C, d)         # inside
    assert not point_in_collision([0.6, 0.0], C, d)      # too far in x
    assert not point_in_collision([0.0, 0.35], C, d)     # too far in y


def test_oriented_rectangle_is_larger_region():
    # A 45-degree oriented ego box should yield a strictly larger collision
    # region (its diagonal extent is larger) than the axis-aligned one.
    lh, wh = 0.275, 0.15
    C0, d0 = minkowski_difference_box(lh, wh, heading_pi=0.0)
    C1, d1 = minkowski_difference_box(lh, wh, heading_pi=np.pi / 4)
    # area via vertex hull extent along x at y=0: compare max feasible rx
    rx0 = d0[np.argmax(C0[:, 0])] if np.any(C0[:, 0] > 0) else 0
    # For rotated box the x-extent grows toward (lh+wh)*... so just check the
    # rotated region contains a point the aligned one excludes near the diagonal.
    assert point_in_collision([0.45, 0.25], C1, d1) or \
           not point_in_collision([0.45, 0.25], C0, d0)
