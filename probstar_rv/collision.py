r"""Collision geometry and unsafe-region constraints (Section 4.2 of the paper).

The ego car pi and an obstacle omega collide when their bounding boxes overlap.
The paper turns that overlap test into a point-in-polytope test with the
Minkowski difference:

    B_pi (at the ego) and B_omega (at the obstacle) overlap
    <=>  (x_omega - x_pi, y_omega - y_pi)  in  B_pi (-) B_omega,

where (-) is the Minkowski difference of the two boxes. The set of relative
offsets that collide is a polytope B_diff = { r : C r <= d } in the xy-plane.
Lifting it to the full stacked state

    X = [x_pi, y_pi, phi_x_pi, phi_y_pi, x_omega, y_omega, phi_x_omega, phi_y_omega]

gives the collision half-space (Algorithm 1, lines 26-32)

    H = [ -C  0_{k,2}   C  0_{k,2} ],      H X <= d,

so H X <= d means C (x_omega - x_pi, y_omega - y_pi) <= d, i.e. the relative
offset is inside B_diff.

For aligned rectangles the Minkowski difference is another rectangle; for rotated
ones it's a convex polygon. Both are computed here. The default collision polygon
uses axis-aligned boxes in the ego frame (simple and slightly conservative, as in
the paper), and an oriented-rectangle routine covers the general case.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .dynamics import STATE_DIM


def _rect_corners(length_half: float, width_half: float,
                  heading: float = 0.0) -> np.ndarray:
    """Return the 4 corners (4x2) of a rectangle centered at the origin.

    Args:
        length_half: half-length along the body x-axis.
        width_half: half-width along the body y-axis.
        heading: rotation of the body (rad); 0 = axis-aligned.
    """
    base = np.array([
        [length_half, width_half],
        [length_half, -width_half],
        [-length_half, -width_half],
        [-length_half, width_half],
    ])
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    return base @ R.T


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """2D convex hull (Andrew's monotone chain). Returns CCW hull vertices."""
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _polygon_halfspaces(vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a CCW convex polygon to half-space form ``C r <= d``.

    Each edge (v_i -> v_{i+1}) gives an outward normal; the interior satisfies
    ``normal . r <= normal . v_i``.
    """
    V = _convex_hull(vertices)
    n = len(V)
    C = np.zeros((n, 2))
    d = np.zeros(n)
    cen = V.mean(axis=0)
    for i in range(n):
        a = V[i]
        b = V[(i + 1) % n]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])  # outward for CCW
        normal = normal / (np.linalg.norm(normal) + 1e-15)
        # ensure it points away from centroid
        if normal @ (a - cen) < 0:
            normal = -normal
        C[i] = normal
        d[i] = normal @ a
    return C, d


def minkowski_difference_box(length_half: float, width_half: float,
                             heading_pi: float = 0.0,
                             heading_omega: float = 0.0,
                             obstacle_length_half: float | None = None,
                             obstacle_width_half: float | None = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute ``B_pi (-) B_omega`` as half-spaces ``{ r : C r <= d }``.

    The Minkowski difference of convex bodies used for collision is equivalent
    (for the overlap test) to the Minkowski **sum** of ``B_pi`` and ``-B_omega``
    (the obstacle body reflected through the origin): two bodies overlap iff the
    relative offset lies in ``B_pi (+) (-B_omega)``.  For centrally symmetric
    rectangles ``-B_omega = B_omega``, so this is ``B_pi (+) B_omega``.

    Args:
        length_half, width_half: ego half-extents.
        heading_pi, heading_omega: body orientations (rad).
        obstacle_length_half, obstacle_width_half: obstacle half-extents
            (default: same as ego).

    Returns:
        (C, d) with ``C r <= d`` the collision region for the relative offset
        ``r = (x_omega - x_pi, y_omega - y_pi)``.
    """
    ol = length_half if obstacle_length_half is None else obstacle_length_half
    ow = width_half if obstacle_width_half is None else obstacle_width_half

    pi_corners = _rect_corners(length_half, width_half, heading_pi)
    # -B_omega reflected through origin (centrally symmetric rectangle => same set)
    omega_corners = -_rect_corners(ol, ow, heading_omega)

    # Minkowski sum of two convex polygons = convex hull of pairwise sums.
    sums = (pi_corners[:, None, :] + omega_corners[None, :, :]).reshape(-1, 2)
    C, d = _polygon_halfspaces(sums)
    return C, d


def collision_half_space(n_vehicles: int, ego_index: int, obs_index: int,
                         length_half: float, width_half: float,
                         heading_pi: float = 0.0, heading_omega: float = 0.0,
                         obstacle_length_half: float | None = None,
                         obstacle_width_half: float | None = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    r"""Build the collision half-space ``H X <= d`` on the full stacked state.

    Implements Algorithm 1 lines 28-31: lift the xy-plane collision polytope
    ``B_diff = { r : C r <= d }`` to the stacked state by placing ``-C`` on the
    ego's (x, y) columns and ``+C`` on the obstacle's (x, y) columns (so the row
    acts on ``x_omega - x_pi, y_omega - y_pi``), with zeros on the phi columns.

    Args:
        n_vehicles: number of vehicles in the stacked state.
        ego_index, obs_index: 0-based vehicle indices for pi and omega.
        length_half, width_half: ego half-extents.
        heading_pi, heading_omega, obstacle_*: see ``minkowski_difference_box``.

    Returns:
        (H, d) with H of shape (k, 4*n_vehicles).
    """
    C, d = minkowski_difference_box(
        length_half, width_half, heading_pi, heading_omega,
        obstacle_length_half, obstacle_width_half)
    k = C.shape[0]
    H = np.zeros((k, STATE_DIM * n_vehicles))
    ego_x = STATE_DIM * ego_index
    obs_x = STATE_DIM * obs_index
    # row acts on (x_omega - x_pi, y_omega - y_pi):  -C on ego xy, +C on obs xy
    H[:, ego_x:ego_x + 2] = -C
    H[:, obs_x:obs_x + 2] = C
    return H, d


def point_in_collision(rel_xy: np.ndarray, C: np.ndarray, d: np.ndarray,
                       tol: float = 1e-9) -> bool:
    """Return True if a relative offset lies in the collision polytope ``C r <= d``."""
    rel_xy = np.asarray(rel_xy, dtype=float)
    return bool(np.all(C @ rel_xy <= d + tol))
