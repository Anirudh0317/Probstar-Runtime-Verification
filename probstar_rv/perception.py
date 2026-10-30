r"""Perception and probabilistic initial conditions (Section 3 of the paper).

What this does NOT include: the paper's perception front-end is a two-stage
probabilistic DNN (a YOLO detector + a Monte-Carlo-Dropout pose head, Figures
2-3) trained on real F1Tenth camera images. We don't have that network, its
dataset, or the hardware, so this module uses a simulated stand-in instead:

    simulate_pose_detection takes a known ground-truth pose and returns a noisy
    Gaussian pose estimate (mean + per-component std), matching the interface and
    uncertainty shape of the real pose DNN (a multivariate-normal pose with
    aleatoric + epistemic spread).

This is not a reproduction of the paper's network. It just lets us drive and
check the rest of the pipeline (uncertainty propagation, ProbStar reachability,
collision verification) with controllable, repeatable uncertainty. Default noise
magnitudes follow the paper's reported error scales (Tables 1-2).

What IS faithful -- everything downstream of the raw pose estimate:
* Figure 4 geometry: recover an obstacle's steering zeta_w and speed V_w from two
  consecutive poses (law of cosines -> radius of curvature -> steering angle;
  arc length / dt -> speed).
* Second-order Taylor propagation of uncertainty through the nonlinear maps:

      E[f(X)]  ~= f(mu) + 0.5 f''(mu) sigma^2
      Var[f(X)] ~= f'(mu)^2 sigma^2 + 0.25 f''(mu)^2 sigma^4

  applied to theta_w = atan2, beta, phi_x = cos(theta+beta), phi_y =
  sin(theta+beta), and to zeta_w, V_w (Figure 4).
* Section 3.3: assemble the initial state as a Gaussian N(mu, Sigma) over
  [x, y, phi_x, phi_y] per vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .dynamics import STATE_DIM


# --------------------------------------------------------------------------- #
# Simulated probabilistic pose detector
# --------------------------------------------------------------------------- #
@dataclass
class PoseEstimate:
    """A probabilistic pose estimate for one vehicle, as the pose DNN would emit.

    The pose is modeled (as in the paper) as a multivariate normal over
    (x, y, theta_i, theta_j) where (theta_i, theta_j) = (cos theta, sin theta).
    We store means and per-component standard deviations (diagonal covariance,
    matching the paper's independent per-output uncertainty).

    Attributes:
        x, y: position mean (metres), in the ego's reference frame.
        theta_i, theta_j: heading unit-vector mean (cos, sin).
        sigma_x, sigma_y: position std.
        sigma_ti, sigma_tj: heading-component std.
    """

    x: float
    y: float
    theta_i: float
    theta_j: float
    sigma_x: float = 0.0
    sigma_y: float = 0.0
    sigma_ti: float = 0.0
    sigma_tj: float = 0.0

    @property
    def theta(self) -> float:
        """Heading angle recovered as atan2(theta_j, theta_i)."""
        return float(np.arctan2(self.theta_j, self.theta_i))

    @property
    def mean_vec(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta_i, self.theta_j])

    @property
    def std_vec(self) -> np.ndarray:
        return np.array([self.sigma_x, self.sigma_y, self.sigma_ti, self.sigma_tj])


# Default per-component noise scales, loosely following the paper's Table 1/2
# error magnitudes (position ~ 1-2 cm std, heading-vector ~ 0.02-0.05).
DEFAULT_POSE_NOISE = {
    "sigma_x": 0.015,
    "sigma_y": 0.015,
    "sigma_ti": 0.03,
    "sigma_tj": 0.03,
}


def simulate_pose_detection(true_x: float, true_y: float, true_theta: float,
                            noise: Optional[dict] = None,
                            jitter: bool = True,
                            rng: Optional[np.random.Generator] = None
                            ) -> PoseEstimate:
    """Simulate a probabilistic pose detection for a vehicle at a known pose.

    This stands in for the paper's pose DNN (see module docstring): it returns a
    Gaussian pose estimate whose mean is the (optionally jittered) ground-truth
    pose and whose per-component std reflects perception uncertainty.

    Args:
        true_x, true_y, true_theta: ground-truth pose (metres, metres, rad).
        noise: dict of std values (keys of DEFAULT_POSE_NOISE); defaults used
            for any missing key.
        jitter: if True, perturb the reported mean by a draw from the noise
            (so the estimate's mean is not exactly the truth, as in reality).
            If False, the mean equals the truth (useful for controlled tests).
        rng: numpy Generator for the jitter.

    Returns:
        A PoseEstimate.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    ns = dict(DEFAULT_POSE_NOISE)
    if noise:
        ns.update(noise)

    ti = np.cos(true_theta)
    tj = np.sin(true_theta)

    x, y = true_x, true_y
    if jitter:
        x = true_x + rng.normal(0.0, ns["sigma_x"])
        y = true_y + rng.normal(0.0, ns["sigma_y"])
        ti = ti + rng.normal(0.0, ns["sigma_ti"])
        tj = tj + rng.normal(0.0, ns["sigma_tj"])

    return PoseEstimate(x=x, y=y, theta_i=ti, theta_j=tj,
                        sigma_x=ns["sigma_x"], sigma_y=ns["sigma_y"],
                        sigma_ti=ns["sigma_ti"], sigma_tj=ns["sigma_tj"])


# --------------------------------------------------------------------------- #
# Second-order Taylor uncertainty propagation (paper's E/Var formulas)
# --------------------------------------------------------------------------- #
def taylor_moments(f, fp, fpp, mu: float, var: float) -> Tuple[float, float]:
    r"""Propagate (mu, var) of a scalar X through a scalar function f.

    Uses the paper's second-order formulas:

        E[f(X)]  ~= f(mu) + 0.5 f''(mu) var
        Var[f(X)] ~= f'(mu)^2 var + 0.25 f''(mu)^2 var^2

    Args:
        f, fp, fpp: the function and its first/second derivatives.
        mu, var: mean and variance of the input X.

    Returns:
        (mean, variance) of f(X).
    """
    m = f(mu) + 0.5 * fpp(mu) * var
    v = fp(mu) ** 2 * var + 0.25 * fpp(mu) ** 2 * var ** 2
    return float(m), float(max(v, 0.0))


def _theta_from_components_moments(ti_mu, ti_var, tj_mu, tj_var):
    r"""Moments of theta = atan2(theta_j, theta_i) via first-order propagation.

    d(atan2(j,i)) = (-j/(i^2+j^2)) di + (i/(i^2+j^2)) dj.  First-order variance
    combines the two independent component variances.  (The paper treats
    theta_w = atan^{-1}(theta_j/theta_i) as a composite function propagated the
    same Taylor way; using atan2 keeps it valid in all quadrants.)
    """
    r2 = ti_mu ** 2 + tj_mu ** 2
    r2 = max(r2, 1e-9)
    theta_mu = np.arctan2(tj_mu, ti_mu)
    dti = -tj_mu / r2
    dtj = ti_mu / r2
    theta_var = dti ** 2 * ti_var + dtj ** 2 * tj_var
    return float(theta_mu), float(theta_var)


def propagate_heading_slip(pose: PoseEstimate, zeta_mu: float, zeta_var: float,
                           ) -> dict:
    r"""Propagate heading + slip into (theta, beta, phi_x, phi_y) moments.

    beta = arctan(tan(zeta)/2).  phi_x = cos(theta+beta), phi_y = sin(theta+beta).
    Each is propagated with the second-order Taylor formulas; theta and beta are
    assumed independent (heading from the detector, slip from steering), so the
    variance of (theta+beta) is the sum.

    Returns a dict with means and variances for theta, beta, phi_x, phi_y.
    """
    theta_mu, theta_var = _theta_from_components_moments(
        pose.theta_i, pose.sigma_ti ** 2, pose.theta_j, pose.sigma_tj ** 2)

    # beta = arctan(tan(zeta)/2)
    def b(z):
        return np.arctan(np.tan(z) / 2.0)

    def bp(z):
        t = np.tan(z)
        sec2 = 1.0 / np.cos(z) ** 2
        return (0.5 * sec2) / (1.0 + (t / 2.0) ** 2)

    def bpp(z):
        # numerical second derivative for robustness
        h = 1e-5
        return (bp(z + h) - bp(z - h)) / (2 * h)

    beta_mu, beta_var = taylor_moments(b, bp, bpp, zeta_mu, zeta_var)

    # angle a = theta + beta
    a_mu = theta_mu + beta_mu
    a_var = theta_var + beta_var

    # phi_x = cos(a), phi_y = sin(a) via second-order Taylor in a
    cos_mu = np.cos(a_mu) - 0.5 * np.cos(a_mu) * a_var
    cos_var = (np.sin(a_mu) ** 2) * a_var + 0.25 * (np.cos(a_mu) ** 2) * a_var ** 2
    sin_mu = np.sin(a_mu) - 0.5 * np.sin(a_mu) * a_var
    sin_var = (np.cos(a_mu) ** 2) * a_var + 0.25 * (np.sin(a_mu) ** 2) * a_var ** 2

    return {
        "theta_mu": theta_mu, "theta_var": theta_var,
        "beta_mu": beta_mu, "beta_var": beta_var,
        "phi_x_mu": float(cos_mu), "phi_x_var": float(max(cos_var, 0.0)),
        "phi_y_mu": float(sin_mu), "phi_y_var": float(max(sin_var, 0.0)),
    }


# --------------------------------------------------------------------------- #
# Figure 4: steering + speed from two consecutive poses
# --------------------------------------------------------------------------- #
def estimate_motion_from_poses(pose0: PoseEstimate, pose1: PoseEstimate,
                               dt: float, wheelbase_L: float) -> dict:
    r"""Estimate (zeta_w, V_w) and their variances from two poses (Figure 4).

    Geometry (paper Figure 4):
        Delta_theta = | theta^1 - theta^0 |
        R = sqrt( (x1-x0)^2 + (y1-y0)^2 / (2 (1 - cos Delta_theta)) )
        zeta_w = arctan( 2 tan( L / (2 R) ) )
        V_w = R Delta_theta / dt

    When Delta_theta -> 0 (straight motion) the radius R -> infinity; we handle
    that limit by falling back to zeta_w = 0 and V_w = chord / dt.

    Uncertainty is propagated by local linearization (a Jacobian-based
    first/second-order approximation) of these maps w.r.t. the pose components,
    consistent with the paper's Taylor-polynomial approach.

    Returns a dict with zeta_mu, zeta_var, V_mu, V_var (and R, dtheta for info).
    """
    x0, y0, th0 = pose0.x, pose0.y, pose0.theta
    x1, y1, th1 = pose1.x, pose1.y, pose1.theta

    dtheta = abs(th1 - th0)
    # wrap into [0, pi]
    dtheta = min(dtheta, 2 * np.pi - dtheta)
    chord2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
    chord = np.sqrt(chord2)

    straight_eps = 1e-4
    if dtheta < straight_eps:
        zeta_mu = 0.0
        R = np.inf
        V_mu = chord / dt
    else:
        R = np.sqrt(chord2 / (2.0 * (1.0 - np.cos(dtheta))))
        zeta_mu = np.arctan(2.0 * np.tan(wheelbase_L / (2.0 * R)))
        V_mu = R * dtheta / dt

    # --- uncertainty via Monte-Carlo-free local linearization -------------
    # Build the input covariance (x0,y0,th0,x1,y1,th1) assuming independence,
    # then push through a finite-difference Jacobian of (zeta, V).
    th0_mu, th0_var = _theta_from_components_moments(
        pose0.theta_i, pose0.sigma_ti ** 2, pose0.theta_j, pose0.sigma_tj ** 2)
    th1_mu, th1_var = _theta_from_components_moments(
        pose1.theta_i, pose1.sigma_ti ** 2, pose1.theta_j, pose1.sigma_tj ** 2)

    inp_mu = np.array([x0, y0, th0_mu, x1, y1, th1_mu])
    inp_var = np.array([pose0.sigma_x ** 2, pose0.sigma_y ** 2, th0_var,
                        pose1.sigma_x ** 2, pose1.sigma_y ** 2, th1_var])

    def motion(inp):
        X0, Y0, T0, X1, Y1, T1 = inp
        dth = abs(T1 - T0)
        dth = min(dth, 2 * np.pi - dth)
        c2 = (X1 - X0) ** 2 + (Y1 - Y0) ** 2
        if dth < straight_eps:
            return np.array([0.0, np.sqrt(c2) / dt])
        Rr = np.sqrt(c2 / (2.0 * (1.0 - np.cos(dth))))
        z = np.arctan(2.0 * np.tan(wheelbase_L / (2.0 * Rr)))
        v = Rr * dth / dt
        return np.array([z, v])

    # finite-difference Jacobian (2 x 6)
    J = np.zeros((2, 6))
    h = 1e-6
    f0 = motion(inp_mu)
    for k in range(6):
        ip = inp_mu.copy(); ip[k] += h
        J[:, k] = (motion(ip) - f0) / h
    cov_out = J @ np.diag(inp_var) @ J.T

    return {
        "zeta_mu": float(zeta_mu), "zeta_var": float(max(cov_out[0, 0], 0.0)),
        "V_mu": float(V_mu), "V_var": float(max(cov_out[1, 1], 0.0)),
        "R": float(R), "dtheta": float(dtheta),
    }


# --------------------------------------------------------------------------- #
# Section 3.3: build the probabilistic initial state for a vehicle
# --------------------------------------------------------------------------- #
@dataclass
class VehicleInit:
    """Probabilistic initial condition for one vehicle's 4D state.

    mean: (4,) mean of [x, y, phi_x, phi_y].
    cov: (4,4) covariance (diagonal from independent perception components).
    zeta_mu, V_mu: estimated steering + speed (used to build the plant matrix).
    """
    mean: np.ndarray
    cov: np.ndarray
    zeta_mu: float
    V_mu: float


def build_initial_distribution(pose: PoseEstimate, zeta_mu: float,
                               zeta_var: float, V_mu: float,
                               min_var: float = 1e-8) -> VehicleInit:
    r"""Assemble N(mu, Sigma) over [x, y, phi_x, phi_y] for one vehicle.

    * x, y means/vars come directly from the pose estimate (paper: taken
      directly from the pose DNN output).
    * phi_x, phi_y means/vars come from Taylor-propagating (theta, beta) through
      cos/sin (``propagate_heading_slip``).

    Args:
        pose: the vehicle's probabilistic pose estimate.
        zeta_mu, zeta_var: steering angle mean/variance (0 for the ego if known
            from actuation, or from ``estimate_motion_from_poses`` for obstacles).
        V_mu: speed mean (enters the plant matrix, not the state itself).
        min_var: floor added to each variance for numerical SPD-ness.

    Returns:
        A VehicleInit.
    """
    hp = propagate_heading_slip(pose, zeta_mu, zeta_var)
    mean = np.array([pose.x, pose.y, hp["phi_x_mu"], hp["phi_y_mu"]])
    var = np.array([
        pose.sigma_x ** 2, pose.sigma_y ** 2,
        hp["phi_x_var"], hp["phi_y_var"],
    ]) + min_var
    cov = np.diag(var)
    return VehicleInit(mean=mean, cov=cov, zeta_mu=zeta_mu, V_mu=V_mu)


def build_initial_probstar(inits, delta: float = 6.0):
    """Stack per-vehicle VehicleInit distributions into one standardized ProbStar.

    Args:
        inits: sequence of VehicleInit (ego first, then obstacles).
        delta: predicate box half-width in sigmas (paper: 6).

    Returns:
        A ProbStar over the stacked 4n-dimensional state, built from the joint
        Gaussian N(mu, Sigma) (block-diagonal Sigma, since vehicles' perception
        errors are independent).
    """
    from .probstar import ProbStar
    from scipy.linalg import block_diag

    means = np.concatenate([vi.mean for vi in inits])
    cov = block_diag(*[vi.cov for vi in inits])
    return ProbStar.from_gaussian(means, cov, delta=delta)
