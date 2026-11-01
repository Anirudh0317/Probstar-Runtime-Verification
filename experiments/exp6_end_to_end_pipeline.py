"""Full perception-to-verification pipeline (extension).

Runs the whole chain end to end, which the other experiments (which build the
initial Gaussian directly) don't:

    ground-truth poses
      -> simulate_pose_detection        (simulated pose DNN, Sec 3.1)
      -> estimate_motion_from_poses      (Figure 4: recover obstacle zeta, V)
      -> build_initial_distribution      (Taylor uncertainty propagation, Sec 3.3)
      -> build_initial_probstar          (standardized ProbStar initial set)
      -> QuantitativeVerifier.verify     (Algorithm 1 reachability + collision prob)
      -> MonteCarloValidator             (independent ground-truth check)

The obstacle moves, so the Figure-4 two-pose motion estimation is actually used.
The simulated pose detector adds noise, so each run is one realization; the seed
is fixed for reproducibility, and we also report how the end-to-end collision
probability varies across several noise realizations.

Note: the pose detector here is the simulated stand-in (Gaussian noise at the
paper's error scales), not the paper's trained network. The point is that the
downstream pipeline takes real pose-estimate objects, propagates their
uncertainty, and verifies -- the pieces fit together end to end.

Outputs:
  results/data/exp6_end_to_end.json/.csv
  results/figures/exp6_end_to_end.png
"""

from __future__ import annotations

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    QuantitativeVerifier, MonteCarloValidator, collision_half_space,
    minkowski_difference_box, load_config,
    simulate_pose_detection, estimate_motion_from_poses,
    build_initial_distribution, build_initial_probstar,
)
from probstar_rv.dynamics import VehicleModel
from probstar_rv.perception import VehicleInit


def run_pipeline(cfg, seed, noise_scale=1.0):
    """Run the full perception->verification pipeline once, return (res, mc, meta).

    Ground-truth world:
      * Ego at origin, heading +x, speed 1.0 m/s, going straight.
      * Obstacle ahead and to the side, moving slowly and turning slightly, so
        the two-pose Figure-4 estimation recovers a nonzero steering and speed.
    """
    L = 0.33
    lh, wh = cfg["vehicle"]["length_half"], cfg["vehicle"]["width_half"]
    dt_r = cfg["reachability"]["dt_r"]
    dt_m = cfg["reachability"]["dt_m"]
    delta = cfg["reachability"]["delta"]
    rng = np.random.default_rng(seed)

    # per-component pose noise (scaled) -- the paper's error magnitudes.
    noise = {k: noise_scale * v for k, v in
             {"sigma_x": 0.015, "sigma_y": 0.015,
              "sigma_ti": 0.03, "sigma_tj": 0.03}.items()}

    # --- ego ground truth: known from actuation (no perception needed) --------
    ego_true = dict(x=0.0, y=0.0, theta=0.0, V=1.0, zeta=0.0)
    ego_pose = simulate_pose_detection(ego_true["x"], ego_true["y"],
                                       ego_true["theta"], noise=noise,
                                       jitter=False, rng=rng)
    ego_init = build_initial_distribution(ego_pose, zeta_mu=ego_true["zeta"],
                                          zeta_var=(0.01) ** 2, V_mu=ego_true["V"])

    # --- obstacle ground truth: must be perceived from two poses --------------
    # Obstacle starts ahead in the ego's lane, facing -x (oncoming), moving
    # toward the ego with a slight steering so the Figure-4 estimate must recover
    # both a speed and a nonzero steering angle.  Geometry is set so the two
    # vehicles are on a genuine collision course.
    obs_V, obs_zeta = 0.6, 0.12
    obs_beta = np.arctan(np.tan(obs_zeta) / 2.0)
    obs_w = obs_V * np.tan(obs_zeta) * np.cos(obs_beta) / L

    def obs_pose_at(t):
        x = 2.5; y = 0.2; th = np.pi  # oncoming in the ego's lane
        n = int(round(t / 5e-4))
        for _ in range(n):
            x += 5e-4 * obs_V * np.cos(th + obs_beta)
            y += 5e-4 * obs_V * np.sin(th + obs_beta)
            th += 5e-4 * obs_w
        return x, y, th

    # two consecutive perceived poses, dt_r apart
    x0, y0, t0 = obs_pose_at(0.0)
    x1, y1, t1 = obs_pose_at(dt_r)
    p0 = simulate_pose_detection(x0, y0, t0, noise=noise, jitter=True, rng=rng)
    p1 = simulate_pose_detection(x1, y1, t1, noise=noise, jitter=True, rng=rng)

    # Figure-4 motion estimation from the two poses
    motion = estimate_motion_from_poses(p0, p1, dt_r, L)
    obs_init = build_initial_distribution(
        p1, zeta_mu=motion["zeta_mu"], zeta_var=motion["zeta_var"],
        V_mu=motion["V_mu"])

    # --- assemble the stacked probabilistic initial state ---------------------
    inits = [ego_init, obs_init]
    P0 = build_initial_probstar(inits, delta=delta)
    mu = np.concatenate([vi.mean for vi in inits])
    from scipy.linalg import block_diag
    Sigma = block_diag(*[vi.cov for vi in inits])

    # --- plant models from the estimated speeds/steer -------------------------
    models = [VehicleModel(L, ego_init.V_mu, ego_init.zeta_mu),
              VehicleModel(L, obs_init.V_mu, obs_init.zeta_mu)]
    k_max = 10
    H, d = collision_half_space(2, 0, 1, lh, wh)
    Cxy, dxy = minkowski_difference_box(lh, wh)

    ver = QuantitativeVerifier(models, dt_r, dt_m, lh, wh,
                               rng=np.random.default_rng(seed + 1))
    res = ver.verify(P0, H, d, k_max, recompute_collision=True)

    mc = MonteCarloValidator(models, dt_r, dt_m, lh, wh,
                             rng=np.random.default_rng(seed + 2))
    mcres = mc.run(mu, Sigma, k_max, 100000, recompute_collision=True)

    meta = {
        "obs_true_V": obs_V, "obs_est_V": motion["V_mu"],
        "obs_true_zeta": obs_zeta, "obs_est_zeta": motion["zeta_mu"],
        "obs_est_zeta_std": float(np.sqrt(motion["zeta_var"])),
    }
    return res, mcres, meta


def main():
    style()
    cfg = load_config()
    seed = cfg["seed"]

    # --- single reference run (full pipeline) ---
    res, mcres, meta = run_pipeline(cfg, seed, noise_scale=1.0)
    maxerr = float(np.max(np.abs(res.per_step_prob - mcres.per_step_prob)))

    # --- robustness across perception-noise realizations ---
    peaks = []
    for s in range(10):
        r, _, _ = run_pipeline(cfg, seed + 1000 + s, noise_scale=1.0)
        peaks.append(float(r.per_step_prob.max()))
    peaks = np.array(peaks)

    rows = [{
        "step": k + 1,
        "time_s": round(float(res.times[k]), 3),
        "probstar_per_step": round(float(res.per_step_prob[k]), 5),
        "mc_per_step": round(float(mcres.per_step_prob[k]), 5),
    } for k in range(len(res.per_step_prob))]
    save_csv("exp6_end_to_end.csv", rows)
    save_json("exp6_end_to_end.json", {
        "motion_estimation": meta,
        "max_abs_err_probstar_vs_mc": maxerr,
        "peak_prob_reference": float(res.per_step_prob.max()),
        "peak_prob_across_noise_realizations": {
            "mean": float(peaks.mean()), "std": float(peaks.std()),
            "min": float(peaks.min()), "max": float(peaks.max()),
            "values": peaks.tolist(),
        },
        "note": "Full pipeline: simulated pose detection -> Figure-4 motion "
                "estimation -> Taylor uncertainty propagation -> ProbStar "
                "reachability -> verification, validated against Monte Carlo.",
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot(res.times, 100 * res.per_step_prob, "o-", label="ProbStar (full pipeline)")
    ax.plot(mcres.times, 100 * mcres.per_step_prob, "s--", label="Monte Carlo")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("per-step collision probability (%)")
    ax.set_title("End-to-end: simulated perception -> verification")
    ax.legend(fontsize=8)
    save_fig(fig, "exp6_end_to_end.png")

    print("\n=== Experiment 6: full perception-to-verification pipeline ===")
    print("Figure-4 motion estimation (obstacle):")
    print("  V:    true=%.3f  estimated=%.3f" % (meta["obs_true_V"], meta["obs_est_V"]))
    print("  zeta: true=%.3f  estimated=%.3f (+-%.3f)" %
          (meta["obs_true_zeta"], meta["obs_est_zeta"], meta["obs_est_zeta_std"]))
    print("Reference run peak collision prob: %.3f" % res.per_step_prob.max())
    print("ProbStar vs MC max per-step error: %.4f" % maxerr)
    print("Peak prob across 10 perception-noise realizations: "
          "mean=%.3f std=%.3f [min %.3f, max %.3f]" %
          (peaks.mean(), peaks.std(), peaks.min(), peaks.max()))
    print("Saved: results/data/exp6_end_to_end.{json,csv}, "
          "results/figures/exp6_end_to_end.png")


if __name__ == "__main__":
    main()
