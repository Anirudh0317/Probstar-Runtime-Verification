"""Verification-guided collision avoidance (extension; Figure 7 / Section 5.3).

Given a predicted collision, check how different control actions change the
collision-probability curve. The ego starts on a collision course with a
stationary obstacle; we re-run the verifier for each action and compare.

Two action types:

* Dynamic braking: the ego decelerates at a constant rate (m/s^2), so its speed
  changes over the horizon -- a real braking profile, not just a lower constant
  speed. Uses a time-varying plant matrix via braking_substep_matrices /
  verify(substep_matrices=...).
* Steering: a fixed steering angle. The ego turns, so its collision box rotates
  and the collision predicate is recomputed each step (Algorithm 1's
  collisionPredicate(Theta_k)).

This is one of the extensions beyond the paper: the verifier is used as a
decision tool to see which actions avoid the collision.

Outputs:
  results/data/exp4_control_actions.json/.csv
  results/figures/exp4_braking.png, exp4_steering.png
"""

from __future__ import annotations

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    ProbStar, QuantitativeVerifier, collision_half_space, load_config,
    braking_substep_matrices, braking_speed_profile,
)
from probstar_rv.dynamics import VehicleModel


def run_action(models, mu, Sigma, cfg, k_max, lh, wh, recompute_collision=False):
    H, d = collision_half_space(2, 0, 1, lh, wh)
    ver = QuantitativeVerifier(models, cfg["reachability"]["dt_r"],
                               cfg["reachability"]["dt_m"], lh, wh,
                               rng=np.random.default_rng(cfg["seed"]))
    P0 = ProbStar.from_gaussian(mu, Sigma, delta=cfg["reachability"]["delta"])
    # When the ego steers, its heading changes over the horizon, so the collision
    # bounding box rotates -- recompute the collision predicate each step
    # (Algorithm 1's collisionPredicate(Theta_k)).  For pure braking (no steering)
    # the heading is fixed and the default fixed-H path is exact.
    return ver.verify(P0, H, d, k_max, record_centers=False,
                      recompute_collision=recompute_collision)


def main():
    style()
    cfg = load_config()
    L = 0.33
    lh, wh = cfg["vehicle"]["length_half"], cfg["vehicle"]["width_half"]
    k_max = 12
    v0 = 1.5                     # initial ego speed (collision course)
    gap = 1.15                   # obstacle distance ahead
    sigma_pos = 0.06

    mu = np.array([0, 0, 1, 0, gap, 0, 1, 0], float)
    Sigma = np.diag([sigma_pos**2, sigma_pos**2, 1e-4, 1e-4,
                     sigma_pos**2, sigma_pos**2, 1e-4, 1e-4])

    # ---------------- Dynamic braking: constant deceleration -----------------
    # The ego decelerates at `decel` m/s^2 from v0; its speed changes each step,
    # so the plant matrix is time-varying (braking_substep_matrices).
    dt_r = cfg["reachability"]["dt_r"]
    dt_m = cfg["reachability"]["dt_m"]
    decels = [0.0, 1.5, 2.0, 3.0, 4.0]      # m/s^2 (0 = no braking)
    H, d = collision_half_space(2, 0, 1, lh, wh)
    ego_template = VehicleModel(L, v0, 0.0)
    other = [VehicleModel(L, 0.0, 0.0)]     # stationary obstacle
    braking = {}
    brake_rows = []
    for a in decels:
        sub_mats = braking_substep_matrices(
            ego_template, other, v0=v0, decel=a, dt_r=dt_r, dt_m=dt_m,
            k_max=k_max, ego_index=0)
        ver = QuantitativeVerifier([ego_template] + other, dt_r, dt_m, lh, wh,
                                   rng=np.random.default_rng(cfg["seed"]))
        P0 = ProbStar.from_gaussian(mu, Sigma, delta=cfg["reachability"]["delta"])
        res = ver.verify(P0, H, d, k_max, record_centers=False,
                         substep_matrices=sub_mats)
        speeds = braking_speed_profile(v0, a, dt_r, k_max)
        stop_step = int(np.argmax(speeds <= 1e-9)) if np.any(speeds <= 1e-9) else -1
        # Validate the time-varying-plant ProbStar against a Monte-Carlo run
        # using the SAME per-step matrices (proves the braking path is correct).
        from probstar_rv import MonteCarloValidator, minkowski_difference_box
        Cxy, dxy = minkowski_difference_box(lh, wh)
        mc = MonteCarloValidator([ego_template] + other, dt_r, dt_m, lh, wh,
                                 rng=np.random.default_rng(cfg["seed"] + 7))
        mcres = mc.run(mu, Sigma, k_max, 100000, collision_C=Cxy, collision_d=dxy,
                       step_matrices=sub_mats)
        mc_maxerr = float(np.max(np.abs(res.per_step_prob - mcres.per_step_prob)))

        braking[f"decel_{a:.0f}"] = res.per_step_prob.tolist()
        brake_rows.append({
            "action": f"decel_{a:.1f}_mps2",
            "decel_mps2": a,
            "stops_at_step": stop_step,
            "peak_collision_prob": round(float(res.per_step_prob.max()), 4),
            "union_lower": round(float(res.union_lower_bound), 4),
            "max_abs_err_vs_mc": round(mc_maxerr, 4),
        })

    # ---------------- Steering: apply a steering angle to the ego ------------
    steer_angles = [0.0, 0.1, 0.2, 0.3, 0.45]  # rad
    steering = {}
    steer_rows = []
    for za in steer_angles:
        models = [VehicleModel(L, v0, za), VehicleModel(L, 0.0, 0.0)]
        # steering turns the ego -> heading-aware collision predicate
        res = run_action(models, mu, Sigma, cfg, k_max, lh, wh,
                         recompute_collision=True)
        steering[f"steer_{za:.2f}rad"] = res.per_step_prob.tolist()
        steer_rows.append({
            "action": f"steer_{za:.2f}rad",
            "steering_rad": za,
            "peak_collision_prob": round(float(res.per_step_prob.max()), 4),
            "union_lower": round(float(res.union_lower_bound), 4),
        })

    times = np.arange(1, k_max + 1) * cfg["reachability"]["dt_r"]

    save_json("exp4_control_actions.json", {
        "braking": braking, "steering": steering,
        "braking_summary": brake_rows, "steering_summary": steer_rows,
        "times_s": times.tolist(),
    })
    save_csv("exp4_braking.csv", brake_rows)
    save_csv("exp4_steering.csv", steer_rows)

    # Plots.
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for a in decels:
        key = f"decel_{a:.0f}"
        label = "no braking" if a == 0 else f"{a:.0f} m/s² braking"
        ax.plot(times, 100 * np.array(braking[key]), "o-", label=label)
    ax.set_xlabel("time (s)"); ax.set_ylabel("per-step collision prob (%)")
    ax.set_title("Dynamic braking (constant decel.) vs. collision probability")
    ax.legend(fontsize=7)
    save_fig(fig, "exp4_braking.png")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for za in steer_angles:
        key = f"steer_{za:.2f}rad"
        ax.plot(times, 100 * np.array(steering[key]), "o-",
                label=f"steer {za:.2f} rad")
    ax.set_xlabel("time (s)"); ax.set_ylabel("per-step collision prob (%)")
    ax.set_title("Steering action vs. predicted collision probability")
    ax.legend(fontsize=7)
    save_fig(fig, "exp4_steering.png")

    print("\n=== Experiment 4: verification-guided collision avoidance ===")
    print("Dynamic braking (constant deceleration from v0=%.1f m/s):" % v0)
    for r in brake_rows:
        print("  %-18s decel=%.1f  stops@step=%d  peak=%.3f  err_vs_mc=%.4f" %
              (r["action"], r["decel_mps2"], r["stops_at_step"],
               r["peak_collision_prob"], r["max_abs_err_vs_mc"]))
    print("Steering:")
    for r in steer_rows:
        print("  %-14s angle=%.2f  peak=%.3f  union_lb=%.3f" %
              (r["action"], r["steering_rad"], r["peak_collision_prob"], r["union_lower"]))
    print("Saved: results/data/exp4_control_actions.json, "
          "results/figures/exp4_{braking,steering}.png")


if __name__ == "__main__":
    main()
