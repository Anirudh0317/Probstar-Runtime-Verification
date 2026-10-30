"""Rear-end collision timeline (Figure 6 style).

Ego approaches a stationary obstacle. Plots the per-step and cumulative collision
probabilities over the horizon, plus the reachable-set centers. Matches the shape
of Figure 6: collision probability rises as the ego nears the obstacle.

Outputs:
  results/data/exp2_rear_end.json/.csv
  results/figures/exp2_rear_end_prob.png, exp2_rear_end_positions.png
"""

from __future__ import annotations

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    ProbStar, QuantitativeVerifier, MonteCarloValidator,
    collision_half_space, minkowski_difference_box, load_config, scenarios,
)
from probstar_rv.dynamics import STATE_DIM


def main():
    style()
    cfg = load_config()
    seed = cfg["seed"]
    dt_r = cfg["reachability"]["dt_r"]
    dt_m = cfg["reachability"]["dt_m"]
    delta = cfg["reachability"]["delta"]

    scn = scenarios.rear_end_dynamic(gap=2.0, v_ego=1.0, sigma_pos=0.05, k_max=12)
    lh, wh = scn.length_half, scn.width_half

    H, d = collision_half_space(2, 0, 1, lh, wh)
    Cxy, dxy = minkowski_difference_box(lh, wh)

    ver = QuantitativeVerifier(scn.models, dt_r, dt_m, lh, wh,
                               rng=np.random.default_rng(seed))
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=delta)
    res = ver.verify(P0, H, d, scn.k_max, record_centers=True)

    mc = MonteCarloValidator(scn.models, dt_r, dt_m, lh, wh,
                             rng=np.random.default_rng(seed + 1))
    mcres = mc.run(scn.mu, scn.Sigma, scn.k_max, 20000,   # validator count; the paper's
                   collision_C=Cxy, collision_d=dxy)       # 1000-trial comparison is exp1

    rows = [{
        "step": k + 1,
        "time_s": round(float(res.times[k]), 3),
        "probstar_per_step": round(float(res.per_step_prob[k]), 5),
        "mc_per_step": round(float(mcres.per_step_prob[k]), 5),
        "probstar_cum_max": round(float(res.cumulative_max[k]), 5),
        "probstar_cum_indep": round(float(res.cumulative_independent[k]), 5),
    } for k in range(scn.k_max)]
    save_csv("exp2_rear_end.csv", rows)
    save_json("exp2_rear_end.json", {
        "scenario": scn.name,
        "per_step_probstar": res.per_step_prob.tolist(),
        "per_step_mc": mcres.per_step_prob.tolist(),
        "union_probstar_lower": res.union_lower_bound,
        "union_probstar_indep": res.union_independent,
        "union_mc_true": mcres.union_prob,
        "max_abs_per_step_error": float(np.max(np.abs(res.per_step_prob - mcres.per_step_prob))),
    })

    # --- probability plot ---
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(res.times, 100 * res.per_step_prob, "o-", label="ProbStar per-step")
    ax.plot(mcres.times, 100 * mcres.per_step_prob, "s--",
            label="Monte Carlo per-step")
    ax.plot(res.times, 100 * res.cumulative_max, "^-", alpha=0.6,
            label="cumulative (max, lower bound)")
    ax.plot(res.times, 100 * res.cumulative_independent, "v:", alpha=0.6,
            label="cumulative (indep. reference)")
    ax.axhline(100 * mcres.union_prob, color="k", ls=":", lw=1,
               label=f"true union (MC) = {100*mcres.union_prob:.1f}%")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("collision probability (%)")
    ax.set_title("Rear-end approach: per-step & cumulative collision probability")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save_fig(fig, "exp2_rear_end_prob.png")

    # --- positions plot (reachable-set centers of ego and obstacle) ---
    C = res.reachable_centers  # (k+1, 8)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(C[:, 0], np.arange(len(C)), "o-", label="ego x-center")
    ax.plot(C[:, STATE_DIM + 0], np.arange(len(C)), "s-", label="obstacle x-center")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("reachability step")
    ax.set_title("Reachable-set centers (x) over the horizon")
    ax.legend(fontsize=8)
    save_fig(fig, "exp2_rear_end_positions.png")

    print("\n=== Experiment 2: rear-end timeline (Figure 6 style) ===")
    print("per-step ProbStar:", np.round(res.per_step_prob, 3))
    print("per-step MC:      ", np.round(mcres.per_step_prob, 3))
    print("max |ProbStar - MC| per-step: %.4f" %
          np.max(np.abs(res.per_step_prob - mcres.per_step_prob)))
    print("union: ProbStar(lower)=%.3f  indep.ref=%.3f  true(MC)=%.3f" %
          (res.union_lower_bound, res.union_independent, mcres.union_prob))
    print("Saved: results/data/exp2_rear_end.{json,csv}, "
          "results/figures/exp2_rear_end_*.png")


if __name__ == "__main__":
    main()
