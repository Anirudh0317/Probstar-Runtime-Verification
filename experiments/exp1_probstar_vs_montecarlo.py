"""Compare ProbStar and Monte-Carlo collision probabilities (reproduces Table 4).

Runs the two hand-computable scenarios from Figure 8:
  * Trial 1: rear-end collision, true probability 50.0%.
  * Trial 2: side collision,     true probability  2.28%.

For each, computes the collision probability both ways and reports accuracy and
timing, as in the paper's Table 4.

Outputs:
  results/data/exp1_table4.json, results/data/exp1_table4.csv
  results/figures/exp1_trial1_curve.png, exp1_trial2_curve.png
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    ProbStar, QuantitativeVerifier, MonteCarloValidator,
    collision_half_space, minkowski_difference_box, load_config, scenarios,
)


def run_trial(scn, cfg, n_mc, seed):
    lh, wh = scn.length_half, scn.width_half
    dt_r = cfg["reachability"]["dt_r"]
    dt_m = cfg["reachability"]["dt_m"]
    delta = cfg["reachability"]["delta"]

    H, d = collision_half_space(len(scn.models), scn.ego_index, scn.obs_index, lh, wh)
    Cxy, dxy = minkowski_difference_box(lh, wh)

    ver = QuantitativeVerifier(
        scn.models, dt_r, dt_m, lh, wh,
        prob_backend=cfg["probability"]["mvn_backend"],
        genz_samples=cfg["probability"]["genz_samples"],
        scipy_maxdim=cfg["probability"]["scipy_maxdim"],
        rng=np.random.default_rng(seed))
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=delta)

    # time just the probability computation at the reporting step (as in paper).
    res = ver.verify(P0, H, d, scn.k_max)

    mc = MonteCarloValidator(scn.models, dt_r, dt_m, lh, wh,
                             rng=np.random.default_rng(seed + 100))
    mcres = mc.run(scn.mu, scn.Sigma, scn.k_max, n_mc,
                   collision_C=Cxy, collision_d=dxy)

    # Reporting probability: final step for Trial 1, first step for Trial 2.
    if scn.expected_kind == "per_step_final":
        ps_prob = float(res.per_step_prob[-1])
        mc_prob = float(mcres.per_step_prob[-1])
    else:
        ps_prob = float(res.per_step_prob[0])
        mc_prob = float(mcres.per_step_prob[0])

    return res, mcres, ps_prob, mc_prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-samples", type=int, default=None,
                    help="override Monte-Carlo sample count")
    args = ap.parse_args()

    style()
    cfg = load_config()
    seed = cfg["seed"]
    n_mc = args.mc_samples or cfg["monte_carlo"]["n_samples"]

    trials = [scenarios.trial1_rear_end(), scenarios.trial2_side()]
    table_rows = []
    summary = {}

    for scn in trials:
        res, mcres, ps_prob, mc_prob = run_trial(scn, cfg, n_mc, seed)
        # Both methods produce the full k_max-step curve, so compare total
        # runtime to total runtime. Per-step figures are reported separately as
        # amortized cost (same basis for both), not as the headline.
        ps_ms_total = 1000.0 * res.runtime_s
        mc_ms_total = 1000.0 * mcres.runtime_s
        ps_ms_step = ps_ms_total / max(scn.k_max, 1)
        mc_ms_step = mc_ms_total / max(scn.k_max, 1)
        speedup = (mc_ms_total / ps_ms_total) if ps_ms_total > 0 else float("nan")

        row = {
            "scenario": scn.name,
            "true_prob_percent": round(100 * scn.expected_prob, 3),
            "probstar_prob_percent": round(100 * ps_prob, 3),
            "montecarlo_prob_percent": round(100 * mc_prob, 3),
            "probstar_err_vs_true_pp": round(100 * abs(ps_prob - scn.expected_prob), 3),
            "montecarlo_err_vs_true_pp": round(100 * abs(mc_prob - scn.expected_prob), 3),
            # like-for-like: total runtime to produce the whole per-step curve
            "probstar_time_ms_total": round(ps_ms_total, 2),
            "montecarlo_time_ms_total": round(mc_ms_total, 2),
            "speedup_mc_over_probstar": round(speedup, 2),
            # amortized per-step (supplementary, same basis for both)
            "probstar_time_ms_per_step": round(ps_ms_step, 3),
            "montecarlo_time_ms_per_step": round(mc_ms_step, 3),
            "mc_samples": n_mc,
        }
        table_rows.append(row)
        summary[scn.name] = row

        # Plot the full per-step curves for both methods.
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(res.times, 100 * res.per_step_prob, "o-", label="ProbStar")
        ax.plot(mcres.times, 100 * mcres.per_step_prob, "s--",
                label=f"Monte Carlo (N={n_mc})")
        ax.axhline(100 * scn.expected_prob, color="k", ls=":", lw=1,
                   label=f"true = {100*scn.expected_prob:.2f}%")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("per-step collision probability (%)")
        ax.set_title(scn.note)
        ax.legend(fontsize=8)
        save_fig(fig, f"exp1_{scn.name}_curve.png")

    save_json("exp1_table4.json", {"mc_samples": n_mc, "trials": summary})
    save_csv("exp1_table4.csv", table_rows)

    # Console report. Timing is TOTAL-vs-TOTAL (both produce the full per-step
    # curve); n_mc is printed so the Monte-Carlo cost is contextualized.
    print("\n=== Experiment 1: ProbStar vs Monte Carlo (reproduces Table 4) ===")
    print(f"(timings are full-run totals for the whole {trials[0].k_max}-step "
          f"curve; Monte Carlo uses N={n_mc} samples)")
    hdr = ("scenario", "true%", "probstar%", "mc%", "ps_err_pp",
           "ps_ms_tot", "mc_ms_tot", "speedup")
    print("{:<18} {:>7} {:>10} {:>8} {:>10} {:>10} {:>10} {:>8}".format(*hdr))
    for r in table_rows:
        print("{:<18} {:>7} {:>10} {:>8} {:>10} {:>10} {:>10} {:>7}x".format(
            r["scenario"], r["true_prob_percent"], r["probstar_prob_percent"],
            r["montecarlo_prob_percent"], r["probstar_err_vs_true_pp"],
            r["probstar_time_ms_total"], r["montecarlo_time_ms_total"],
            r["speedup_mc_over_probstar"]))
    print("\nNote: the speedup here is modest because N is small; it grows with "
          "the sample count needed for rare-event accuracy (cf. the paper's "
          "hundreds-of-seconds Monte-Carlo runs vs ~15 ms ProbStar).")
    print("Saved: results/data/exp1_table4.{json,csv}, "
          "results/figures/exp1_*_curve.png")


if __name__ == "__main__":
    main()
