"""Runtime vs. number of reachability steps (shape of Table 3).

Shows how verification runtime grows with the horizon length k_max. The paper
splits runtime into object detection, pose estimation, initial-state estimation,
and probstar calculation, with the probstar probability part dominating and
growing with k_max. We only measure the parts we implement (initial-state
construction and probstar reachability + probability); the perception DNN timings
are not reproduced, since we don't run the real networks.

Outputs:
  results/data/exp3_timing.json/.csv
  results/figures/exp3_timing.png
"""

from __future__ import annotations

import time

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    ProbStar, QuantitativeVerifier, collision_half_space, load_config, scenarios,
)


def main():
    style()
    cfg = load_config()
    seed = cfg["seed"]
    dt_r = cfg["reachability"]["dt_r"]
    dt_m = cfg["reachability"]["dt_m"]
    delta = cfg["reachability"]["delta"]

    scn = scenarios.rear_end_dynamic(gap=3.0, v_ego=1.0, k_max=100)
    lh, wh = scn.length_half, scn.width_half
    H, d = collision_half_space(2, 0, 1, lh, wh)

    k_values = [5, 10, 20, 100]
    rows = []
    for k in k_values:
        ver = QuantitativeVerifier(scn.models, dt_r, dt_m, lh, wh,
                                   prob_backend=cfg["probability"]["mvn_backend"],
                                   genz_samples=cfg["probability"]["genz_samples"],
                                   scipy_maxdim=cfg["probability"]["scipy_maxdim"],
                                   rng=np.random.default_rng(seed))

        # time initial-state (probstar construction) separately.
        t0 = time.perf_counter()
        for _ in range(20):
            P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=delta)
        init_ms = 1000.0 * (time.perf_counter() - t0) / 20

        # time the whole reachability+probability loop (median of a few runs).
        best = np.inf
        for _ in range(3):
            P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=delta)
            t0 = time.perf_counter()
            ver.verify(P0, H, d, k, record_centers=False)
            best = min(best, time.perf_counter() - t0)
        total_ms = 1000.0 * best
        rows.append({
            "k_max": k,
            "init_state_ms": round(init_ms, 3),
            "reach_plus_prob_ms_total": round(total_ms, 3),
            "reach_plus_prob_ms_per_step": round(total_ms / k, 3),
        })

    save_csv("exp3_timing.csv", rows)
    save_json("exp3_timing.json", {"rows": rows,
                                   "note": "Only initial-state and probstar "
                                   "timings are measured; perception-DNN "
                                   "timings from Table 3 are not reproduced "
                                   "(real networks not run)."})

    # Rare-event cost crossover: why the paper reports big speedups.
    # ProbStar's cost does not depend on the probability magnitude. Monte Carlo,
    # to estimate p to a target relative std error `relerr` at ~95% confidence,
    # needs N >= z^2 (1 - p) / (p * relerr^2) with z = 1.96, which blows up as
    # p -> 0. We measure ProbStar's per-query cost and MC's per-sample-per-step
    # cost, then project the MC time for a 10% relative error across a range of p.
    # That's the rare-collision regime where ProbStar wins by orders of magnitude,
    # and it's why at the moderate probabilities and small sample counts in exp1
    # vectorized MC is competitive instead.
    from probstar_rv import MonteCarloValidator, minkowski_difference_box

    # measured ProbStar per-query (single reachable set) probability time
    ps_probe = QuantitativeVerifier(scn.models, dt_r, dt_m, lh, wh,
                                    rng=np.random.default_rng(cfg["seed"]))
    P0 = ProbStar.from_gaussian(scn.mu, scn.Sigma, delta=delta)
    t0 = time.perf_counter()
    ps_probe.verify(P0, H, d, 10, record_centers=False)
    ps_per_query_s = (time.perf_counter() - t0) / 10.0

    # measured MC cost per (sample * step)
    Cxy, dxy = minkowski_difference_box(lh, wh)
    mc_probe = MonteCarloValidator(scn.models, dt_r, dt_m, lh, wh,
                                   rng=np.random.default_rng(cfg["seed"]))
    n_probe = 200000
    t0 = time.perf_counter()
    mc_probe.run(scn.mu, scn.Sigma, 10, n_probe, collision_C=Cxy, collision_d=dxy)
    mc_per_sample_step_s = (time.perf_counter() - t0) / (n_probe * 10)

    z, relerr = 1.96, 0.10
    probs = [0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-6]
    crossover = []
    for p in probs:
        n_needed = z ** 2 * (1 - p) / (p * relerr ** 2)
        mc_time_s = n_needed * mc_per_sample_step_s  # for a single-step estimate
        crossover.append({
            "true_prob": p,
            "mc_samples_needed_10pct": int(n_needed),
            "mc_time_s_est": round(mc_time_s, 4),
            "probstar_time_s": round(ps_per_query_s, 6),
            "probstar_faster": bool(ps_per_query_s < mc_time_s),
        })
    save_csv("exp3_rare_event_cost.csv", crossover)
    save_json("exp3_rare_event_cost.json", {
        "probstar_per_query_s": ps_per_query_s,
        "mc_per_sample_step_s": mc_per_sample_step_s,
        "target_relative_error": relerr,
        "rows": crossover,
        "note": "MC sample count for 10%% relative std error at 95%% confidence; "
                "ProbStar cost is independent of p. The crossover shows where "
                "ProbStar wins (rare events), consistent with the paper.",
    })

    fig2, ax2 = plt.subplots(figsize=(6, 3.6))
    ps = [c["true_prob"] for c in crossover]
    ax2.loglog(ps, [c["mc_time_s_est"] for c in crossover], "o-",
               label="Monte Carlo (10% rel. error)")
    ax2.axhline(ps_per_query_s, color="C1", ls="--",
                label="ProbStar (independent of p)")
    ax2.set_xlabel("true collision probability p")
    ax2.set_ylabel("time to estimate p (s)")
    ax2.set_title("Rare-event cost: MC explodes as p -> 0, ProbStar is flat")
    ax2.legend(fontsize=8)
    ax2.invert_xaxis()
    save_fig(fig2, "exp3_rare_event_cost.png")

    ks = [r["k_max"] for r in rows]
    tot = [r["reach_plus_prob_ms_total"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, tot, "o-")
    ax.set_xlabel("k_max (number of reachability steps)")
    ax.set_ylabel("reachability + probability time (ms)")
    ax.set_title("Verification runtime grows with horizon (cf. Table 3)")
    save_fig(fig, "exp3_timing.png")

    print("\n=== Experiment 3: timing vs k_max (Table 3 shape) ===")
    print("{:>6} {:>14} {:>22} {:>22}".format(
        "k_max", "init_ms", "reach+prob_ms", "per_step_ms"))
    for r in rows:
        print("{:>6} {:>14} {:>22} {:>22}".format(
            r["k_max"], r["init_state_ms"],
            r["reach_plus_prob_ms_total"], r["reach_plus_prob_ms_per_step"]))
    print("\n--- Rare-event cost crossover (why the paper sees big speedups) ---")
    print("ProbStar per-query: %.4f s (independent of p)" % ps_per_query_s)
    print("{:>12} {:>22} {:>16} {:>16}".format(
        "true_prob", "mc_samples_for_10%", "mc_time_s", "probstar_faster"))
    for c in crossover:
        print("{:>12.0e} {:>22,} {:>16.4f} {:>16}".format(
            c["true_prob"], c["mc_samples_needed_10pct"],
            c["mc_time_s_est"], str(c["probstar_faster"])))
    print("Saved: results/data/exp3_timing.{json,csv}, "
          "results/data/exp3_rare_event_cost.{json,csv}, "
          "results/figures/exp3_timing.png, exp3_rare_event_cost.png")


if __name__ == "__main__":
    main()
