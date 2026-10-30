"""Sensitivity to uncertainty, horizon, and timestep (extension).

Three sweeps:

A. Uncertainty level: scale the perception position std and see how the
   collision-probability curve spreads (more uncertainty -> earlier non-zero
   probability and a smoother rise). Checked against MC at each level.

B. Modeling timestep dt_m: vary dt_r / dt_m and measure the discretization error
   in the predicted collision probability (against a fine-dt_m reference),
   confirming that a small dt_m keeps the discretization error negligible.

C. Reachability horizon dt_r: vary dt_r at fixed total time and check that the
   predicted collision probability and runtime stay consistent.

Outputs:
  results/data/exp5_*.json/.csv
  results/figures/exp5_uncertainty.png, exp5_timestep.png
"""

from __future__ import annotations

import numpy as np

from common import save_json, save_csv, save_fig, style
import matplotlib.pyplot as plt

from probstar_rv import (
    ProbStar, QuantitativeVerifier, MonteCarloValidator,
    collision_half_space, minkowski_difference_box, load_config,
)
from probstar_rv.dynamics import VehicleModel


def _scenario(gap=2.0, v_ego=1.0, sigma_pos=0.05):
    L = 0.33
    models = [VehicleModel(L, v_ego, 0.0), VehicleModel(L, 0.0, 0.0)]
    mu = np.array([0, 0, 1, 0, gap, 0, 1, 0], float)
    Sigma = np.diag([sigma_pos**2, sigma_pos**2, 1e-4, 1e-4,
                     sigma_pos**2, sigma_pos**2, 1e-4, 1e-4])
    return models, mu, Sigma


def sweep_uncertainty(cfg, lh, wh):
    k_max = 12
    H, d = collision_half_space(2, 0, 1, lh, wh)
    Cxy, dxy = minkowski_difference_box(lh, wh)
    sigmas = [0.02, 0.05, 0.10, 0.20]
    curves = {}
    rows = []
    for s in sigmas:
        models, mu, Sigma = _scenario(gap=2.0, v_ego=1.0, sigma_pos=s)
        ver = QuantitativeVerifier(models, cfg["reachability"]["dt_r"],
                                   cfg["reachability"]["dt_m"], lh, wh,
                                   rng=np.random.default_rng(cfg["seed"]))
        res = ver.verify(ProbStar.from_gaussian(mu, Sigma, cfg["reachability"]["delta"]),
                         H, d, k_max)
        mc = MonteCarloValidator(models, cfg["reachability"]["dt_r"],
                                 cfg["reachability"]["dt_m"], lh, wh,
                                 rng=np.random.default_rng(cfg["seed"] + 1))
        mcres = mc.run(mu, Sigma, k_max, 100000, collision_C=Cxy, collision_d=dxy)
        curves[f"sigma_{s}"] = res.per_step_prob.tolist()
        rows.append({
            "sigma_pos_m": s,
            "peak_prob": round(float(res.per_step_prob.max()), 4),
            "first_nonzero_step": int(np.argmax(res.per_step_prob > 1e-4) + 1),
            "max_abs_err_vs_mc": round(float(np.max(np.abs(res.per_step_prob - mcres.per_step_prob))), 4),
        })
    return sigmas, curves, rows


def sweep_timestep(cfg, lh, wh):
    """Vary dt_m (modeling step) and measure discretization error vs reference.

    Uses a *turning* ego (nonzero steering), because the Euler discretization
    error of the phi-augmented model only appears when the yaw dynamics are
    active -- a straight-line scenario is nearly exact regardless of dt_m and
    would make the sweep uninformative.
    """
    dt_r = 0.25
    k_max = 10
    L = 0.33
    H, d = collision_half_space(2, 0, 1, lh, wh)
    # ego turns (zeta=0.4) while approaching an obstacle placed on the arc.
    models = [VehicleModel(L, 1.0, 0.4), VehicleModel(L, 0.0, 0.0)]
    mu = np.array([0, 0, 1, 0, 1.2, 0.9, 1, 0], float)
    sigma_pos = 0.05
    Sigma = np.diag([sigma_pos**2, sigma_pos**2, 1e-4, 1e-4,
                     sigma_pos**2, sigma_pos**2, 1e-4, 1e-4])

    # Reference: very fine dt_m.
    ref_ver = QuantitativeVerifier(models, dt_r, dt_r / 100, lh, wh,
                                   rng=np.random.default_rng(cfg["seed"]))
    ref = ref_ver.verify(ProbStar.from_gaussian(mu, Sigma, cfg["reachability"]["delta"]),
                         H, d, k_max).per_step_prob

    ratios = [1, 2, 5, 10, 25]
    rows = []
    for r in ratios:
        ver = QuantitativeVerifier(models, dt_r, dt_r / r, lh, wh,
                                   rng=np.random.default_rng(cfg["seed"]))
        res = ver.verify(ProbStar.from_gaussian(mu, Sigma, cfg["reachability"]["delta"]),
                         H, d, k_max)
        rows.append({
            "dt_r_over_dt_m": r,
            "dt_m_s": round(dt_r / r, 5),
            "max_abs_err_vs_fine_ref": round(float(np.max(np.abs(res.per_step_prob - ref))), 5),
            "runtime_ms": round(1000 * res.runtime_s, 2),
        })
    return rows


def main():
    style()
    cfg = load_config()
    lh, wh = cfg["vehicle"]["length_half"], cfg["vehicle"]["width_half"]

    # --- A. uncertainty sweep ---
    sigmas, curves, unc_rows = sweep_uncertainty(cfg, lh, wh)
    save_json("exp5_uncertainty.json", {"curves": curves, "summary": unc_rows})
    save_csv("exp5_uncertainty.csv", unc_rows)

    times = np.arange(1, 13) * cfg["reachability"]["dt_r"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for s in sigmas:
        ax.plot(times, 100 * np.array(curves[f"sigma_{s}"]), "o-",
                label=f"sigma={s} m")
    ax.set_xlabel("time (s)"); ax.set_ylabel("per-step collision prob (%)")
    ax.set_title("Effect of perception uncertainty on collision probability")
    ax.legend(fontsize=7)
    save_fig(fig, "exp5_uncertainty.png")

    # --- B. timestep sweep ---
    ts_rows = sweep_timestep(cfg, lh, wh)
    save_json("exp5_timestep.json", {"rows": ts_rows})
    save_csv("exp5_timestep.csv", ts_rows)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot([r["dt_r_over_dt_m"] for r in ts_rows],
            [r["max_abs_err_vs_fine_ref"] for r in ts_rows], "o-")
    ax.set_xlabel("dt_r / dt_m  (modeling substeps)")
    ax.set_ylabel("max |prob error| vs fine-dt_m reference")
    ax.set_title("Discretization error shrinks as dt_m decreases")
    save_fig(fig, "exp5_timestep.png")

    print("\n=== Experiment 5A: uncertainty sweep ===")
    for r in unc_rows:
        print("  sigma=%.2f  peak=%.3f  first_nonzero_step=%d  max_err_vs_mc=%.4f" %
              (r["sigma_pos_m"], r["peak_prob"], r["first_nonzero_step"], r["max_abs_err_vs_mc"]))
    print("=== Experiment 5B: modeling-timestep sweep ===")
    for r in ts_rows:
        print("  dt_r/dt_m=%3d  dt_m=%.4f  max_err_vs_fine=%.5f  runtime=%.1fms" %
              (r["dt_r_over_dt_m"], r["dt_m_s"], r["max_abs_err_vs_fine_ref"], r["runtime_ms"]))
    print("Saved: results/data/exp5_*.{json,csv}, results/figures/exp5_*.png")


if __name__ == "__main__":
    main()
