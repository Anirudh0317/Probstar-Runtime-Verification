"""Run every experiment end-to-end, in order.

Usage:
    python experiments/run_all.py [--quick]

--quick reduces Monte-Carlo sample counts for a faster smoke run.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import common  # noqa: F401  (sets up sys.path)


EXPERIMENTS = [
    ("exp1_probstar_vs_montecarlo", "ProbStar vs Monte Carlo (Table 4)"),
    ("exp2_rear_end_timeline", "Rear-end collision timeline (Figure 6)"),
    ("exp3_timing", "Timing vs horizon (Table 3 shape)"),
    ("exp4_control_actions", "Verification-guided collision avoidance (Figure 7)"),
    ("exp5_sensitivity", "Uncertainty / horizon / timestep sensitivity"),
    ("exp6_end_to_end_pipeline", "Full perception-to-verification pipeline"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller MC sample counts for a fast run")
    args = ap.parse_args()

    # In quick mode, monkeypatch exp1's default MC samples via argv.
    t_start = time.perf_counter()
    for mod_name, title in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"RUNNING {mod_name}: {title}")
        print("=" * 70)
        mod = importlib.import_module(mod_name)
        old_argv = sys.argv[:]
        if args.quick and mod_name == "exp1_probstar_vs_montecarlo":
            sys.argv = [mod_name, "--mc-samples", "50000"]
        else:
            sys.argv = [mod_name]
        try:
            mod.main()
        finally:
            sys.argv = old_argv

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE in {time.perf_counter() - t_start:.1f}s")
    print("Outputs in results/data/ and results/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()
