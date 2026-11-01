# Perception-based Quantitative Runtime Verification

A Python implementation of

> R. Brown, L. V. Nguyen, W. Xiang, M. Wolf, H.-D. Tran.
> Perception-based Quantitative Runtime Verification for Learning-enabled
> Cyber-Physical Systems. ICCPS '25.

It implements the paper's ProbStar reachability pipeline for a driving scenario:
turn perception/sensing/actuation uncertainty into a probabilistic initial
condition, propagate a probabilistic reachable set with the vehicle dynamics,
intersect it with a collision region, and compute the collision probability over
a future horizon. The results are checked against an independent Monte-Carlo
simulation.

CS637 course project. This repo is the codebase only (no slides or video).

---

## 1. What's implemented, and how close it is to the paper

| Paper component | Status | Notes |
|---|---|---|
| ProbStar set, affine map (Prop. 2), half-space intersection (Prop. 1), probability (Def. 2) | Reimplemented | Checked against the official StarV `probstar.py`. See §4. |
| Truncated multivariate-normal probability integral | Replacement | StarV uses `scipy.stats.mvn.mvnun`, removed in SciPy ≥1.13. We use `multivariate_normal.cdf` (low-dim) + a Genz separation-of-variables MC estimator with variable reordering and antithetic variates (general). Not Botev tilting — see §4. |
| Vehicle dynamics: kinematic bicycle, φ-augmented linear model (Eq. 1–4), discretization | Implemented | 4D state `[x, y, φx, φy]`, block-diagonal multi-vehicle matrix, Euler `A_d=I+A_c·dt` (plus exact `expm` for testing). |
| Uncertainty propagation (2nd-order Taylor; Fig. 4 steering/speed from two poses) | Implemented | Paper's `E[f]`, `Var[f]` formulas; checked against Monte Carlo. |
| Collision geometry: Minkowski-difference half-space (Sec. 4.2, Alg. 1 lines 26–32) | Implemented | Oriented-rectangle Minkowski sum → half-space on the stacked state. |
| Quantitative verification loop (Algorithm 1, `dt_m`/`dt_r` substeps) | Implemented | Per-step and cumulative collision probability. |
| Monte-Carlo comparison (Sec. 5.4, Table 4) | Implemented | Independent sampler; reproduces Trial 1 (50%) and Trial 2 (2.28%). |
| Pose-detection DNN (Fig. 2–3, YOLO + MC-Dropout) | Simulated stand-in, not reproduced | We don't have the trained networks, dataset, or F1Tenth hardware. `perception.py` is a clearly-labelled simulated pose detector that emits a Gaussian pose at the paper's error scales. It does not reproduce the network. |
| F1Tenth hardware experiments, VICON ground truth, real driving logs | Not reproduced | No hardware or data. We reproduce the two hand-computable analytic trials and add controlled simulation experiments. |

Nothing here claims the simulated perception reproduces the paper's real
perception pipeline, and no numbers are made up — every value in `results/` comes
from running the code.

---

## 2. Install

Needs Python ≥ 3.9. From the project root:

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
pip install -e .                                        # optional: import probstar_rv anywhere
```

Dependencies are just the standard scientific stack: `numpy`, `scipy`,
`matplotlib`, `PyYAML` (plus `pytest` for tests). No `gurobipy`, no StarV, no
`onnx`/`torch` — see §4 for why.

---

## 3. Running it

Run everything from the project root. The experiment scripts add the repo root to
`sys.path` via `experiments/common.py`, so they work even without installing the
package.

### Tests

```bash
python -m pytest                 # 46 tests, ~15 s
```

### All experiments

```bash
python experiments/run_all.py            # full run
python experiments/run_all.py --quick    # smaller Monte-Carlo counts, faster
```

### One experiment at a time

```bash
python experiments/exp1_probstar_vs_montecarlo.py   # Table 4: ProbStar vs MC
python experiments/exp2_rear_end_timeline.py        # Figure 6: collision timeline
python experiments/exp3_timing.py                   # Table 3 shape: timing vs k_max
python experiments/exp4_control_actions.py          # Figure 7: braking/steering
python experiments/exp5_sensitivity.py              # uncertainty / horizon / dt_m sweeps
python experiments/exp6_end_to_end_pipeline.py      # full perception -> verification chain
```

`exp1` takes `--mc-samples N` to change the Monte-Carlo sample count.

### Outputs

* `results/data/*.json`, `results/data/*.csv` — numbers.
* `results/figures/*.png` — plots.

Each experiment also prints a short summary table to the console.

---

## 4. StarV vs. this implementation

The brief says to prefer the official ProbStar/StarV toolkit when it's workable.
We looked at it (`https://github.com/V2A2/StarV`) and it doesn't run reliably in a
current environment, for three reasons:

1. Not on PyPI, and it pins `numpy<=1.26.4`, `scipy<1.13.1`, `pycddlib<=2.1.8` —
   conflicts with current versions (this repo was developed on NumPy 2.5 /
   SciPy 1.18).
2. Its probability routine calls `scipy.stats.mvn.mvnun`, removed in SciPy ≥1.13,
   so even the reference probability code won't run on a current SciPy.
3. Its LP backend defaults to `gurobipy`, a commercial solver.

So this is a self-contained reimplementation. What matches StarV and what differs
(also in `probstar_rv/probstar.py`'s docstring):

* Same set representation and operations: the augmented basis matrix
  `V = [c | v_1 … v_m]`, predicate `(C, d)`, Gaussian `(μ, Σ)`, box bounds
  `(l, u)`; `affine_map`, `intersect_halfspace`, and the box bound-tightening
  (`updatePredicateRanges`) follow StarV's logic.
* `estimate_probability` follows StarV's `estimateProbability`: the stack-the-box
  → half-space-normal-integral transform and the SVD path for a singular
  `C Σ Cᵀ`. Only the final integral call is swapped for the modern truncated-MVN
  evaluator. That evaluator (`truncated_mvn.genz_sov`) is Genz's
  separation-of-variables Monte-Carlo estimator with Gibbs variable reordering
  and antithetic variates. It is not Botev minimax exponential tilting (an
  earlier name, `genz_botev`, overstated this and is kept only as an alias). SOV
  with reordering matches brute-force MC to 3–4 decimals at the dimensions here
  (4–8), which the tests check; it is not claimed to be identical to StarV's
  solver.
* LP solver: StarV uses Gurobi for exact predicate ranges; our pipeline doesn't
  need an LP (it uses the same analytic bound propagation StarV ships), and the
  optional range helper uses SciPy's HiGHS `linprog`.

The reimplementation is checked against brute-force Monte Carlo across dimensions
and probability magnitudes (see `tests/` and §6).

---

## 5. How it works

State. Each vehicle uses the kinematic bicycle model, with the heading replaced
by `φx = cos(θ+β)`, `φy = sin(θ+β)` (paper Eq. 2), giving the 4D linear state
`s = [x, y, φx, φy]`. This avoids the poor `[x, y, θ]` linearization the authors
report. The continuous model `ṡ = A_c s` (Eq. 3) is Euler-discretized
`A_d = I + A_c·dt`; vehicles are stacked block-diagonally (Eq. 4).

Initial condition (Sec. 3.3). `x, y` means and variances come from the
(simulated) pose detector; `φx, φy` come from second-order Taylor propagation of
`(θ, β)` through `cos`/`sin`. For obstacles, steering `ζ_ω` and speed `V_ω` are
recovered from two poses with the Figure-4 geometry (law of cosines → radius of
curvature → steering; arc length/Δt → speed), and their uncertainty is Taylor-
propagated. The joint initial state `X₀ ~ N(μ, Σ)` becomes a standardized
ProbStar (±δσ box, δ=6, Sec. 4.1).

Reachability (Algorithm 1). For each step `dt_r`, the ProbStar is affine-mapped
by the plant matrix `dt_r/dt_m` times (modeling substeps at the smaller `dt_m`,
which keeps discretization error low — Sec. 4.1). The reachable set is intersected
with the collision half-space (Minkowski difference of the two boxes, lifted to
the stacked state, Sec. 4.2), and the intersection probability (Def. 2) is the
per-step collision probability. The collision predicate can be recomputed each
step from the current reachable-set headings (`verify(..., recompute_collision=
True)`), matching Algorithm 1's `collisionPredicate(Θ_k)`: when a vehicle turns,
its box rotates, so the Minkowski polygon is rebuilt from the step-`k` heading.
For straight-line motion the heading never changes and the fixed half-space is
already exact (and faster), so that's the default; the steering experiment
(`exp4`) turns recomputation on. The Monte-Carlo validator has the same option so
the two stay comparable.

Per-step vs. cumulative risk. We report the per-step `p_k` and also the
probability of at least one collision over the horizon. The per-step events share
one Gaussian initial condition, so they're strongly positively correlated; we
report `max_k p_k` as a valid lower bound on the union and `1-∏(1-p_k)` only as an
independence upper reference, and get the true union from Monte Carlo (which
handles the temporal dependence exactly). More detail in
`probstar_rv/verification.py`.

Extensions beyond the paper:
* End-to-end perception pipeline (`exp6`): runs the whole chain — simulated pose
  detection → Figure-4 motion estimation → Taylor propagation → ProbStar →
  verification → MC check — instead of injecting the initial Gaussian directly.
* Verification-guided collision avoidance (`exp4`): the verifier as a decision
  tool. Braking is a real time-varying-plant constant-deceleration profile (the
  ego's speed changes over the horizon), and steering uses the heading-aware
  collision predicate; each action is scored by its predicted curve (cf. Fig. 7).
* Heading-aware collision predicate (`verify(recompute_collision=True)`): rebuilds
  the Minkowski polygon from the step-`k` heading when vehicles turn, matching
  Algorithm 1's `collisionPredicate(Θ_k)`.
* Rare-event cost analysis (`exp3`): the ProbStar-vs-MC crossover, showing where
  the paper's efficiency advantage comes from.
* Sensitivity studies (`exp5`): sweeps over perception uncertainty, modeling
  timestep `dt_m` (error drops as `dt_m` drops), and reachability step size.
* Full-trajectory ProbStar-vs-MC agreement (`exp2`): the whole rise-and-fall
  collision-probability curve, not just the single reported step.
* Correlated-Σ initial conditions: `ProbStar.from_gaussian` uses a full Cholesky
  factor, generalizing StarV's diagonal `diag(σ)` standardization.

---

## 6. What was run, and the results

Everything below comes from running this repo (`results/`) on a laptop CPU.
Timings are machine-dependent; the relative comparison is the point.

### Tests — `python -m pytest`

46 passed. Coverage: truncated-MVN vs analytic/MC across 1D–8D; ProbStar
affine-map covariance and single/stacked/diagonal half-space probabilities vs MC;
dynamics vs nonlinear ground truth and φ-norm growth; the braking speed profile
and time-varying-plant matrices; Taylor propagation vs MC; Figure-4 motion
recovery; collision half-space vs analytic rectangle overlap; heading-aware and
time-varying-braking verification vs MC; the full perception→verification pipeline
end to end; and Trials 1 & 2.

### Experiment 1 — ProbStar vs Monte Carlo (reproduces Table 4)

Monte Carlo uses 1000 trials, matching the paper's Table 4 ("Simulation (1000
Trials)"):

| scenario | true | ProbStar | Monte Carlo (1000) | ProbStar err | MC err |
|---|---|---|---|---|---|
| Trial 1 (rear-end) | 50.00% | 49.99% | 50.8% | 0.008 pp | 0.8 pp |
| Trial 2 (side)     |  2.28% |  2.27% |  2.0% | 0.009 pp | 0.28 pp |

ProbStar is within ~0.01 pp of the hand-computed true probabilities. The
1000-trial Monte Carlo is noisier (0.3–0.8 pp off), which is exactly the point of
Table 4: ProbStar is more accurate than a 1000-trial simulation (the paper's own
1000-trial MC gives 47.8% and 2.10%). MC values vary run to run with the seed.

On timing: the experiment compares total runtime to total runtime (both methods
produce the whole per-step curve). At the moderate probabilities and small sample
counts here, this correctness-focused ProbStar implementation is slower than
vectorized Monte Carlo (each ProbStar step runs a ~20k-sample Genz integral over
the constrained set; a 1000-sample vectorized MC run is ~1 ms). We don't claim a
blanket speedup. The paper's large speedups come from the
rare-event regime: ProbStar's cost doesn't depend on the probability magnitude,
while Monte Carlo needs `N ≈ z²(1−p)/(p·relerr²)` samples to hold a target
relative error, which blows up as `p → 0`. Experiment 3 measures this crossover:
to estimate `p = 10⁻⁶` to 10% relative error, MC needs ~3.8×10⁸ samples (~46 s)
versus ProbStar's fixed ~0.05 s (about 1000×), while at `p = 0.1` MC is faster.
That rare-collision regime is where the paper reports hundreds of seconds for MC
versus milliseconds for ProbStar.

### Experiment 2 — rear-end timeline (Figure 6)

ProbStar tracks the Monte-Carlo per-step curve to within max |error| ≈ 0.0013
across the horizon, with the collision probability rising to 100% as the ego
reaches the obstacle.

### Experiment 3 — timing vs horizon (Table 3 shape) + rare-event crossover

Reachability+probability runtime grows linearly with `k_max` (roughly constant
per-step cost), matching Table 3, where the per-step probstar probability is the
dominant repeated cost. (The perception-DNN timings in Table 3 are not reproduced
— we don't run the real networks.)

A second analysis (`exp3_rare_event_cost.*`) measures the ProbStar-vs-MC cost
crossover: for a 10% target relative error, MC's required sample count (hence
time) grows as `1/p`, so ProbStar — whose cost is independent of `p` — wins once
events are rare (p ≲ 10⁻³ here) and loses for common events. That's why the
paper's efficiency advantage shows up in its rare-collision setting but not at the
moderate probabilities in exp1.

### Experiment 4 — verification-guided avoidance (Figure 7)

The verifier scores actions by their predicted collision-probability curve.
Dynamic braking is a constant-deceleration profile (the ego's speed changes over
the horizon via a time-varying plant matrix, `braking_substep_matrices` +
`verify(substep_matrices=…)`), not just a lower constant speed: as deceleration
increases from 0, the peak collision probability drops through a graded threshold
(≈1.00 → 0.96 → 0.33 → 0.12 for 0/2/3/4 m/s² here), and the time-varying path is
checked against a Monte-Carlo run using the same per-step matrices (max error <
0.001). Steering turns the ego, so the collision predicate is recomputed per step
from the current heading; larger angles push the collision probability to ~0. This
matches Figure 7's conclusions, with braking modeled as constant deceleration.

### Experiment 5 — sensitivity

* Uncertainty: higher perception σ gives earlier non-zero collision probability
  and a lower, smoother peak; ProbStar stays within <0.005 of MC throughout.
* Timestep: with a turning ego, discretization error falls monotonically from
  ~0.19 (one Euler step per `dt_r`) to ~0.009 (`dt_m=dt_r/25`); at the paper's
  `dt_m=dt_r/10` it's already small (~0.02), consistent with Sec. 4.1.

### Experiment 6 — full perception-to-verification pipeline

Runs the whole chain end to end, which exp1–exp5 don't: simulated pose detection →
Figure-4 two-pose motion estimation (recovers the obstacle's speed and steering:
e.g. V 0.55 vs 0.60 true, ζ 0.06 vs 0.12 true under pose noise) → Taylor
uncertainty propagation → standardized ProbStar initial set → Algorithm-1
verification → Monte-Carlo check (agreement to ~1e-3). It also reports how the
end-to-end collision probability varies across 10 independent perception-noise
realizations, exercising `perception.py` in composition rather than on its own.

---

## 7. Known limitations and approximations

* Perception is simulated, not learned. The YOLO + MC-Dropout pose network
  (Fig. 2–3) is replaced by a Gaussian-noise pose simulator at the paper's error
  scales. The downstream math is the real thing; the front-end is not the paper's
  network. No real images, dataset, or F1Tenth hardware were used.
* No hardware/VICON results. We reproduce the two hand-computable analytic trials
  (Table 4) and controlled simulations, not the physical F1Tenth runs (Tables
  1–2, Figs. 5–6 real data).
* Fixed plant matrix over a horizon. Like the paper, `V, ζ, β` (hence `A`) are
  fixed per reachability window from the current estimate. Over long, hard-turning
  horizons the heading `θ` drifts (position error is still ~1 cm over 1 s at
  `dt_m=0.025`, in line with the paper's ~2 cm/2.5 s). The method is meant for
  short horizons re-estimated each step.
* The Euler φ-model doesn't preserve `‖φ‖` exactly. It drifts slowly (<2% over 1 s
  at `dt_m=0.025`); the paper notes a small `dt_m` keeps this in check.
* The truncated-MVN estimator is Monte-Carlo in high dimension. At the dimensions
  here (4–8) it matches analytic/MC to 3–4 decimals; much higher dimension would
  need more samples.
* The cumulative union probability is reported as a lower bound (`max_k p_k`) and
  an independence reference; the exact value comes from Monte Carlo. A fully
  analytic union over correlated steps is left for later.

---

## 8. Project layout

```
probstar_rv/
  config.yaml                 # all tunable parameters
  requirements.txt
  pyproject.toml
  README.md
  probstar_rv/                # the package
    __init__.py
    config.py                 # config loader (+ built-in defaults)
    truncated_mvn.py          # truncated multivariate-normal probability
    probstar.py               # ProbStar set (StarV reimplementation)
    dynamics.py               # kinematic bicycle / phi-augmented model + braking schedule
    perception.py             # simulated pose detector + Taylor uncertainty prop.
    collision.py              # Minkowski-difference collision half-space
    verification.py           # Algorithm 1 verifier (heading-aware + time-varying plant)
    monte_carlo.py            # independent MC validator (matching options)
    scenarios.py              # Figure-8 trials + rear-end builders
  tests/                      # pytest suite (46 tests)
  experiments/                # exp1..exp6 + run_all + common helpers
  results/                    # generated data/ and figures/ (committed samples)
```
