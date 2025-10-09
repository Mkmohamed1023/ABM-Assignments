# DATASET_CARD.md — Ant Foraging ABM Synthetic Runs (Assignment 2)

## Overview
Synthetic datasets produced by the ant-foraging agent-based model (ABM) for Assignment 2.
Intended for research/teaching on search, trail formation, and first‑passage time analysis.
Not field data.

## Maintainer
Course project team • Contact: <maintainer@example.edu>

## Source Code
- `abm_ch1_ants.py` — core ABM
- `abm_ch1_experiments.py` — CLI harness (runs, sweeps, output)
- `abm_ch1_tk.py` — optional GUI for exploration
- `requirements.txt` — pinned package versions

## Versioning & Reproducibility
- RNG seeded via `numpy.random.default_rng(seed)` per run.
- Save a manifest per run: seed, CLI command, config dict, package versions, git commit (if available),
  and SHA256 hashes of saved CSV/PNG artifacts.
- Recommended environment: Python 3.10+, packages pinned by `requirements.txt`.

## Files per Run (Artifacts)
- Tables (CSV): `agents.csv`, `events.csv`, `env.csv`, `global.csv`
- Figures (PNG): `ant_paths.png`, `pheromone.png`
- `manifest.json` (suggested): seed, command, config, package versions, file hashes

## Generation Process (Summary)
Agents deposit, diffuse, and evaporate pheromone on a grid (5‑point stencil).
Unloaded ants follow pheromone gradients (plus small jitter/deterministic rule depending on experiment);
loaded ants home to the nest. Food picked near sources is carried to the nest.
The harness iterates parameter sweeps and saves outputs for each seed and scenario.

## Parameters (Key)
- Population / geometry: ants `N`, sources `M`, grid size `H×W`, step `ds`, jitter `σ`
- Dynamics: diffusion `α`, evaporation `λ`, deposit rate, home/trail gains
- Interaction radii: pick radius, nest radius
- Layout: `corners` (clustered), `line` (ring), `random` (obstacles) — names follow the harness
- Seed: integer passed to RNG
- Horizon: ticks `T`
(See run config in `manifest.json` for exact values per run.)

## Splits
- By scenario: baseline, high diffusion, obstacles, ablations (heterogeneity, memory, both)
- By seed: at least 20 per cell

## Metrics Provided / Derivable
- Time to first discovery, empirical CDFs
- Trail stability (autocorrelation of top‑q pheromone edges)
- Path diversity (edge‑traversal entropy)
- Workload equity (Gini of per‑ant pickups)
- Surrogate fits (Part V), residual diagnostics, RMSE/MAE

## Known Gaps vs. Field Data
1. No explicit collisions/queuing (agents can overlap on a grid cell).
2. Simple pickup/handling; no fatigue or variable load.
3. Isotropic diffusion on 5‑point stencil; drift/weather simplified.

## Caveats (Synthetic vs. Real)
- Absolute rates depend on grid/time scaling and parameterization.
- Small‑scale behaviors (micro‑turning, pheromone chemistry) are stylized.
- Use trends and comparative results; avoid literal biological claims.

## Ethics & Privacy
- Fully synthetic; no personal data.
- Share code, configs, and manifests to enable verification and reuse.

## How to Reproduce
Example single run:
```bash
python abm_ch1_experiments.py --seed 17 --N 200 --diffusion 0.15   --evaporation 0.01 --map_size 100 --layout corners
```
Sweep skeleton:
```bash
for N in 50 200 800; do
  for A in 0.05 0.15 0.30; do
    for LAM in 0.00 0.01 0.03; do
      for L in corners line random; do
        for S in $(seq 1 20); do
          python abm_ch1_experiments.py --seed $S --N $N             --diffusion $A --evaporation $LAM --map_size 100 --layout $L
        done
      done
    endo
  done
done
```

## License
CC BY 4.0 for the dataset outputs, unless your course policy overrides.
Code uses the license specified in the repository.

## Citation
Please cite the project repository and this dataset card in reports.