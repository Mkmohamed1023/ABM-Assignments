# ================================================================
# abm_ch1_experiments.py — Procedural Experiment Harness
# Mirrors CLI options and outputs of the class-based version, but calls functions.
# ================================================================

import argparse                        # for CLI argument parsing
import os                              # for making directories and paths
import numpy as np                     # RNG
import pandas as pd                    # saving tables
from abm_ch1_ants import (             # import procedural API from ants module
    init_world, step_world, to_tables, plot_paths, plot_pheromone
)

def make_cfg(H, W, N, alpha, lam, layout):
    """Return a config dict for the ant ABM run."""
    return {
        "H": H, "W": W, "N": N, "M": 2,          # grid and counts
        "T": 200,                                # number of ticks
        "ds": 1.0, "dt": 1.0,                    # step length and time step
        "sigma": 0.30,                           # heading jitter
        "kappa_trail": 0.70, "kappa_home": 1.20, # drift weights
        "alpha": alpha, "lam": lam,              # diffusion/evaporation
        "deposit": 0.80,                         # deposition amount
        "pick_radius": 2.0, "nest_radius": 2.0,  # interaction radii
        "food_per_source": 200,                  # initial food
        "env_stride": 5,                         # env sampling stride
        "layout": layout                         # layout option
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ant Foraging ABM — Procedural harness")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducibility")
    parser.add_argument("--N", type=int, default=50, help="Number of ants")
    parser.add_argument("--diffusion", type=float, default=0.15, help="Discrete diffusion factor alpha")
    parser.add_argument("--evaporation", type=float, default=0.02, help="Evaporation rate lambda")
    parser.add_argument("--map_size", type=int, default=100, help="World size (H=W=map_size)")
    parser.add_argument("--layout", type=str, default="corners", choices=["corners", "line", "random"],
                        help="Food source layout strategy")

    args = parser.parse_args()                      # parse CLI args
    rng = np.random.default_rng(args.seed)          # instantiate RNG
    H = W = int(args.map_size)                      # square grid
    cfg = make_cfg(H, W, int(args.N), float(args.diffusion), float(args.evaporation), args.layout)  # build config

    os.makedirs("data", exist_ok=True)              # ensure data dir exists
    os.makedirs("figures", exist_ok=True)           # ensure figures dir exists

    state = init_world(cfg, rng)                    # initialize world state
    for t in range(cfg["T"]):                       # loop over ticks
        step_world(state, t)                        # advance one tick

    agents, events, env, glob = to_tables(state)    # collect tables
    agents.to_csv("data/agents.csv", index=False)   # save agents
    events.to_csv("data/events.csv", index=False)   # save events
    env.to_csv("data/env.csv", index=False)         # save env
    glob.to_csv("data/global.csv", index=False)     # save global

    plot_paths(state, "figures/ant_paths.png")      # save paths figure
    plot_pheromone(state, "figures/pheromone.png")  # save pheromone figure

    print("Run complete.")
    print("Config:", cfg)
    print("Saved tables: data/agents.csv, data/events.csv, data/env.csv, data/global.csv")
    print("Saved figures: figures/ant_paths.png, figures/pheromone.png")
