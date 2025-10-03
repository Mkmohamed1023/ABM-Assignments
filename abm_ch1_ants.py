# ================================================================
# abm_ch1_ants.py — Ant Foraging ABM (Assignment 2, Part I)
# Procedural version (NO classes), with detailed comments on each line.
# Variables follow ABN symbols: ds, dt, kappa, P, lam, alpha, etc.
# ================================================================

# ----- Imports -----
import math                              # math functions like cos, sin, atan2, hypot
import numpy as np                       # fast numerical arrays
import pandas as pd                      # tidy tables (agents, events, env, global)
import matplotlib.pyplot as plt          # plotting library (matplotlib as required)
from typing import Dict, Tuple           # type hints for clarity

# ----- Utility: reflecting boundary -----
def reflect_bounds(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    """Reflect (x,y) back inside the domain [0,W)×[0,H)."""
    if x < 0:                             # if x is less than 0
        x = -x                            # reflect across x=0
    if x >= W:                            # if x is at or beyond the right edge
        x = 2 * W - x - 1e-9              # reflect back inside with epsilon
    if y < 0:                             # if y is less than 0
        y = -y                            # reflect across y=0
    if y >= H:                            # if y is at or beyond the top edge
        y = 2 * H - y - 1e-9              # reflect back inside with epsilon
    return x, y                           # return corrected coordinates

# ----- Initialization: build a world state dict -----
def init_world(cfg: Dict, rng: np.random.Generator) -> Dict:
    """Create and return the entire simulation state as a dict."""
    state = {}                                        # create an empty dictionary to hold everything
    state["cfg"] = cfg                                # store the configuration dict for later access
    state["rng"] = rng                                # store the random number generator for reproducibility

    state["H"] = cfg["H"]                             # grid height
    state["W"] = cfg["W"]                             # grid width
    state["N"] = cfg["N"]                             # number of ants
    state["M"] = cfg["M"]                             # number of food sources
    state["ds"] = cfg["ds"]                           # step length per tick (Δs)
    state["dt"] = cfg["dt"]                           # discrete time step (Δt)
    state["sigma"] = cfg["sigma"]                     # heading jitter stdev σ (radians)
    state["kappa_trail"] = cfg["kappa_trail"]         # κ for trail following
    state["kappa_home"]  = cfg["kappa_home"]          # κ for homing to nest
    state["alpha"] = cfg["alpha"]                     # discrete diffusion factor for P
    state["lam"] = cfg["lam"]                         # evaporation rate λ
    state["deposit"] = cfg["deposit"]                 # pheromone deposited per carrier per tick
    state["pick_radius"] = cfg["pick_radius"]         # radius to pick up food
    state["nest_radius"] = cfg["nest_radius"]         # radius to consider "inside nest"
    state["food_per_source"] = cfg["food_per_source"] # units of food at each source initially
    state["env_stride"] = cfg.get("env_stride", 5)    # env downsampling stride
    state["layout"] = cfg.get("layout", "corners")    # placement strategy for food

    state["P"] = np.zeros((state["H"], state["W"]), dtype=np.float32)   # pheromone field initialized to zero
    state["nest"] = np.array([state["W"]/2.0, state["H"]/2.0], dtype=np.float32)  # nest at center

    if state["layout"] == "corners":                  # if using corner layout
        base = np.array([                             # define two anchor sources
            [int(0.15 * state["W"]), int(0.15 * state["H"])],
            [int(0.85 * state["W"]), int(0.85 * state["H"])]
        ], dtype=np.float32)
    elif state["layout"] == "line":                   # if using line layout
        base = np.array([                             # two points along a diagonal
            [int(0.2 * state["W"]), int(0.2 * state["H"])],
            [int(0.4 * state["W"]), int(0.4 * state["H"])]
        ], dtype=np.float32)
    else:                                             # otherwise random placement
        base = rng.uniform(low=0, high=[state["W"], state["H"]], size=(min(2, state["M"]), 2)).astype(np.float32)

    if state["M"] <= len(base):                       # if we need ≤ base sources
        state["food_sources"] = base[: state["M"]]    # take the first M
    else:                                             # if we need more than base
        extra = rng.uniform(low=0, high=[state["W"], state["H"]], size=(state["M"] - len(base), 2))  # random extras
        state["food_sources"] = np.vstack([base, extra]).astype(np.float32)  # stack to get M sources

    state["food_remaining"] = np.full(state["M"], state["food_per_source"], dtype=np.int64)  # counters for remaining food

    state["x"] = np.full(state["N"], state["nest"][0], dtype=np.float32)   # x positions initialized at nest x
    state["y"] = np.full(state["N"], state["nest"][1], dtype=np.float32)   # y positions initialized at nest y
    state["theta"] = rng.uniform(0.0, 2*np.pi, size=state["N"]).astype(np.float32)  # random headings θ
    state["carry"] = np.zeros(state["N"], dtype=bool)   # carrying flags start as False (not carrying)

    state["prev_in_nest"] = np.ones(state["N"], dtype=bool)  # since we start at nest, all ants are in_nest=True

    state["path_x"] = []                                  # list to store x positions each tick for plotting
    state["path_y"] = []                                  # list to store y positions each tick for plotting

    state["records"] = []                                 # list to store per-tick DataFrames
    state["first_discovery_time"] = None                  # None means no pickup has happened yet

    return state                                          # return the complete state dict

# ----- Gradient sampling from P -----
def sample_gradient(P: np.ndarray, xs: np.ndarray, ys: np.ndarray, W: int, H: int):
    """Return gx, gy sampled from ∇P at nearest cells to (xs, ys)."""
    grad_y, grad_x = np.gradient(P)                       # compute ∂P/∂y and ∂P/∂x on the grid
    ix = np.clip(xs.astype(int), 0, W - 1)                # convert x to indices and clip
    iy = np.clip(ys.astype(int), 0, H - 1)                # convert y to indices and clip
    gx = grad_x[iy, ix]                                   # gather dP/dx per agent
    gy = grad_y[iy, ix]                                   # gather dP/dy per agent
    return gx, gy                                         # return gradient components

# ----- Pheromone field update (discrete diffusion + evaporation + source) -----
def update_pheromone(P: np.ndarray, alpha: float, lam: float, source: np.ndarray) -> np.ndarray:
    """Apply five-point stencil diffusion and evaporation, then add source; return updated P."""
    north = np.roll(P,  1, axis=0)                        # neighbor up via roll
    south = np.roll(P, -1, axis=0)                        # neighbor down via roll
    west  = np.roll(P,  1, axis=1)                        # neighbor left via roll
    east  = np.roll(P, -1, axis=1)                        # neighbor right via roll
    lap5 = (-4.0 * P) + north + south + east + west       # compute -4P + N + S + E + W
    P_new = (1.0 - lam) * (P + alpha * lap5) + source     # multiply evaporation and add diffusion + source
    return P_new                                          # return updated field

# ----- Distance helper -----
def distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between (ax, ay) and (bx, by)."""
    return float(math.hypot(ax - bx, ay - by))            # hypot returns sqrt((dx)^2 + (dy)^2)

# ----- Nearest food index/distance -----
def nearest_food(x: float, y: float, food_sources: np.ndarray) -> Tuple[int, float]:
    """Compute index and distance to the nearest food source from (x, y)."""
    d = np.sqrt((food_sources[:, 0] - x) ** 2 + (food_sources[:, 1] - y) ** 2)  # distances to each source
    j = int(np.argmin(d))                                   # index of closest source
    return j, float(d[j])                                   # return (index, distance)

# ----- One simulation step -----
def step_world(state: Dict, t: int) -> None:
    """Advance the simulation by one tick and record data into state."""
    source = np.zeros_like(state["P"], dtype=np.float32)     # initialize deposition field for this tick
    gx, gy = sample_gradient(state["P"], state["x"], state["y"], state["W"], state["H"])  # sense ∇P at ant positions
    tick_events = []                                         # list to collect event rows

    x_prev = state["x"].copy()                               # copy previous x for velocity computation
    y_prev = state["y"].copy()                               # copy previous y for velocity computation

    for i in range(state["N"]):                              # loop over each ant i
        jitter = state["rng"].normal(0.0, state["sigma"])    # draw angular noise from Normal(0, σ)

        if not state["carry"][i]:                            # if ant is not carrying food
            grad_dir = math.atan2(gy[i], gx[i])              # compute direction of gradient via atan2(gy, gx)
            state["theta"][i] = state["theta"][i] + jitter + state["kappa_trail"] * (grad_dir - state["theta"][i])  # bias
        else:                                                # if ant is carrying food
            dxn = state["nest"][0] - state["x"][i]           # x-component pointing from ant to nest
            dyn = state["nest"][1] - state["y"][i]           # y-component pointing from ant to nest
            nest_dir = math.atan2(dyn, dxn)                  # direction to the nest
            state["theta"][i] = state["theta"][i] + jitter + state["kappa_home"] * (nest_dir - state["theta"][i])   # bias

        state["x"][i] = state["x"][i] + state["ds"] * math.cos(state["theta"][i])  # move x by step along heading
        state["y"][i] = state["y"][i] + state["ds"] * math.sin(state["theta"][i])  # move y by step along heading

        state["x"][i], state["y"][i] = reflect_bounds(state["x"][i], state["y"][i], state["W"], state["H"])  # reflect at walls

        in_nest = (distance(state["x"][i], state["y"][i], state["nest"][0], state["nest"][1]) <= state["nest_radius"])  # nest check

        if in_nest and not state["prev_in_nest"][i]:         # if ant just entered the nest
            tick_events.append({"t": t, "agent_id": i, "event": "enter_nest"})  # record enter event
        if (not in_nest) and state["prev_in_nest"][i]:       # if ant just exited the nest
            tick_events.append({"t": t, "agent_id": i, "event": "exit_nest"})   # record exit event
        state["prev_in_nest"][i] = in_nest                   # update previous nest flag

        if not state["carry"][i]:                            # if not carrying, check for pickup at nearest source
            j, dj = nearest_food(state["x"][i], state["y"][i], state["food_sources"])  # get nearest source and distance
            if (dj <= state["pick_radius"]) and (state["food_remaining"][j] > 0):      # within pickup radius and food left
                state["carry"][i] = True                     # start carrying food
                state["food_remaining"][j] -= 1              # decrement food at source j
                if state["first_discovery_time"] is None:    # if first pickup in the whole sim
                    state["first_discovery_time"] = t        # set first discovery time
                tick_events.append({"t": t, "agent_id": i, "event": "pickup", "food_source": int(j)})  # record pickup
        else:                                                # if carrying
            if in_nest:                                      # if inside nest now
                state["carry"][i] = False                    # drop food implicitly by clearing carrying flag
                tick_events.append({"t": t, "agent_id": i, "event": "drop", "food_source": None})  # record drop
            else:                                            # still traveling back
                ix = int(np.clip(state["x"][i], 0, state["W"] - 1))  # compute integer x index
                iy = int(np.clip(state["y"][i], 0, state["H"] - 1))  # compute integer y index
                source[iy, ix] += state["deposit"]           # deposit pheromone at current grid cell

    state["P"] = update_pheromone(state["P"], state["alpha"], state["lam"], source)  # update pheromone field P

    vx = (state["x"] - x_prev) / state["dt"]                 # compute velocity x component
    vy = (state["y"] - y_prev) / state["dt"]                 # compute velocity y component

    ix = np.clip(state["x"].astype(int), 0, state["W"] - 1)  # discretize x positions for sampling P
    iy = np.clip(state["y"].astype(int), 0, state["H"] - 1)  # discretize y positions for sampling P
    sensed = state["P"][iy, ix]                               # pheromone sensed at each ant location

    agents_df = pd.DataFrame({                                # build agents DataFrame for this tick
        "t": t, "agent_id": np.arange(state["N"]), "x": state["x"], "y": state["y"],
        "vx": vx, "vy": vy, "heading": state["theta"], "carrying_food": state["carry"],
        "sensed_pheromone": sensed, "step_len": state["ds"]
    })

    events_df = pd.DataFrame(tick_events) if len(tick_events) > 0 else pd.DataFrame(
        columns=["t", "agent_id", "event", "food_source"])    # build events DataFrame (or empty with columns)

    env_rows = []                                             # list to collect downsampled env rows
    for yy in range(0, state["H"], state["env_stride"]):      # iterate y in strides of env_stride
        for xx in range(0, state["W"], state["env_stride"]):  # iterate x in strides of env_stride
            env_rows.append({"t": t, "x": xx, "y": yy, "pheromone": float(state["P"][yy, xx])})  # append record
    env_df = pd.DataFrame(env_rows)                           # convert env rows to DataFrame

    P_thr = (state["P"] > 0.10)                               # threshold mask for trail cells
    trail_len = int(P_thr.sum())                              # count of trail cells
    if trail_len > 0:                                         # if any trail cells exist
        deg = (np.roll(P_thr, 1, 0).astype(int) +             # sum of neighbor trail flags (up)
               np.roll(P_thr, -1, 0).astype(int) +            # down
               np.roll(P_thr, 1, 1).astype(int) +             # left
               np.roll(P_thr, -1, 1).astype(int))             # right
        mean_deg = float(deg[P_thr].mean())                   # mean degree over trail cells
    else:                                                     # if no trail cells
        mean_deg = 0.0                                        # mean degree is zero

    global_data = {                                           # form global summary row
        "t": t,
        "food_remaining_total": int(state["food_remaining"].sum()),  # sum remaining food
        "trail_length": trail_len,                             # number of trail cells
        "mean_trail_degree": mean_deg,                        # average 4-neighbor degree over trails
        "time_to_first_discovery": (None if state["first_discovery_time"] is None else int(state["first_discovery_time"])) # FPT
    }
    for j in range(state["M"]):                                # for each food source
        global_data[f"food_{j}"] = int(state["food_remaining"][j])   # add per-source remaining
    global_df = pd.DataFrame([global_data])                    # wrap as one-row DataFrame

    state["records"].append({"agents": agents_df, "events": events_df, "env": env_df, "global": global_df})  # store records
    state["path_x"].append(state["x"].copy())                  # save current x for paths
    state["path_y"].append(state["y"].copy())                  # save current y for paths

# ----- Concatenate per-tick records into full tables -----
def to_tables(state: Dict):
    """Concatenate per-tick DataFrames into full tables (agents, events, env, global)."""
    agents = pd.concat([r["agents"] for r in state["records"]], ignore_index=True)  # concat agents over ticks
    ev = [r["events"] for r in state["records"] if len(r["events"]) > 0]            # filter empty events
    events = pd.concat(ev, ignore_index=True) if len(ev) > 0 else pd.DataFrame(
        columns=["t", "agent_id", "event", "food_source"])                           # concat or empty
    env = pd.concat([r["env"] for r in state["records"]], ignore_index=True)        # concat env
    glob = pd.concat([r["global"] for r in state["records"]], ignore_index=True)    # concat global
    return agents, events, env, glob                                                # return all tables

# ----- Plot paths and pheromone -----
def plot_paths(state: Dict, out_path: str):
    """Plot ant paths over time and save a PNG figure."""
    plt.figure(figsize=(6, 6))                                   # open figure
    X = np.array(state["path_x"]); Y = np.array(state["path_y"]) # convert paths to arrays shape (T, N)
    for i in range(state["N"]):                                   # iterate over ants
        plt.plot(X[:, i], Y[:, i], linewidth=0.8)                 # draw polyline for ant i
    plt.scatter([state["nest"][0]], [state["nest"][1]], s=40)     # mark the nest
    plt.scatter(state["food_sources"][:, 0], state["food_sources"][:, 1], s=40)  # mark food sources
    plt.title("Ant Paths"); plt.xlabel("x"); plt.ylabel("y")      # annotate axes
    plt.gca().set_aspect("equal", adjustable="box")               # square aspect
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()  # save and close

def plot_pheromone(state: Dict, out_path: str):
    """Plot final pheromone field as a heatmap and save a PNG figure."""
    plt.figure(figsize=(6, 6))                    # open figure
    plt.imshow(state["P"], origin="lower")        # draw P image (default colormap)
    plt.title("Pheromone (final)"); plt.xlabel("x"); plt.ylabel("y")  # annotate
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()  # save and close
