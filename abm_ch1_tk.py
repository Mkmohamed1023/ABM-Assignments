# ================================================================
# abm_ch1_tk_fixed.py — Tkinter GUI (scrollable controls + live updates)
# Procedural (NO classes). Uses Tk .after(...) + forced canvas.draw()
# Resizable panes and a scrollable control panel.
# ================================================================

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math

import matplotlib
matplotlib.use("TkAgg")  # select Tk backend before importing canvas pieces
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------
# Core ABM helpers
# -------------------------
def reflect_bounds(x, y, W, H):
    if x < 0: x = -x
    if x >= W: x = 2 * W - x - 1e-9
    if y < 0: y = -y
    if y >= H: y = 2 * H - y - 1e-9
    return x, y

def sample_gradient(P, xs, ys, W, H):
    grad_y, grad_x = np.gradient(P)
    ix = np.clip(xs.astype(int), 0, W-1)
    iy = np.clip(ys.astype(int), 0, H-1)
    return grad_x[iy, ix], grad_y[iy, ix]

def update_pheromone(P, alpha, lam, source):
    north = np.roll(P,  1, axis=0)
    south = np.roll(P, -1, axis=0)
    west  = np.roll(P,  1, axis=1)
    east  = np.roll(P, -1, axis=1)
    lap5 = (-4.0 * P) + north + south + east + west
    return (1.0 - lam) * (P + alpha * lap5) + source

def distance(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by))

def nearest_food(x, y, food_sources):
    d = np.sqrt((food_sources[:,0]-x)**2 + (food_sources[:,1]-y)**2)
    j = int(np.argmin(d))
    return j, float(d[j])

# -------------------------
# World init/step
# -------------------------
def init_world(cfg, rng):
    state = {"cfg": dict(cfg), "rng": rng}
    state["H"] = int(cfg["W"]); state["W"] = int(cfg["W"])
    state["N"] = int(cfg["N"]); state["M"] = int(cfg["M"])
    state["ds"] = float(cfg["ds"]); state["dt"] = 1.0
    state["sigma"] = float(cfg["sigma"])
    state["kappa_trail"] = float(cfg["kappa_trail"])
    state["kappa_home"]  = float(cfg["kappa_home"])
    state["alpha"] = float(cfg["alpha"]); state["lam"] = float(cfg["lam"])
    state["deposit"] = float(cfg["deposit"])
    state["pick_radius"] = float(cfg["pick_radius"])
    state["nest_radius"] = float(cfg["nest_radius"])
    state["food_per_source"] = int(cfg["food_per_source"])
    state["env_stride"] = int(cfg["env_stride"])
    state["layout"] = cfg["layout"]

    state["P"] = np.zeros((state["H"], state["W"]), dtype=np.float32)
    state["nest"] = np.array([state["W"]/2.0, state["H"]/2.0], dtype=np.float32)

    if state["layout"] == "corners":
        base = np.array([[int(0.15*state["W"]), int(0.15*state["H"])],
                         [int(0.85*state["W"]), int(0.85*state["H"])]], dtype=np.float32)
    elif state["layout"] == "line":
        base = np.array([[int(0.2*state["W"]), int(0.2*state["H"])],
                         [int(0.4*state["W"]), int(0.4*state["H"])]], dtype=np.float32)
    else:
        base = rng.uniform(low=0, high=[state["W"], state["H"]], size=(min(2,state["M"]),2)).astype(np.float32)
    if state["M"] <= len(base):
        state["food_sources"] = base[:state["M"]]
    else:
        extra = rng.uniform(low=0, high=[state["W"], state["H"]], size=(state["M"]-len(base),2))
        state["food_sources"] = np.vstack([base, extra]).astype(np.float32)

    state["food_remaining"] = np.full(state["M"], state["food_per_source"], dtype=np.int64)
    state["x"] = np.full(state["N"], state["nest"][0], dtype=np.float32)
    state["y"] = np.full(state["N"], state["nest"][1], dtype=np.float32)
    state["theta"] = rng.uniform(0.0, 2*np.pi, size=state["N"]).astype(np.float32)
    state["carry"] = np.zeros(state["N"], dtype=bool)
    state["prev_in_nest"] = np.ones(state["N"], dtype=bool)

    state["path_x"] = []; state["path_y"] = []; state["records"] = []
    state["first_discovery_time"] = None; state["t"] = 0
    return state

def step_world(state):
    t = state["t"]
    source = np.zeros_like(state["P"], dtype=np.float32)
    gx, gy = sample_gradient(state["P"], state["x"], state["y"], state["W"], state["H"])
    tick_events = []

    x_prev = state["x"].copy(); y_prev = state["y"].copy()

    for i in range(state["N"]):
        jitter = state["rng"].normal(0.0, state["sigma"])
        if not state["carry"][i]:
            grad_dir = math.atan2(gy[i], gx[i])
            state["theta"][i] = state["theta"][i] + jitter + state["kappa_trail"]*(grad_dir - state["theta"][i])
        else:
            dxn = state["nest"][0] - state["x"][i]; dyn = state["nest"][1] - state["y"][i]
            nest_dir = math.atan2(dyn, dxn)
            state["theta"][i] = state["theta"][i] + jitter + state["kappa_home"]*(nest_dir - state["theta"][i])

        state["x"][i] = state["x"][i] + state["ds"]*math.cos(state["theta"][i])
        state["y"][i] = state["y"][i] + state["ds"]*math.sin(state["theta"][i])
        state["x"][i], state["y"][i] = reflect_bounds(state["x"][i], state["y"][i], state["W"], state["H"])

        in_nest = distance(state["x"][i], state["y"][i], state["nest"][0], state["nest"][1]) <= state["nest_radius"]
        if in_nest and not state["prev_in_nest"][i]:
            tick_events.append({"t": t, "agent_id": i, "event": "enter_nest"})
        if (not in_nest) and state["prev_in_nest"][i]:
            tick_events.append({"t": t, "agent_id": i, "event": "exit_nest"})
        state["prev_in_nest"][i] = in_nest

        if not state["carry"][i]:
            j, dj = nearest_food(state["x"][i], state["y"][i], state["food_sources"])
            if (dj <= state["pick_radius"]) and (state["food_remaining"][j] > 0):
                state["carry"][i] = True
                state["food_remaining"][j] -= 1
                if state["first_discovery_time"] is None: state["first_discovery_time"] = t
                tick_events.append({"t": t, "agent_id": i, "event": "pickup", "food_source": int(j)})
        else:
            if in_nest:
                state["carry"][i] = False
                tick_events.append({"t": t, "agent_id": i, "event": "drop", "food_source": None})
            else:
                ix = int(np.clip(state["x"][i], 0, state["W"]-1))
                iy = int(np.clip(state["y"][i], 0, state["H"]-1))
                source[iy, ix] += state["deposit"]

    state["P"] = update_pheromone(state["P"], state["alpha"], state["lam"], source)

    vx = (state["x"] - x_prev)/state["dt"]; vy = (state["y"] - y_prev)/state["dt"]
    ix = np.clip(state["x"].astype(int), 0, state["W"]-1)
    iy = np.clip(state["y"].astype(int), 0, state["H"]-1)
    sensed = state["P"][iy, ix]

    agents_df = pd.DataFrame({
        "t": t, "agent_id": np.arange(state["N"]),
        "x": state["x"], "y": state["y"], "vx": vx, "vy": vy,
        "heading": state["theta"], "carrying_food": state["carry"],
        "sensed_pheromone": sensed, "step_len": state["ds"]
    })
    events_df = pd.DataFrame(tick_events) if tick_events else pd.DataFrame(
        columns=["t","agent_id","event","food_source"]
    )
    env_rows = [{"t": t, "x": xx, "y": yy, "pheromone": float(state["P"][yy, xx])}
                for yy in range(0, state["H"], state["env_stride"])
                for xx in range(0, state["W"], state["env_stride"])]
    env_df = pd.DataFrame(env_rows)

    P_thr = (state["P"] > 0.10)
    trail_len = int(P_thr.sum())
    if trail_len > 0:
        deg = (np.roll(P_thr,1,0).astype(int) + np.roll(P_thr,-1,0).astype(int) +
               np.roll(P_thr,1,1).astype(int) + np.roll(P_thr,-1,1).astype(int))
        mean_deg = float(deg[P_thr].mean())
    else:
        mean_deg = 0.0

    global_data = {
        "t": t,
        "food_remaining_total": int(state["food_remaining"].sum()),
        "trail_length": trail_len,
        "mean_trail_degree": mean_deg,
        "time_to_first_discovery": (None if state["first_discovery_time"] is None else int(state["first_discovery_time"]))
    }
    for j in range(state["M"]):
        global_data[f"food_{j}"] = int(state["food_remaining"][j])
    global_df = pd.DataFrame([global_data])

    state["records"].append({"agents": agents_df, "events": events_df, "env": env_df, "global": global_df})
    state["path_x"].append(state["x"].copy()); state["path_y"].append(state["y"].copy())
    state["t"] += 1

def to_tables(state):
    if not state["records"]:
        return pd.DataFrame(), pd.DataFrame(columns=["t","agent_id","event","food_source"]), pd.DataFrame(), pd.DataFrame()
    agents = pd.concat([r["agents"] for r in state["records"]], ignore_index=True)
    ev = [r["events"] for r in state["records"] if len(r["events"]) > 0]
    events = pd.concat(ev, ignore_index=True) if ev else pd.DataFrame(columns=["t","agent_id","event","food_source"])
    env = pd.concat([r["env"] for r in state["records"]], ignore_index=True)
    glob = pd.concat([r["global"] for r in state["records"]], ignore_index=True)
    return agents, events, env, glob

# -------------------------
# Tk GUI: window & layout
# -------------------------
root = tk.Tk()
root.title("Assignment 2 — Ant Foraging ABM (Tk GUI)")
root.geometry("1200x800")  # larger default window
root.minsize(900, 600)

# Use a PanedWindow so user can resize controls vs plots
panes = ttk.PanedWindow(root, orient="horizontal")
panes.pack(fill="both", expand=True)

# ---- Left: scrollable controls frame ----
controls_outer = ttk.Frame(panes)
panes.add(controls_outer, weight=1)

# Create a canvas + scrollbar to make controls scrollable
ctrl_canvas = tk.Canvas(controls_outer, highlightthickness=0)
ctrl_scroll = ttk.Scrollbar(controls_outer, orient="vertical", command=ctrl_canvas.yview)
controls_inner = ttk.Frame(ctrl_canvas)

controls_inner.bind(
    "<Configure>",
    lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
)
ctrl_canvas.create_window((0, 0), window=controls_inner, anchor="nw")
ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

ctrl_canvas.pack(side="left", fill="both", expand=True)
ctrl_scroll.pack(side="right", fill="y")

# ---- Right: plots frame (fills remaining space) ----
plots = ttk.Frame(panes)
panes.add(plots, weight=3)
plots.rowconfigure(0, weight=1)
plots.rowconfigure(1, weight=1)
plots.columnconfigure(0, weight=1)

# Status bar
status = ttk.Frame(root, padding=4)
status.pack(fill="x")
status_text = tk.StringVar(value="Ready.")
ttk.Label(status, textvariable=status_text, anchor="w").pack(fill="x")

# -------------------------
# Controls (with helpers)
# -------------------------
def labeled_scale(parent, text, frm, to, init, step=None, fmt="%.2f"):
    row = ttk.Frame(parent); row.pack(fill="x", pady=2)
    ttk.Label(row, text=text).pack(anchor="w")
    var = tk.DoubleVar(value=init)
    scale = ttk.Scale(row, from_=frm, to=to, variable=var, orient="horizontal")
    scale.pack(fill="x")
    box = ttk.Entry(row, width=8); box.insert(0, fmt % init); box.pack(anchor="e")
    def sync_from_scale(_=None):
        box.delete(0, tk.END); box.insert(0, fmt % var.get())
    def sync_from_box(_=None):
        try:
            val = float(box.get()); val = min(max(val, frm), to); var.set(val); sync_from_scale()
        except:
            pass
    scale.bind("<ButtonRelease-1>", sync_from_scale)
    box.bind("<Return>", sync_from_box)
    return var

ttk.Label(controls_inner, text="Parameters", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(6,6))

seed_var = tk.IntVar(value=1234)
row_seed = ttk.Frame(controls_inner); row_seed.pack(fill="x", pady=2)
ttk.Label(row_seed, text="Seed").pack(side="left")
seed_entry = ttk.Entry(row_seed, width=10)
seed_entry.insert(0, str(seed_var.get())); seed_entry.pack(side="left", padx=4)

W_var     = labeled_scale(controls_inner, "Map size W (H=W)", 50, 300, 120, fmt="%.0f")
N_var     = labeled_scale(controls_inner, "Number of ants N", 10, 500, 120, fmt="%.0f")
M_var     = labeled_scale(controls_inner, "Food sources M", 1, 8, 2, fmt="%.0f")

row_layout = ttk.Frame(controls_inner); row_layout.pack(fill="x", pady=(6,0))
ttk.Label(row_layout, text="Food layout").pack(anchor="w")
layout_var = tk.StringVar(value="corners")
ttk.Combobox(row_layout, textvariable=layout_var, values=["corners","line","random"], state="readonly").pack(fill="x")

ds_var     = labeled_scale(controls_inner, "Step length ds", 0.2, 3.0, 1.0)
sigma_var  = labeled_scale(controls_inner, "Heading jitter σ", 0.0, 1.5, 0.30)
ktrail_var = labeled_scale(controls_inner, "κ (trail following)", 0.0, 2.0, 0.70)
khome_var  = labeled_scale(controls_inner, "κ (homing to nest)", 0.0, 2.0, 1.20)

alpha_var  = labeled_scale(controls_inner, "α (diffusion)", 0.0, 0.5, 0.15)
lam_var    = labeled_scale(controls_inner, "λ (evaporation)", 0.0, 0.2, 0.02)
deposit_var= labeled_scale(controls_inner, "Deposit per tick", 0.0, 2.0, 0.80)

pick_var   = labeled_scale(controls_inner, "Pickup radius", 1.0, 6.0, 2.0)
nest_var   = labeled_scale(controls_inner, "Nest radius", 1.0, 8.0, 2.5)
food_src_var = labeled_scale(controls_inner, "Food per source", 20, 2000, 300, fmt="%.0f")
stride_var = labeled_scale(controls_inner, "Env stride (downsample)", 1, 12, 5, fmt="%.0f")

# runtime/animation controls
ttk.Separator(controls_inner).pack(fill="x", pady=8)
ttk.Label(controls_inner, text="Animation").pack(anchor="w")

batch_var = labeled_scale(controls_inner, "Steps per frame (speed)", 1, 50, 5, fmt="%.0f")
interval_var = labeled_scale(controls_inner, "Frame interval (ms)", 10, 200, 30, fmt="%.0f")

ttk.Label(controls_inner, text="Run K ticks").pack(anchor="w", pady=(6,0))
K_entry = ttk.Entry(controls_inner, width=10); K_entry.insert(0, "200"); K_entry.pack(anchor="w")

# Buttons
btns = ttk.Frame(controls_inner); btns.pack(fill="x", pady=8)
btn_reset = ttk.Button(btns, text="Reset", width=12)
btn_step  = ttk.Button(btns, text="Step 1", width=12)
btn_runK  = ttk.Button(btns, text="Run K", width=12)
btn_run   = ttk.Button(btns, text="Run ∞", width=12)
btn_stop  = ttk.Button(btns, text="Stop", width=12)

btn_reset.grid(row=0, column=0, padx=2, pady=2)
btn_step.grid (row=0, column=1, padx=2, pady=2)
btn_runK.grid (row=1, column=0, padx=2, pady=2)
btn_run.grid  (row=1, column=1, padx=2, pady=2)
btn_stop.grid (row=1, column=2, padx=2, pady=2)

ttk.Separator(controls_inner).pack(fill="x", pady=8)
btn_csv  = ttk.Button(controls_inner, text="Export CSVs")
btn_figs = ttk.Button(controls_inner, text="Export Figures")
btn_csv.pack(fill="x", pady=2); btn_figs.pack(fill="x", pady=2)

# -------------------------
# Plots (two figures)
# -------------------------
fig_paths = Figure(figsize=(5.5,5.0)); ax_paths = fig_paths.add_subplot(111)
canvas_paths = FigureCanvasTkAgg(fig_paths, master=plots)
canvas_paths.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

fig_P = Figure(figsize=(5.5,5.0)); ax_P = fig_P.add_subplot(111)
canvas_P = FigureCanvasTkAgg(fig_P, master=plots)
canvas_P.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

# -------------------------
# Build world from current UI
# -------------------------
def read_cfg():
    try:
        seed_val = int(seed_entry.get()); seed_var.set(seed_val)
    except:
        seed_val = seed_var.get()
    return {
        "W": float(W_var.get()), "N": float(N_var.get()), "M": float(M_var.get()),
        "ds": float(ds_var.get()), "sigma": float(sigma_var.get()),
        "kappa_trail": float(ktrail_var.get()), "kappa_home": float(khome_var.get()),
        "alpha": float(alpha_var.get()), "lam": float(lam_var.get()), "deposit": float(deposit_var.get()),
        "pick_radius": float(pick_var.get()), "nest_radius": float(nest_var.get()),
        "food_per_source": float(food_src_var.get()), "env_stride": float(stride_var.get()),
        "layout": layout_var.get(), "seed": int(seed_var.get())
    }

cfg0 = read_cfg()
STATE = init_world(cfg0, np.random.default_rng(cfg0["seed"]))

# -------------------------
# Drawing & status
# -------------------------
def redraw_plots():
    ax_paths.clear()
    if len(STATE["path_x"]) > 0:
        X = np.array(STATE["path_x"]); Y = np.array(STATE["path_y"])
        for i in range(STATE["N"]):
            ax_paths.plot(X[:, i], Y[:, i], linewidth=0.8)
    ax_paths.scatter([STATE["nest"][0]], [STATE["nest"][1]], s=40)
    ax_paths.scatter(STATE["food_sources"][:,0], STATE["food_sources"][:,1], s=40)
    ax_paths.set_title("Ant Paths"); ax_paths.set_xlabel("x"); ax_paths.set_ylabel("y")
    ax_paths.set_aspect("equal", adjustable="box"); fig_paths.tight_layout()
    canvas_paths.draw()  # FORCE redraw

    ax_P.clear()
    ax_P.imshow(STATE["P"], origin="lower")
    ax_P.set_title("Pheromone (current)"); ax_P.set_xlabel("x"); ax_P.set_ylabel("y")
    fig_P.tight_layout()
    canvas_P.draw()      # FORCE redraw

def update_status():
    t = STATE["t"]; food_total = int(STATE["food_remaining"].sum())
    trail_len = 0
    if STATE["records"]:
        trail_len = int(STATE["records"][-1]["global"].iloc[0]["trail_length"])
    fpt = "-" if STATE["first_discovery_time"] is None else int(STATE["first_discovery_time"])
    status_text.set(f"t={t} | food_remaining_total={food_total} | trail_length={trail_len} | first_discovery={fpt}")

# initial paint
redraw_plots(); update_status()

# -------------------------
# Run loops via after()
# -------------------------
RUNNING = False
RUN_K_LEFT = 0

def schedule_continuous():
    if not RUNNING: return
    steps = int(batch_var.get())
    for _ in range(max(1, steps)):
        step_world(STATE)
    redraw_plots(); update_status()
    root.after(int(max(1, interval_var.get())), schedule_continuous)

def schedule_k():
    global RUN_K_LEFT
    if RUN_K_LEFT <= 0: return
    steps = int(batch_var.get())
    # clamp steps so we don't overshoot
    to_do = min(steps, RUN_K_LEFT)
    for _ in range(to_do):
        step_world(STATE)
    RUN_K_LEFT -= to_do
    redraw_plots(); update_status()
    if RUN_K_LEFT > 0:
        root.after(int(max(1, interval_var.get())), schedule_k)
    else:
        status_text.set("Finished Run K.")

# -------------------------
# Button handlers
# -------------------------
def on_reset():
    global STATE, RUNNING, RUN_K_LEFT
    RUNNING = False; RUN_K_LEFT = 0
    cfg = read_cfg()
    STATE = init_world(cfg, np.random.default_rng(cfg["seed"]))
    redraw_plots(); update_status()
    status_text.set("World reset.")

def on_step():
    global RUNNING, RUN_K_LEFT
    RUNNING = False; RUN_K_LEFT = 0
    step_world(STATE)
    redraw_plots(); update_status()

def on_run():
    global RUNNING, RUN_K_LEFT
    RUN_K_LEFT = 0
    if RUNNING: return
    RUNNING = True
    schedule_continuous()
    status_text.set("Running continuously… (Stop to halt)")

def on_run_k():
    global RUNNING, RUN_K_LEFT
    RUNNING = False
    try:
        RUN_K_LEFT = max(1, int(K_entry.get()))
    except:
        RUN_K_LEFT = 100
    schedule_k()
    status_text.set(f"Running K ticks: {RUN_K_LEFT} remaining…")

def on_stop():
    global RUNNING, RUN_K_LEFT
    RUNNING = False; RUN_K_LEFT = 0
    status_text.set("Stopped.")

def on_export_csvs():
    if not STATE["records"]:
        messagebox.showinfo("Export CSVs", "No data yet. Run or Step first.")
        return
    agents, events, env, glob = to_tables(STATE)
    folder = filedialog.askdirectory(title="Choose folder to save CSVs")
    if not folder: return
    try:
        agents.to_csv(f"{folder}/agents.csv", index=False)
        events.to_csv(f"{folder}/events.csv", index=False)
        env.to_csv(f"{folder}/env.csv", index=False)
        glob.to_csv(f"{folder}/global.csv", index=False)
        messagebox.showinfo("Export CSVs", f"Saved to {folder}")
    except Exception as e:
        messagebox.showerror("Export CSVs", str(e))

def on_export_figs():
    folder = filedialog.askdirectory(title="Choose folder to save PNGs")
    if not folder: return
    try:
        fig_paths.savefig(f"{folder}/ant_paths.png", dpi=150, bbox_inches="tight")
        fig_P.savefig(f"{folder}/pheromone.png", dpi=150, bbox_inches="tight")
        messagebox.showinfo("Export Figures", f"Saved to {folder}")
    except Exception as e:
        messagebox.showerror("Export Figures", str(e))

# wire buttons
btn_reset.configure(command=on_reset)
btn_step.configure(command=on_step)
btn_run.configure(command=on_run)
btn_runK.configure(command=on_run_k)
btn_stop.configure(command=on_stop)
btn_csv.configure(command=on_export_csvs)
btn_figs.configure(command=on_export_figs)

# Mouse wheel scroll for controls
def _on_mousewheel(event):
    ctrl_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
ctrl_canvas.bind_all("<MouseWheel>", _on_mousewheel)       # Windows/Mac
ctrl_canvas.bind_all("<Button-4>", lambda e: ctrl_canvas.yview_scroll(-1, "units"))  # Linux up
ctrl_canvas.bind_all("<Button-5>", lambda e: ctrl_canvas.yview_scroll( 1, "units"))  # Linux down

root.mainloop()
