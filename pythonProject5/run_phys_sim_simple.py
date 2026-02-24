#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_phys_sim_simple.py — Robust-final vs Deterministic dual-suite simulation
Γ in {0..6}, each Γ runs N times with global-gamma random delays.
Outputs per-variant stats (min/mean/std/max) for Cmax(ws) and Cmax(drop),
and 8 comparison bar charts (ASCII labels).

Dependencies expected in your project:
- map_generator.load_map_csv, create_map_from_components
- movement_manager.MovementManager
"""

import os
import ast
import math
import heapq
import random
import sys
import io
import contextlib
from itertools import product
import pandas as pd
import re
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from map_generator import load_map_csv, create_map_from_components
from movement_manager import MovementManager

# ===============================
# ======== CONFIG (EDIT) ========
# ===============================
PREFIX = "demo01"           # scenario/<PREFIX>/
PATH_MODE = "jump"          # "jump": Manhattan & jump to endpoints; "astar": use A* path
D_SETUP = 2.0               # setup time between consecutive WS jobs
SUPPRESS_LIB_LOGS = True    # mute loaders' prints

# Simulation design
GAMMA_SET = list(range(0, 7))  # Γ = 0..6
BATCH_RUNS = 10000              # runs per Γ (reduce for smoke test)
GLOBAL_GAMMA = True             # True: select γ segments across ALL AGVs; False: per-AGV ≤ γ
EXACT_GAMMA = True              # True: exactly γ segments; False: uniform in [0, γ]

SEGMENTS_ALLOWED = {"move1", "move2", "move3"}  # delayable segments
FACTOR_RANGE = (2.0, 2.0)       # multiplicative delay factor range (use None if additive)
ADD_RANGE = None                # additive seconds range (use None if multiplicative)

# Robust vs Deterministic scheduling policies
POLICY_ROBUST = "MILP_ORDER"    # robust: gate by WS order only (randomness affects ws_s)
POLICY_DET    = "MILP_ORDER"          # deterministic: gate by ws order + not earlier than p_opt

# File selection priority
USE_ROBUST_FINAL = True
ROBUST_FINAL_TAG = "robust_final"

# Deterministic fallback: use robust_it1_gamma0 triplet if no standalone deterministic triplet exists
USE_ROBUST_AS_DET_FALLBACK = True   # <=== 关键开关

# ===============================
# ======== UTILITIES ============
# ===============================
BIG_M = 10000.0
VIRTUAL_BASE = 3000

@contextlib.contextmanager
def _mute_print(active: bool):
    if not active:
        yield
        return
    old = sys.stdout
    try:
        sys.stdout = io.StringIO()
        yield
    finally:
        sys.stdout = old

def idx2rc(idx: int, W: int):
    r, c = divmod(int(idx) - 1, int(W))
    return r, c

def manhattan_idx(a: int, b: int, W: int) -> int:
    ra, ca = idx2rc(a, W); rb, cb = idx2rc(b, W)
    return abs(ra - rb) + abs(ca - cb)

# ---------- locate robust final / itN-max (same-source triplet) ----------
def resolve_robust_set(prefix: str, gamma: int = 0) -> dict:
    """
    Return robust triplet paths (same source):
      prefer robust_final triplet, else robust_itN max triplet.
    """
    exp_dir = "solution_exports"

    def pick_final(stem):
        cands = [
            os.path.join(exp_dir, f"{prefix}_{ROBUST_FINAL_TAG}_{stem}_gamma{gamma}.csv"),
            os.path.join(exp_dir, f"{prefix}_{ROBUST_FINAL_TAG}_{stem}.csv"),
            os.path.join(exp_dir, f"{prefix}_{ROBUST_FINAL_TAG}_{stem}_gamma0.csv"),
        ]
        for p in cands:
            if os.path.exists(p): return p
        return None

    # final
    if USE_ROBUST_FINAL:
        final = {s: pick_final(s) for s in ("wsPlan","taskSeq","taskEndShelf")}
        if all(final.values()):
            return final

    # itN max (complete set)
    pat = os.path.join(exp_dir, f"{prefix}_robust_it*_*.csv")
    paths = glob.glob(pat); it_set = set()
    for p in paths:
        m = re.search(rf"{re.escape(prefix)}_robust_it(\d+)_", os.path.basename(p))
        if m: it_set.add(int(m.group(1)))
    for N in sorted(it_set, reverse=True):
        hit, ok = {}, True
        for stem in ("wsPlan","taskSeq","taskEndShelf"):
            cands = [
                os.path.join(exp_dir, f"{prefix}_robust_it{N}_{stem}_gamma{gamma}.csv"),
                os.path.join(exp_dir, f"{prefix}_robust_it{N}_{stem}.csv"),
                os.path.join(exp_dir, f"{prefix}_robust_it{N}_{stem}_gamma0.csv"),
            ]
            p = next((c for c in cands if os.path.exists(c)), None)
            if p: hit[stem] = p
            else: ok = False; break
        if ok: return hit

    raise FileNotFoundError("Robust final or itN-max complete triplet not found.")

# ---------- locate deterministic gamma0/no-suffix (+ robust_it1_gamma0 fallback) ----------
def resolve_deterministic_set(prefix: str, gamma: int = 0) -> dict:
    """
    Return deterministic triplet paths; wsPlan & taskSeq required; taskEndShelf optional.
    Fallback (if enabled): use robust_it1_gamma0 (or earliest itN_gamma0) as deterministic.
    """
    exp_dir = "solution_exports"

    def pick_det(stem):
        cands = [
            os.path.join(exp_dir, f"{prefix}_{stem}_gamma{gamma}.csv"),
            os.path.join(exp_dir, f"{prefix}_{stem}.csv"),
            os.path.join(exp_dir, f"{prefix}_{stem}_gamma0.csv"),
        ]
        for p in cands:
            if os.path.exists(p): return p
        return None

    res = {s: pick_det(s) for s in ("wsPlan","taskSeq","taskEndShelf")}
    if res["wsPlan"] and res["taskSeq"]:
        return res

    if not USE_ROBUST_AS_DET_FALLBACK:
        raise FileNotFoundError("Deterministic wsPlan/taskSeq not found (gamma0 or no-suffix).")

    # ---- Fallback 1: robust_it1_gamma0
    hit = {}
    ok = True
    for stem in ("wsPlan","taskSeq","taskEndShelf"):
        cands = [
            os.path.join(exp_dir, f"{prefix}_robust_it1_{stem}_gamma{gamma}.csv"),
            os.path.join(exp_dir, f"{prefix}_robust_it1_{stem}_gamma0.csv"),
        ]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p: hit[stem] = p
        else: ok = False; break
    if ok:
        print("[INFO] deterministic triplet not found; fallback to robust_it1_gamma0 triplet.")
        return hit

    # ---- Fallback 2: earliest robust_itN_gamma0 (min N)
    pat = os.path.join(exp_dir, f"{prefix}_robust_it*_*.csv")
    paths = glob.glob(pat); it_set = set()
    for p in paths:
        m = re.search(rf"{re.escape(prefix)}_robust_it(\d+)_", os.path.basename(p))
        if m: it_set.add(int(m.group(1)))
    for N in sorted(it_set):
        hit, ok = {}, True
        for stem in ("wsPlan","taskSeq","taskEndShelf"):
            cands = [
                os.path.join(exp_dir, f"{prefix}_robust_it{N}_{stem}_gamma{gamma}.csv"),
                os.path.join(exp_dir, f"{prefix}_robust_it{N}_{stem}_gamma0.csv"),
            ]
            p = next((c for c in cands if os.path.exists(c)), None)
            if p: hit[stem] = p
            else: ok = False; break
        if ok:
            print(f"[INFO] deterministic triplet not found; fallback to robust_it{N}_gamma0 triplet.")
            return hit

    # still not found
    raise FileNotFoundError("Deterministic wsPlan/taskSeq not found (gamma0 or no-suffix), "
                            "and robust_itN_gamma0 fallback failed.")

# ---------- CSV parsers ----------
def load_ws_plan_from_file(plan_csv: str, print_banner=True):
    df = pd.read_csv(plan_csv)
    missing = {"WS","Task"} - set(df.columns)
    if missing:
        raise KeyError(f"[PLAN] {os.path.basename(plan_csv)} missing columns: {missing}")
    sort_cols = [c for c in ("WS","p_opt","Task") if c in df.columns]
    order_by_ws, p_opt, q_opt = {}, {}, {}
    for _, r in df.sort_values(sort_cols).iterrows():
        ws = int(r["WS"]); t = int(r["Task"])
        order_by_ws.setdefault(ws, []).append(t)
        if "p_opt" in df.columns: p_opt[t] = float(r["p_opt"])
        if "q_opt" in df.columns: q_opt[t] = float(r.get("q_opt", r["p_opt"]))
    d_setup_in_file = float(df["D_setup"].iloc[0]) if "D_setup" in df.columns else None
    if print_banner:
        print(f"[PLAN] Loaded: {os.path.basename(plan_csv)}")
        for ws in sorted(order_by_ws):
            print(f"  WS{ws}: {order_by_ws[ws]}")
    return {"order_by_ws": order_by_ws, "p_opt": p_opt, "q_opt": q_opt, "D_setup": d_setup_in_file}

def parse_task_seq_csv(seq_csv: str, verbose=True):
    df = pd.read_csv(seq_csv)
    if verbose:
        print(f"[SEQ] Loaded: {os.path.basename(seq_csv)}")
    agv_col = next((c for c in ["AGV_ID","AGV","agv_id","agv"] if c in df.columns), None)
    seq_col = next((c for c in ["seq","sequence","tasks","task_list"] if c in df.columns), None)
    if agv_col is None or seq_col is None:
        raise KeyError(f"[SEQ] {os.path.basename(seq_csv)} missing AGV_ID/seq; got {list(df.columns)}")
    task_seq = {}
    for _, r in df.iterrows():
        agv_id = int(r[agv_col])
        raw = str(r[seq_col]).strip()
        if raw in ("", "[]", "None", "nan"):
            seq = []
        else:
            try:
                seq = list(ast.literal_eval(raw))
            except Exception:
                seq = [int(x) for x in re.findall(r"\d+", raw)]
        task_seq[agv_id] = [int(x) for x in seq]
    if verbose:
        for agv in sorted(task_seq):
            print(f"  AGV{agv}: {task_seq[agv]}")
    return task_seq

def parse_end_shelf_csv(end_csv: str, verbose=True):
    if end_csv is None or (not os.path.exists(end_csv)):
        if verbose: print("[X] end-shelf: <none>, default to home_before")
        return {}
    df = pd.read_csv(end_csv)
    cols_lower = {c.lower(): c for c in df.columns}
    task_col  = cols_lower.get("task")
    shelf_col = cols_lower.get("endshelf") or cols_lower.get("end_shelf")
    if not task_col or not shelf_col:
        raise KeyError(f"[X] {os.path.basename(end_csv)} missing Task/EndShelf; got {list(df.columns)}")
    end_map = {int(r[task_col]): int(r[shelf_col]) for _, r in df.iterrows()}
    if verbose: print(f"[X] end-shelf loaded: {os.path.basename(end_csv)} ({len(end_map)} rows)")
    return end_map

# ===============================
# ======== CORE SIM ============
# ===============================
def run_single_sim(prefix, plan_csv, seq_csv, end_csv,
                   d_setup, path_mode, policy,
                   gamma_budget, rng_seed,
                   global_gamma=True, exact_gamma=True,
                   segments_allowed=("move1","move2","move3"),
                   factor_range=(2.0,2.0), add_range=None,
                   suppress_lib_logs=True):
    """
    Run a single simulation with random delays.
    Returns (cmax_ws, cmax_drop).
    """
    # --- map & managers ---
    with _mute_print(suppress_lib_logs):
        shelf_init, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)
        shelf_cur = {int(k): int(v) for k, v in shelf_init.items()}
        map_obj = create_map_from_components(width=W, height=H,
                                             sp_indices=sp_indices,
                                             ws_indices=ws_indices,
                                             shelf_data=shelf_cur,
                                             agv_data=agv_data)
        mm = MovementManager(map_obj, time_manager=None)

    # --- plan & sequence ---
    plan = load_ws_plan_from_file(plan_csv, print_banner=False) if policy in ("MILP","MILP_ORDER") else None
    if plan and plan.get("D_setup") is not None:
        d_setup = float(plan["D_setup"])

    task_seq = parse_task_seq_csv(seq_csv, verbose=False)
    if not task_seq or all(len(v)==0 for v in task_seq.values()):
        raise RuntimeError(f"[SEQ] empty sequence parsed from {os.path.basename(seq_csv)}")

    end_map  = parse_end_shelf_csv(end_csv, verbose=False)

    # --- tasks meta (from scenario or saved inputs) ---
    exp_dir = "solution_exports"
    scen_tasks_csv = os.path.join("scenario", prefix, "tasks.csv")
    if os.path.exists(scen_tasks_csv):
        tdf = pd.read_csv(scen_tasks_csv)
        need = {"Task","Shelf","Workstation","Duration"}
        if not need.issubset(tdf.columns) or tdf.isna().any().any():
            raise ValueError(f"[TASK] {scen_tasks_csv} invalid")
        info_df  = tdf[["Task","Workstation","Duration"]].copy()
        shelf_df = tdf[["Task","Shelf"]].copy()
    else:
        info_csv  = os.path.join(exp_dir, f"{prefix}_taskInfo.csv")
        shelf_csv = os.path.join(exp_dir, f"{prefix}_taskShelf.csv")
        if not (os.path.exists(info_csv) and os.path.exists(shelf_csv)):
            raise FileNotFoundError("taskInfo/taskShelf not found")
        info_df  = pd.read_csv(info_csv)
        shelf_df = pd.read_csv(shelf_csv)

    tasks_info = {int(r.Task):(int(r.Workstation), float(r.Duration)) for _, r in info_df.iterrows()}
    task_shelf = {int(r.Task): int(r.Shelf) for _, r in shelf_df.iterrows()}

    # --- WS cells & path funcs ---
    ws_cells = {ws_id: int(ws_indices[ws_id - 1]) for ws_id in range(1, len(ws_indices) + 1)}

    def get_path_astar(a: int, b: int) -> list[int]:
        key = (int(a), int(b))
        return mm.get_path(key[0], key[1])

    def get_dt_and_path(a: int, b: int):
        a, b = int(a), int(b)
        if path_mode == "jump":
            dt = manhattan_idx(a, b, W); path = [a, b]
        else:
            path = get_path_astar(a, b); dt = max(0, len(path) - 1)
        return float(dt), path

    # --- schedule structures ---
    ws_order = plan["order_by_ws"] if plan else {}
    ws_cursor = {ws: 0 for ws in ws_cells}
    ws_park   = {ws: {} for ws in ws_cells}
    ws_free_time = {ws_id: 0.0 for ws_id in ws_cells}
    ws_last_task  = {ws_id: None for ws_id in ws_cells}

    agv_cur_cell = {int(a): int(s) for a, s in agv_data.items()}
    agv_clock    = {int(a): 0.0 for a in sorted(task_seq.keys())}
    seq_ptr      = {int(a): 0   for a in sorted(task_seq.keys())}

    timeline_rows, seg_paths = [], {}

    # --- sample random delays (global/per-AGV gamma) ---
    rng = random.Random(rng_seed)

    def sample_delay_plan(task_seq_dict):
        def _assign(plan_dict, key):
            if factor_range is not None:
                f = rng.uniform(factor_range[0], factor_range[1]); plan_dict[key] = ("mult", f)
            elif add_range is not None:
                s = rng.uniform(add_range[0], add_range[1]); plan_dict[key] = ("add", s)
            else:
                plan_dict[key] = ("mult", 2.0)

        plan_local = {}
        if gamma_budget <= 0:
            return plan_local
        segs = tuple(sorted(list(segments_allowed)))

        if global_gamma:
            cand_all = []
            for agv, seq in task_seq_dict.items():
                for t in seq:
                    ti = int(t)
                    for seg in segs: cand_all.append((int(agv), ti, seg))
            if not cand_all: return plan_local
            kmax  = min(int(gamma_budget), len(cand_all))
            kpick = kmax if exact_gamma else rng.randint(0, kmax)
            if kpick > 0:
                for pick in rng.sample(cand_all, kpick): _assign(plan_local, pick)
            return plan_local

        # per-AGV ≤ γ
        for agv, seq in task_seq_dict.items():
            cand = []
            for t in seq:
                ti = int(t)
                for seg in segs: cand.append((int(agv), ti, seg))
            if not cand: continue
            kmax = min(int(gamma_budget), len(cand))
            if kmax <= 0: continue
            kpick = kmax if exact_gamma else rng.randint(0, kmax)
            if kpick == 0: continue
            for pick in rng.sample(cand, kpick): _assign(plan_local, pick)
        return plan_local

    delay_plan = sample_delay_plan(task_seq)

    def apply_delay_dt(plan_dict, agv, task, seg, dt_nominal: float):
        meta = plan_dict.get((agv, task, seg))
        if not meta: return float(dt_nominal), False
        kind, val = meta
        return (float(dt_nominal) * float(val), True) if kind == "mult" else (float(dt_nominal)+float(val), True)

    # --- event heap (arrival to WS) ---
    heap = []  # (move2_e, counter, agv, payload)
    counter = 0

    def dispatch_next_task(agv: int):
        nonlocal counter
        seq = task_seq.get(agv, [])
        ptr = seq_ptr.get(agv, 0)
        if ptr >= len(seq): return
        t = int(seq[ptr])
        shelf_id = int(task_shelf[t])
        ws_id, duration = tasks_info[t]
        home_before = int(shelf_cur[shelf_id])
        ws_idx = int(ws_cells[ws_id])
        cur_idx = int(agv_cur_cell[agv])
        t0 = float(agv_clock[agv])

        # move1
        dt1_nom, p1 = get_dt_and_path(cur_idx, home_before)
        dt1_exec, _ = apply_delay_dt(delay_plan, agv, t, "move1", dt1_nom)
        move1_s, move1_e = t0, t0 + dt1_exec
        pick_s = pick_e = move1_e
        # move2
        dt2_nom, p2 = get_dt_and_path(home_before, ws_idx)
        dt2_exec, _ = apply_delay_dt(delay_plan, agv, t, "move2", dt2_nom)
        move2_s, move2_e = pick_e, pick_e + dt2_exec

        payload = {
            "Task": t, "AGV": agv, "Shelf": shelf_id, "WS": ws_id,
            "ws_idx": ws_idx, "duration": float(duration),
            "home_before": home_before,
            "move1_s": move1_s, "move1_e": move1_e, "t_move1": dt1_exec, "p1": p1,
            "pick_s": pick_s, "pick_e": pick_e,
            "move2_s": move2_s, "move2_e": move2_e, "t_move2": dt2_exec, "p2": p2,
        }
        heapq.heappush(heap, (move2_e, counter, agv, payload)); counter += 1

    for agv in sorted(task_seq.keys()):
        dispatch_next_task(agv)

    def schedule_one(rec, ws_s):
        nonlocal timeline_rows, seg_paths
        t = rec["Task"]; agv = rec["AGV"]; shelf_id = rec["Shelf"]; ws_id = rec["WS"]; ws_idx = rec["ws_idx"]
        duration = rec["duration"]; home_before = rec["home_before"]
        arrival_to_ws = float(rec["move2_e"])
        ws_e = ws_s + duration

        # move3 to end shelf (from plan if provided, else back home_before)
        home_after = int(end_map.get(t, home_before))
        dt3_nom, p3 = get_dt_and_path(ws_idx, home_after)
        dt3_exec, _ = apply_delay_dt(delay_plan, agv, t, "move3", dt3_nom)
        move3_s, move3_e = ws_e, ws_e + dt3_exec
        drop_s = drop_e = move3_e

        # timeline row
        row = {
            "Task": t, "AGV": agv, "Shelf": shelf_id, "WS": ws_id,
            "home_before": home_before, "end_shelf": home_after,
            "move1_s": rec["move1_s"], "move1_e": rec["move1_e"],
            "pick_s": rec["pick_s"], "pick_e": rec["pick_e"],
            "move2_s": rec["move2_s"], "move2_e": rec["move2_e"],
            "ws_s": ws_s, "ws_e": ws_e, "wait_ws": ws_s - arrival_to_ws,
            "move3_s": move3_s, "move3_e": move3_e,
            "drop_s": drop_s, "drop_e": drop_e,
            "t_move1": rec["t_move1"], "t_move2": rec["t_move2"], "t_ws": duration, "t_move3": dt3_exec,
            "p": ws_s, "q": ws_e
        }
        timeline_rows.append(row)
        seg_paths[(agv, t)] = {"p1": rec.get("p1", []), "p2": rec.get("p2", []), "p3": p3}

        # update states
        agv_clock[agv] = drop_e
        agv_cur_cell[agv] = home_after
        shelf_cur[shelf_id] = home_after
        ws_free_time[ws_id] = ws_e
        ws_last_task[ws_id] = t
        seq_ptr[agv] += 1
        dispatch_next_task(agv)

    def try_schedule_from_park(ws_id, policy_local):
        # after scheduling one task, if following tasks in order already arrived (in park), schedule them serially
        order = ws_order.get(ws_id, [])
        cur = ws_cursor.get(ws_id, 0)
        while cur < len(order):
            next_task = order[cur]
            rec = ws_park[ws_id].get(next_task)
            if rec is None: break
            arrival_to_ws = float(rec["move2_e"])
            base = max(arrival_to_ws, float(ws_free_time[ws_id])) + d_setup
            if policy_local == "MILP" and plan:
                ws_s = max(base, float(plan["p_opt"].get(next_task, base)))
            else:
                ws_s = base
            schedule_one(rec, ws_s)
            ws_park[ws_id].pop(next_task, None)
            cur += 1
            ws_cursor[ws_id] = cur

    # main loop
    while heap:
        _, _, agv, rec = heapq.heappop(heap)
        t = rec["Task"]; ws_id = rec["WS"]

        if policy == "FCFS":
            arrival_to_ws = float(rec["move2_e"])
            base = max(arrival_to_ws, float(ws_free_time[ws_id])) + d_setup
            schedule_one(rec, base)
            continue

        order = ws_order.get(ws_id, [])
        cur = ws_cursor.get(ws_id, 0)
        next_task = order[cur] if cur < len(order) else None

        if next_task is None:
            # no plan for this WS -> FCFS
            arrival_to_ws = float(rec["move2_e"])
            base = max(arrival_to_ws, float(ws_free_time[ws_id])) + d_setup
            schedule_one(rec, base)
            continue

        if t != next_task:
            # not the next in order: park
            ws_park[ws_id][t] = rec
            try_schedule_from_park(ws_id, policy)
            continue

        # it's the correct next task
        arrival_to_ws = float(rec["move2_e"])
        base = max(arrival_to_ws, float(ws_free_time[ws_id])) + d_setup
        if policy == "MILP" and plan:
            ws_s = max(base, float(plan["p_opt"].get(t, base)))
        else:  # MILP_ORDER
            ws_s = base
        schedule_one(rec, ws_s)
        ws_cursor[ws_id] = cur + 1
        try_schedule_from_park(ws_id, policy)

    if not timeline_rows:
        raise RuntimeError("[SIM] no timeline")

    df_tl = pd.DataFrame(timeline_rows)
    cmax_ws   = float(df_tl["ws_e"].max())
    cmax_drop = float(df_tl["drop_e"].max())
    return cmax_ws, cmax_drop

# ===============================
# ======== RUN SUITE ============
# ===============================
def run_suite(prefix, plan_csv, seq_csv, end_csv,
              gamma_list, runs_per_gamma,
              policy, path_mode):
    rows = []
    for gb in gamma_list:
        vals = []
        for i in range(runs_per_gamma):
            seed_i = random.randrange(1, 10**9)
            cws, cdrop = run_single_sim(
                prefix=prefix,
                plan_csv=plan_csv,
                seq_csv=seq_csv,
                end_csv=end_csv,
                d_setup=D_SETUP,
                path_mode=path_mode,
                policy=policy,
                gamma_budget=gb,
                rng_seed=seed_i,
                global_gamma=GLOBAL_GAMMA,
                exact_gamma=EXACT_GAMMA,
                segments_allowed=tuple(sorted(list(SEGMENTS_ALLOWED))),
                factor_range=FACTOR_RANGE,
                add_range=ADD_RANGE,
                suppress_lib_logs=SUPPRESS_LIB_LOGS
            )
            vals.append((cws, cdrop))
        df = pd.DataFrame(vals, columns=["cws","cdrop"])
        print(f"[{('robust' if policy==POLICY_ROBUST else 'det'):>7}] Γ={gb}: "
              f"CmaxWS min/mean/std/max = {df.cws.min():.2f}/{df.cws.mean():.2f}/{df.cws.std(ddof=1):.2f}/{df.cws.max():.2f}, "
              f"CmaxDrop min/mean/std/max = {df.cdrop.min():.2f}/{df.cdrop.mean():.2f}/{df.cdrop.std(ddof=1):.2f}/{df.cdrop.max():.2f}")
        rows.append({
            "variant": ("robust" if policy==POLICY_ROBUST else "deterministic"),
            "gamma": gb,
            "ws_min": df.cws.min(), "ws_mean": df.cws.mean(), "ws_std": df.cws.std(ddof=1), "ws_max": df.cws.max(),
            "drop_min": df.cdrop.min(), "drop_mean": df.cdrop.mean(), "drop_std": df.cdrop.std(ddof=1), "drop_max": df.cdrop.max(),
        })
    return pd.DataFrame(rows)

def save_compare_bars(df_sum, prefix, outdir="solution_exports"):
    os.makedirs(outdir, exist_ok=True)
    metrics = [
        ("ws_min",   "CmaxWS_min"),
        ("ws_mean",  "CmaxWS_mean"),
        ("ws_std",   "CmaxWS_std"),
        ("ws_max",   "CmaxWS_max"),
        ("drop_min", "CmaxDrop_min"),
        ("drop_mean","CmaxDrop_mean"),
        ("drop_std", "CmaxDrop_std"),
        ("drop_max", "CmaxDrop_max"),
    ]
    variants = ["robust","deterministic"]
    x = np.arange(len(GAMMA_SET)); width = 0.38
    for mcol, mname in metrics:
        fig, ax = plt.subplots()
        for j, var in enumerate(variants):
            vals, xs = [], []
            for k, g in enumerate(GAMMA_SET):
                sub = df_sum[(df_sum["variant"]==var) & (df_sum["gamma"]==g)]
                if not sub.empty:
                    vals.append(float(sub[mcol].iloc[0]))
                    xs.append(x[k] + (j-0.5)*width)
            if vals:
                bars = ax.bar(xs, vals, width, label=var)
                for b in bars:
                    h = b.get_height()
                    ax.text(b.get_x()+b.get_width()/2, h, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlabel("Gamma"); ax.set_ylabel(mname); ax.set_title(f"{mname} vs Gamma")
        ax.set_xticks(x); ax.set_xticklabels(GAMMA_SET); ax.legend(); plt.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(outdir, f"{prefix}_{mname}_compare_bar_{ts}.png")
        plt.savefig(fpath, dpi=150); plt.close(fig)
        print(f"[PLOT] saved: {fpath}")

# ===============================
# =============== MAIN ==========
# ===============================
if __name__ == "__main__":
    exp_dir = "solution_exports"; os.makedirs(exp_dir, exist_ok=True)

    # Resolve two consistent triplets (no mixing)
    robust_paths = resolve_robust_set(PREFIX, gamma=0)
    det_paths    = resolve_deterministic_set(PREFIX, gamma=0)

    print("[PLAN-robust] using:")
    print("  - taskSeq     :", os.path.basename(robust_paths["taskSeq"]))
    print("  - taskEndShelf:", os.path.basename(robust_paths["taskEndShelf"]))
    print("  - wsPlan      :", os.path.basename(robust_paths["wsPlan"]))
    print("[PLAN-det] using:")
    print("  - taskSeq     :", os.path.basename(det_paths["taskSeq"]))
    print("  - taskEndShelf:", os.path.basename(det_paths.get("taskEndShelf") or "<none>"))
    print("  - wsPlan      :", os.path.basename(det_paths["wsPlan"]))

    # Run both variants
    print("\n===== Variant: robust =====")
    df_r = run_suite(PREFIX, robust_paths["wsPlan"], robust_paths["taskSeq"], robust_paths["taskEndShelf"],
                     gamma_list=GAMMA_SET, runs_per_gamma=BATCH_RUNS,
                     policy=POLICY_ROBUST, path_mode=PATH_MODE)
    out_r = os.path.join(exp_dir, f"{PREFIX}_dual_compare_stats_robust.csv")
    df_r.to_csv(out_r, index=False); print(f"[SAVE] robust stats -> {out_r}")

    print("\n===== Variant: deterministic =====")
    df_d = run_suite(PREFIX, det_paths["wsPlan"], det_paths["taskSeq"], det_paths.get("taskEndShelf"),
                     gamma_list=GAMMA_SET, runs_per_gamma=BATCH_RUNS,
                     policy=POLICY_DET, path_mode=PATH_MODE)
    out_d = os.path.join(exp_dir, f"{PREFIX}_dual_compare_stats_det.csv")
    df_d.to_csv(out_d, index=False); print(f"[SAVE] det stats -> {out_d}")

    # Merge summary & plot
    df_sum = pd.concat([df_r, df_d], ignore_index=True).sort_values(["variant","gamma"])
    out_sum = os.path.join(exp_dir, f"{PREFIX}_dual_compare_summary.csv")
    df_sum.to_csv(out_sum, index=False); print(f"[SAVE] combined summary -> {out_sum}")

    save_compare_bars(df_sum, PREFIX, outdir=exp_dir)

    print("\n[OK] done.")
