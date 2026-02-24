# main_align_bridge.py
from __future__ import annotations

import os
import json
import math
import argparse
import importlib
import inspect
from typing import Dict, List, Tuple, Optional, Set, Any

import pandas as pd
from collections import defaultdict

from heuristic_init import build_initial_solution_basic
from pythonProject5.solution_exports.solution_structures import InitialSolution
from warmstart_checks import check_v_consistency
from map_generator import load_map_csv, create_map_from_components
from movement_manager import MovementManager
from time_manager import TimeManager
from scenario import build_scenario_from_prefix, print_scenario_brief
from utils import distance
from evaluator import RobustEvaluator
from export_eval_diag import export_evaluator_diagnostics


# ----------------------------
# bundle helpers
# ----------------------------
def _bundle_intify(bundle: dict) -> dict:
    routes_raw = bundle.get("routes") or bundle.get("routes_by_agv") or {}
    shelf_seq_raw = bundle.get("shelf_seq") or {}
    place_raw = bundle.get("place") or {}
    sigma_raw = bundle.get("cell_sigma") or {}

    routes = {int(r): [int(t) for t in (seq or [])] for r, seq in routes_raw.items()}
    shelf_seq = {int(c): [int(t) for t in (seq or [])] for c, seq in shelf_seq_raw.items()}
    place = {int(j): int(s) for j, s in place_raw.items()}

    cell_sigma = None
    if isinstance(sigma_raw, dict) and sigma_raw:
        cell_sigma = {int(s): [int(x) for x in (seq or [])] for s, seq in sigma_raw.items()}

    out = dict(bundle)
    out["routes"] = routes
    out["shelf_seq"] = shelf_seq
    out["place"] = place
    out["cell_sigma"] = cell_sigma
    return out


def load_bundle_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    return _bundle_intify(b)


def _pack_3key_map_to_nested(m: dict) -> dict:
    out: dict = {}
    for k, v in (m or {}).items():
        if not (isinstance(k, tuple) and len(k) == 3):
            continue
        a, b, c = k
        aa = int(a); bb = int(b); cc = int(c)
        out.setdefault(aa, {}).setdefault(bb, {})[cc] = float(v)
    return out


def _make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj and all(isinstance(k, tuple) and len(k) == 3 for k in obj.keys()):
            return _pack_3key_map_to_nested(obj)

        newd = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k2 = "|".join(str(x) for x in k)
            elif isinstance(k, (str, int, float, bool)) or k is None:
                k2 = k
            else:
                k2 = str(k)
            newd[k2] = _make_json_safe(v)
        return newd

    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(x) for x in obj]

    return obj


def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj2 = dict(obj)

    for key in ("g", "h"):
        if key in obj2 and isinstance(obj2[key], dict):
            if obj2[key] and all(isinstance(k, tuple) and len(k) == 3 for k in obj2[key].keys()):
                obj2[key] = _pack_3key_map_to_nested(obj2[key])

    obj2 = _make_json_safe(obj2)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj2, f, ensure_ascii=False, indent=2)


def find_bundle_path(prefix: str, gamma: int, outdir: str, kind: str) -> Optional[str]:
    cand = []
    if kind == "alns":
        cand = [
            os.path.join(outdir, f"{prefix}_alns_bundle_gamma{gamma}.json"),
            os.path.join(outdir, f"{prefix}_alns_bundle_gamma{gamma}.JSON"),
        ]
    elif kind == "milp":
        cand = [
            os.path.join(outdir, f"{prefix}_milp_bundle_gamma{gamma}.json"),
            os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json"),
        ]
    else:
        return None

    for p in cand:
        if os.path.exists(p):
            return p
    return None


# ----------------------------
# sigma builder
# ----------------------------
def build_cell_sigma_from_event_diag(
    *,
    diag: dict,
    J_set: Set[int],
) -> Dict[int, List[int]]:
    end_final = diag.get("end_shelf_final", {}) or {}
    arrive_act = diag.get("arrive_cell_act", {}) or {}

    try:
        end2 = {int(k): int(v) for k, v in end_final.items()}
    except Exception:
        end2 = {}
    try:
        arr2 = {int(k): float(v) for k, v in arrive_act.items()}
    except Exception:
        arr2 = {}

    by_cell: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for j, s in end2.items():
        jj = int(j)
        if jj not in J_set:
            continue
        if jj not in arr2:
            continue
        by_cell[int(s)].append((float(arr2[jj]), jj))

    cell_sigma: Dict[int, List[int]] = {}
    for s, lst in by_cell.items():
        lst.sort(key=lambda x: (x[0], x[1]))
        cell_sigma[int(s)] = [int(j) for _, j in lst]
    return cell_sigma


# ----------------------------
# warmstart export
# ----------------------------
def build_warm_hint_from_eval(
    *,
    J: Set[int],
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    details: dict,
    ws_fixed_seq: Dict[int, List[int]],
    J0: Dict[int, int],
    Jd: Dict[int, int],
    J_I: Dict[int, int],
    pi: Dict[int, int],
    lock_x: bool = True,
    place_override: Optional[Dict[int, int]] = None,
) -> dict:
    BIG_M_TIME = 10000.0
    warm_hint: dict = {}

    # ---- w ----
    w_map: Dict[int, int] = {}
    for r, seq in routes.items():
        rr = int(r)
        for j in (seq or []):
            jj = int(j)
            if jj in J:
                w_map[jj] = rr
    warm_hint["w"] = w_map

    # ---- z ----
    z_list: List[Tuple[int, int, int]] = []
    for r, seq in routes.items():
        rr = int(r)
        seq_clean = [int(j) for j in (seq or []) if int(j) in J]
        j0 = J0.get(rr)
        jd = Jd.get(rr)

        prev = j0
        for j in seq_clean:
            if prev is not None:
                z_list.append((int(prev), int(j), rr))
            prev = j
        if prev is not None and jd is not None:
            z_list.append((int(prev), int(jd), rr))
    warm_hint["z"] = z_list

    # ---- v ----
    V_raw = details.get("V_arcs", []) or []
    v_list: List[Tuple[int, int, int, int]] = []
    for arc in V_raw:
        if not isinstance(arc, (list, tuple)) or len(arc) != 4:
            continue
        i, j, s, sp = arc
        v_list.append((int(i), int(j), int(s), int(sp)))
    warm_hint["v"] = v_list

    # ---- x ----
    x_map: Dict[int, int] = {}
    if place_override is not None:
        for j, s in (place_override or {}).items():
            jj = int(j)
            if jj in J:
                x_map[jj] = int(s)
    else:
        end_final = details.get("end_shelf_final", {}) or {}
        for j, s in end_final.items():
            jj = int(j)
            if jj in J:
                x_map[jj] = int(s)

    warm_hint["x"] = x_map
    warm_hint["lock_x"] = bool(lock_x)

    # ---- immediate edges (chain) ----
    imm_edges: List[Tuple[int, int, int]] = []
    for c, seq in (shelf_seq or {}).items():
        cc = int(c)
        vt_c = J_I.get(cc)
        if vt_c is None:
            continue
        seq_clean = [int(j) for j in (seq or []) if int(j) in J]
        if not seq_clean:
            continue
        imm_edges.append((int(vt_c), int(seq_clean[0]), cc))
        for a, b in zip(seq_clean[:-1], seq_clean[1:]):
            imm_edges.append((int(a), int(b), cc))
    warm_hint["immediate"] = imm_edges

    # ---- p/q ----
    p_map = {int(j): float(t) for j, t in (details.get("p", {}) or {}).items() if int(j) in J}
    q_map = {int(j): float(t) for j, t in (details.get("q", {}) or {}).items() if int(j) in J}
    warm_hint["p"] = p_map
    warm_hint["q"] = q_map

    # ---- g/h ----
    timeline = details.get("timeline", []) or []
    rec_by_task: Dict[int, dict] = {}
    for rec in timeline:
        try:
            jj = int(rec.get("Task"))
        except Exception:
            continue
        rec_by_task[jj] = rec

    def _get_pick_start0(j: int) -> float:
        rec = rec_by_task.get(int(j))
        if isinstance(rec, dict):
            if rec.get("pick_start_0") is not None:
                try:
                    return float(rec["pick_start_0"])
                except Exception:
                    pass
            if rec.get("pick_start") is not None:
                try:
                    return float(rec["pick_start"])
                except Exception:
                    pass
        try:
            return float((details.get("pick_start") or {}).get(int(j), BIG_M_TIME))
        except Exception:
            return BIG_M_TIME

    def _get_arrive_cell0(j: int) -> float:
        rec = rec_by_task.get(int(j))
        if isinstance(rec, dict):
            if rec.get("arrive_cell_act_0") is not None:
                try:
                    return float(rec["arrive_cell_act_0"])
                except Exception:
                    pass
            if rec.get("arrive_cell_act") is not None:
                try:
                    return float(rec["arrive_cell_act"])
                except Exception:
                    pass
        try:
            return float((details.get("arrive_cell_act") or {}).get(int(j), 0.0))
        except Exception:
            return 0.0

    succ_chain: Dict[int, int] = {}
    chain_of: Dict[int, int] = {}
    tail_of_chain: Dict[int, int] = {}
    for c, seq in (shelf_seq or {}).items():
        cc = int(c)
        seq_clean = [int(j) for j in (seq or []) if int(j) in J]
        if not seq_clean:
            continue
        tail_of_chain[cc] = int(seq_clean[-1])
        for a, b in zip(seq_clean[:-1], seq_clean[1:]):
            succ_chain[int(a)] = int(b)
            chain_of[int(a)] = cc
            chain_of[int(b)] = cc

    g_map: Dict[Tuple[int, int, int], float] = {}
    h_map: Dict[Tuple[int, int, int], float] = {}

    for j, s in x_map.items():
        jj = int(j)
        ss = int(s)
        g0 = _get_arrive_cell0(jj)

        cc = chain_of.get(jj, None)
        is_tail = (cc is not None and tail_of_chain.get(int(cc)) == jj)
        nxt = succ_chain.get(jj, None)

        if is_tail or (nxt is None):
            h0 = BIG_M_TIME
        else:
            h0 = _get_pick_start0(int(nxt))
            if (not math.isfinite(h0)) or h0 <= 0.0:
                h0 = BIG_M_TIME
        if h0 < g0:
            h0 = g0

        g_map[(jj, 0, ss)] = float(g0)
        h_map[(jj, 0, ss)] = float(h0)

    for c, vt in (J_I or {}).items():
        cc = int(c)
        vt_id = int(vt)
        seq = (shelf_seq or {}).get(cc, []) or []
        seq_clean = [int(j) for j in seq if int(j) in J]
        if not seq_clean:
            continue
        j_first = int(seq_clean[0])
        rec = rec_by_task.get(j_first, {})
        init_cell = rec.get("home_before", None)
        if init_cell is None:
            continue
        init_cell = int(init_cell)
        h0 = _get_pick_start0(j_first)
        if (not math.isfinite(h0)) or h0 <= 0.0:
            h0 = BIG_M_TIME

        g_map[(vt_id, 0, init_cell)] = 0.0
        h_map[(vt_id, 0, init_cell)] = float(h0)

    warm_hint["g"] = g_map
    warm_hint["h"] = h_map
    warm_hint["cmax_eval"] = float(details.get("C_task_max", float("inf")))
    return warm_hint


# ----------------------------
# evaluator factory
# ----------------------------
def make_evaluator(
    *,
    J: Set[int],
    R: dict,
    S: dict,
    pi: Dict[int, int],
    D: Dict[int, float],
    J0: Dict[int, int],
    Jd: Dict[int, int],
    J_I: Dict[int, int],
    shelf_data: Dict[int, int],
    agv_data: Dict[int, int],
    d_s_pi: dict,
    d_pi_s: dict,
    d_s_s: dict,
    Delta_s_pi: dict,
    Delta_pi_s: dict,
    Delta_s_s: dict,
    ws_fixed_seq: Dict[int, List[int]],
    gamma: int,
    cell_sigma: Optional[Dict[int, List[int]]],
    timeline_mode: str,
    collect_v_arcs: bool,
    cell_gate_mode: str,
) -> RobustEvaluator:
    return RobustEvaluator(
        J=J, R=R, S=S,
        pi=pi,
        D=D,
        J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        gamma=int(gamma),
        ws_fixed_seq=ws_fixed_seq,
        ws_setup_rule="flat",

        lock_place=True,
        detach_on_mismatch=False,
        envelope_shared_resources=True,

        enable_chain_repair=True,
        chain_repair_max_iters=8,
        enable_cell_repair=True,
        cell_repair_max_iters=15,

        cell_conflict_mode="hard",
        allow_incomplete=False,

        cell_gate_mode=str(cell_gate_mode),
        cell_sigma=cell_sigma,

        timeline_mode=str(timeline_mode),
        collect_v_arcs=bool(collect_v_arcs),
        record_cell_repair_log=False,
    )


# ----------------------------
# MILP runner hook (dynamic import)
# ----------------------------
def run_milp_with_warmstart(
    *,
    prefix: str,
    gamma: int,
    warm_hint: Optional[dict],
    outdir: str,
    # problem data
    J: Set[int],
    R: dict,
    S: dict,
    pi: Dict[int, int],
    D: Dict[int, float],
    J0: Dict[int, int],
    Jd: Dict[int, int],
    J_I: Dict[int, int],
    shelf_data: Dict[int, int],
    agv_data: Dict[int, int],
    d_s_pi: dict,
    d_pi_s: dict,
    d_s_s: dict,
    Delta_s_pi: dict,
    Delta_pi_s: dict,
    Delta_s_s: dict,
    ws_fixed_seq: Dict[int, List[int]],
    # structure (optional)
    fixed_routes: Optional[Dict[int, List[int]]] = None,
    fixed_shelf_seq: Optional[Dict[int, List[int]]] = None,
    fixed_place: Optional[Dict[int, int]] = None,
    fix_structure: bool = True,
    time_limit: float = 300.0,
    verbose: bool = False,
) -> Optional[dict]:
    """
    你只要在工程里提供一个可被 import 的函数即可（推荐名：solve_milp_bridge）：
        solve_milp_bridge(..., warm_hint=..., fix_structure=..., time_limit=...)
    这个脚本会自动尝试导入并调用。
    """
    _ = outdir

    candidates = [
        ("milp_solver", ["solve_milp_bridge", "solve_milp", "solve"]),
        ("optimization_model", ["solve_milp_bridge", "solve_milp", "solve"]),
        ("milp_model", ["solve_milp_bridge", "solve_milp", "solve"]),
        ("gurobi_model", ["solve_milp_bridge", "solve_milp", "solve"]),
    ]

    fn = None
    for mod_name, fn_names in candidates:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        for name in fn_names:
            if hasattr(mod, name):
                fn = getattr(mod, name)
                break
        if fn is not None:
            break

    if fn is None:
        if verbose:
            print("[MILP-HOOK] not found. Please provide solve_milp_bridge() in milp_solver.py (recommended).")
        return None

    call_kwargs = dict(
        prefix=prefix,
        gamma=int(gamma),
        warm_hint=warm_hint,
        fix_structure=bool(fix_structure),
        time_limit=float(time_limit),
        verbose=bool(verbose),

        J=set(int(x) for x in J),
        R=R, S=S,
        pi=pi, D=D,
        J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        ws_fixed_seq=ws_fixed_seq,

        fixed_routes=fixed_routes,
        fixed_shelf_seq=fixed_shelf_seq,
        fixed_place=fixed_place,
    )

    sig = inspect.signature(fn)
    filtered = {k: v for k, v in call_kwargs.items() if k in sig.parameters}

    try:
        return fn(**filtered)
    except Exception as e:
        print(f"[MILP-HOOK] call failed: {e}")
        return None


# ----------------------------
# main
# ----------------------------
def parse_gamma_list(s: str) -> List[int]:
    if not s:
        return [0]
    arr: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            arr.extend(range(int(a), int(b) + 1))
        else:
            arr.append(int(tok))
    return sorted(set(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="demo01")
    ap.add_argument("--src-gammas", default="", help="源解 γ 列表（用 ALNS bundle 的 γ）。默认等于 --gammas")
    ap.add_argument("--eval-gammas", default="", help="目标 MILP γ 列表。默认等于 --gammas")
    ap.add_argument("--gammas", default="0", help="若不单独指定 src/eval，就用这个列表")
    ap.add_argument("--outdir", default="solution_exports")

    ap.add_argument("--run-milp", action="store_true", help="已接入 MILP 求解入口时，执行 feasibility check")
    ap.add_argument("--fix-structure", action="store_true", help="MILP 中固定结构变量（w/z/x/immediate/...），仅检验可行性")
    ap.add_argument("--milp-time-limit", type=float, default=300.0)

    ap.add_argument("--export-eval-diag", action="store_true", help="导出 evaluator diag/timeline/pq 等")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    prefix = args.prefix
    outdir = args.outdir

    base_list = parse_gamma_list(args.gammas)
    src_list = parse_gamma_list(args.src_gammas) if args.src_gammas.strip() else base_list
    eval_list = parse_gamma_list(args.eval_gammas) if args.eval_gammas.strip() else base_list

    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"未找到 {tasks_csv}")

    # ========== load map + tasks ==========
    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)
    tasks_df = pd.read_csv(tasks_csv)

    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1
    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )

    map_obj = create_map_from_components(
        width=W, height=H,
        sp_indices=sp_indices,
        ws_indices=ws_indices,
        shelf_data=shelf_data,
        agv_data=agv_data
    )
    time_manager = TimeManager()
    _ = MovementManager(map_obj, time_manager)

    # build task structures
    shelf_ids = sorted(shelf_data.keys())
    tasks: Dict[int, Tuple[Optional[int], float, Any, Any]] = {}
    task_shelf_mapping: Dict[int, Optional[int]] = {}
    J: Set[int] = set()

    for _, r in tasks_df.sort_values("Task").iterrows():
        tid = int(r["Task"])
        sid = int(r["Shelf"])
        ws = int(r["Workstation"])
        dur = float(r["Duration"])
        tasks[tid] = (ws, dur, None, None)
        task_shelf_mapping[tid] = sid
        J.add(tid)

    # virtual tasks
    J0: Dict[int, int] = {}
    Jd: Dict[int, int] = {}
    for aid in agv_data:
        J0[int(aid)] = 1000 + int(aid)
        Jd[int(aid)] = 2000 + int(aid)

    J_I: Dict[int, int] = {}
    for sid in shelf_ids:
        J_I[int(sid)] = 3000 + int(sid)

    # sets
    R = map_obj.extract_AGVs()
    def idx2rc(idx: int): return divmod(int(idx) - 1, W)
    S = {int(sp): idx2rc(int(sp)) for sp in sp_indices}
    K = {i + 1: idx2rc(ws_indices[i]) for i in range(len(ws_indices))}

    # scenario brief
    pi = {j: int(tasks[j][0]) for j in J}
    sc = build_scenario_from_prefix(prefix=prefix, distance=distance, D_setup=2.0, gamma_budget=0, K_near=12)
    print_scenario_brief(sc)

    # distance matrices
    d_s_pi, d_pi_s, d_s_s = {}, {}, {}
    for s in S:
        for j in J:
            d_s_pi[(s, j)] = distance(s, pi[j], S, K)
    for j in J:
        for s in S:
            d_pi_s[(j, s)] = distance(pi[j], s, K, S)
    for s in S:
        for sp in S:
            d_s_s[(s, sp)] = distance(s, sp, S, S)

    Delta_s_pi = {k: d_s_pi[k] for k in d_s_pi}
    Delta_pi_s = {k: d_pi_s[k] for k in d_pi_s}
    Delta_s_s = {k: d_s_s[k] for k in d_s_s}

    # init solution sanity
    init_sol = build_initial_solution_basic(
        J=J, R=R,
        task_shelf_mapping=task_shelf_mapping,
        shelf_data=shelf_data,
        J_I=J_I,
        pi=pi
    )
    check_v_consistency(
        routes=init_sol.routes, shelf_seq=init_sol.shelf_seq, place=init_sol.place,
        J=J, R=R, S=S, J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data
    )
    print("[CHECK] v-arc consistency passed.")

    timeline_mode = "full" if args.export_eval_diag else "min"
    collect_v = bool(args.export_eval_diag)

    # ------------------------------------------------------------
    # CROSS: ALNS(src_gamma) -> MILP(eval_gamma)
    # ------------------------------------------------------------
    rows = []
    matrix_obj = pd.DataFrame(index=src_list, columns=eval_list, dtype=float)
    matrix_feas = pd.DataFrame(index=src_list, columns=eval_list, dtype=object)

    for src_g in src_list:
        alns_path = find_bundle_path(prefix, src_g, outdir, "alns")
        if not alns_path:
            print(f"[SKIP] ALNS bundle missing for src_gamma={src_g}")
            continue

        alns_bundle = load_bundle_json(alns_path)
        routes = alns_bundle["routes"]
        shelf_seq = alns_bundle["shelf_seq"]
        place = alns_bundle["place"]

        print("\n" + "=" * 90)
        print(f"[CROSS] src_gamma(ALNS)={src_g}  -> eval_gammas={eval_list}")
        print("=" * 90)

        for tgt_g in eval_list:
            # ---- Step 1: build sigma via a ROUGH legacy run (always buildable) ----
            ev_rough = make_evaluator(
                J=J, R=R, S=S, pi=pi,
                D={j: float(tasks[j][1]) for j in J},
                J0=J0, Jd=Jd, J_I=J_I,
                shelf_data=shelf_data, agv_data=agv_data,
                d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
                Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
                ws_fixed_seq=ws_fixed_seq,
                gamma=int(tgt_g),
                cell_sigma=None,
                timeline_mode="off",
                collect_v_arcs=False,
                cell_gate_mode="legacy",
            )
            ms0, diag0 = ev_rough.evaluate(routes, shelf_seq, place, verbose=False)
            if not isinstance(diag0, dict):
                diag0 = {}

            sigma = build_cell_sigma_from_event_diag(diag=diag0, J_set=set(int(x) for x in J))

            # ---- Step 2: event-gate exact run with sigma ----
            ev = make_evaluator(
                J=J, R=R, S=S, pi=pi,
                D={j: float(tasks[j][1]) for j in J},
                J0=J0, Jd=Jd, J_I=J_I,
                shelf_data=shelf_data, agv_data=agv_data,
                d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
                Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
                ws_fixed_seq=ws_fixed_seq,
                gamma=int(tgt_g),
                cell_sigma=sigma,
                timeline_mode=timeline_mode,
                collect_v_arcs=collect_v,
                cell_gate_mode="event",
            )
            ms, diag = ev.evaluate(routes, shelf_seq, place, verbose=False)
            if not isinstance(diag, dict):
                diag = {}

            feas_eval = math.isfinite(float(ms))
            print(f"[EVAL] srcG={src_g} -> tgtG={tgt_g} | evaluator={ms}")

            # ---- export evaluator diag (optional) ----
            if args.export_eval_diag:
                export_evaluator_diagnostics(
                    prefix=prefix,
                    export_tag=f"cross_eval_srcG{src_g}_tgtG{tgt_g}",
                    source_gamma=int(src_g),
                    eval_gamma=int(tgt_g),
                    routes=routes,
                    shelf_seq=shelf_seq,
                    place=place,
                    milp_cmax=float("nan"),
                    eval_cmax=float(ms),
                    diag=diag,
                    outdir=outdir,
                )

            # ---- export warmstart for THIS target gamma ----
            warm_hint = None
            warm_path = os.path.join(outdir, f"{prefix}_warmstart_from_alns_srcG{src_g}_to_milpG{tgt_g}.json")

            if feas_eval:
                warm_hint = build_warm_hint_from_eval(
                    J=set(int(x) for x in J),
                    routes=routes,
                    shelf_seq=shelf_seq,
                    place=place,
                    details=diag,
                    ws_fixed_seq=ws_fixed_seq,
                    J0=J0, Jd=Jd, J_I=J_I,
                    pi=pi,
                    lock_x=True,
                    place_override=place,  # 强制用结构 place
                )
                # 附带 sigma（给你的 MILP 如需要用）
                warm_hint["cell_sigma"] = sigma
                save_json(warm_path, warm_hint)
                print(f"[EXPORT] warmstart -> {warm_path}")
            else:
                print(f"[WARN] evaluator infeasible at tgtG={tgt_g}. warmstart not exported.")

            # ---- Step 3: run MILP feasibility check (if hooked) ----
            milp_res = None
            if args.run_milp:
                milp_res = run_milp_with_warmstart(
                    prefix=prefix,
                    gamma=int(tgt_g),
                    warm_hint=warm_hint,
                    outdir=outdir,

                    J=set(int(x) for x in J),
                    R=R, S=S,
                    pi=pi,
                    D={j: float(tasks[j][1]) for j in J},
                    J0=J0, Jd=Jd, J_I=J_I,
                    shelf_data=shelf_data, agv_data=agv_data,
                    d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
                    Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
                    ws_fixed_seq=ws_fixed_seq,

                    fixed_routes=routes,
                    fixed_shelf_seq=shelf_seq,
                    fixed_place=place,
                    fix_structure=bool(args.fix_structure),
                    time_limit=float(args.milp_time_limit),
                    verbose=args.verbose,
                )

            # ---- record ----
            row = {
                "src_gamma": int(src_g),
                "tgt_gamma": int(tgt_g),
                "eval_feasible": bool(feas_eval),
                "eval_cmax": float(ms) if math.isfinite(float(ms)) else float("inf"),
                "warmstart_path": warm_path if (warm_hint is not None) else "",
            }

            if milp_res is None:
                row.update({
                    "milp_ran": False,
                    "milp_feasible": None,
                    "milp_status": "NO_HOOK" if args.run_milp else "SKIP",
                    "milp_obj": None,
                    "milp_runtime": None,
                })
                matrix_feas.loc[int(src_g), int(tgt_g)] = row["milp_status"]
            else:
                status = str(milp_res.get("status", "UNKNOWN"))
                feasible = bool(milp_res.get("feasible", False))
                obj = milp_res.get("obj", None)
                rt = milp_res.get("runtime", None)

                row.update({
                    "milp_ran": True,
                    "milp_feasible": feasible,
                    "milp_status": status,
                    "milp_obj": float(obj) if obj is not None else None,
                    "milp_runtime": float(rt) if rt is not None else None,
                })
                matrix_feas.loc[int(src_g), int(tgt_g)] = "OK" if feasible else f"INFEAS({status})"
                matrix_obj.loc[int(src_g), int(tgt_g)] = float(obj) if (obj is not None and feasible) else float("inf")

            rows.append(row)

    # export summary
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df_path = os.path.join(outdir, f"{prefix}_crossGamma_milp_feas_detail.csv")
    df.to_csv(df_path, index=False, encoding="utf-8-sig")
    print(f"\n[EXPORT] detail -> {df_path}")

    feas_path = os.path.join(outdir, f"{prefix}_crossGamma_milp_feas_matrix.csv")
    matrix_feas.to_csv(feas_path, encoding="utf-8-sig")
    print(f"[EXPORT] feas matrix -> {feas_path}")

    obj_path = os.path.join(outdir, f"{prefix}_crossGamma_milp_obj_matrix.csv")
    matrix_obj.to_csv(obj_path, encoding="utf-8-sig")
    print(f"[EXPORT] obj matrix -> {obj_path}")

    print("\n[DONE] cross-gamma MILP validation finished.")


if __name__ == "__main__":
    main()