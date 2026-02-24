# occ_simulate_and_audit.py
from __future__ import annotations
import os, math, argparse, ast, json, re, time
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

# ==== 依赖你工程里已有的模块 ====
from evaluator import RobustEvaluator
from map_generator import load_map_csv, create_map_from_components
from utils import distance

EPS = 1e-9

# ---------- 通用工具 ----------
def _warn(msg: str): print(f"[OCC][WARN] {msg}")
def _info(msg: str): print(f"[OCC] {msg}")

def _sanitize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag: return ""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)
    return tag.strip("-_")

def _safe_path(path: str, *, overwrite: bool) -> str:
    """若 path 已存在且不允许覆盖，则自动追加 _v1/_v2/..."""
    if overwrite or (not os.path.exists(path)): return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_v{k}{ext}"
        if not os.path.exists(cand): return cand
        k += 1

def _parse_list_cell(s) -> List[int]:
    if isinstance(s, list): return [int(x) for x in s]
    if not isinstance(s, str): return []
    s = s.strip()
    if not s: return []
    for parser in (ast.literal_eval, json.loads):
        try:
            v = parser(s)
            if isinstance(v, list): return [int(x) for x in v]
        except Exception:
            pass
    try:
        return [int(x) for x in s.replace("[","").replace("]","").split(",") if x.strip()]
    except Exception:
        return []

def _finite(x) -> bool:
    try:
        v = float(x); return math.isfinite(v)
    except Exception:
        return False

def _finite_or(*cands, default=None):
    for v in cands:
        if _finite(v): return float(v)
    return default

def _finite_max(vals) -> float:
    m = 0.0
    for v in vals:
        if _finite(v): m = max(m, float(v))
    return m

def _coerce_int(x) -> Optional[int]:
    try: return int(x)
    except Exception: return None

# ---------- 通用的“多别名健壮取值器” ----------
def _G(rec: Dict[str, Any], *names: str, default=None, finite_only=True) -> Optional[float]:
    """
    从 rec 中按顺序取第一个存在的键；如果 finite_only=True 则仅返回有限值。
    例：_G(rec, "arrive_cell_act","arrCell_act","arrive_cell_nom","arrCell_nom", default=H)
    """
    for n in names:
        if n in rec:
            v = rec[n]
            if not finite_only:
                return v
            if _finite(v):
                return float(v)
    return default

# ---------- 读取结构：优先 bundle ----------
def load_struct(prefix: str, gamma: int, J: set[int], R: Dict[int,int], S: Dict[int,tuple], shelf_ids: List[int]):
    outdir = "solution_exports"
    # 1) 首选 bundle
    f_bundle = os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json")
    if os.path.exists(f_bundle):
        with open(f_bundle, "r", encoding="utf-8") as f:
            b = json.load(f)
        routes = {int(r): [int(t) for t in (b.get("routes", {}).get(str(r), []) or []) if int(t) in J] for r in R}
        place  = {int(j): int(s) for j, s in (b.get("place", {}) or {}).items()
                  if _coerce_int(j) in J and _coerce_int(s) in S}
        shelf_seq = {int(c): [int(t) for t in (b.get("shelf_seq", {}).get(str(c), []) or []) if int(t) in J]
                     for c in shelf_ids}
        return routes, place, shelf_seq, f_bundle

    # 2) 回退：routes_by_agv 或 taskSeq
    f_routes_agv = os.path.join(outdir, f"{prefix}_routes_by_agv_gamma{gamma}.csv")
    f_taskseq    = os.path.join(outdir, f"{prefix}_taskSeq_gamma{gamma}.csv")

    routes: Dict[int, List[int]] = {int(r): [] for r in R}
    if os.path.exists(f_routes_agv):
        df = pd.read_csv(f_routes_agv)
        for _, row in df.iterrows():
            r = int(row["AGV_ID"])
            if r in R:
                seq = _parse_list_cell(row["seq"])
                routes[r] = [int(j) for j in seq if int(j) in J]
    elif os.path.exists(f_taskseq):
        df = pd.read_csv(f_taskseq)
        for _, row in df.iterrows():
            r = int(row["AGV_ID"])
            if r in R:
                seq = _parse_list_cell(row["seq"])
                routes[r] = [int(j) for j in seq if int(j) in J]
    else:
        raise FileNotFoundError("未找到 routes_by_agv 或 taskSeq 导出。")

    # place（x 的结果；无则 fromV 兜底）
    place: Dict[int,int] = {}
    f_end = os.path.join(outdir, f"{prefix}_taskEndShelf_gamma{gamma}.csv")
    if not os.path.exists(f_end):
        f_end = os.path.join(outdir, f"{prefix}_taskEndShelf_fromV_gamma{gamma}.csv")
    if os.path.exists(f_end):
        df = pd.read_csv(f_end)
        Sset = {int(s) for s in S}
        for _, row in df.iterrows():
            j = int(row["Task"]); s = int(row["EndShelf"])
            if j in J and s in Sset: place[j] = s
    else:
        raise FileNotFoundError("未找到 taskEndShelf(_fromV) 导出。")

    # shelf_seq
    shelf_seq: Dict[int, List[int]] = {int(c): [] for c in shelf_ids}
    f_seq = os.path.join(outdir, f"{prefix}_shelf_seq_gamma{gamma}.csv")
    if os.path.exists(f_seq):
        df = pd.read_csv(f_seq)
        for _, row in df.iterrows():
            c = int(row["shelf_id"])
            seq = _parse_list_cell(row["seq"])
            shelf_seq[c] = [int(j) for j in seq if int(j) in J]

    return routes, place, shelf_seq, None

# ---------- 构造 evaluator 所需数据 ----------
def build_data(prefix: str):
    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv): raise FileNotFoundError(f"未找到 {tasks_csv}")

    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)
    map_obj = create_map_from_components(width=W, height=H,
                                         sp_indices=sp_indices, ws_indices=ws_indices,
                                         shelf_data=shelf_data, agv_data=agv_data)
    tasks_df = pd.read_csv(tasks_csv)
    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1

    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"].apply(lambda s: [int(x) for x in s.tolist()]).to_dict()
    )

    shelf_ids = sorted(shelf_data.keys())
    J = set(int(t) for t in tasks_df["Task"].tolist())
    R = map_obj.extract_AGVs()

    def idx2rc(idx: int): return divmod(int(idx) - 1, W)
    S = {int(sp): idx2rc(int(sp)) for sp in sp_indices}
    K = {i + 1: idx2rc(ws_indices[i]) for i in range(len(ws_indices))}

    J0, Jd, J_I = {}, {}, {}
    for aid in R.keys():
        J0[aid] = 1000 + int(aid)
        Jd[aid] = 2000 + int(aid)
    for sid in shelf_ids:
        J_I[sid] = 3000 + int(sid)

    pi = {int(r["Task"]): int(r["Workstation"]) for _, r in tasks_df.iterrows()}
    D  = {int(r["Task"]): float(r["Duration"]) for _, r in tasks_df.iterrows()}

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
    Delta_s_s  = {k: d_s_s[k]  for k in d_s_s}

    return {
        "tasks_df": tasks_df, "ws_fixed_seq": ws_fixed_seq,
        "shelf_data": shelf_data, "agv_data": agv_data, "shelf_ids": shelf_ids,
        "J": J, "R": R, "S": S, "K": K,
        "J0": J0, "Jd": Jd, "J_I": J_I,
        "pi": pi, "D": D,
        "d_s_pi": d_s_pi, "d_pi_s": d_pi_s, "d_s_s": d_s_s,
        "Delta_s_pi": Delta_s_pi, "Delta_pi_s": Delta_pi_s, "Delta_s_s": Delta_s_s,
        "map_obj": map_obj
    }

# ---------- 由 timeline 生成占用表（含名义回退，统一键名） ----------
def build_tables(evaluator: RobustEvaluator, details: Dict):
    tl = list(details.get("timeline", []))
    if not tl:
        return {name: pd.DataFrame() for name in ["cells","agvs","workstations","shelves"]}

    # H：吸收所有可见的有限时间，含名义回退键
    H = _finite_max([
        details.get("C_agv_max", 0.0),
        details.get("C_task_max", 0.0),
        *[_G(r, "arrive_cell_act","arrCell_act", default=0.0) for r in tl],
        *[_G(r, "arrive_cell_nom","arrCell_nom", default=0.0) for r in tl],
        *[_G(r, "ws_end", default=0.0) for r in tl],
        *[_G(r, "arrival_ws","arrWS", default=0.0) for r in tl],
        *[_G(r, "pick_start","pick", default=0.0) for r in tl],
    ])

    # 1) AGV 占用（从离格子到回库到达，半开区间）
    rows_agv = []
    for rec in tl:
        r  = int(_G(rec, "AGV", finite_only=False) or rec.get("AGV"))
        j  = int(_G(rec, "Task", finite_only=False) or rec.get("Task"))
        c  = int(_G(rec, "Chain", finite_only=False) or rec.get("Chain"))
        ws = int(_G(rec, "WS", finite_only=False) or rec.get("WS"))

        dt1      = _G(rec, "dt1","dt1_eff","dt1_nom")
        dt2_eff  = _G(rec, "dt2_eff","dt2_nom", default=0.0)
        dt3_nom  = _G(rec, "dt3_nom","dt3_eff")

        t_pick   = _G(rec, "pick_start","pick")
        t_arr_ws = _G(rec, "arrival_ws","arrWS", default=_finite_or(t_pick, 0.0))
        t_ws_st  = _G(rec, "ws_start", default=t_arr_ws)
        t_ws_ed  = _G(rec, "ws_end",   default=t_ws_st)

        t_arr_shelf = _G(rec, "arrive_shelf","arr_shelf", default=_finite_or(t_pick - dt1 if (_finite(t_pick) and _finite(dt1)) else None, 0.0))
        # 回库到达：act → nom → ws_end + dt3_nom → H
        t_arr_cell = _finite_or(
            _G(rec, "arrive_cell_act","arrCell_act"),
            _G(rec, "arrive_cell_nom","arrCell_nom"),
            (t_ws_ed + dt3_nom) if (_finite(t_ws_ed) and _finite(dt3_nom)) else None,
            H
        )

        # 段起点：优先 arrive_shelf - dt1，否则 pick - dt1，再否则 0
        t0 = _finite_or(
            (t_arr_shelf - dt1) if (_finite(t_arr_shelf) and _finite(dt1)) else None,
            (t_pick - dt1) if (_finite(t_pick) and _finite(dt1)) else None,
            0.0
        )
        start = float(max(0.0, t0))
        end   = float(max(start, t_arr_cell))

        if _finite(t_pick) and _finite(dt2_eff) and _finite(t_arr_ws):
            diff = abs(t_arr_ws - (t_pick + dt2_eff))
            if diff > 1e-6:
                _warn(f"AGV{r}-Task{j}: arrival_ws != pick + dt2_eff (Δ={diff:.6f})")

        rows_agv.append({
            "AGV": r, "Task": j, "Chain": c, "WS": ws,
            "start": start, "end": end, "duration": max(0.0, end - start),
            "home_before": int(rec.get("home_before", 0)),
            "end_s": int(rec.get("end_s")) if _coerce_int(rec.get("end_s")) is not None else None,
            "arrive_shelf": float(t_arr_shelf) if _finite(t_arr_shelf) else None,
            "pick_start": float(t_pick) if _finite(t_pick) else None,
            "arrival_ws": float(t_arr_ws) if _finite(t_arr_ws) else None,
            "ws_start": float(t_ws_st) if _finite(t_ws_st) else None,
            "ws_end": float(t_ws_ed) if _finite(t_ws_ed) else None,
            "arrive_cell": float(t_arr_cell) if _finite(t_arr_cell) else None
        })
    df_agv = pd.DataFrame(rows_agv).sort_values(["AGV","start","Task"]).reset_index(drop=True)

    # 2) 工作站占用（闭开区间）
    rows_ws = []
    for rec in tl:
        ws = int(_G(rec, "WS", finite_only=False) or rec.get("WS"))
        j  = int(_G(rec, "Task", finite_only=False) or rec.get("Task"))
        c  = int(_G(rec, "Chain", finite_only=False) or rec.get("Chain"))
        r  = int(_G(rec, "AGV", finite_only=False) or rec.get("AGV"))

        t_arr_ws = _G(rec, "arrival_ws","arrWS")
        t_ws_st  = _G(rec, "ws_start", default=t_arr_ws)
        t_ws_ed  = _G(rec, "ws_end",   default=t_ws_st)

        rows_ws.append({
            "WS": ws, "Task": j, "Chain": c, "AGV": r,
            "start": float(t_ws_st) if _finite(t_ws_st) else None,
            "end": float(t_ws_ed) if _finite(t_ws_ed) else None,
            "process_dur": float(t_ws_ed - t_ws_st) if (_finite(t_ws_st) and _finite(t_ws_ed)) else None,
            "setup_dur_obs": float(t_ws_st - t_arr_ws) if (_finite(t_arr_ws) and _finite(t_ws_st)) else None,
            "arrival_ws": float(t_arr_ws) if _finite(t_arr_ws) else None
        })
    df_ws = pd.DataFrame(rows_ws).sort_values(["WS","start","Task"]).reset_index(drop=True)

    # 3) 货架 / 4) 格子占用（半开区间，含名义回退）
    shelf_init = {int(k): int(v) for k, v in getattr(evaluator, "shelf_init", {}).items()}

    # 按链聚合并按 pick_start（或 ws_start）排序
    by_chain: Dict[int, List[Dict]] = {}
    for rec in tl:
        by_chain.setdefault(int(rec["Chain"]), []).append(rec)
    for c in list(by_chain.keys()):
        def key_fn(r):
            return _finite_or(_G(r, "pick_start","pick"),
                              _G(r, "ws_start"),
                              1e6)
        by_chain[c].sort(key=key_fn)

    initial_occupancy = str(getattr(evaluator, "initial_occupancy", "hard")).lower()
    treat_initial_as_busy = (initial_occupancy == "hard")

    rows_shelf, rows_cell = [], []
    for c, seq in by_chain.items():
        if c not in shelf_init:
            _warn(f"Shelf chain {c} 缺少初始位置，跳过。")
            continue
        s_cur = int(shelf_init[c]); t_since = 0.0
        for idx, rec in enumerate(seq):
            j  = int(rec["Task"]); ws = int(rec["WS"])
            dt3_nom = _G(rec, "dt3_nom","dt3_eff")

            t_pick   = _G(rec, "pick_start","pick", default=t_since)
            t_arr_ws = _G(rec, "arrival_ws","arrWS", default=t_pick)
            t_ws_st  = _G(rec, "ws_start", default=t_arr_ws)
            t_ws_ed  = _G(rec, "ws_end",   default=t_ws_st)
            end_s    = int(rec["end_s"]) if _coerce_int(rec.get("end_s")) is not None else s_cur

            t_arr_cell = _finite_or(
                _G(rec, "arrive_cell_act","arrCell_act"),
                _G(rec, "arrive_cell_nom","arrCell_nom"),
                (t_ws_ed + dt3_nom) if (_finite(t_ws_ed) and _finite(dt3_nom)) else None,
                H
            )

            # cell: 在 home_before 驻留到 pick
            rows_cell.append({
                "cell": int(s_cur), "Shelf": int(c), "Task_leave": j,
                "start": float(t_since), "end": float(t_pick),
                "duration": max(0.0, float(t_pick - t_since)),
                "kind": "AT_CELL_INIT" if (idx==0 and not treat_initial_as_busy) else "AT_CELL",
            })
            # shelf: 去工位
            rows_shelf.append({
                "Shelf": int(c), "state": "TO_WS", "Task": j, "WS": ws,
                "from_cell": int(s_cur), "to_cell": None,
                "start": float(t_pick), "end": float(t_arr_ws),
                "duration": max(0.0, float(t_arr_ws - t_pick)),
            })
            # shelf: 在工位
            rows_shelf.append({
                "Shelf": int(c), "state": "IN_WS", "Task": j, "WS": ws,
                "from_cell": None, "to_cell": None,
                "start": float(t_ws_st), "end": float(t_ws_ed),
                "duration": max(0.0, float(t_ws_ed - t_ws_st)),
            })
            # shelf: 回库在路上
            rows_shelf.append({
                "Shelf": int(c), "state": "TO_CELL", "Task": j, "WS": ws,
                "from_cell": None, "to_cell": int(end_s),
                "start": float(t_ws_ed), "end": float(t_arr_cell),
                "duration": max(0.0, float(t_arr_cell - t_ws_ed)),
            })
            # cell: 回到 end_s 的驻留（直到下一次 pick或 Horizon）
            next_pick = _finite_or(
                _G(seq[idx+1], "pick_start","pick") if (idx+1) < len(seq) else None,
                H
            )
            rows_cell.append({
                "cell": int(end_s), "Shelf": int(c),
                "Task_leave": (int(seq[idx+1]["Task"]) if (idx+1) < len(seq) else None),
                "start": float(t_arr_cell), "end": float(next_pick),
                "duration": max(0.0, float(next_pick - t_arr_cell)),
                "kind": "AT_CELL",
            })

            s_cur = int(end_s)
            t_since = float(t_arr_cell)

    df_shelf = pd.DataFrame(rows_shelf).sort_values(["Shelf","start","Task"]).reset_index(drop=True)
    df_cell  = pd.DataFrame(rows_cell ).sort_values(["cell","start","Shelf"]).reset_index(drop=True)

    return {"cells": df_cell, "agvs": df_agv, "workstations": df_ws, "shelves": df_shelf}

# ---------- 最严格重叠审计（半开区间 [start, end)） ----------
def _interval_overlaps(df: pd.DataFrame, group_cols: List[str], start_col="start", end_col="end") -> pd.DataFrame:
    issues = []
    if df.empty:
        return pd.DataFrame(columns=["where","key","i","j","start_i","end_i","start_j","end_j","type"])
    g = df.groupby(group_cols, dropna=False, sort=False)
    for key, sub in g:
        sub = sub.copy()
        sub["_ord"] = range(len(sub))
        sub = sub.sort_values([start_col, end_col, "_ord"])
        prev_row = None
        prev_end = None
        for _, row in sub.iterrows():
            s = row[start_col]; e = row[end_col]
            if not (_finite(s) and _finite(e)):
                issues.append({"where": str(group_cols), "key": str(key),
                               "i": int(row.get("_ord", -1)), "j": None,
                               "start_i": s, "end_i": e,
                               "start_j": None, "end_j": None,
                               "type": "non_finite_or_missing"})
            elif float(e) < float(s) - EPS:
                issues.append({"where": str(group_cols), "key": str(key),
                               "i": int(row.get("_ord", -1)), "j": None,
                               "start_i": s, "end_i": e,
                               "start_j": None, "end_j": None,
                               "type": "negative_duration"})
            if prev_row is not None and _finite(prev_end) and _finite(s):
                # 半开区间重叠：s < prev_end - EPS
                if float(s) < float(prev_end) - EPS:
                    issues.append({"where": str(group_cols), "key": str(key),
                                   "i": int(prev_row.get("_ord", -1)), "j": int(row.get("_ord", -1)),
                                   "start_i": prev_row[start_col], "end_i": prev_row[end_col],
                                   "start_j": s, "end_j": e,
                                   "type": "overlap"})
            prev_row = row
            prev_end = e
    return pd.DataFrame(issues)

def audit_all_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    audits = {}
    audits["agvs"]         = _interval_overlaps(tables["agvs"], ["AGV"])
    audits["workstations"] = _interval_overlaps(tables["workstations"], ["WS"])
    audits["cells"]        = _interval_overlaps(tables["cells"], ["cell"])
    audits["shelves"]      = _interval_overlaps(tables["shelves"], ["Shelf"])
    return audits

# ---------- 主程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--initial-occupancy", default="hard", choices=["hard","soft"])
    ap.add_argument("--ignore-return-block", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--timestamp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    prefix = args.prefix
    gamma  = int(args.gamma)
    tag    = _sanitize_tag(args.tag)
    ts     = time.strftime("%Y%m%d-%H%M%S") if args.timestamp else ""
    suffix = ("_" + tag if tag else "") + ("_" + ts if ts else "")

    # 基础数据
    data = build_data(prefix)
    shelf_ids = data["shelf_ids"]; J = data["J"]; R = data["R"]; S = data["S"]

    # 读取结构（优先 bundle）
    routes, place, shelf_seq, used_bundle = load_struct(prefix, gamma, J, R, S, shelf_ids)
    if used_bundle:
        _info(f"Using bundle: {used_bundle}")
    else:
        _info("Using CSV exports (routes/place/shelf_seq)")

    # 评估器（贴近 MILP）
    evaluator = RobustEvaluator(
        J=J, R=R, S=S,
        pi=data["pi"], D=data["D"],
        J0=data["J0"], Jd=data["Jd"], J_I=data["J_I"],
        shelf_data=data["shelf_data"], agv_data=data["agv_data"],
        d_s_pi=data["d_s_pi"], d_pi_s=data["d_pi_s"], d_s_s=data["d_s_s"],
        Delta_s_pi=data["Delta_s_pi"], Delta_pi_s=data["Delta_pi_s"], Delta_s_s=data["Delta_s_s"],
        gamma=0,
        ws_fixed_seq=data["ws_fixed_seq"],
        ws_setup_rule="flat",       # 与 MILP 相同口径：每单 +2
        lock_place=True,            # 固定回库位
        detach_on_mismatch=False,
        initial_occupancy=args.initial_occupancy,
        ignore_return_block=bool(args.ignore_return_block),
    )

    # 评估 + 占用表（带名义回退消灭 inf/None）
    cmax, details = evaluator.evaluate(routes, shelf_seq, place, verbose=False)
    tables = build_tables(evaluator, details)

    # 导出（不覆盖命名）
    outdir = "solution_exports"; os.makedirs(outdir, exist_ok=True)
    name_base = f"{prefix}_occ_{{kind}}_gamma{gamma}{suffix}.csv"
    out_paths = {}
    for kind, df in tables.items():
        raw = os.path.join(outdir, name_base.format(kind=kind))
        path = _safe_path(raw, overwrite=args.overwrite)
        df.to_csv(path, index=False)
        out_paths[kind] = path
        _info(f"[EXPORT] {kind:12s} -> {path} (rows={len(df)})")

    # 最严格重叠审计
    audits = audit_all_tables(tables)
    audit_rows = []
    for k, df in audits.items():
        if df.empty:
            audit_rows.append({"table": k, "issues": 0})
        else:
            df_path = _safe_path(os.path.join(outdir, f"{prefix}_audit_{k}_gamma{gamma}{suffix}.csv"),
                                 overwrite=args.overwrite)
            df.to_csv(df_path, index=False)
            audit_rows.append({"table": k, "issues": int(len(df)), "csv": df_path})
            _warn(f"[AUDIT] {k} issues={len(df)} -> {df_path}")
    audit_df = pd.DataFrame(audit_rows)
    rpt_path = _safe_path(os.path.join(outdir, f"{prefix}_audit_summary_gamma{gamma}{suffix}.csv"),
                          overwrite=args.overwrite)
    audit_df.to_csv(rpt_path, index=False)
    _info(f"[AUDIT] summary -> {rpt_path}")

    # 简要汇报
    print(f"[OK] C_task_max={details.get('C_task_max')}, C_agv_max={details.get('C_agv_max')}, cmax_eval={cmax}")
    print("[DONE] 占用导出 + 严格审计 完成。")

if __name__ == "__main__":
    main()
