# simulate_and_export_occupancy.py
from __future__ import annotations
import os, math, argparse, ast, json, re, time
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ==== 你工程里已有的依赖 ====
from evaluator import RobustEvaluator                 # 评估器（已实现 timeline 等）
from map_generator import load_map_csv, create_map_from_components
from scenario import build_scenario_from_prefix
from utils import distance

# ----------------- 小工具 -----------------
EPS = 1e-9

def _parse_list_cell(s: str) -> List[int]:
    if isinstance(s, list):
        return [int(x) for x in s]
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [int(x) for x in v]
    except Exception:
        pass
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [int(x) for x in v]
    except Exception:
        pass
    try:
        return [int(x) for x in s.replace("[","").replace("]","").split(",") if x.strip()]
    except Exception:
        return []

def _finite_max(vals) -> float:
    m = 0.0
    for v in vals:
        try:
            x = float(v)
            if math.isfinite(x):
                m = max(m, x)
        except Exception:
            pass
    return m

def _warn(msg: str):
    print(f"[OCC][WARN] {msg}")

def _sanitize_tag(tag: str) -> str:
    tag = str(tag or "").strip()
    if not tag:
        return ""
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", tag)
    return tag.strip("-_")

def _safe_path(path: str, *, overwrite: bool) -> str:
    """若 path 已存在且不允许覆盖，则追加 _v1/_v2/..."""
    if overwrite or (not os.path.exists(path)):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        cand = f"{base}_v{k}{ext}"
        if not os.path.exists(cand):
            return cand
        k += 1

# ----------------- 从导出里恢复结构（routes/place/shelf_seq） -----------------
def load_struct_from_exports(prefix: str, gamma: int, J: set[int], R: Dict[int,int], S: Dict[int,tuple], shelf_ids: List[int]):
    outdir = "solution_exports"
    # routes
    f_routes_z = os.path.join(outdir, f"{prefix}_routes_by_z_gamma{gamma}.csv")
    f_taskseq  = os.path.join(outdir, f"{prefix}_taskSeq_gamma{gamma}.csv")
    routes: Dict[int, List[int]] = {int(r): [] for r in R}
    if os.path.exists(f_routes_z):
        df = pd.read_csv(f_routes_z)
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
        raise FileNotFoundError("未找到 routes_by_z 或 taskSeq 导出。")

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
            if j in J and s in Sset:
                place[j] = s
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

    return routes, place, shelf_seq

# ----------------- 构造 evaluator 所需的基本数据 -----------------
def build_data_for_evaluator(prefix: str):
    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"未找到 {tasks_csv}")

    # 地图
    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)
    map_obj = create_map_from_components(
        width=W, height=H, sp_indices=sp_indices, ws_indices=ws_indices,
        shelf_data=shelf_data, agv_data=agv_data
    )

    # 任务
    tasks_df = pd.read_csv(tasks_csv)
    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1

    # 固定工位顺序
    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )

    # 基本集合
    shelf_ids = sorted(shelf_data.keys())
    J = set(int(t) for t in tasks_df["Task"].tolist())
    R = map_obj.extract_AGVs()

    def idx2rc(idx: int): return divmod(int(idx) - 1, W)
    S = {int(sp): idx2rc(int(sp)) for sp in sp_indices}
    K = {i + 1: idx2rc(ws_indices[i]) for i in range(len(ws_indices))}

    # 虚拟点
    J0, Jd, J_I = {}, {}, {}
    for aid in R.keys():
        J0[aid] = 1000 + int(aid)
        Jd[aid] = 2000 + int(aid)
    for sid in shelf_ids:
        J_I[sid] = 3000 + int(sid)

    # pi, D
    pi = {int(r["Task"]): int(r["Workstation"]) for _, r in tasks_df.iterrows()}
    D  = {int(r["Task"]): float(r["Duration"]) for _, r in tasks_df.iterrows()}

    # 距离
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

    # Δ（这里用名义值；如需鲁棒段可在 evaluator.gamma>0 时使用）
    Delta_s_pi = {k: d_s_pi[k] for k in d_s_pi}
    Delta_pi_s = {k: d_pi_s[k] for k in d_pi_s}
    Delta_s_s  = {k: d_s_s[k]  for k in d_s_s}

    return {
        "tasks_df": tasks_df,
        "ws_fixed_seq": ws_fixed_seq,
        "shelf_data": shelf_data,
        "agv_data": agv_data,
        "shelf_ids": shelf_ids,
        "J": J, "R": R, "S": S, "K": K,
        "J0": J0, "Jd": Jd, "J_I": J_I,
        "pi": pi, "D": D,
        "d_s_pi": d_s_pi, "d_pi_s": d_pi_s, "d_s_s": d_s_s,
        "Delta_s_pi": Delta_s_pi, "Delta_pi_s": Delta_pi_s, "Delta_s_s": Delta_s_s,
        "map_obj": map_obj
    }

# ----------------- 从时间线构造四类占用表 -----------------
def build_occupancy_tables_from_timeline(evaluator: RobustEvaluator, details: Dict):
    tl = list(details.get("timeline", []))
    if not tl:
        return {name: pd.DataFrame() for name in ["cells","agvs","workstations","shelves"]}

    # horizon
    H = _finite_max([
        details.get("C_agv_max", 0.0), details.get("C_task_max", 0.0),
        *[r.get("arrive_cell_act", 0.0) for r in tl],
        *[r.get("ws_end", 0.0) for r in tl],
        *[r.get("arrival_ws", 0.0) for r in tl],
        *[r.get("pick_start", 0.0) for r in tl],
    ])

    # 1) AGV
    rows_agv = []
    for rec in tl:
        r, j, c, ws = int(rec["AGV"]), int(rec["Task"]), int(rec["Chain"]), int(rec["WS"])
        dt1 = float(rec["dt1"])
        t_arr_shelf = float(rec["arrive_shelf"])
        t_pick      = float(rec["pick_start"])
        t_arr_ws    = float(rec["arrival_ws"])
        t_ws_st     = float(rec["ws_start"])
        t_ws_ed     = float(rec["ws_end"])
        t_arr_cell  = float(rec["arrive_cell_act"])
        home_before = int(rec["home_before"])
        end_s       = int(rec["end_s"])

        t0 = t_arr_shelf - dt1
        start = float(t0)
        end   = float(t_arr_cell if math.isfinite(t_arr_cell) else H)

        if abs(t_arr_ws - (t_pick + float(rec["dt2_eff"]))) > 1e-6:
            _warn(f"AGV{r}-Task{j}: arrival_ws != pick + dt2_eff (Δ={t_arr_ws - (t_pick + float(rec['dt2_eff'])):.6f})")

        rows_agv.append({
            "AGV": r, "Task": j, "Chain": c, "WS": ws,
            "start": start, "end": end, "duration": max(0.0, end - start),
            "home_before": home_before, "end_s": end_s,
            "arrive_shelf": t_arr_shelf, "pick_start": t_pick,
            "arrival_ws": t_arr_ws, "ws_start": t_ws_st, "ws_end": t_ws_ed,
            "arrive_cell_act": t_arr_cell
        })
    df_agv = pd.DataFrame(rows_agv).sort_values(["AGV","start","Task"]).reset_index(drop=True)

    # 2) WS
    rows_ws = []
    for rec in tl:
        ws, j, c, r = int(rec["WS"]), int(rec["Task"]), int(rec["Chain"]), int(rec["AGV"])
        t_ws_st = float(rec["ws_start"]); t_ws_ed = float(rec["ws_end"]); t_arr_ws = float(rec["arrival_ws"])
        rows_ws.append({
            "WS": ws, "Task": j, "Chain": c, "AGV": r,
            "start": t_ws_st, "end": t_ws_ed,
            "process_dur": max(0.0, t_ws_ed - t_ws_st),
            "setup_dur_obs": max(0.0, t_ws_st - t_arr_ws),
            "arrival_ws": t_arr_ws
        })
    df_ws = pd.DataFrame(rows_ws).sort_values(["WS","start","Task"]).reset_index(drop=True)

    # 3) Shelves & 4) Cells
    shelf_init = {int(k): int(v) for k, v in getattr(evaluator, "shelf_init", {}).items()}
    by_chain: Dict[int, List[Dict]] = {}
    for rec in tl:
        by_chain.setdefault(int(rec["Chain"]), []).append(rec)
    for c in list(by_chain.keys()):
        by_chain[c].sort(key=lambda r: float(r["pick_start"]))

    initial_occupancy = getattr(evaluator, "initial_occupancy", "hard")
    treat_initial_as_busy = (str(initial_occupancy).lower() == "hard")

    rows_shelf, rows_cell = [], []
    for c, seq in by_chain.items():
        if c not in shelf_init:
            _warn(f"Shelf chain {c} 缺少初始位置，跳过。")
            continue
        s_cur = int(shelf_init[c]); t_since = 0.0
        for idx, rec in enumerate(seq):
            j = int(rec["Task"]); ws = int(rec["WS"])
            t_pick   = float(rec["pick_start"])
            t_arr_ws = float(rec["arrival_ws"])
            t_ws_st  = float(rec["ws_start"])
            t_ws_ed  = float(rec["ws_end"])
            end_s    = int(rec["end_s"])
            t_arr_cell = float(rec["arrive_cell_act"])

            # cell: 在 home_before 驻留到 pick
            rows_cell.append({
                "cell": int(s_cur), "Shelf": int(c), "Task_leave": j,
                "start": float(t_since), "end": float(t_pick),
                "duration": max(0.0, float(t_pick) - float(t_since)),
                "kind": "AT_CELL_INIT" if (idx==0 and not treat_initial_as_busy) else "AT_CELL",
            })

            # shelf: 去工位
            rows_shelf.append({
                "Shelf": int(c), "state": "TO_WS", "Task": j, "WS": ws,
                "from_cell": int(s_cur), "to_cell": None,
                "start": float(t_pick), "end": float(t_arr_ws),
                "duration": max(0.0, float(t_arr_ws) - float(t_pick)),
            })
            # shelf: 在工位
            rows_shelf.append({
                "Shelf": int(c), "state": "IN_WS", "Task": j, "WS": ws,
                "from_cell": None, "to_cell": None,
                "start": float(t_ws_st), "end": float(t_ws_ed),
                "duration": max(0.0, float(t_ws_ed) - float(t_ws_st)),
            })
            # shelf: 回库在路上
            rows_shelf.append({
                "Shelf": int(c), "state": "TO_CELL", "Task": j, "WS": ws,
                "from_cell": None, "to_cell": int(end_s),
                "start": float(t_ws_ed), "end": float(t_arr_cell if math.isfinite(t_arr_cell) else H),
                "duration": max(0.0, float((t_arr_cell if math.isfinite(t_arr_cell) else H) - t_ws_ed)),
            })

            # cell: 回到 end_s 的驻留（直到下一次 pick或 Horizon）
            next_pick = float(seq[idx+1]["pick_start"]) if (idx+1) < len(seq) else H
            rows_cell.append({
                "cell": int(end_s), "Shelf": int(c), "Task_leave": (seq[idx+1]["Task"] if (idx+1) < len(seq) else None),
                "start": float(t_arr_cell if math.isfinite(t_arr_cell) else H),
                "end": float(next_pick if math.isfinite(next_pick) else H),
                "duration": max(0.0, float((next_pick if math.isfinite(next_pick) else H) - (t_arr_cell if math.isfinite(t_arr_cell) else H))),
                "kind": "AT_CELL",
            })

            s_cur = int(end_s)
            t_since = float(t_arr_cell if math.isfinite(t_arr_cell) else H)

    df_shelf = pd.DataFrame(rows_shelf).sort_values(["Shelf","start","Task"]).reset_index(drop=True)
    df_cell  = pd.DataFrame(rows_cell ).sort_values(["cell","start","Shelf"]).reset_index(drop=True)

    return {
        "cells": df_cell,
        "agvs": df_agv,
        "workstations": df_ws,
        "shelves": df_shelf
    }

# ----------------- 主程：读导出 -> 构建 evaluator -> 评估 -> 导出 4 表（不覆盖命名） -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="场景前缀，如 demo01")
    ap.add_argument("--gamma", type=int, default=0, help="读取的 MILP/ALNS gamma（默认0）")
    # 评估口径开关（不改 main.py；单独控制）
    ap.add_argument("--initial-occupancy", default="hard", choices=["hard","soft"], help="初始格子的口径（仅影响占用导出第一段的标记）")
    ap.add_argument("--ignore-return-block", action="store_true", help="评估口径：回库不受占位阻塞")
    # === 新增：不覆盖命名控制 ===
    ap.add_argument("--tag", default="", help="给输出文件追加自定义标记（如 run1 / expB）。会被清洗为 [A-Za-z0-9_.-]")
    ap.add_argument("--timestamp", action="store_true", help="在文件名后追加时间戳（YYYYMMDD-HHMMSS）以避免覆盖")
    ap.add_argument("--overwrite", action="store_true", help="允许覆盖同名文件（默认不覆盖）")
    args = ap.parse_args()

    prefix = args.prefix
    gamma  = int(args.gamma)
    tag    = _sanitize_tag(args.tag)
    ts     = time.strftime("%Y%m%d-%H%M%S") if args.timestamp else ""
    suffix = ""
    if tag:
        suffix += f"_{tag}"
    if ts:
        suffix += f"_{ts}"

    # 1) 基本数据
    data = build_data_for_evaluator(prefix)
    shelf_ids = data["shelf_ids"]; J = data["J"]; R = data["R"]; S = data["S"]

    # 2) 读取结构（routes/place/shelf_seq）来自 solution_exports
    routes, place, shelf_seq = load_struct_from_exports(prefix, gamma, J, R, S, shelf_ids)

    # 3) 构造 evaluator（保持与 MILP 贴近：flat + lock_place）
    evaluator = RobustEvaluator(
        J=J, R=R, S=S,
        pi=data["pi"], D=data["D"],
        J0=data["J0"], Jd=data["Jd"], J_I=data["J_I"],
        shelf_data=data["shelf_data"], agv_data=data["agv_data"],
        d_s_pi=data["d_s_pi"], d_pi_s=data["d_pi_s"], d_s_s=data["d_s_s"],
        Delta_s_pi=data["Delta_s_pi"], Delta_pi_s=data["Delta_pi_s"], Delta_s_s=data["Delta_s_s"],
        gamma=0,
        ws_fixed_seq=data["ws_fixed_seq"],
        ws_setup_rule="flat",       # 与 MILP 口径一致：每单 +2
        lock_place=True,            # 固定回库位（采用文件中的 x）
        detach_on_mismatch=False,
        initial_occupancy=args.initial_occupancy,
        ignore_return_block=bool(args.ignore_return_block),
    )

    # 4) 评估 + 构建四张表
    cmax, details = evaluator.evaluate(routes, shelf_seq, place, verbose=False)
    tables = build_occupancy_tables_from_timeline(evaluator, details)

    # 5) 导出（不覆盖命名）
    outdir = "solution_exports"
    os.makedirs(outdir, exist_ok=True)

    name_base = f"{prefix}_occ_{{kind}}_gamma{gamma}{suffix}.csv"
    out_paths = {}
    for kind, df in tables.items():
        raw = os.path.join(outdir, name_base.format(kind=kind))
        path = _safe_path(raw, overwrite=args.overwrite)
        df.to_csv(path, index=False)
        out_paths[kind] = path
        print(f"[EXPORT][occ] {kind:12s} -> {path} (rows={len(df)})")

    # 简单汇报
    print(f"[OK] Horizon C_task_max={details.get('C_task_max')}  C_agv_max={details.get('C_agv_max')}")
    print("[DONE] 占用导出完成。")

if __name__ == "__main__":
    main()
