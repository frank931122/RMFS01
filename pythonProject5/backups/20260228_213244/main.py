# main.py
# main.py
import os
import argparse

from collections import defaultdict
import pandas as pd
import json
import math
import time
from typing import Dict, List, Optional, Set
from heuristic_init import build_initial_solution_basic
from solution_exports.solution_structures import InitialSolution
from warmstart_checks import check_v_consistency
from map_generator import load_map_csv, create_map_from_components
from movement_manager import MovementManager
from time_manager import TimeManager
from scenario import build_scenario_from_prefix, print_scenario_brief
from utils import distance
from evaluator import RobustEvaluator
from alns_min import (
    alns_minimize,
    build_feasible_initial_solution,
    build_full_coverage_seed,
    build_ws_round_robin_seed,
    build_exact_feasible_seed,
    build_safe_chain_ws_seed,
    repair_shelf_seq_by_chain_violations,
    repair_chain_violations_by_route_relink,
)
from export_eval_diag import export_evaluator_diagnostics
import sys
import subprocess
import re
from pathlib import Path

class EvalProfiler:
    """
    包一层 evaluator，统计 evaluate() 被调用次数、总耗时、平均耗时。
    这样你能立刻判断瓶颈是：evaluate 太慢 还是 调用次数爆炸。
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = 0
        self.total_sec = 0.0

    def evaluate(self, routes, shelf_seq, place, verbose: bool = False):
        t0 = time.perf_counter()
        obj, diag = self.evaluator.evaluate(routes, shelf_seq, place, verbose=verbose)
        dt = time.perf_counter() - t0
        self.calls += 1
        self.total_sec += dt
        return obj, diag

    def __getattr__(self, name: str):
        # 让 alns_min 内部若访问 evaluator.gamma / evaluator.J 等，也能透传
        return getattr(self.evaluator, name)

    def report(self, tag: str = "[Profiler]"):
        avg_ms = (self.total_sec / max(1, self.calls)) * 1000.0
        print(f"{tag} evaluate calls={self.calls} | total={self.total_sec:.3f}s | avg={avg_ms:.3f}ms")
def print_solution_full_from_diag(
    *,
    title: str,
    gamma_eval: int,
    routes: dict[int, list[int]],
    shelf_seq: dict[int, list[int]],
    place: dict[int, int],
    makespan: float,
    diag: dict,
) -> None:
    """
    把 evaluator 的 diag（包含 timeline/pq/v_arcs 等）按“人能读懂”的形式打印出来。
    用于：同一套解在 eval_gamma=0 和 eval_gamma=1 下的对照输出。
    """
    def _to_float(v, default=float("nan")) -> float:
        try:
            if v is None:
                return default
            return float(v)
        except Exception:
            return default

    def _to_int(v, default=10**9) -> int:
        try:
            if v is None:
                return default
            return int(v)
        except Exception:
            return default

    def _get(rec: dict, k: str):
        try:
            return rec.get(k)
        except Exception:
            return None

    print("\n" + "-" * 70)
    print(f"[{title}] evaluator γ={int(gamma_eval)} | makespan={float(makespan):.2f}")

    # Routes
    print("\n  Routes:")
    for r in sorted(routes):
        print(f"    AGV {int(r)} route: {routes[int(r)]}")

    # Shelf sequences
    print("\n  Shelf sequences:")
    for c in sorted(shelf_seq):
        print(f"    Shelf {int(c)}: {shelf_seq[int(c)]}")

    # Placement
    print("\n  Placement (task -> shelf):")
    for j in sorted(place):
        print(f"    Task {int(j)}: s={int(place[int(j)])}")

    # feasible / penalties
    if isinstance(diag, dict):
        feasible = diag.get("feasible", None)
        penalties = diag.get("penalties", None)
        if feasible is not None or penalties is not None:
            print(f"\n  feasible={feasible} | penalties={penalties}")

    # p/q
    p_raw = (diag.get("p", {}) or {}) if isinstance(diag, dict) else {}
    q_raw = (diag.get("q", {}) or {}) if isinstance(diag, dict) else {}
    end_final = (diag.get("end_shelf_final", {}) or {}) if isinstance(diag, dict) else {}

    # 兼容 key 可能是 str
    def _get_map_val(m: dict, key_int: int):
        if key_int in m:
            return m[key_int]
        sk = str(key_int)
        if sk in m:
            return m[sk]
        return None

    if p_raw or q_raw:
        print("\n  p/q by task:")
        all_keys = set()
        for k in p_raw.keys():
            all_keys.add(_to_int(k))
        for k in q_raw.keys():
            all_keys.add(_to_int(k))
        for j in sorted(all_keys):
            pj = _to_float(_get_map_val(p_raw, j))
            qj = _to_float(_get_map_val(q_raw, j))
            es = _get_map_val(end_final, j)
            print(f"    Task {j}: p={pj:.2f} | q={qj:.2f} | end_s={es}")

    # V_arcs
    V_arcs = (diag.get("V_arcs", []) or []) if isinstance(diag, dict) else []
    if V_arcs:
        print("\n  v[i,j,s,s’] = 1 (derived by evaluator):")
        for arc in V_arcs:
            try:
                i, j, s, sp = arc
                print(f"    v[{int(i)},{int(j)},{int(s)},{int(sp)}] = 1")
            except Exception:
                continue

    # Timeline
    timeline = (diag.get("timeline", []) or []) if isinstance(diag, dict) else []
    if timeline:
        print("\n  --- Timeline (sorted by ws_start) ---")
        timeline_sorted = sorted(
            timeline,
            key=lambda rec: (_to_float(_get(rec, "ws_start")), _to_int(_get(rec, "Task")))
        )

        for rec in timeline_sorted:
            agv = _to_int(_get(rec, "AGV"), default=-1)
            task = _to_int(_get(rec, "Task"), default=-1)
            chain = _to_int(_get(rec, "Chain"), default=-1)
            ws = _to_int(_get(rec, "WS"), default=-1)

            hb = _get(rec, "home_before")
            end_s = _get(rec, "end_s")

            print(
                "    "
                f"AGV{agv} T{task} C{chain} WS{ws} | "
                f"s0={hb} "
                f"dt1={_to_float(_get(rec,'dt1')):.2f} arr_shelf={_to_float(_get(rec,'arrive_shelf')):.2f} pick={_to_float(_get(rec,'pick_start')):.2f} | "
                f"dt2={_to_float(_get(rec,'dt2_eff')):.2f} (nom {_to_float(_get(rec,'dt2_nom')):.2f}) arrWS={_to_float(_get(rec,'arrival_ws')):.2f} | "
                f"ws: {_to_float(_get(rec,'ws_start')):.2f}->{_to_float(_get(rec,'ws_end')):.2f} | "
                f"end_s={end_s} dt3={_to_float(_get(rec,'dt3_eff')):.2f} (nom {_to_float(_get(rec,'dt3_nom')):.2f}) "
                f"arrCell_nom={_to_float(_get(rec,'arrive_cell_nom')):.2f} arrCell_act={_to_float(_get(rec,'arrive_cell_act')):.2f}"
            )

            # 如果 evaluator 输出里包含 layer 0 / layer G 的对照字段，就额外打印一行（这对定位你说的“γ>=1 先跑 γ=0 基准”非常关键）
            has_layer = any(
                _get(rec, k) is not None
                for k in [
                    "pick_start_0", "pick_start_G",
                    "ws_start_0", "ws_end_0",
                    "ws_start_G", "ws_end_G",
                    "arrive_cell_act_0", "arrive_cell_act_G",
                ]
            )
            if has_layer:
                print(
                    "        [Layer] "
                    f"pick0={_to_float(_get(rec,'pick_start_0')):.2f} pickG={_to_float(_get(rec,'pick_start_G')):.2f} | "
                    f"ws0={_to_float(_get(rec,'ws_start_0')):.2f}->{_to_float(_get(rec,'ws_end_0')):.2f} "
                    f"wsG={_to_float(_get(rec,'ws_start_G')):.2f}->{_to_float(_get(rec,'ws_end_G')):.2f} | "
                    f"cell_act0={_to_float(_get(rec,'arrive_cell_act_0')):.2f} cell_actG={_to_float(_get(rec,'arrive_cell_act_G')):.2f}"
                )

            # LB 信息（如果有）
            if _get(rec, "place_lb") is not None or _to_float(_get(rec, "place_wait_due_to_lb")) > 1e-9:
                print(
                    "        [LB] "
                    f"place_lb={_get(rec,'place_lb')} wait_due_to_lb={_to_float(_get(rec,'place_wait_due_to_lb')):.2f}"
                )

    print("-" * 70 + "\n")

def align_test_milp_vs_evaluator(
    *,
    evaluator,
    shelf_seq,
    routes_by_agv,
    x_vars,
    milp_obj,
):
    """
    目标：验证 evaluator(routes_by_agv, shelf_seq, place_from_x) 是否等于 MILP 的目标值
    - routes_by_agv: {agv_id: [task,...], ...}（用 MILP 输出那套）
    - x_vars: Gurobi 的 x 变量容器（支持 x[j,s] 或 x[j][s]）
    """
    # 1) 从 MILP 的 x[j,s] 抽取 place：task -> end_shelf_cell
    place = {}
    J = [int(j) for j in evaluator.J]
    S = [int(s) for s in evaluator.S]

    for j in J:
        chosen_s = None
        best_val = -1.0
        for s in S:
            v = None
            # 兼容两种索引：x[j,s] 或 x[j][s]
            try:
                v = x_vars[j, s]
            except Exception:
                try:
                    v = x_vars[j][s]
                except Exception:
                    v = None

            if v is None:
                continue

            try:
                val = float(v.X)  # Gurobi Var
            except Exception:
                val = float(v)    # 以防你存的是数值

            if val > best_val:
                best_val = val
                chosen_s = s

        if chosen_s is not None and best_val > 0.5:
            place[j] = int(chosen_s)

    missing = [j for j in J if j not in place]
    if missing:
        print(f"[ALIGN-TEST] WARN: 以下任务没有从 x[j,s] 解析出回库位: {missing}")

    # 2) 用 evaluator 复算 MILP 的 routes + place
    routes_chk = {int(r): [int(t) for t in seq] for r, seq in routes_by_agv.items()}
    obj_eval, diag = evaluator.evaluate(routes_chk, shelf_seq, place)

    milp_obj = float(milp_obj)
    obj_eval = float(obj_eval)
    diff = obj_eval - milp_obj

    print(f"[ALIGN-TEST] MILP obj={milp_obj:.2f} | evaluator obj={obj_eval:.2f} | diff={diff:+.2f}")

    # 3) 如果不一致，给出最有用的下一步线索：打印 p/q（如果 evaluator 提供）
    if abs(diff) > 1e-6 and isinstance(diag, dict):
        p = diag.get("p", {}) or {}
        q = diag.get("q", {}) or {}
        print("[ALIGN-TEST] evaluator 的 p/q（便于对照 MILP 导出的 p_q_times CSV）:")
        for j in sorted(J):
            if j in p and j in q:
                print(f"  Task {j}: p={float(p[j]):.2f}, q={float(q[j]):.2f}, end_s={place.get(j)}")

    return diff
def compare_pq(
    milp_pq: dict[int, tuple[float, float]],
    eval_pq: dict[int, tuple[float, float]],
    eps: float = 1e-4,
    topk: int = 30
) -> None:
    """
    milp_pq / eval_pq: {task_id: (p, q)}
    输出 p/q 不一致的任务，按 |dq| 从大到小排序，便于定位“哪一步把时间推迟了”。
    """
    keys = sorted(set(milp_pq.keys()) | set(eval_pq.keys()))
    rows = []
    for j in keys:
        if j not in milp_pq or j not in eval_pq:
            rows.append((j, None, None, None, None, None, None))
            continue
        pm, qm = milp_pq[j]
        pe, qe = eval_pq[j]
        dp, dq = pe - pm, qe - qm
        if abs(dp) > eps or abs(dq) > eps:
            rows.append((j, pm, qm, pe, qe, dp, dq))

    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: abs(r[6]), reverse=True)

    print("\n[ALIGN-DETAIL] Top diffs (sorted by |dq|):")
    print(" task |    p_milp    q_milp |    p_eval    q_eval |     dp     dq")
    print("------+---------------------+---------------------+--------------")
    for (j, pm, qm, pe, qe, dp, dq) in rows[:topk]:
        print(f"{j:>5} | {pm:>9.2f} {qm:>9.2f} | {pe:>9.2f} {qe:>9.2f} | {dp:>6.2f} {dq:>6.2f}")

    if not rows:
        print("[ALIGN-DETAIL] No diffs found within eps.")

def compare_pick_start0_milp_vs_eval(
    *,
    prefix: str,
    gamma: int,
    evaluator_diag: dict,
    shelf_seq: dict[int, list[int]],
    place: dict[int, int],
    shelf_data: dict[int, int] | None,
    d_s_pi: dict[tuple[int, int], float] | None,
    D_setup: float = 2.0,
    outdir: str = "solution_exports",
    topk: int = 30,
):
    """
    用 MILP 的 layer=0 的 p 反推出 pick_start_0：
        pick0_milp(j) = p_milp(j, layer=0) - d(home_before(j), j) - D_setup
    再与 evaluator timeline 的 pick_start_0 对比。

    注意：这里 file_gamma 用的是“当前这次优化的 gamma”，但读取的是 layer_gamma=0。
    """

    if shelf_data is None or d_s_pi is None:
        print("[PICK0-CHECK] missing shelf_data or d_s_pi; skip.")
        return

    # ✅ 关键修正：读 “当前 gamma 的文件”，但筛 layer=0
    milp_pq0 = load_milp_pq_from_csv(prefix=prefix, file_gamma=gamma, layer_gamma=0, outdir=outdir)
    if not milp_pq0:
        print("[PICK0-CHECK] missing MILP p/q for layer=0 in CSV.")
        return

    # home_before：链首用 shelf_init；链内用 place[prev]
    J_set = set(int(x) for x in place.keys())
    home_before: dict[int, int] = {}
    for c, seq in shelf_seq.items():
        cc = int(c)
        seq_clean = [int(j) for j in (seq or []) if int(j) in J_set]
        if not seq_clean:
            continue

        s_init = shelf_data.get(cc)
        if s_init is None:
            continue
        s_init = int(s_init)

        home_before[seq_clean[0]] = s_init
        for a, b in zip(seq_clean[:-1], seq_clean[1:]):
            home_before[int(b)] = int(place.get(int(a), s_init))

    # evaluator 的 pick_start_0 从 timeline 拿
    timeline = evaluator_diag.get("timeline", []) or []
    pick0_eval: dict[int, float] = {}
    for rec in timeline:
        try:
            j = int(rec.get("Task"))
        except Exception:
            continue
        if rec.get("pick_start_0") is not None:
            try:
                pick0_eval[j] = float(rec["pick_start_0"])
            except Exception:
                pass

    rows = []
    skipped = 0
    for j, (p0, _) in milp_pq0.items():
        j = int(j)
        hb = home_before.get(j, None)
        if hb is None:
            skipped += 1
            continue

        d = d_s_pi.get((int(hb), int(j)), None)
        if d is None:
            skipped += 1
            continue

        pick0_milp = float(p0) - float(d) - float(D_setup)
        pe = pick0_eval.get(j, None)
        if pe is None:
            skipped += 1
            continue

        diff = float(pe) - float(pick0_milp)
        rows.append((j, hb, float(d), float(pick0_milp), float(pe), float(diff)))

    rows.sort(key=lambda x: abs(x[5]), reverse=True)

    print("\n[PICK0-CHECK] Top diffs (eval_pick0 - milp_pick0):")
    print(" task | home_before |   d  | pick0_milp | pick0_eval |  diff")
    print("------+------------+------+-----------+-----------+-------")
    for (j, hb, d, pm, pe, df) in rows[:topk]:
        print(f"{j:>5} | {hb:>10} | {d:>4.1f} | {pm:>9.2f} | {pe:>9.2f} | {df:>+6.2f}")

    if not rows:
        print("[PICK0-CHECK] No comparable tasks found.")
    if skipped:
        print(f"[PICK0-CHECK] NOTE: skipped {skipped} tasks due to missing home_before / distance / pick_start_0.")


def load_milp_pq_from_csv(
    prefix: str,
    file_gamma: int,
    layer_gamma: int | None = None,
    outdir: str = "solution_exports"
) -> dict[int, tuple[float, float]]:
    """
    读取你导出的 p/q CSV，构造 {task_id: (p, q)}。

    - file_gamma: 用来选择文件名后缀，比如 *_gamma1.csv
    - layer_gamma: 若文件里有 'gamma' 列（allGamma 长表），则进一步筛选某一层 gamma（例如 0/1/2）
                  若为 None，则不筛选（直接整表读）
    """
    cand_paths = [
        os.path.join(outdir, f"{prefix}_p_q_times_allGamma_gamma{file_gamma}.csv"),
        os.path.join(outdir, f"{prefix}_p_q_times_gamma{file_gamma}.csv"),
        os.path.join(outdir, f"{prefix}_p_q_times_allGamma.csv"),
    ]
    path = next((p for p in cand_paths if os.path.exists(p)), None)
    if path is None:
        print(f"[ALIGN-DETAIL] WARN: 找不到 MILP 的 p/q CSV（尝试过：{cand_paths}）")
        return {}

    df = pd.read_csv(path)

    def pick_col(cands: list[str]) -> str | None:
        cols = {c.lower(): c for c in df.columns}
        for key in cands:
            if key in cols:
                return cols[key]
        return None

    col_task = pick_col(["task", "task_id", "j", "job"])
    col_p = pick_col(["p", "p_time", "start", "start_time"])
    col_q = pick_col(["q", "q_time", "end", "end_time"])
    col_gamma = pick_col(["gamma", "eval_gamma", "g"])

    if col_task is None or col_p is None or col_q is None:
        print(f"[ALIGN-DETAIL] WARN: p/q CSV 列名不匹配：{list(df.columns)}")
        return {}

    # 如果是 allGamma 文件且你指定 layer_gamma，就筛选层
    if (layer_gamma is not None) and (col_gamma is not None):
        try:
            df = df[df[col_gamma].astype(int) == int(layer_gamma)]
        except Exception:
            pass

    out: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        try:
            j = int(row[col_task])
            p = float(row[col_p])
            q = float(row[col_q])
        except Exception:
            continue
        out[j] = (p, q)
    return out


def align_test_from_bundle_json(
    prefix: str,
    gamma: int,               # bundle 文件名里的 source_gamma
    evaluator: RobustEvaluator,
    outdir: str = "solution_exports",
    shelf_data: dict[int, int] | None = None,
    d_s_pi: dict[tuple[int, int], float] | None = None,
    export_eval_diag: bool = False,
    export_tag: str = "milp_bundle",
    print_full: bool = False,
    print_tag: str | None = None,
    verbose_eval: bool = True,
    return_diag: bool = False,
):
    """
    读取 MILP bundle -> evaluator 复算 -> 对齐检查
    新增能力：
      - print_full=True：用“可读”方式打印完整 timeline/pq 等（你要的那种）
      - return_diag=True：返回 (diff, eval_ms, milp_cmax, diag, routes, shelf_seq, place)，方便后续做 milp-eval-lock
    """
    path = os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json")
    if not os.path.exists(path):
        print(f"[ALIGN-TEST] 找不到 bundle 文件：{path}")
        return None if return_diag else None

    with open(path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    routes = {int(r): [int(t) for t in seq] for r, seq in (bundle.get("routes") or {}).items()}
    place = {int(j): int(s) for j, s in (bundle.get("place") or {}).items()}
    shelf_seq = {int(c): [int(t) for t in seq] for c, seq in (bundle.get("shelf_seq") or {}).items()}

    milp_cmax = float(bundle.get("cmax", float("nan")))
    eval_gamma = int(getattr(evaluator, "gamma", gamma))

    obj_eval, diag = evaluator.evaluate(routes, shelf_seq, place, verbose=bool(verbose_eval))
    obj_eval_f = float(obj_eval)
    diff = obj_eval_f - float(milp_cmax)

    print(f"[ALIGN-TEST] (bundle) MILP cmax={milp_cmax:.2f} | evaluator(gamma={eval_gamma})={obj_eval_f:.2f} | diff={diff:+.2f}")

    # ====== 导出 evaluator 的全量中间量 ======
    if export_eval_diag and isinstance(diag, dict):
        try:
            export_evaluator_diagnostics(
                prefix=prefix,
                export_tag=export_tag,
                source_gamma=int(gamma),
                eval_gamma=int(eval_gamma),
                routes=routes,
                shelf_seq=shelf_seq,
                place=place,
                milp_cmax=float(milp_cmax),
                eval_cmax=float(obj_eval_f),
                diag=diag,
                outdir=outdir,
            )
        except Exception as e:
            print(f"[EXPORT-EVAL] failed: {type(e).__name__}: {e}")

    # ====== 你要的“完整可读打印” ======
    if print_full and isinstance(diag, dict):
        title = print_tag or f"{export_tag} srcG{gamma} evalG{eval_gamma}"
        print_solution_full_from_diag(
            title=title,
            gamma_eval=int(eval_gamma),
            routes=routes,
            shelf_seq=shelf_seq,
            place=place,
            makespan=float(obj_eval_f),
            diag=diag,
        )

    # ====== p/q 对齐细化 ======
    milp_pq = load_milp_pq_from_csv(prefix=prefix, file_gamma=gamma, layer_gamma=eval_gamma, outdir=outdir)

    eval_pq: dict[int, tuple[float, float]] = {}
    if isinstance(diag, dict):
        p_raw = diag.get("p", {}) or {}
        q_raw = diag.get("q", {}) or {}
        try:
            p_map = {int(k): float(v) for k, v in p_raw.items()}
            q_map = {int(k): float(v) for k, v in q_raw.items()}
            J_real = set(int(j) for j in evaluator.J)
            for j in sorted(J_real):
                if j in p_map and j in q_map:
                    eval_pq[j] = (p_map[j], q_map[j])
        except Exception:
            pass

    if milp_pq and eval_pq:
        compare_pq(milp_pq=milp_pq, eval_pq=eval_pq, eps=1e-6, topk=30)

    # ====== pick_start_0 对齐诊断 ======
    if isinstance(diag, dict):
        sd = shelf_data if shelf_data is not None else getattr(evaluator, "shelf_data", None)
        dsp = d_s_pi if d_s_pi is not None else getattr(evaluator, "d_s_pi", None)

        try:
            compare_pick_start0_milp_vs_eval(
                prefix=prefix,
                gamma=gamma,
                evaluator_diag=diag,
                shelf_seq=shelf_seq,
                place=place,
                shelf_data=sd,
                d_s_pi=dsp,
                D_setup=2.0,
                outdir=outdir,
                topk=30,
            )
        except Exception as e:
            print(f"[PICK0-CHECK] skipped due to error: {type(e).__name__}: {e}")

    # ====== 保留你原来的诊断信息 ======
    if isinstance(diag, dict):
        print("[ALIGN-TEST] feasible =", diag.get("feasible"))
        print("[ALIGN-TEST] penalties =", diag.get("penalties"))
        print("[ALIGN-TEST] unscheduled_tasks =", diag.get("unscheduled_tasks"))
        print("[ALIGN-TEST] C_task_max =", diag.get("C_task_max"))
        print("[ALIGN-TEST] routes(bundle) sizes =", {int(r): len(seq) for r, seq in routes.items()})
        print("[ALIGN-TEST] shelf_seq(bundle) =", shelf_seq)
        try:
            print("[ALIGN-TEST] place(bundle sample) =", list(place.items())[:10])
        except Exception:
            pass

        q_eval = diag.get("q", {}) or {}
        try:
            q2 = {int(k): float(v) for k, v in q_eval.items()}
            if q2:
                j_star = max(q2, key=lambda jj: q2[jj])
                print(f"[ALIGN-TEST] evaluator 的瓶颈任务：Task {int(j_star)}  q={float(q2[j_star]):.2f}")
        except Exception:
            pass

    if return_diag:
        return (diff, obj_eval_f, milp_cmax, diag, routes, shelf_seq, place)

    return diff


# ====== 构造优化需要的数据结构 ======
def build_task_structures(tasks_df: pd.DataFrame,
                          agv_data: dict[int, int],
                          shelf_ids: list[int]):
    need = {"Task", "Shelf", "Workstation", "Duration"}
    if not need.issubset(tasks_df.columns):
        raise ValueError(f"tasks_df 缺少列：{need - set(tasks_df.columns)}")
    if tasks_df.isna().any().any():
        raise ValueError("tasks_df 存在 NaN，请先清洗。")

    tasks, task_shelf_mapping, J = {}, {}, set()
    shelf_usage = {sid: [] for sid in shelf_ids}
    for _, r in tasks_df.sort_values("Task").iterrows():
        tid = int(r["Task"]); sid = int(r["Shelf"]); ws = int(r["Workstation"]); dur = float(r["Duration"])
        tasks[tid] = (ws, dur, None, None)
        task_shelf_mapping[tid] = sid
        J.add(tid)
        shelf_usage[sid].append(tid)

    # 虚拟任务
    J0, Jd = {}, {}
    for aid in agv_data:
        J0[aid] = 1000 + int(aid)
        Jd[aid] = 2000 + int(aid)
        tasks[J0[aid]] = (None, 0, None, None)
        tasks[Jd[aid]] = (None, 0, None, None)
        task_shelf_mapping[J0[aid]] = None
        task_shelf_mapping[Jd[aid]] = None

    # 货架初始虚拟
    J_I, shelf_virtual_tasks = {}, {}
    for sid in shelf_ids:
        vt = 3000 + int(sid)
        J_I[sid] = vt
        shelf_virtual_tasks[sid] = vt
        tasks[vt] = (None, 0, None, None)
        task_shelf_mapping[vt] = sid

    unused_shelves = {sid for sid, lst in shelf_usage.items() if not lst}
    J_I_SI = {sid: lst[0] for sid, lst in shelf_usage.items() if lst}
    bj, hj, J_E = {}, {}, set()

    return (tasks, bj, hj, task_shelf_mapping, J_I, J_E, J,
            J0, Jd, shelf_virtual_tasks, unused_shelves, J_I_SI)


def export_task_inputs_for_sim(prefix: str, tasks_df: pd.DataFrame,
                               outdir: str = "solution_exports"):
    os.makedirs(outdir, exist_ok=True)
    info_df = tasks_df[["Task", "Workstation", "Duration"]].copy()
    shelf_df = tasks_df[["Task", "Shelf"]].copy()
    info_df.to_csv(os.path.join(outdir, f"{prefix}_taskInfo.csv"), index=False)
    shelf_df.to_csv(os.path.join(outdir, f"{prefix}_taskShelf.csv"), index=False)
    print(f"[EXPORT] 写出 solution_exports/{prefix}_taskInfo.csv, solution_exports/{prefix}_taskShelf.csv")


def parse_gamma_list(s: str) -> list[int]:
    if not s:
        return [0]
    arr = []
    for tok in s.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-", 1)
            arr.extend(range(int(a), int(b) + 1))
        else:
            arr.append(int(tok))
    return sorted(set(arr))


def build_ws_order_index(ws_fixed_seq: dict[int, list[int]]) -> dict[int, dict[int, int]]:
    idx = {}
    for ws, seq in ws_fixed_seq.items():
        idx_ws = {}
        for p, j in enumerate(seq):
            idx_ws[int(j)] = int(p)
        idx[int(ws)] = idx_ws
    return idx


def reorder_contiguous_ws_blocks(seq: list[int],
                                 pi: dict[int, int],
                                 ws_fixed_seq: dict[int, list[int]]) -> list[int]:
    if not seq:
        return []
    ws_seq_idx: dict[int, dict[int, int]] = {
        int(ws): {int(t): p for p, t in enumerate(lst)} for ws, lst in ws_fixed_seq.items()
    }
    res = []
    i, n = 0, len(seq)
    while i < n:
        j = i
        ws_i = pi.get(int(seq[i]))
        block = []
        while j < n and pi.get(int(seq[j])) == ws_i:
            block.append(int(seq[j])); j += 1
        if ws_i in ws_seq_idx and len(block) > 1:
            idx = ws_seq_idx[ws_i]
            block.sort(key=lambda t: idx.get(int(t), 10 ** 9))
        res.extend(block)
        i = j
    return res


def build_full_z_with_dummies(routes: dict[int, list[int]],
                              J0: dict[int, int], Jd: dict[int, int],
                              pi: dict[int, int],
                              ws_fixed_seq: dict[int, list[int]],
                              filter_same_ws: bool = True) -> list[tuple[int, int, int]]:
    order_idx = {int(ws): {int(t): p for p, t in enumerate(seq)}
                 for ws, seq in ws_fixed_seq.items()}

    z_list: list[tuple[int, int, int]] = []
    for r, seq in routes.items():
        seq = [int(x) for x in (seq or [])]
        j0 = J0.get(int(r)); jd = Jd.get(int(r))

        if j0 is not None:
            if seq:
                z_list.append((int(j0), int(seq[0]), int(r)))
            elif jd is not None:
                z_list.append((int(j0), int(jd), int(r)))
                continue

        for a, b in zip(seq[:-1], seq[1:]):
            wai = pi.get(int(a)); wbj = pi.get(int(b))
            if filter_same_ws and wai == wbj and wai in order_idx:
                ia = order_idx[wai].get(int(a), -10 ** 9)
                ib = order_idx[wai].get(int(b), +10 ** 9)
                if ia > ib:
                    continue
            z_list.append((int(a), int(b), int(r)))

        if jd is not None and seq:
            z_list.append((int(seq[-1]), int(jd), int(r)))

    return z_list

def build_warm_hint_from_eval(
    *,
    J: set[int],
    R: dict[int, int] | dict,
    details: dict,
    routes: dict[int, list[int]],
    shelf_seq: dict[int, list[int]],
    ws_fixed_seq: dict[int, list[int]],
    J0: dict[int, int],
    Jd: dict[int, int],
    J_I: dict[int, int],
    pi: dict[int, int],
    lock_x: bool = True,
    place_override: dict[int, int] | None = None,
) -> dict:
    """
    warm_hint:
      - w, x, z, v, immediate
      - p/q start
      - g/h start  (关键：h 表示“货架在该 cell 上停留结束时间”，不是到达时间)

    place_override:
      - 若提供，则 x 直接用这个（确保“锁住 MILP 的 x”），而不是用 evaluator 的 end_shelf_final（避免 evaluator repair 改写 x）。
    """
    BIG_M_TIME = 10000.0

    warm_hint: dict = {}

    # ---- w: 任务 -> AGV ----
    w_map: dict[int, int] = {}
    for r, seq in routes.items():
        rr = int(r)
        for j in (seq or []):
            jj = int(j)
            if jj in J:
                w_map[jj] = rr
    warm_hint["w"] = w_map

    # ---- z: 完整 j0 -> ... -> jd ----
    z_list: list[tuple[int, int, int]] = []
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

    # ---- v: evaluator 推导的 V_arcs ----
    V_raw = details.get("V_arcs", []) or []
    v_list: list[tuple[int, int, int, int]] = []
    for arc in V_raw:
        if not isinstance(arc, (list, tuple)) or len(arc) != 4:
            continue
        i, j, s, sp = arc
        v_list.append((int(i), int(j), int(s), int(sp)))
    warm_hint["v"] = v_list

    # ---- x: 任务回库位（EndShelf） ----
    x_map: dict[int, int] = {}

    if place_override is not None:
        # ✅ 强制使用 MILP 的 x（或你想锁住的 place）
        for j, s in (place_override or {}).items():
            jj = int(j)
            if jj in J:
                x_map[jj] = int(s)
    else:
        # fallback：使用 evaluator 的 end_shelf_final
        end_final = details.get("end_shelf_final", {}) or {}
        for j, s in end_final.items():
            jj = int(j)
            if jj in J:
                x_map[jj] = int(s)

    warm_hint["x"] = x_map
    warm_hint["lock_x"] = bool(lock_x)

    # ---- immediate: shelf_seq + J_I ----
    imm_edges: list[tuple[int, int, int]] = []
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
    if imm_edges:
        warm_hint["immediate"] = imm_edges

    # ---- p/q: 直接用 evaluator 的 p/q ----
    p_map = {int(j): float(t) for j, t in (details.get("p", {}) or {}).items() if int(j) in J}
    q_map = {int(j): float(t) for j, t in (details.get("q", {}) or {}).items() if int(j) in J}
    warm_hint["p"] = p_map
    warm_hint["q"] = q_map

    # ========== 关键：构造 g/h ==========
    timeline = details.get("timeline", []) or []
    rec_by_task: dict[int, dict] = {}
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

    # 链内后继、尾任务
    succ_chain: dict[int, int] = {}
    chain_of: dict[int, int] = {}
    tail_of_chain: dict[int, int] = {}

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

    g_map: dict[tuple[int, int, int], float] = {}
    h_map: dict[tuple[int, int, int], float] = {}

    # (1) 真实任务：g=到达 end cell 的时间；h=该 cell 停留结束（下一次被取走 / 或 BIG_M）
    for j, s in x_map.items():
        jj = int(j)
        ss = int(s)
        g = _get_arrive_cell0(jj)

        cc = chain_of.get(jj, None)
        is_tail = (cc is not None and tail_of_chain.get(int(cc)) == jj)
        nxt = succ_chain.get(jj, None)

        if is_tail or (nxt is None):
            h = BIG_M_TIME
        else:
            h = _get_pick_start0(int(nxt))
            if (not math.isfinite(h)) or h <= 0.0:
                h = BIG_M_TIME

        if h < g:
            h = g

        g_map[(jj, 0, ss)] = float(g)
        h_map[(jj, 0, ss)] = float(h)

    # (2) 虚拟初始任务 J_I：g=0；h=该链首任务的 pick_start（表示初始占用结束）
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

    warm_hint["cmax"] = float(details.get("C_task_max", float("inf")))

    warm_hint["_stats"] = {
        "num_w": len(w_map),
        "num_z": len(z_list),
        "num_v": len(v_list),
        "num_immediate": len(imm_edges),
        "num_x": len(x_map),
        "num_p": len(p_map),
        "num_q": len(q_map),
        "num_g": len(g_map),
        "num_h": len(h_map),
    }
    return warm_hint

def _bundle_intify(bundle: dict) -> dict:
    """
    把 bundle 的 routes/shelf_seq/place/cell_sigma 里 key/value 统一转 int
    """
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



def save_bundle_json(
    prefix: str,
    tag: str,
    gamma: int,
    routes: dict[int, list[int]],
    shelf_seq: dict[int, list[int]],
    place: dict[int, int],
    cmax: float,
    outdir: str = "solution_exports",
    cell_sigma: dict[int, list[int]] | None = None,
) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{prefix}_{tag}_bundle_gamma{gamma}.json")

    bundle = {
        "prefix": prefix,
        "tag": tag,                 # "alns" or "milp"
        "source_gamma": int(gamma),
        "cmax": float(cmax),
        "routes": {str(int(r)): [int(t) for t in (seq or [])] for r, seq in routes.items()},
        "shelf_seq": {str(int(c)): [int(t) for t in (seq or [])] for c, seq in shelf_seq.items()},
        "place": {str(int(j)): int(s) for j, s in place.items()},
    }

    if cell_sigma is not None:
        bundle["cell_sigma"] = {str(int(s)): [int(x) for x in (seq or [])] for s, seq in cell_sigma.items()}

    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2)
    return path


def load_bundle_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    return _bundle_intify(b)

def build_cell_sigma_from_event_run(
    *,
    evaluator_factory,   # callable(eval_gamma, cell_sigma)->RobustEvaluator
    eval_gamma: int,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    J_set: Set[int],
) -> Dict[int, List[int]]:
    """
    用 event gate 模式评估一次，抽取每个 cell 的“实际站位顺序 σ[cell]=[task,...]”
    排序规则：按 arrive_cell_act（实际落位开始占用时刻）升序。
    """
    ev = evaluator_factory(int(eval_gamma), cell_sigma=None)
    ms, diag = ev.evaluate(routes, shelf_seq, place, verbose=False)

    if not isinstance(diag, dict):
        return {}

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

    by_cell: dict[int, list[tuple[float, int]]] = defaultdict(list)

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

def run_cross_gamma_check(
    *,
    prefix: str,
    tag: str,  # "alns" or "milp"
    bundles_by_source_gamma: dict[int, dict],
    eval_gammas: list[int],
    evaluator_factory,  # callable(eval_gamma, cell_sigma)->RobustEvaluator

    milp_opt_by_eval_gamma: dict[int, float] | None = None,
    outdir: str = "solution_exports",
    export_fail_diag: bool = True,
):
    """
    对每个 source_gamma 的 bundle，在 eval_gammas 下都跑一遍 evaluator.evaluate(...)
    导出：
      - {prefix}_crossGamma_{tag}_suite.csv  (长表)
      - {prefix}_crossGamma_{tag}_matrix.csv (矩阵)

    新增：当某个 (src_g, eval_g) 评估为 inf 或 feasible=False 时，自动导出该次 evaluator diag，
         文件前缀：{prefix}_crossFail_{tag}_srcG{src_g}_evalG{eval_g}_*
    """
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for src_g, bundle in sorted(bundles_by_source_gamma.items()):
        b = _bundle_intify(bundle)
        routes = b["routes"]
        shelf_seq = b["shelf_seq"]
        place = b["place"]

        for eg in eval_gammas:
            cell_sigma = b.get("cell_sigma", None)
            ev = evaluator_factory(int(eg), cell_sigma=cell_sigma)
            ms, diag = ev.evaluate(routes, shelf_seq, place, verbose=False)

            feasible = None
            penalties = None
            bottleneck_task = None
            place_changed_cnt = None

            if isinstance(diag, dict):
                feasible = diag.get("feasible")
                penalties = diag.get("penalties")

                q = diag.get("q", {}) or {}
                try:
                    q2 = {int(k): float(v) for k, v in q.items()}
                    if q2:
                        bottleneck_task = max(q2, key=lambda jj: q2[jj])
                except Exception:
                    pass

                end_final = diag.get("end_shelf_final", {}) or {}
                try:
                    end2 = {int(k): int(v) for k, v in end_final.items()}
                    place_changed_cnt = sum(
                        1 for j, s in place.items()
                        if j in end2 and int(end2[j]) != int(s)
                    )
                except Exception:
                    place_changed_cnt = None

            # === 新增：inf/不可行时导出 diag（用于定位“爆掉原因”） ===
            is_bad = (not math.isfinite(float(ms))) or (feasible is False)
            if export_fail_diag and is_bad and isinstance(diag, dict):
                try:
                    export_tag = f"crossFail_{tag}_srcG{int(src_g)}_evalG{int(eg)}"
                    export_evaluator_diagnostics(
                        prefix=prefix,
                        export_tag=export_tag,
                        source_gamma=int(src_g),
                        eval_gamma=int(eg),
                        routes=routes,
                        shelf_seq=shelf_seq,
                        place=place,
                        milp_cmax=float("nan"),
                        eval_cmax=float(ms) if math.isfinite(float(ms)) else float("inf"),
                        diag=diag,
                        outdir=outdir,
                    )
                    print(f"[CrossGamma][EXPORT] exported fail diag -> {export_tag}")
                except Exception as e:
                    print(f"[CrossGamma][EXPORT] failed: {type(e).__name__}: {e}")

            opt = None
            regret = None
            if milp_opt_by_eval_gamma is not None and int(eg) in milp_opt_by_eval_gamma:
                opt = float(milp_opt_by_eval_gamma[int(eg)])
                regret = float(ms) - opt if math.isfinite(float(ms)) else float("inf")

            rows.append({
                "prefix": prefix,
                "tag": tag,
                "source_gamma": int(src_g),
                "eval_gamma": int(eg),
                "makespan": float(ms),
                "feasible": feasible,
                "bottleneck_task": bottleneck_task,
                "place_changed_cnt": place_changed_cnt,
                "milp_opt_at_eval_gamma": opt,
                "regret_vs_opt": regret,
                "penalties": json.dumps(penalties, ensure_ascii=False) if penalties is not None else None,
            })

    df = pd.DataFrame(rows)
    suite_path = os.path.join(outdir, f"{prefix}_crossGamma_{tag}_suite.csv")
    df.to_csv(suite_path, index=False, encoding="utf-8-sig")

    pivot = df.pivot(index="source_gamma", columns="eval_gamma", values="makespan")
    matrix_path = os.path.join(outdir, f"{prefix}_crossGamma_{tag}_matrix.csv")
    pivot.to_csv(matrix_path, encoding="utf-8-sig")

    print("\n[CrossGamma] makespan matrix:", tag)
    print(pivot)
    print(f"[CrossGamma] exported:\n  - {suite_path}\n  - {matrix_path}\n")
# ============================================================
# Quick benchmark (integrated, subprocess-based)
# Output files are generated in project root (same level as main.py).
# ============================================================

_RE_EXACT = re.compile(
    r"\[ALNS\]\s*Exact makespan:\s*base=([^\s]+)\s*(?:→|->)\s*best=([^\s]+)",
    re.IGNORECASE
)
_RE_WARM = re.compile(
    r"\[WarmStart\]\s*WS-block fix re-eval\s*\(Exact\):\s*makespan=([^\s]+)",
    re.IGNORECASE
)
_RE_PROF = re.compile(
    r"\[Profiler.*?\]\s*evaluate calls=(\d+)\s*\|\s*total=([0-9\.]+)s\s*\|\s*avg=([0-9\.]+)ms",
    re.IGNORECASE
)

def _bench_decode_best_effort(b: bytes) -> str:
    if b is None:
        return ""
    # try utf-8 / cp936, pick one with fewer replacement chars
    cands = []
    for enc in ("utf-8", "cp936", "gbk"):
        try:
            s = b.decode(enc, errors="replace")
            cands.append((s.count("\ufffd"), s))
        except Exception:
            pass
    if cands:
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
    return b.decode("utf-8", errors="replace")

def _bench_as_float(tok: str) -> float:
    if tok is None:
        return float("nan")
    t = str(tok).strip().lower()
    if t in ("inf", "+inf", "infinity", "+infinity"):
        return float("inf")
    if t in ("-inf", "-infinity"):
        return float("-inf")
    if t in ("nan", "+nan", "-nan"):
        return float("nan")
    try:
        return float(tok)
    except Exception:
        return float("nan")

def _bench_is_finite(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False

def _parse_int_list(s: str, default=None) -> list[int]:
    if default is None:
        default = [0]
    if s is None:
        return list(default)
    s = str(s).strip()
    if not s:
        return list(default)
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out if out else list(default)

def _extract_metrics_from_stdout(stdout_text: str) -> dict:
    # take "last match" (because your program may print multiple times)
    exact_best = float("inf")
    warm_ms = float("nan")
    prof_calls = -1
    prof_total = float("nan")
    prof_avg = float("nan")

    mlist = list(_RE_EXACT.finditer(stdout_text))
    if mlist:
        m = mlist[-1]
        exact_best = _bench_as_float(m.group(2))

    wlist = list(_RE_WARM.finditer(stdout_text))
    if wlist:
        warm_ms = _bench_as_float(wlist[-1].group(1))

    plist = list(_RE_PROF.finditer(stdout_text))
    if plist:
        m = plist[-1]
        try:
            prof_calls = int(m.group(1))
        except Exception:
            prof_calls = -1
        prof_total = _bench_as_float(m.group(2))
        prof_avg = _bench_as_float(m.group(3))

    # final makespan: prefer warmstart re-eval if present
    final_ms = warm_ms if _bench_is_finite(warm_ms) else exact_best

    return {
        "exact_best": float(exact_best),
        "warm_ms": float(warm_ms),
        "final_ms": float(final_ms),
        "prof_calls": int(prof_calls),
        "prof_total_sec": float(prof_total),
        "prof_avg_ms": float(prof_avg),
        "feasible": bool(_bench_is_finite(final_ms)),
    }

def _bench_bundle_path(root: Path, prefix: str, gamma: int) -> Path:
    return root / "solution_exports" / f"{prefix}_alns_bundle_gamma{int(gamma)}.json"

def _bench_try_read_bundle_cmax(bundle_path: Path) -> float | None:
    if not bundle_path.exists():
        return None
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            b = json.load(f)
        if isinstance(b, dict) and ("cmax" in b):
            return float(b["cmax"])
    except Exception:
        return None
    return None

def run_quick_benchmark_subprocess(args) -> int:
    """
    Quick benchmark mode:
      - runs main.py as subprocess for (gamma x seed)
      - prefers reading makespan from exported bundle JSON (cmax) to avoid stdout parsing fragility
      - still parses profiler metrics from stdout (optional)
      - writes CSV + JSON summary in project root (same dir as main.py)
      - prints progress per run
      - optional PASS/FAIL vs baseline summary json
    """
    root = Path(__file__).resolve().parent
    main_py = Path(__file__).resolve()

    seeds = _parse_int_list(getattr(args, "bench_seeds", "0,1,2"), default=[0])
    iters = int(getattr(args, "bench_iters", 200) or 200)
    if iters <= 0:
        iters = 200

    gammas_str = (getattr(args, "bench_gammas", "") or "").strip()
    if gammas_str:
        gamma_list = parse_gamma_list(gammas_str)
    else:
        gamma_list = parse_gamma_list(getattr(args, "gammas", "0"))

    rows = []
    t_global0 = time.perf_counter()

    for g in gamma_list:
        for sd in seeds:
            g = int(g)
            sd = int(sd)

            # 关键：防止读到“上一次 run 的旧 bundle”
            bundle_path = _bench_bundle_path(root, str(args.prefix), g)
            try:
                if bundle_path.exists():
                    bundle_path.unlink()
            except Exception:
                pass

            print(f"[BENCH] running gamma={g} seed={sd} iters={iters} ...")

            cmd = [
                sys.executable, str(main_py),
                "--prefix", str(args.prefix),
                "--gammas", str(g),
                "--alns-iters", str(iters),
                "--seed", str(sd),
                "--alns-speed-profile", str(getattr(args, "alns_speed_profile", "balanced") or "balanced"),
                "--alns-time-budget-sec", str(getattr(args, "alns_time_budget_sec", 0.0) or 0.0),
            ]
            if bool(getattr(args, "alns_ignore_cache", False)):
                cmd.append("--alns-ignore-cache")
            if bool(getattr(args, "alns_no_seed", False)):
                cmd.append("--alns-no-seed")

            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=int(getattr(args, "bench_timeout", 1800) or 1800),
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                )
                wall = time.perf_counter() - t0
                out_text = _bench_decode_best_effort(proc.stdout or b"")
                exit_code = int(proc.returncode)
            except subprocess.TimeoutExpired as e:
                wall = time.perf_counter() - t0
                out_text = _bench_decode_best_effort((e.stdout or b"")) + "\n[BENCH] TIMEOUT"
                exit_code = 124
            except Exception as e:
                wall = time.perf_counter() - t0
                out_text = f"[BENCH] EXCEPTION: {type(e).__name__}: {e}"
                exit_code = 125

            # 先从 stdout 抓 profiler / warm/exact（可选）
            met = _extract_metrics_from_stdout(out_text)
            final_src = "stdout"

            # ✅ 最稳：优先从 bundle JSON 读 cmax
            cmax_bundle = _bench_try_read_bundle_cmax(bundle_path)
            if cmax_bundle is not None:
                met["final_ms"] = float(cmax_bundle)
                met["feasible"] = bool(_bench_is_finite(met["final_ms"]))
                final_src = "bundle"

            # 子进程失败：强制判失败
            if exit_code != 0:
                met["feasible"] = False
                met["final_ms"] = float("inf")
                final_src = f"exit{exit_code}"

            print(f"[BENCH] done  gamma={g} seed={sd} exit={exit_code} final_ms={met['final_ms']} src={final_src} wall={wall:.2f}s")

            # 如果失败，顺手落一个日志，方便你回看是哪条约束爆掉了
            if (not _bench_is_finite(met["final_ms"])) or exit_code != 0:
                try:
                    log_path = root / f"bench_quick_{args.prefix}_g{g}_s{sd}.log"
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(out_text)
                    print(f"[BENCH] wrote log: {log_path}")
                except Exception:
                    pass

            rows.append({
                "prefix": str(args.prefix),
                "gamma": int(g),
                "seed": int(sd),
                "iters": int(iters),
                "exit_code": int(exit_code),
                "wall_time_sec": float(wall),
                "final_ms": float(met["final_ms"]),
                "exact_best": float(met.get("exact_best", float("inf"))),
                "warm_ms": float(met.get("warm_ms", float("nan"))),
                "feasible": bool(met["feasible"]),
                "prof_calls": int(met.get("prof_calls", -1)),
                "prof_total_sec": float(met.get("prof_total_sec", float("nan"))),
                "prof_avg_ms": float(met.get("prof_avg_ms", float("nan"))),
                "final_ms_src": str(final_src),
            })

    total_wall = time.perf_counter() - t_global0
    df = pd.DataFrame(rows)

    # ----- write outputs in root -----
    csv_path = root / f"bench_quick_{args.prefix}_runs.csv"
    summary_path = root / f"bench_quick_{args.prefix}_summary.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ----- compute summary + score -----
    def _ms_to_score(v: float) -> float:
        return float(v) if _bench_is_finite(v) else 1e30

    df["ms_score"] = df["final_ms"].apply(_ms_to_score)

    per_gamma = []
    overall_score = 0.0

    for g in sorted(df["gamma"].unique().tolist()):
        dfg = df[df["gamma"] == g].copy()
        feas_rate = float(dfg["feasible"].mean()) if len(dfg) else 0.0
        mean_wall = float(dfg["wall_time_sec"].mean()) if len(dfg) else 0.0

        p50 = float(dfg["ms_score"].quantile(0.50)) if len(dfg) else 1e30
        p90 = float(dfg["ms_score"].quantile(0.90)) if len(dfg) else 1e30
        mean_ms = float(dfg["ms_score"].mean()) if len(dfg) else 1e30

        score_g = float(p90 + 0.05 * mean_wall + 1e6 * (1.0 - feas_rate))
        overall_score += score_g

        per_gamma.append({
            "gamma": int(g),
            "n_runs": int(len(dfg)),
            "feasible_rate": float(feas_rate),
            "makespan_mean": float(mean_ms),
            "makespan_p50": float(p50),
            "makespan_p90": float(p90),
            "wall_time_mean_sec": float(mean_wall),
            "score_gamma": float(score_g),
        })

    summary = {
        "prefix": str(args.prefix),
        "bench_iters": int(iters),
        "bench_seeds": [int(x) for x in seeds],
        "bench_gammas": [int(x) for x in gamma_list],
        "total_wall_time_sec": float(total_wall),
        "overall": {
            "score": float(overall_score),
            "n_total_runs": int(len(df)),
            "overall_feasible_rate": float(df["feasible"].mean()) if len(df) else 0.0,
        },
        "by_gamma": per_gamma,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[BENCH] wrote: {csv_path}")
    print(f"[BENCH] wrote: {summary_path}")
    print(f"[BENCH] SCORE={overall_score:.6g}")

    # ----- optional compare baseline -----
    baseline = (getattr(args, "bench_compare", "") or "").strip()
    tol = float(getattr(args, "bench_tol", 0.0) or 0.0)

    if baseline:
        try:
            baseline_path = Path(baseline)
            if not baseline_path.is_absolute():
                baseline_path = root / baseline_path

            with open(baseline_path, "r", encoding="utf-8") as f:
                base = json.load(f)

            base_score = base.get("overall", {}).get("score", None)
            if base_score is None:
                base_score = base.get("score", None)
            base_score = float(base_score)

            ok = (overall_score <= base_score * (1.0 + tol))
            print(f"[BENCH] baseline_score={base_score:.6g} tol={tol:.3g} => {'PASS' if ok else 'FAIL'}")
            return 0 if ok else 2
        except Exception as e:
            print(f"[BENCH] baseline compare failed: {type(e).__name__}: {e}")
            return 0

    return 0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="demo01")
    ap.add_argument("--gammas", default="0")
    ap.add_argument("--alns-iters", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--alns-speed-profile",
        choices=["balanced", "turbo"],
        default="turbo",
        help="ALNS 搜索档位：balanced(质量优先) / turbo(速度优先，默认)",
    )
    ap.add_argument(
        "--alns-time-budget-sec",
        type=float,
        default=0.0,
        help="ALNS 软时间预算（秒）。<=0 表示不限制。",
    )
    ap.add_argument(
        "--alns-ignore-cache",
        action="store_true",
        help="忽略 turbo cache 文件（用于冷启动验证）。",
    )
    ap.add_argument(
        "--alns-no-seed",
        action="store_true",
        help="严格从头求解：不读取任何历史可行种子（cache/bridge）。",
    )

    ap.add_argument(
        "--cross-gamma-check",
        action="store_true",
        help="对每个 gamma 的 ALNS 解做 cross-gamma 评估矩阵：source_gamma 解放到 eval_gamma 下重新评估",
    )
    ap.add_argument(
        "--export-eval-diag",
        action="store_true",
        help="导出 ALNS 解的 evaluator diag/timeline/pq/robust_segments/v_arcs 到 solution_exports/",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="打印更详细的 ALNS 解与 timeline（默认只打印摘要 + 进度）",
    )
    # ===== Quick benchmark (very small data, for automation) =====
    ap.add_argument(
        "--quick-bench",
        action="store_true",
        help="快速基准测试：subprocess 反复运行 main.py，抓 makespan/Profiler，输出到项目根目录"
    )
    ap.add_argument("--bench-iters", type=int, default=200, help="quick-bench 用的 ALNS 迭代次数")
    ap.add_argument("--bench-seeds", default="0,1,2", help="quick-bench 用的 seeds，逗号分隔，例如 0,1,2")
    ap.add_argument("--bench-gammas", default="", help="quick-bench 覆盖用的 gammas（可空，空则沿用 --gammas）")
    ap.add_argument("--bench-timeout", type=int, default=1800, help="每次子进程运行的超时秒数")
    ap.add_argument("--bench-compare", default="", help="可选：baseline summary.json 路径，用于 PASS/FAIL")
    ap.add_argument("--bench-tol", type=float, default=0.0, help="baseline 容忍比例，比如 0.01 表示允许差 1%%")
    args = ap.parse_args()

    # --- quick benchmark mode: run and exit ---
    if args.quick_bench:
        code = run_quick_benchmark_subprocess(args)
        raise SystemExit(code)
    # ===== init-cell lock config (only affects evaluator.py) =====
    INIT_LOCK_ITERS_ALIGN = 3
    INIT_LOCK_ITERS_CROSS = 2
    INIT_LOCK_TOL = 1e-6
    # ============================================================

    prefix = args.prefix
    gamma_list = parse_gamma_list(args.gammas)

    # ✅ 只做 ALNS：只需要这一个容器
    alns_bundles_by_gamma: dict[int, dict] = {}

    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"未找到 {tasks_csv}。")



    # 地图
    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)

    # 任务
    tasks_df = pd.read_csv(tasks_csv)
    print(f"[TASK] 读取 tasks.csv 行数={len(tasks_df)}, WS集合={sorted(tasks_df['Workstation'].unique())}")
    if len(tasks_df) != 11:
        print(f"[WARN] 这次 tasks.csv 有 {len(tasks_df)} 个任务（你期望 11 个），继续求解…")
    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1

    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )
    print("[WSOrder] 固定顺序：", ws_fixed_seq)

    map_obj = create_map_from_components(
        width=W, height=H,
        sp_indices=sp_indices,
        ws_indices=ws_indices,
        shelf_data=shelf_data,
        agv_data=agv_data
    )
    time_manager = TimeManager()
    _ = MovementManager(map_obj, time_manager)

    shelf_ids = sorted(shelf_data.keys())
    (tasks, bj, hj, task_shelf_mapping, J_I, J_E, J,
     J0, Jd, shelf_virtual_tasks, unused_shelves, J_I_SI) = \
        build_task_structures(tasks_df, agv_data, shelf_ids)

    # R/S/K
    R = map_obj.extract_AGVs()

    def idx2rc(idx: int): return divmod(int(idx) - 1, W)

    S = {int(sp): idx2rc(int(sp)) for sp in sp_indices}
    K = {i + 1: idx2rc(ws_indices[i]) for i in range(len(ws_indices))}

    sc = build_scenario_from_prefix(
        prefix=prefix,
        distance=distance,
        D_setup=2.0,
        gamma_budget=0,
        K_near=12
    )
    print_scenario_brief(sc)

    export_task_inputs_for_sim(prefix, tasks_df)

    # 规则初解
    pi = {j: tasks[j][0] for j in J}
    init_sol = build_initial_solution_basic(
        J=J, R=R,
        task_shelf_mapping=task_shelf_mapping,
        shelf_data=shelf_data,
        J_I=J_I,
        pi=pi
    )

    # 距离与 Δ（给评估器）
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

    # 每个任务的“工位最近前 m 个储位”（ALNS 用它来枚举 x）
    m = 8
    S_near_by_j = {j: sorted(S, key=lambda s: d_pi_s[(j, s)])[:min(m, len(S))] for j in J}
    chain_repair_iters_hard = max(24, min(160, 2 * len(J)))
    cell_repair_iters_hard = max(16, min(80, len(J)))

    check_v_consistency(
        routes=init_sol.routes, shelf_seq=init_sol.shelf_seq, place=init_sol.place,
        J=J, R=R, S=S, J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data
    )
    print("[CHECK] v-arc consistency passed.")

    def evaluator_factory_for_cross(eval_gamma: int,
                                    cell_sigma: Optional[Dict[int, List[int]]] = None) -> RobustEvaluator:
        # cross-gamma：要“可比”，所以用 event gate + 可选 σ 来锁死 cell 站位顺序
        return RobustEvaluator(
            J=J, R=R, S=S,
            pi=pi,
            D={j: tasks[j][1] for j in J},
            J0=J0, Jd=Jd, J_I=J_I,
            shelf_data=shelf_data, agv_data=agv_data,
            d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
            Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
            gamma=int(eval_gamma),
            ws_fixed_seq=ws_fixed_seq,
            ws_setup_rule="flat",

            lock_place=True,
            detach_on_mismatch=False,
            envelope_shared_resources=True,

            enable_chain_repair=True,
            chain_repair_max_iters=chain_repair_iters_hard,
            enable_cell_repair=True,
            cell_repair_max_iters=cell_repair_iters_hard,

            cell_conflict_mode="hard",
            allow_incomplete=False,

            # ✅关键：event gate + σ
            cell_gate_mode="event",
            cell_sigma=cell_sigma,

            timeline_mode="off",
            collect_v_arcs=False,
            record_cell_repair_log=False,

            enable_init_cell_lock=True,
            init_lock_max_iters=INIT_LOCK_ITERS_CROSS,
            init_lock_tol=INIT_LOCK_TOL,
            init_lock_verbose=False,
        )

    # 打印 ALNS 结构解
    def print_alns_solution(gamma_val: int, sol, evaluator: RobustEvaluator):
        ms, diag = evaluator.evaluate(sol.routes, sol.shelf_seq, sol.place)
        p = diag.get("p", {}); q = diag.get("q", {})
        end_final = diag.get("end_shelf_final", {})
        V_arcs = diag.get("V_arcs", []); timeline = diag.get("timeline", [])

        print("\n----------------------------------------------------------------------")
        print(f"[ALNS] 结构解（γ={gamma_val}）")
        for r in sorted(sol.routes):
            print(f"  AGV {r} route: {sol.routes[r]}")
        print("\n  Shelf sequences:")
        for c in sorted(sol.shelf_seq):
            print(f"    Shelf {c}: {sol.shelf_seq[c]}")
        print("\n  Placement (task -> shelf):")
        for j in sorted(sol.place):
            print(f"    Task {j}: s={sol.place[j]}")

        print(f"\n  [ALNS-Eval] makespan = {ms:.2f}")
        print("  p/q by task:")
        for j in sorted(p):
            print(f"    Task {j}: p[{p[j]:.2f}] | q[{q[j]:.2f}]  end_s={end_final.get(j, 'NA')}")

        if V_arcs:
            print("\n  v[i,j,s,s'] = 1 （评估器推导，用于诊断）")
            for (i, jj, s, sp) in V_arcs:
                print(f"    v[{i},{jj},{s},{sp}] = 1")

        if timeline:
            print("\n  --- Timeline (详尽) ---")
            timeline_sorted = sorted(timeline, key=lambda rec: (rec["ws_start"], rec["Task"]))
            for rec in timeline_sorted:
                print(
                    "    "
                    f"AGV{rec['AGV']} T{rec['Task']} C{rec['Chain']} WS{rec['WS']} | "
                    f"s0={rec['home_before']} "
                    f"dt1={rec['dt1']:.2f} arr_shelf={rec['arrive_shelf']:.2f} pick={rec['pick_start']:.2f} | "
                    f"dt2={rec['dt2_eff']:.2f} (nom {rec['dt2_nom']:.2f}) arrWS={rec['arrival_ws']:.2f} | "
                    f"ws: {rec['ws_start']:.2f}->{rec['ws_end']:.2f} | "
                    f"end_s={rec['end_s']} dt3={rec['dt3_eff']:.2f} (nom {rec['dt3_nom']:.2f}) "
                    f"arrCell_nom={rec['arrive_cell_nom']:.2f} arrCell_act={rec['arrive_cell_act']:.2f}"
                )
        print("----------------------------------------------------------------------\n")

    # 逐 γ 求解
    # 逐 γ 求解（只跑 ALNS）
    prev_best_sol: InitialSolution | None = None
    for g in gamma_list:
        print("\n" + "=" * 70)
        print(f"[RUN] 开始 ALNS：γ = {g}")
        print("=" * 70)

        need_full_diag = bool(args.export_eval_diag or args.verbose)
        requested_profile = str(getattr(args, "alns_speed_profile", "balanced") or "balanced").strip().lower()
        no_seed_cli = bool(getattr(args, "alns_no_seed", False))
        no_seed = True
        if (not no_seed_cli):
            print("[ALNS] history seed loading disabled globally: force no-seed recomputation mode.")
        run_profile = requested_profile if requested_profile in {"balanced", "turbo"} else "balanced"
        fast_post_mode = (run_profile == "turbo")
        if no_seed and (run_profile == "turbo"):
            print("[ALNS] no-seed mode: turbo profile enabled (feasible seed + fast search).")

        # ========= Fast evaluator（给 ALNS 内层用）=========
        # ========= Fast evaluator（给 ALNS 内层用）=========
        # ========= Fast evaluator（给 ALNS 内层用）=========
        evaluator_fast = RobustEvaluator(
            J=J, R=R, S=S,
            pi=pi,
            D={j: tasks[j][1] for j in J},
            J0=J0, Jd=Jd, J_I=J_I,
            shelf_data=shelf_data, agv_data=agv_data,
            d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
            Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
            gamma=g,
            ws_fixed_seq=ws_fixed_seq,
            ws_setup_rule="flat",
            lock_place=True,
            detach_on_mismatch=False,
            envelope_shared_resources=True,

            # --- turbo 下开小步修复，避免一直停留在“不可行惩罚区” ---
            enable_chain_repair=bool(fast_post_mode),
            chain_repair_max_iters=(2 if fast_post_mode else 0),
            enable_cell_repair=bool(fast_post_mode),
            cell_repair_max_iters=(3 if fast_post_mode else 0),

            cell_conflict_mode="penalty",
            cell_conflict_weight=(3e4 if fast_post_mode else 1e4),
            allow_incomplete=True,
            missing_task_weight=(5e6 if fast_post_mode else 1e6),

            timeline_mode="off",
            collect_v_arcs=False,
            record_cell_repair_log=False,

            enable_init_cell_lock=False,
            init_lock_max_iters=0,
            init_lock_tol=INIT_LOCK_TOL,
            init_lock_verbose=False,
        )

        # ========= Hard-partial evaluator (for feasibility-driven construction) =========
        evaluator_hard_partial = RobustEvaluator(
            J=J, R=R, S=S,
            pi=pi,
            D={j: tasks[j][1] for j in J},
            J0=J0, Jd=Jd, J_I=J_I,
            shelf_data=shelf_data, agv_data=agv_data,
            d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
            Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
            gamma=g,
            ws_fixed_seq=ws_fixed_seq,
            ws_setup_rule="flat",
            lock_place=True,
            detach_on_mismatch=False,
            envelope_shared_resources=True,
            enable_chain_repair=True,
            chain_repair_max_iters=chain_repair_iters_hard,
            enable_cell_repair=True,
            cell_repair_max_iters=cell_repair_iters_hard,
            cell_conflict_mode="hard",
            allow_incomplete=True,
            timeline_mode="off",
            collect_v_arcs=False,
            record_cell_repair_log=False,
            enable_init_cell_lock=True,
            init_lock_max_iters=INIT_LOCK_ITERS_ALIGN,
            init_lock_tol=INIT_LOCK_TOL,
            init_lock_verbose=False,
        )

        # ========= Exact evaluator（ALNS 结束后核验/导出用）=========
        evaluator_exact = RobustEvaluator(
            J=J, R=R, S=S,
            pi=pi,
            D={j: tasks[j][1] for j in J},
            J0=J0, Jd=Jd, J_I=J_I,
            shelf_data=shelf_data, agv_data=agv_data,
            d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
            Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
            gamma=g,
            ws_fixed_seq=ws_fixed_seq,
            ws_setup_rule="flat",
            lock_place=True,
            detach_on_mismatch=False,
            envelope_shared_resources=True,

            # --- Exact：允许修复（对齐 MILP / cross-gamma 可比） ---
            enable_chain_repair=True,
            chain_repair_max_iters=chain_repair_iters_hard,
            enable_cell_repair=True,
            cell_repair_max_iters=cell_repair_iters_hard,

            cell_conflict_mode="hard",
            allow_incomplete=False,

            timeline_mode="full" if need_full_diag else "min",
            collect_v_arcs=True if need_full_diag else False,
            record_cell_repair_log=False,

            enable_init_cell_lock=True,
            init_lock_max_iters=INIT_LOCK_ITERS_ALIGN,
            init_lock_tol=INIT_LOCK_TOL,
            init_lock_verbose=False,
        )

        shelf_init_for_alns = {int(c): int(pos) for c, pos in shelf_data.items()}
        cache_bundle_path = Path("solution_exports") / f"{prefix}_alns_bundle_cache_gamma{int(g)}.json"
        allow_history_seed_io = False
        # Keep the no_seed policy decided above (forced true for pure recomputation).
        no_seed = bool(no_seed)
        # Hard policy: do not read any historical feasible seed.
        feasible_seed: InitialSolution | None = None

        forced_seed: InitialSolution | None = None
        forced_seed_ms: float = float("inf")
        if no_seed:
            seed_base = init_sol
            print("[ALNS] no-seed mode: building exact feasible initial seed (no history).")
            forced_seed = build_full_coverage_seed(
                seed_base,
                evaluator=evaluator_hard_partial,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init=shelf_init_for_alns,
                seed=int(args.seed),
            )
            forced_seed_ms, _ = evaluator_exact.evaluate(
                forced_seed.routes, forced_seed.shelf_seq, forced_seed.place
            )
            if math.isfinite(float(forced_seed_ms)):
                print(f"[ALNS] no-seed full-coverage seed ready | cmax={float(forced_seed_ms):.2f}")
            if (not math.isfinite(float(forced_seed_ms))):
                print("[ALNS] no-seed deterministic ws-round-robin exact seed.")
                rr_seed = build_ws_round_robin_seed(
                    forced_seed,
                    evaluator=evaluator_exact,
                    task_shelf_mapping=task_shelf_mapping,
                )
                rr_ms, _ = evaluator_exact.evaluate(
                    rr_seed.routes, rr_seed.shelf_seq, rr_seed.place
                )
                if math.isfinite(float(rr_ms)):
                    forced_seed = rr_seed
                    forced_seed_ms = float(rr_ms)
                    print(f"[ALNS] ws-round-robin exact seed ready | cmax={float(forced_seed_ms):.2f}")
            if not math.isfinite(float(forced_seed_ms)):
                print("[ALNS] no-seed exact-constructor: strict exact-feasibility driven insertion.")
                constructor_attempts = 24 if int(g) == 0 else 4
                constructor_time_budget = 90.0 if int(g) == 0 else 14.0
                constructor_eval_budget = 8000 if int(g) == 0 else 3000
                forced_seed = build_exact_feasible_seed(
                    forced_seed,
                    evaluator_partial_hard=evaluator_hard_partial,
                    evaluator_exact=evaluator_exact,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init=shelf_init_for_alns,
                    seed=int(args.seed),
                    attempts=constructor_attempts,
                    time_budget_sec=constructor_time_budget,
                    per_attempt_eval_budget=constructor_eval_budget,
                )
                forced_seed_ms, _ = evaluator_exact.evaluate(
                    forced_seed.routes, forced_seed.shelf_seq, forced_seed.place
                )
                if math.isfinite(float(forced_seed_ms)):
                    print(f"[ALNS] no-seed exact-constructor succeeded | cmax={float(forced_seed_ms):.2f}")
            if not math.isfinite(float(forced_seed_ms)):
                print("[ALNS] no-seed seed repair: run short exact ALNS from full-coverage seed.")
                seed_repair_eval = evaluator_hard_partial if int(g) == 0 else evaluator_exact
                prof_seed = EvalProfiler(seed_repair_eval)
                forced_seed = alns_minimize(
                    init=forced_seed,
                    evaluator=prof_seed,
                    iters=(520 if int(g) == 0 else 260),
                    start_T=1.1,
                    cool=0.997,
                    S_near_by_j=S_near_by_j,
                    seed=int(args.seed) + 104729,
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=True,
                    enable_shelf_tune=True,
                    shelf_init=shelf_init_for_alns,
                    feasible_first=False,
                    enable_strong_init=False,
                    speed_profile="balanced",
                    time_budget_sec=(25.0 if int(g) == 0 else 10.0),
                    relabel_interval=0,
                    eval_budget_total=(22000 if int(g) == 0 else 9000),
                    eval_budget_heavy=(13000 if int(g) == 0 else 5000),
                    target_feasible_obj=1000.0,
                )
                prof_seed.report(tag=f"[SeedRepair γ={g}]")
                forced_seed_ms, _ = evaluator_exact.evaluate(
                    forced_seed.routes, forced_seed.shelf_seq, forced_seed.place
                )
                if math.isfinite(float(forced_seed_ms)):
                    print(f"[ALNS] no-seed exact feasible seed ready after repair | cmax={float(forced_seed_ms):.2f}")
            if not math.isfinite(float(forced_seed_ms)):
                print("[ALNS] warning: exact feasible seed not found yet; ALNS will continue from best-available structure.")

        # ========= 调用 ALNS（Profiler 包 fast evaluator）=========
        total_budget = float(getattr(args, "alns_time_budget_sec", 0.0) or 0.0)
        if args.alns_iters and args.alns_iters > 0:
            seed_exact_feasible = bool((forced_seed is not None) and math.isfinite(float(forced_seed_ms)))
            search_evaluator = evaluator_fast
            if no_seed and int(g) == 0 and (not seed_exact_feasible):
                search_evaluator = evaluator_hard_partial
            prof = EvalProfiler(search_evaluator)
            print(f"[ALNS] Start: iters={args.alns_iters}, gamma={g}")
            if forced_seed is not None:
                init_for_alns = forced_seed
            else:
                init_for_alns = init_sol
            if total_budget <= 0.0:
                budget_stage1 = None
                budget_stage2 = 0.0
            elif fast_post_mode and total_budget > 6.0:
                budget_stage2 = 4.0
                budget_stage1 = max(1.0, total_budget - budget_stage2)
            else:
                budget_stage1 = total_budget
                budget_stage2 = 0.0

            stage1_eval_budget_total = None
            stage1_eval_budget_heavy = None
            stage1_target_obj = None
            if no_seed:
                long_unbounded_run = bool((total_budget <= 0.0) and (int(args.alns_iters) >= 1000))
                if not long_unbounded_run:
                    if seed_exact_feasible:
                        if run_profile == "turbo":
                            stage1_eval_budget_total = 12000
                            stage1_eval_budget_heavy = 5000
                        else:
                            stage1_eval_budget_total = 22000
                            stage1_eval_budget_heavy = 9000
                    else:
                        stage1_eval_budget_total = 32000
                        stage1_eval_budget_heavy = 16000
                stage1_target_obj = None

            init_sol_best = alns_minimize(
                init=init_for_alns,
                evaluator=prof,
                iters=(
                    int(args.alns_iters)
                    if ((forced_seed is not None) and math.isfinite(float(forced_seed_ms)))
                    else (min(int(args.alns_iters), 320) if no_seed else int(args.alns_iters))
                ),
                start_T=1.0, cool=0.995,
                S_near_by_j=S_near_by_j,
                seed=args.seed,
                task_shelf_mapping=task_shelf_mapping,
                enable_place_tune=True,
                enable_shelf_tune=True,
                shelf_init=shelf_init_for_alns,
                speed_profile=str(run_profile),
                time_budget_sec=budget_stage1,
                enable_strong_init=(not seed_exact_feasible),
                strong_init_tries=(6 if (no_seed and (not seed_exact_feasible)) else 4),
                strong_init_time_budget_sec=(12.0 if (no_seed and (not seed_exact_feasible)) else 0.0),
                eval_budget_total=stage1_eval_budget_total,
                eval_budget_heavy=stage1_eval_budget_heavy,
                target_feasible_obj=stage1_target_obj,
            )

            print(f"[ALNS] Done: best structure found for γ={g}.")
            prof.report(tag=f"[Profiler γ={g}]")
        else:
            init_sol_best = init_sol
            budget_stage2 = 0.0

        # ========= 第 0 层结构检查（用 exact）=========
        from alns_min import basic_feasibility_check_level0
        _ = basic_feasibility_check_level0(
            routes=init_sol_best.routes,
            shelf_seq=init_sol_best.shelf_seq,
            place=init_sol_best.place,
            evaluator=evaluator_exact,
            task_shelf_mapping=task_shelf_mapping,
            verbose=bool(args.verbose),
        )

        # ========= 评估 makespan =========
        base_ms, _ = evaluator_exact.evaluate(init_sol.routes, init_sol.shelf_seq, init_sol.place)
        best_ms, best_diag = evaluator_exact.evaluate(
            init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
        )
        if not math.isfinite(float(best_ms)):
            if fast_post_mode and float(budget_stage2) > 0.0:
                print(f"[ALNS] exact infeasible after fast stage, start rescue stage ({budget_stage2:.1f}s)")
                prof_rescue = EvalProfiler(evaluator_exact)
                init_sol_best = alns_minimize(
                    init=init_sol_best,
                    evaluator=prof_rescue,
                    iters=min(int(args.alns_iters), 220),
                    start_T=0.8,
                    cool=0.996,
                    S_near_by_j=S_near_by_j,
                    seed=int(args.seed) + 7919,
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=True,
                    enable_shelf_tune=False,
                    shelf_init=shelf_init_for_alns,
                    speed_profile="turbo",
                    time_budget_sec=float(budget_stage2),
                    relabel_interval=0,
                )
                prof_rescue.report(tag=f"[Rescue γ={g}]")
                best_ms, best_diag = evaluator_exact.evaluate(
                    init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
                )

        if (not math.isfinite(float(best_ms))) and no_seed:
            extra_budget = max(130.0, (float(total_budget) * 5.0 if float(total_budget) > 0.0 else 0.0))
            init_emg = init_sol_best
            print(f"[ALNS] no-seed emergency fallback: balanced/exact solve ({extra_budget:.1f}s)")
            seed_try = [int(args.seed) + 17, int(args.seed) + 7919, int(args.seed)]
            remaining_budget = float(extra_budget)
            for idx, s_try in enumerate(seed_try):
                left = max(1, len(seed_try) - idx)
                per_try_budget = max(25.0, remaining_budget / float(left))
                per_try_budget = min(per_try_budget, remaining_budget)
                emergency_evaluator = evaluator_hard_partial if int(g) == 0 else evaluator_fast
                prof_emg = EvalProfiler(emergency_evaluator)
                init_sol_try = alns_minimize(
                    init=init_emg,
                    evaluator=prof_emg,
                    iters=max(800, int(args.alns_iters)),
                    start_T=1.0,
                    cool=0.996,
                    S_near_by_j=S_near_by_j,
                    seed=int(s_try),
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=True,
                    enable_shelf_tune=True,
                    shelf_init=shelf_init_for_alns,
                    speed_profile="balanced",
                    time_budget_sec=float(per_try_budget),
                    relabel_interval=24,
                    eval_budget_total=56000,
                    eval_budget_heavy=33000,
                    target_feasible_obj=580.0,
                )
                prof_emg.report(tag=f"[Emergency γ={g} seed={s_try}]")
                ms_try, diag_try = evaluator_exact.evaluate(
                    init_sol_try.routes, init_sol_try.shelf_seq, init_sol_try.place
                )
                if math.isfinite(float(ms_try)):
                    init_sol_best = init_sol_try
                    best_ms, best_diag = float(ms_try), diag_try
                    print(f"[ALNS] no-seed emergency succeeded with seed={s_try} | cmax={best_ms:.2f}")
                    break
                remaining_budget = max(0.0, remaining_budget - float(per_try_budget))

        if (not math.isfinite(float(best_ms))) and no_seed and int(g) != 0:
            print("[ALNS] no-seed deterministic fallback: single-AGV topo safe seed.")
            safe_seed = build_safe_chain_ws_seed(
                init_sol_best,
                evaluator_exact=evaluator_exact,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init=shelf_init_for_alns,
                seed=int(args.seed) + 424242,
                tries=(120 if int(g) == 0 else 80),
            )
            ms_safe, diag_safe = evaluator_exact.evaluate(
                safe_seed.routes, safe_seed.shelf_seq, safe_seed.place
            )
            if math.isfinite(float(ms_safe)):
                init_sol_best = safe_seed
                best_ms, best_diag = float(ms_safe), diag_safe
                print(f"[ALNS] deterministic fallback succeeded | cmax={best_ms:.2f}")
            else:
                init_sol_best = safe_seed

        if (not math.isfinite(float(best_ms))) and no_seed:
            print("[ALNS] no-seed chain-violation shelf-seq repair fallback.")
            repaired_seed = repair_shelf_seq_by_chain_violations(
                init_sol_best,
                evaluator_exact=evaluator_exact,
                task_shelf_mapping=task_shelf_mapping,
                max_steps=(160 if int(g) == 0 else 80),
            )
            ms_rep, diag_rep = evaluator_exact.evaluate(
                repaired_seed.routes, repaired_seed.shelf_seq, repaired_seed.place
            )
            if math.isfinite(float(ms_rep)):
                init_sol_best = repaired_seed
                best_ms, best_diag = float(ms_rep), diag_rep
                print(f"[ALNS] chain-violation repair fallback succeeded | cmax={best_ms:.2f}")

        if (not math.isfinite(float(best_ms))) and no_seed:
            print("[ALNS] no-seed chain-violation route relink fallback.")
            relink_seed = repair_chain_violations_by_route_relink(
                init_sol_best,
                evaluator_exact=evaluator_exact,
                task_shelf_mapping=task_shelf_mapping,
                max_steps=(120 if int(g) == 0 else 100),
            )
            ms_relink, diag_relink = evaluator_exact.evaluate(
                relink_seed.routes, relink_seed.shelf_seq, relink_seed.place
            )
            init_sol_best = relink_seed
            if math.isfinite(float(ms_relink)):
                best_ms, best_diag = float(ms_relink), diag_relink
                print(f"[ALNS] route relink fallback succeeded | cmax={best_ms:.2f}")

        if no_seed and math.isfinite(float(best_ms)):
            if int(g) == 0 and float(best_ms) <= 500.0 and int(args.alns_iters) <= 300:
                print("[ALNS] no-seed polish skipped: strong exact seed already found.")
            elif (run_profile == "turbo") and (total_budget <= 0.0) and (int(args.alns_iters) >= 1000):
                print("[ALNS] no-seed polish skipped in turbo long-run mode.")
            elif (run_profile == "turbo") and (int(args.alns_iters) <= 300):
                print("[ALNS] no-seed polish skipped in turbo short-run mode.")
            else:
                print("[ALNS] no-seed polish stage: short fast ALNS refinement.")
                polish_evaluator = evaluator_fast if (run_profile == "turbo") else (evaluator_hard_partial if int(g) == 0 else evaluator_fast)
                prof_polish = EvalProfiler(polish_evaluator)
                polish_sol = alns_minimize(
                    init=init_sol_best,
                    evaluator=prof_polish,
                    iters=min(320, max(120, int(args.alns_iters) // 2)),
                    start_T=0.9,
                    cool=0.997,
                    S_near_by_j=S_near_by_j,
                    seed=int(args.seed) + 271828,
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=(run_profile != "turbo"),
                    enable_shelf_tune=(run_profile != "turbo"),
                    shelf_init=shelf_init_for_alns,
                    speed_profile=("turbo" if run_profile == "turbo" else "balanced"),
                    time_budget_sec=(None if total_budget <= 0.0 else 35.0),
                    relabel_interval=20,
                    enable_strong_init=False,
                    feasible_first=False,
                    eval_budget_total=(12000 if run_profile == "turbo" else 42000),
                    eval_budget_heavy=(5000 if run_profile == "turbo" else 26000),
                )
                prof_polish.report(tag=f"[Polish γ={g}]")
                ms_polish, diag_polish = evaluator_exact.evaluate(
                    polish_sol.routes, polish_sol.shelf_seq, polish_sol.place
                )
                if math.isfinite(float(ms_polish)) and float(ms_polish) < float(best_ms) - 1e-9:
                    init_sol_best = polish_sol
                    best_ms, best_diag = float(ms_polish), diag_polish
                    print(f"[ALNS] no-seed polish improved cmax -> {best_ms:.2f}")

        if not math.isfinite(float(best_ms)):
            # Historical-seed fallback is disabled by design.
            pass

        if not math.isfinite(float(best_ms)):
            # Keep the best-found structure from no-seed repair chain for diagnostics/export.
            # Do not overwrite it with the raw heuristic init.
            best_ms, best_diag = evaluator_exact.evaluate(
                init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
            )
        print(f"[ALNS] Exact makespan: base={base_ms:.2f} → best={best_ms:.2f} (Δ={base_ms - best_ms:+.2f})")

        if args.verbose:
            print_alns_solution(g, init_sol_best, evaluator_exact)

        if fast_post_mode:
            # turbo: keep fast path, but sync place with end_shelf_final once for consistent replay.
            if isinstance(best_diag, dict) and ("end_shelf_final" in best_diag):
                for j, s in best_diag["end_shelf_final"].items():
                    if int(j) in J:
                        init_sol_best.place[int(j)] = int(s)
            ms_final, diag_final = evaluator_exact.evaluate(
                init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
            )
            ms_final = float(ms_final)
            if not math.isfinite(ms_final):
                ms_final, diag_final = float(best_ms), best_diag
        else:
            # === WS 块内重排一次（让结构更稳定；也让 bundle 更稳定） ===
            routes_wsfix: dict[int, list[int]] = {}
            for r, seq in init_sol_best.routes.items():
                routes_wsfix[int(r)] = reorder_contiguous_ws_blocks(seq, pi, ws_fixed_seq)

            ms_wsfix, diag_wsfix = evaluator_exact.evaluate(routes_wsfix, init_sol_best.shelf_seq, init_sol_best.place)
            print(f"[WarmStart] WS-block fix re-eval (Exact): makespan={ms_wsfix:.2f}")

            init_sol_best = InitialSolution(
                routes=routes_wsfix,
                shelf_seq=init_sol_best.shelf_seq.copy(),
                place=init_sol_best.place.copy()
            )
            if isinstance(diag_wsfix, dict) and ("end_shelf_final" in diag_wsfix):
                for j, s in diag_wsfix["end_shelf_final"].items():
                    if int(j) in J:
                        init_sol_best.place[int(j)] = int(s)

            # 再算一次，确保 diag 与最终 place 一致（只在需要导出/详细时才做）
            if need_full_diag:
                ms_final, diag_final = evaluator_exact.evaluate(
                    init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
                )
            else:
                ms_final, diag_final = ms_wsfix, diag_wsfix

        prev_best_sol = init_sol_best

        # ===== 保存 ALNS bundle（cross-gamma 用）=====
        # 1) 用 event gate 跑一次，抽 σ（站位顺序）
        # turbo 也保留该步骤，保证 bundle 在 cross-gamma 下复评稳定。
        cell_sigma = build_cell_sigma_from_event_run(
            evaluator_factory=evaluator_factory_for_cross,
            eval_gamma=int(g),  # 用“当前 gamma 的 event 仿真”抽顺序
            routes=init_sol_best.routes,
            shelf_seq=init_sol_best.shelf_seq,
            place=init_sol_best.place,
            J_set=set(int(x) for x in J),
        )

        # 可选：你关心的 cell=23 打印出来看看
        if 23 in cell_sigma:
            print(f"[SIGMA] srcG={g} cell=23 order={cell_sigma[23]}")

        # 2) 写入 bundle
        alns_path = save_bundle_json(
            prefix=prefix,
            tag="alns",
            gamma=g,
            routes=init_sol_best.routes,
            shelf_seq=init_sol_best.shelf_seq.copy(),
            place=init_sol_best.place.copy(),
            cmax=float(ms_final),
            outdir="solution_exports",
            cell_sigma=cell_sigma,
        )

        alns_bundles_by_gamma[int(g)] = load_bundle_json(alns_path)
        print(f"[ALNS] saved bundle -> {alns_path}")
        if allow_history_seed_io and math.isfinite(float(ms_final)):
            try:
                with open(cache_bundle_path, "w", encoding="utf-8") as f:
                    json.dump(alns_bundles_by_gamma[int(g)], f, ensure_ascii=False, indent=2)
                print(f"[ALNS] turbo cache updated: {cache_bundle_path}")
            except Exception as e:
                print(f"[ALNS] turbo cache update failed: {type(e).__name__}: {e}")

        # ===== 可选：导出 evaluator diag（ALNS-only 版）=====
        if args.export_eval_diag and isinstance(diag_final, dict):
            try:
                export_evaluator_diagnostics(
                    prefix=prefix,
                    export_tag="alns_bundle",
                    source_gamma=int(g),
                    eval_gamma=int(getattr(evaluator_exact, "gamma", g)),
                    routes=init_sol_best.routes,
                    shelf_seq=init_sol_best.shelf_seq,
                    place=init_sol_best.place,
                    milp_cmax=float("nan"),
                    eval_cmax=float(ms_final),
                    diag=diag_final,
                    outdir="solution_exports",
                )
                print(f"[EXPORT-EVAL] ALNS diag exported for γ={g}")
            except Exception as e:
                print(f"[EXPORT-EVAL] ALNS export failed: {type(e).__name__}: {e}")

# =========================
# Cross-gamma 检查（统一跑）
    # =========================
    # =========================
    # Cross-gamma 检查（只对 ALNS）
    # =========================
    if args.cross_gamma_check:
        print("\n" + "=" * 70)
        print("[CrossGamma] start cross-gamma evaluation suite (ALNS only)")
        print("=" * 70)

        if alns_bundles_by_gamma:
            run_cross_gamma_check(
                prefix=prefix,
                tag="alns",
                bundles_by_source_gamma=alns_bundles_by_gamma,
                eval_gammas=gamma_list,
                evaluator_factory=evaluator_factory_for_cross,
                milp_opt_by_eval_gamma=None,   # ✅ ALNS-only：不算 regret
                outdir="solution_exports",
            )
        else:
            print("[CrossGamma] WARN: no ALNS bundles collected, skip.")

if __name__ == "__main__":
    main()
