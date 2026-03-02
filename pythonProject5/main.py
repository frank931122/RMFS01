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
    鍖呬竴灞?evaluator锛岀粺璁?evaluate() 琚皟鐢ㄦ鏁般€佹€昏€楁椂銆佸钩鍧囪€楁椂銆?
    杩欐牱浣犺兘绔嬪埢鍒ゆ柇鐡堕鏄細evaluate 澶參 杩樻槸 璋冪敤娆℃暟鐖嗙偢銆?
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
        # 璁?alns_min 鍐呴儴鑻ヨ闂?evaluator.gamma / evaluator.J 绛夛紝涔熻兘閫忎紶
        return getattr(self.evaluator, name)

    def report(self, tag: str = "[Profiler]"):
        avg_ms = (self.total_sec / max(1, self.calls)) * 1000.0
        print(f"{tag} evaluate calls={self.calls} | total={self.total_sec:.3f}s | avg={avg_ms:.3f}ms")


def clone_initial_solution(sol: InitialSolution) -> InitialSolution:
    return InitialSolution(
        routes={int(r): [int(x) for x in (seq or [])] for r, seq in (getattr(sol, "routes", {}) or {}).items()},
        shelf_seq={int(c): [int(x) for x in (seq or [])] for c, seq in (getattr(sol, "shelf_seq", {}) or {}).items()},
        place={int(j): int(s) for j, s in (getattr(sol, "place", {}) or {}).items()},
    )


def initial_solution_signature(sol: InitialSolution) -> tuple:
    routes = getattr(sol, "routes", {}) or {}
    shelf_seq = getattr(sol, "shelf_seq", {}) or {}
    place = getattr(sol, "place", {}) or {}
    key_routes = tuple(
        (int(r), tuple(int(x) for x in (routes.get(r, []) or [])))
        for r in sorted(routes.keys())
    )
    key_shelf = tuple(
        (int(c), tuple(int(x) for x in (shelf_seq.get(c, []) or [])))
        for c in sorted(shelf_seq.keys())
    )
    key_place = tuple((int(j), int(place[j])) for j in sorted(place.keys()))
    return key_routes, key_shelf, key_place


def bundle_to_initial_solution(bundle: dict) -> InitialSolution:
    b = _bundle_intify(bundle)
    return InitialSolution(
        routes={int(r): [int(x) for x in (seq or [])] for r, seq in (b.get("routes", {}) or {}).items()},
        shelf_seq={int(c): [int(x) for x in (seq or [])] for c, seq in (b.get("shelf_seq", {}) or {}).items()},
        place={int(j): int(s) for j, s in (b.get("place", {}) or {}).items()},
    )


def build_fixed_shelf_seq_from_ws(
    *,
    task_shelf_mapping: Dict[int, object],
    ws_fixed_seq: Dict[int, List[int]],
    shelf_ids: List[int],
    tasks: Set[int],
) -> Dict[int, List[int]]:
    """
    Build a deterministic shelf order from:
      - task->shelf mapping
      - global ws fixed order
    Missing tasks are appended by task id within their shelf chain.
    """
    out: Dict[int, List[int]] = {int(c): [] for c in shelf_ids}
    seen: Set[int] = set()
    task_set = set(int(j) for j in tasks)

    def _safe_chain(j: int) -> Optional[int]:
        v = task_shelf_mapping.get(int(j), None)
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    for ws in sorted(int(w) for w in ws_fixed_seq.keys()):
        for j_raw in (ws_fixed_seq.get(int(ws), []) or []):
            j = int(j_raw)
            if (j in seen) or (j not in task_set):
                continue
            c = _safe_chain(j)
            if c is None:
                continue
            if int(c) not in out:
                out[int(c)] = []
            out[int(c)].append(int(j))
            seen.add(int(j))

    for j in sorted(task_set):
        if j in seen:
            continue
        c = _safe_chain(j)
        if c is None:
            continue
        if int(c) not in out:
            out[int(c)] = []
        out[int(c)].append(int(j))
        seen.add(int(j))

    return out


def force_solution_shelf_seq(sol: InitialSolution, fixed_shelf_seq: Dict[int, List[int]]) -> InitialSolution:
    sol.shelf_seq = {int(c): [int(x) for x in seq] for c, seq in (fixed_shelf_seq or {}).items()}
    return sol


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
    鎶?evaluator 鐨?diag锛堝寘鍚?timeline/pq/v_arcs 绛夛級鎸夆€滀汉鑳借鎳傗€濈殑褰㈠紡鎵撳嵃鍑烘潵銆?
    鐢ㄤ簬锛氬悓涓€濂楄В鍦?eval_gamma=0 鍜?eval_gamma=1 涓嬬殑瀵圭収杈撳嚭銆?
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
    print(f"[{title}] evaluator 纬={int(gamma_eval)} | makespan={float(makespan):.2f}")

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

    # 鍏煎 key 鍙兘鏄?str
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
        print("\n  v[i,j,s,s鈥橾 = 1 (derived by evaluator):")
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

            # 濡傛灉 evaluator 杈撳嚭閲屽寘鍚?layer 0 / layer G 鐨勫鐓у瓧娈碉紝灏遍澶栨墦鍗颁竴琛岋紙杩欏瀹氫綅浣犺鐨勨€溛?=1 鍏堣窇 纬=0 鍩哄噯鈥濋潪甯稿叧閿級
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

            # LB 淇℃伅锛堝鏋滄湁锛?
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
    鐩爣锛氶獙璇?evaluator(routes_by_agv, shelf_seq, place_from_x) 鏄惁绛変簬 MILP 鐨勭洰鏍囧€?
    - routes_by_agv: {agv_id: [task,...], ...}锛堢敤 MILP 杈撳嚭閭ｅ锛?
    - x_vars: Gurobi 鐨?x 鍙橀噺瀹瑰櫒锛堟敮鎸?x[j,s] 鎴?x[j][s]锛?
    """
    # 1) 浠?MILP 鐨?x[j,s] 鎶藉彇 place锛歵ask -> end_shelf_cell
    place = {}
    J = [int(j) for j in evaluator.J]
    S = [int(s) for s in evaluator.S]

    for j in J:
        chosen_s = None
        best_val = -1.0
        for s in S:
            v = None
            # 鍏煎涓ょ绱㈠紩锛歺[j,s] 鎴?x[j][s]
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
                val = float(v)    # 浠ラ槻浣犲瓨鐨勬槸鏁板€?

            if val > best_val:
                best_val = val
                chosen_s = s

        if chosen_s is not None and best_val > 0.5:
            place[j] = int(chosen_s)

    missing = [j for j in J if j not in place]
    if missing:
        print(f"[ALIGN-TEST] WARN: 浠ヤ笅浠诲姟娌℃湁浠?x[j,s] 瑙ｆ瀽鍑哄洖搴撲綅: {missing}")

    # 2) 鐢?evaluator 澶嶇畻 MILP 鐨?routes + place
    routes_chk = {int(r): [int(t) for t in seq] for r, seq in routes_by_agv.items()}
    obj_eval, diag = evaluator.evaluate(routes_chk, shelf_seq, place)

    milp_obj = float(milp_obj)
    obj_eval = float(obj_eval)
    diff = obj_eval - milp_obj

    print(f"[ALIGN-TEST] MILP obj={milp_obj:.2f} | evaluator obj={obj_eval:.2f} | diff={diff:+.2f}")

    # 3) 濡傛灉涓嶄竴鑷达紝缁欏嚭鏈€鏈夌敤鐨勪笅涓€姝ョ嚎绱細鎵撳嵃 p/q锛堝鏋?evaluator 鎻愪緵锛?
    if abs(diff) > 1e-6 and isinstance(diag, dict):
        p = diag.get("p", {}) or {}
        q = diag.get("q", {}) or {}
        print("[ALIGN-TEST] evaluator 鐨?p/q锛堜究浜庡鐓?MILP 瀵煎嚭鐨?p_q_times CSV锛?")
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
    杈撳嚭 p/q 涓嶄竴鑷寸殑浠诲姟锛屾寜 |dq| 浠庡ぇ鍒板皬鎺掑簭锛屼究浜庡畾浣嶁€滃摢涓€姝ユ妸鏃堕棿鎺ㄨ繜浜嗏€濄€?
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
    鐢?MILP 鐨?layer=0 鐨?p 鍙嶆帹鍑?pick_start_0锛?
        pick0_milp(j) = p_milp(j, layer=0) - d(home_before(j), j) - D_setup
    鍐嶄笌 evaluator timeline 鐨?pick_start_0 瀵规瘮銆?

    娉ㄦ剰锛氳繖閲?file_gamma 鐢ㄧ殑鏄€滃綋鍓嶈繖娆′紭鍖栫殑 gamma鈥濓紝浣嗚鍙栫殑鏄?layer_gamma=0銆?
    """

    if shelf_data is None or d_s_pi is None:
        print("[PICK0-CHECK] missing shelf_data or d_s_pi; skip.")
        return

    # 鉁?鍏抽敭淇锛氳 鈥滃綋鍓?gamma 鐨勬枃浠垛€濓紝浣嗙瓫 layer=0
    milp_pq0 = load_milp_pq_from_csv(prefix=prefix, file_gamma=gamma, layer_gamma=0, outdir=outdir)
    if not milp_pq0:
        print("[PICK0-CHECK] missing MILP p/q for layer=0 in CSV.")
        return

    # home_before锛氶摼棣栫敤 shelf_init锛涢摼鍐呯敤 place[prev]
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

    # evaluator 鐨?pick_start_0 浠?timeline 鎷?
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
    璇诲彇浣犲鍑虹殑 p/q CSV锛屾瀯閫?{task_id: (p, q)}銆?

    - file_gamma: 鐢ㄦ潵閫夋嫨鏂囦欢鍚嶅悗缂€锛屾瘮濡?*_gamma1.csv
    - layer_gamma: 鑻ユ枃浠堕噷鏈?'gamma' 鍒楋紙allGamma 闀胯〃锛夛紝鍒欒繘涓€姝ョ瓫閫夋煇涓€灞?gamma锛堜緥濡?0/1/2锛?
                  鑻ヤ负 None锛屽垯涓嶇瓫閫夛紙鐩存帴鏁磋〃璇伙級
    """
    cand_paths = [
        os.path.join(outdir, f"{prefix}_p_q_times_allGamma_gamma{file_gamma}.csv"),
        os.path.join(outdir, f"{prefix}_p_q_times_gamma{file_gamma}.csv"),
        os.path.join(outdir, f"{prefix}_p_q_times_allGamma.csv"),
    ]
    path = next((p for p in cand_paths if os.path.exists(p)), None)
    if path is None:
        print(f"[ALIGN-DETAIL] WARN: cannot find MILP p/q CSV (tried: {cand_paths})")
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
        print(f"[ALIGN-DETAIL] WARN: p/q CSV 鍒楀悕涓嶅尮閰嶏細{list(df.columns)}")
        return {}

    # 濡傛灉鏄?allGamma 鏂囦欢涓斾綘鎸囧畾 layer_gamma锛屽氨绛涢€夊眰
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
    gamma: int,               # bundle 鏂囦欢鍚嶉噷鐨?source_gamma
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
    璇诲彇 MILP bundle -> evaluator 澶嶇畻 -> 瀵归綈妫€鏌?
    鏂板鑳藉姏锛?
      - print_full=True锛氱敤鈥滃彲璇烩€濇柟寮忔墦鍗板畬鏁?timeline/pq 绛夛紙浣犺鐨勯偅绉嶏級
      - return_diag=True锛氳繑鍥?(diff, eval_ms, milp_cmax, diag, routes, shelf_seq, place)锛屾柟渚垮悗缁仛 milp-eval-lock
    """
    path = os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json")
    if not os.path.exists(path):
        print(f"[ALIGN-TEST] cannot find bundle file: {path}")
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

    # ====== 瀵煎嚭 evaluator 鐨勫叏閲忎腑闂撮噺 ======
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

    # ====== 浣犺鐨勨€滃畬鏁村彲璇绘墦鍗扳€?======
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

    # ====== p/q 瀵归綈缁嗗寲 ======
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

    # ====== pick_start_0 瀵归綈璇婃柇 ======
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

    # ====== 淇濈暀浣犲師鏉ョ殑璇婃柇淇℃伅 ======
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
                print(f"[ALIGN-TEST] evaluator 鐨勭摱棰堜换鍔★細Task {int(j_star)}  q={float(q2[j_star]):.2f}")
        except Exception:
            pass

    if return_diag:
        return (diff, obj_eval_f, milp_cmax, diag, routes, shelf_seq, place)

    return diff


# ====== 鏋勯€犱紭鍖栭渶瑕佺殑鏁版嵁缁撴瀯 ======
def build_task_structures(tasks_df: pd.DataFrame,
                          agv_data: dict[int, int],
                          shelf_ids: list[int]):
    need = {"Task", "Shelf", "Workstation", "Duration"}
    if not need.issubset(tasks_df.columns):
        raise ValueError(f"tasks_df 缂哄皯鍒楋細{need - set(tasks_df.columns)}")
    if tasks_df.isna().any().any():
        raise ValueError("tasks_df has NaN values; please clean inputs first.")

    tasks, task_shelf_mapping, J = {}, {}, set()
    shelf_usage = {sid: [] for sid in shelf_ids}
    for _, r in tasks_df.sort_values("Task").iterrows():
        tid = int(r["Task"]); sid = int(r["Shelf"]); ws = int(r["Workstation"]); dur = float(r["Duration"])
        tasks[tid] = (ws, dur, None, None)
        task_shelf_mapping[tid] = sid
        J.add(tid)
        shelf_usage[sid].append(tid)

    # 铏氭嫙浠诲姟
    J0, Jd = {}, {}
    for aid in agv_data:
        J0[aid] = 1000 + int(aid)
        Jd[aid] = 2000 + int(aid)
        tasks[J0[aid]] = (None, 0, None, None)
        tasks[Jd[aid]] = (None, 0, None, None)
        task_shelf_mapping[J0[aid]] = None
        task_shelf_mapping[Jd[aid]] = None

    # 璐ф灦鍒濆铏氭嫙
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
    print(f"[EXPORT] 鍐欏嚭 solution_exports/{prefix}_taskInfo.csv, solution_exports/{prefix}_taskShelf.csv")


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
      - g/h start  (鍏抽敭锛歨 琛ㄧず鈥滆揣鏋跺湪璇?cell 涓婂仠鐣欑粨鏉熸椂闂粹€濓紝涓嶆槸鍒拌揪鏃堕棿)

    place_override:
      - 鑻ユ彁渚涳紝鍒?x 鐩存帴鐢ㄨ繖涓紙纭繚鈥滈攣浣?MILP 鐨?x鈥濓級锛岃€屼笉鏄敤 evaluator 鐨?end_shelf_final锛堥伩鍏?evaluator repair 鏀瑰啓 x锛夈€?
    """
    BIG_M_TIME = 10000.0

    warm_hint: dict = {}

    # ---- w: 浠诲姟 -> AGV ----
    w_map: dict[int, int] = {}
    for r, seq in routes.items():
        rr = int(r)
        for j in (seq or []):
            jj = int(j)
            if jj in J:
                w_map[jj] = rr
    warm_hint["w"] = w_map

    # ---- z: 瀹屾暣 j0 -> ... -> jd ----
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

    # ---- v: evaluator 鎺ㄥ鐨?V_arcs ----
    V_raw = details.get("V_arcs", []) or []
    v_list: list[tuple[int, int, int, int]] = []
    for arc in V_raw:
        if not isinstance(arc, (list, tuple)) or len(arc) != 4:
            continue
        i, j, s, sp = arc
        v_list.append((int(i), int(j), int(s), int(sp)))
    warm_hint["v"] = v_list

    # ---- x: 浠诲姟鍥炲簱浣嶏紙EndShelf锛?----
    x_map: dict[int, int] = {}

    if place_override is not None:
        # 鉁?寮哄埗浣跨敤 MILP 鐨?x锛堟垨浣犳兂閿佷綇鐨?place锛?
        for j, s in (place_override or {}).items():
            jj = int(j)
            if jj in J:
                x_map[jj] = int(s)
    else:
        # fallback锛氫娇鐢?evaluator 鐨?end_shelf_final
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

    # ---- p/q: 鐩存帴鐢?evaluator 鐨?p/q ----
    p_map = {int(j): float(t) for j, t in (details.get("p", {}) or {}).items() if int(j) in J}
    q_map = {int(j): float(t) for j, t in (details.get("q", {}) or {}).items() if int(j) in J}
    warm_hint["p"] = p_map
    warm_hint["q"] = q_map

    # ========== 鍏抽敭锛氭瀯閫?g/h ==========
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

    # 閾惧唴鍚庣户銆佸熬浠诲姟
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

    # (1) 鐪熷疄浠诲姟锛歡=鍒拌揪 end cell 鐨勬椂闂达紱h=璇?cell 鍋滅暀缁撴潫锛堜笅涓€娆¤鍙栬蛋 / 鎴?BIG_M锛?
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

    # (2) 铏氭嫙鍒濆浠诲姟 J_I锛歡=0锛沨=璇ラ摼棣栦换鍔＄殑 pick_start锛堣〃绀哄垵濮嬪崰鐢ㄧ粨鏉燂級
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
    鎶?bundle 鐨?routes/shelf_seq/place/cell_sigma 閲?key/value 缁熶竴杞?int
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
    鐢?event gate 妯″紡璇勪及涓€娆★紝鎶藉彇姣忎釜 cell 鐨勨€滃疄闄呯珯浣嶉『搴?蟽[cell]=[task,...]鈥?
    鎺掑簭瑙勫垯锛氭寜 arrive_cell_act锛堝疄闄呰惤浣嶅紑濮嬪崰鐢ㄦ椂鍒伙級鍗囧簭銆?
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
    瀵规瘡涓?source_gamma 鐨?bundle锛屽湪 eval_gammas 涓嬮兘璺戜竴閬?evaluator.evaluate(...)
    瀵煎嚭锛?
      - {prefix}_crossGamma_{tag}_suite.csv  (闀胯〃)
      - {prefix}_crossGamma_{tag}_matrix.csv (鐭╅樀)

    鏂板锛氬綋鏌愪釜 (src_g, eval_g) 璇勪及涓?inf 鎴?feasible=False 鏃讹紝鑷姩瀵煎嚭璇ユ evaluator diag锛?
         鏂囦欢鍓嶇紑锛歿prefix}_crossFail_{tag}_srcG{src_g}_evalG{eval_g}_*
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

            # === 鏂板锛歩nf/涓嶅彲琛屾椂瀵煎嚭 diag锛堢敤浜庡畾浣嶁€滅垎鎺夊師鍥犫€濓級 ===
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
    r"\[ALNS\]\s*Exact makespan:\s*base=([^\s]+)\s*(?:鈫抾->)\s*best=([^\s]+)",
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

            # 鍏抽敭锛氶槻姝㈣鍒扳€滀笂涓€娆?run 鐨勬棫 bundle鈥?
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
                "--eval-layering", str(int(getattr(args, "eval_layering", 0) or 0)),
                "--max-exact-evals-per-iter", str(int(getattr(args, "max_exact_evals_per_iter", 1) or 1)),
                "--use-eval-cache", str(int(getattr(args, "use_eval_cache", 0) or 0)),
                "--use-shallow-copy", str(int(getattr(args, "use_shallow_copy", 0) or 0)),
                "--alns-history-seed", str(int(getattr(args, "alns_history_seed", 1) or 0)),
                "--alns-multistart-restarts", str(int(getattr(args, "alns_multistart_restarts", 2) or 1)),
                "--alns-enable-ejection-chain", str(int(getattr(args, "alns_enable_ejection_chain", 1) or 0)),
                "--alns-ejection-prob", str(float(getattr(args, "alns_ejection_prob", 0.12) or 0.0)),
                "--alns-enable-ws-micro-reorder", str(int(getattr(args, "alns_enable_ws_micro_reorder", 1) or 0)),
                "--alns-ws-micro-prob", str(float(getattr(args, "alns_ws_micro_prob", 0.16) or 0.0)),
                "--verbose", str(int(getattr(args, "verbose", 0) or 0)),
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

            # 鍏堜粠 stdout 鎶?profiler / warm/exact锛堝彲閫夛級
            met = _extract_metrics_from_stdout(out_text)
            final_src = "stdout"

            # 鉁?鏈€绋筹細浼樺厛浠?bundle JSON 璇?cmax
            cmax_bundle = _bench_try_read_bundle_cmax(bundle_path)
            if cmax_bundle is not None:
                met["final_ms"] = float(cmax_bundle)
                met["feasible"] = bool(_bench_is_finite(met["final_ms"]))
                final_src = "bundle"

            # 瀛愯繘绋嬪け璐ワ細寮哄埗鍒ゅけ璐?
            if exit_code != 0:
                met["feasible"] = False
                met["final_ms"] = float("inf")
                final_src = f"exit{exit_code}"

            print(f"[BENCH] done  gamma={g} seed={sd} exit={exit_code} final_ms={met['final_ms']} src={final_src} wall={wall:.2f}s")

            # 濡傛灉澶辫触锛岄『鎵嬭惤涓€涓棩蹇楋紝鏂逛究浣犲洖鐪嬫槸鍝潯绾︽潫鐖嗘帀浜?
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
        help="ALNS speed profile: balanced (quality) or turbo (speed).",
    )
    ap.add_argument(
        "--alns-time-budget-sec",
        type=float,
        default=0.0,
        help="ALNS time budget in seconds; 0 means no limit.",
    )
    ap.add_argument(
        "--eval-layering",
        type=int,
        choices=[0, 1],
        default=0,
        help="ALNS dual-layer evaluation: 0=off, 1=fast prescreen + exact recheck.",
    )
    ap.add_argument(
        "--max-exact-evals-per-iter",
        type=int,
        default=1,
        help="Maximum exact evaluations per ALNS iteration when eval-layering=1.",
    )
    ap.add_argument(
        "--use-eval-cache",
        type=int,
        choices=[0, 1],
        default=0,
        help="ALNS evaluation cache switch: 0=off, 1=on.",
    )
    ap.add_argument(
        "--use-shallow-copy",
        type=int,
        choices=[0, 1],
        default=0,
        help="ALNS copy strategy: 0=deepcopy, 1=structured shallow copy.",
    )
    ap.add_argument(
        "--alns-ignore-cache",
        action="store_true",
        help="Ignore ALNS cache files (cold-start validation).",
    )
    ap.add_argument(
        "--alns-no-seed",
        action="store_true",
        help="strict no-seed mode: do not load historical feasible seeds (cache/bundle/prev).",
    )
    ap.add_argument(
        "--alns-history-seed",
        type=int,
        choices=[0, 1],
        default=1,
        help="history seed switch: 0=off, 1=on (disabled when --alns-no-seed is set)",
    )
    ap.add_argument(
        "--alns-multistart-restarts",
        type=int,
        default=2,
        help="number of stage1 multi-start restarts (total iters are split across restarts)",
    )
    ap.add_argument(
        "--alns-enable-ejection-chain",
        type=int,
        choices=[0, 1],
        default=1,
        help="enable cross-vehicle large-segment ejection-chain operator",
    )
    ap.add_argument(
        "--alns-ejection-prob",
        type=float,
        default=0.12,
        help="trigger probability of ejection-chain in heavy-local stage",
    )
    ap.add_argument(
        "--alns-enable-ws-micro-reorder",
        type=int,
        choices=[0, 1],
        default=1,
        help="enable WS-neighborhood idle-window micro reorder operator",
    )
    ap.add_argument(
        "--alns-ws-micro-prob",
        type=float,
        default=0.16,
        help="trigger probability of WS micro-reorder in heavy-local stage",
    )
    ap.add_argument(
        "--shelf-seq-intermediate",
        action="store_true",
        help="Experimental: treat shelf_seq as fixed intermediate state (not searched by ALNS).",
    )

    ap.add_argument(
        "--cross-gamma-check",
        action="store_true",
        help="Run cross-gamma evaluation matrix for each ALNS solution.",
    )
    ap.add_argument(
        "--cross-gamma-independent",
        type=int,
        choices=[0, 1],
        default=0,
        help="Cross-gamma solve mode: 0=keep history-seed coupling (default), 1=independent per gamma.",
    )
    ap.add_argument(
        "--export-eval-diag",
        action="store_true",
        help="瀵煎嚭 ALNS 瑙ｇ殑 evaluator diag/timeline/pq/robust_segments/v_arcs 鍒?solution_exports/",
    )
    ap.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1],
        default=0,
        help="Verbose logs: 0=off, 1=on.",
    )
    # ===== Quick benchmark (very small data, for automation) =====
    ap.add_argument(
        "--quick-bench",
        action="store_true",
        help="Quick benchmark mode: run subprocess loops and export summary.",
    )
    ap.add_argument("--bench-iters", type=int, default=200, help="ALNS iterations per quick-bench subprocess.")
    ap.add_argument("--bench-seeds", default="0,1,2", help="Comma-separated seeds for quick-bench.")
    ap.add_argument("--bench-gammas", default="", help="Optional gamma list override for quick-bench.")
    ap.add_argument("--bench-timeout", type=int, default=1800, help="Timeout (seconds) per quick-bench subprocess.")
    ap.add_argument("--bench-compare", default="", help="Optional baseline summary JSON for PASS/FAIL comparison.")
    ap.add_argument("--bench-tol", type=float, default=0.0, help="Relative tolerance for baseline comparison.")
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
    cross_gamma_independent_mode = bool(
        int(getattr(args, "cross_gamma_independent", 0) or 0) == 1
        and bool(getattr(args, "cross_gamma_check", False))
        and (len(gamma_list) > 1)
    )
    alns_iters_limit = max(1, int(getattr(args, "alns_iters", 0) or 1))
    eval_layering_flag = int(getattr(args, "eval_layering", 0) or 0)
    max_exact_evals_per_iter_flag = int(getattr(args, "max_exact_evals_per_iter", 1) or 1)
    use_eval_cache_flag = int(getattr(args, "use_eval_cache", 0) or 0)
    use_shallow_copy_flag = int(getattr(args, "use_shallow_copy", 0) or 0)
    verbose_flag = bool(int(getattr(args, "verbose", 0) or 0) == 1)
    enable_ejection_chain_flag = int(getattr(args, "alns_enable_ejection_chain", 1) or 0)
    ejection_prob_flag = float(getattr(args, "alns_ejection_prob", 0.12) or 0.0)
    enable_ws_micro_reorder_flag = int(getattr(args, "alns_enable_ws_micro_reorder", 1) or 0)
    ws_micro_prob_flag = float(getattr(args, "alns_ws_micro_prob", 0.16) or 0.0)
    history_seed_flag = int(getattr(args, "alns_history_seed", 1) or 0)
    multistart_restarts_flag = max(1, int(getattr(args, "alns_multistart_restarts", 2) or 1))

    # 鉁?鍙仛 ALNS锛氬彧闇€瑕佽繖涓€涓鍣?
    alns_bundles_by_gamma: dict[int, dict] = {}

    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"tasks.csv not found: {tasks_csv}")



    # 鍦板浘
    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)

    # 浠诲姟
    tasks_df = pd.read_csv(tasks_csv)
    print(f"[TASK] 璇诲彇 tasks.csv 琛屾暟={len(tasks_df)}, WS闆嗗悎={sorted(tasks_df['Workstation'].unique())}")
    if len(tasks_df) != 11:
        print(f"[WARN] tasks.csv has {len(tasks_df)} tasks (expected 11); continue solving.")
    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1

    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )
    print("[WSOrder] fixed order:", ws_fixed_seq)

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

    # 瑙勫垯鍒濊В
    pi = {j: tasks[j][0] for j in J}
    init_sol = build_initial_solution_basic(
        J=J, R=R,
        task_shelf_mapping=task_shelf_mapping,
        shelf_data=shelf_data,
        J_I=J_I,
        pi=pi
    )

    # Experiment mode (opt-in): treat shelf_seq as an intermediate fixed variable.
    shelf_seq_intermediate_mode = bool(getattr(args, "shelf_seq_intermediate", False))
    fixed_shelf_seq: Dict[int, List[int]] = {}
    if shelf_seq_intermediate_mode:
        fixed_shelf_seq = build_fixed_shelf_seq_from_ws(
            task_shelf_mapping=task_shelf_mapping,
            ws_fixed_seq=ws_fixed_seq,
            shelf_ids=shelf_ids,
            tasks=J,
        )
        init_sol = force_solution_shelf_seq(init_sol, fixed_shelf_seq)
    if shelf_seq_intermediate_mode:
        print("[ALNS] shelf_seq-as-intermediate mode ON: fixed derived shelf order (ws + task->shelf).")

    # 璺濈涓?螖锛堢粰璇勪及鍣級
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

    # 姣忎釜浠诲姟鐨勨€滃伐浣嶆渶杩戝墠 m 涓偍浣嶁€濓紙ALNS 鐢ㄥ畠鏉ユ灇涓?x锛?
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
        # cross-gamma锛氳鈥滃彲姣斺€濓紝鎵€浠ョ敤 event gate + 鍙€?蟽 鏉ラ攣姝?cell 绔欎綅椤哄簭
        sigma_eff: Optional[Dict[int, List[int]]] = None
        if isinstance(cell_sigma, dict) and len(cell_sigma) > 0:
            sigma_eff = {int(s): [int(x) for x in (seq or [])] for s, seq in cell_sigma.items()}
        cross_cell_gate_mode = "event" if sigma_eff else "off"
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

            # Keep event-gate only when a non-empty sigma is provided; fallback to exact mode otherwise.
            cell_gate_mode=cross_cell_gate_mode,
            cell_sigma=sigma_eff,

            timeline_mode="off",
            collect_v_arcs=False,
            record_cell_repair_log=False,

            enable_init_cell_lock=True,
            init_lock_max_iters=INIT_LOCK_ITERS_CROSS,
            init_lock_tol=INIT_LOCK_TOL,
            init_lock_verbose=False,
            mode="exact",
        )

    # 鎵撳嵃 ALNS 缁撴瀯瑙?
    def print_alns_solution(gamma_val: int, sol, evaluator: RobustEvaluator):
        ms, diag = evaluator.evaluate(sol.routes, sol.shelf_seq, sol.place)
        p = diag.get("p", {}); q = diag.get("q", {})
        end_final = diag.get("end_shelf_final", {})
        V_arcs = diag.get("V_arcs", []); timeline = diag.get("timeline", [])

        print("\n----------------------------------------------------------------------")
        print(f"[ALNS] solution summary (gamma={gamma_val})")
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
            print("\n  v[i,j,s,s'] = 1 锛堣瘎浼板櫒鎺ㄥ锛岀敤浜庤瘖鏂級")
            for (i, jj, s, sp) in V_arcs:
                print(f"    v[{i},{jj},{s},{sp}] = 1")

        if timeline:
            print("\n  --- Timeline (璇﹀敖) ---")
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

    # 閫?纬 姹傝В
    # 閫?纬 姹傝В锛堝彧璺?ALNS锛?
    prev_best_sol: InitialSolution | None = None
    for g in gamma_list:
        print("\n" + "=" * 70)
        print(f"[RUN] Start ALNS: gamma={g}")
        print("=" * 70)

        need_full_diag = bool(args.export_eval_diag or verbose_flag)
        requested_profile = str(getattr(args, "alns_speed_profile", "balanced") or "balanced").strip().lower()
        no_seed_cli = bool(getattr(args, "alns_no_seed", False))
        no_seed = bool(no_seed_cli)
        if no_seed:
            print("[ALNS] strict no-seed mode: skip all historical seeds.")
        else:
            if cross_gamma_independent_mode:
                print("[ALNS] cross-gamma independent mode: disable history-seed/prev-gamma coupling.")
            else:
                print("[ALNS] history-seed mode enabled: cache/prev-gamma/multi-start active.")
        run_profile = "balanced" if no_seed else requested_profile
        fast_post_mode = (run_profile == "turbo")
        if no_seed and requested_profile == "turbo":
            print("[ALNS] no-seed mode: force balanced profile for from-scratch feasibility.")

        # ========= Fast evaluator锛堢粰 ALNS 鍐呭眰鐢級=========
        # ========= Fast evaluator锛堢粰 ALNS 鍐呭眰鐢級=========
        # ========= Fast evaluator锛堢粰 ALNS 鍐呭眰鐢級=========
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

            # --- turbo 涓嬪紑灏忔淇锛岄伩鍏嶄竴鐩村仠鐣欏湪鈥滀笉鍙鎯╃綒鍖衡€?---
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
            mode="fast",
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
            mode="exact",
        )

        # ========= Exact evaluator锛圓LNS 缁撴潫鍚庢牳楠?瀵煎嚭鐢級=========
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

            # --- Exact锛氬厑璁镐慨澶嶏紙瀵归綈 MILP / cross-gamma 鍙瘮锛?---
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
            mode="exact",
        )

        shelf_init_for_alns = {int(c): int(pos) for c, pos in shelf_data.items()}
        cache_bundle_path = Path("solution_exports") / f"{prefix}_alns_bundle_cache_gamma{int(g)}.json"
        use_history_seed = bool((not no_seed) and int(history_seed_flag) == 1 and (not cross_gamma_independent_mode))
        allow_history_seed_io = bool(use_history_seed and (not bool(getattr(args, "alns_ignore_cache", False))))
        feasible_seed: InitialSolution | None = None
        if use_history_seed:
            seed_candidates: list[tuple[str, InitialSolution]] = []
            seen_seed_keys: set[tuple] = set()

            def _try_add_seed(tag: str, sol_obj: InitialSolution | None) -> None:
                if sol_obj is None:
                    return
                try:
                    sig = initial_solution_signature(sol_obj)
                except Exception:
                    return
                if sig in seen_seed_keys:
                    return
                seen_seed_keys.add(sig)
                seed_candidates.append((str(tag), clone_initial_solution(sol_obj)))

            if prev_best_sol is not None:
                _try_add_seed("prev_gamma", prev_best_sol)

            if allow_history_seed_io and cache_bundle_path.exists():
                try:
                    cache_bundle = load_bundle_json(str(cache_bundle_path))
                    _try_add_seed("cache_gamma", bundle_to_initial_solution(cache_bundle))
                except Exception as e:
                    print(f"[ALNS] history cache load failed: {type(e).__name__}: {e}")

            live_bundle_path = Path("solution_exports") / f"{prefix}_alns_bundle_gamma{int(g)}.json"
            if (not bool(getattr(args, "alns_ignore_cache", False))) and live_bundle_path.exists():
                try:
                    live_bundle = load_bundle_json(str(live_bundle_path))
                    _try_add_seed("bundle_gamma", bundle_to_initial_solution(live_bundle))
                except Exception as e:
                    print(f"[ALNS] history bundle load failed: {type(e).__name__}: {e}")

            best_seed_ms = float("inf")
            for seed_tag, seed_sol in seed_candidates:
                ms_seed, _ = evaluator_exact.evaluate(seed_sol.routes, seed_sol.shelf_seq, seed_sol.place)
                if math.isfinite(float(ms_seed)):
                    print(f"[ALNS] history seed accepted ({seed_tag}) | cmax={float(ms_seed):.2f}")
                    if float(ms_seed) < float(best_seed_ms):
                        best_seed_ms = float(ms_seed)
                        feasible_seed = clone_initial_solution(seed_sol)
            if feasible_seed is None and seed_candidates:
                print("[ALNS] history seeds found but none are exact-feasible; fallback to fresh init.")

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
            if not math.isfinite(float(forced_seed_ms)):
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
                    evaluator_exact=evaluator_exact,
                    iters=min(alns_iters_limit, (520 if int(g) == 0 else 260)),
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
                    eval_layering=eval_layering_flag,
                    max_exact_evals_per_iter=max_exact_evals_per_iter_flag,
                    use_eval_cache=use_eval_cache_flag,
                    use_shallow_copy=use_shallow_copy_flag,
                    verbose=verbose_flag,
                    enable_ejection_chain=enable_ejection_chain_flag,
                    ejection_chain_prob=ejection_prob_flag,
                    enable_ws_micro_reorder=enable_ws_micro_reorder_flag,
                    ws_micro_reorder_prob=ws_micro_prob_flag,
                )
                prof_seed.report(tag=f"[SeedRepair gamma={g}]")
                forced_seed_ms, _ = evaluator_exact.evaluate(
                    forced_seed.routes, forced_seed.shelf_seq, forced_seed.place
                )
                if math.isfinite(float(forced_seed_ms)):
                    print(f"[ALNS] no-seed exact feasible seed ready after repair | cmax={float(forced_seed_ms):.2f}")
            if not math.isfinite(float(forced_seed_ms)):
                print("[ALNS] warning: exact feasible seed not found yet; ALNS will continue from best-available structure.")

        # ========= 璋冪敤 ALNS锛圥rofiler 鍖?fast evaluator锛?========
        total_budget = float(getattr(args, "alns_time_budget_sec", 0.0) or 0.0)
        if args.alns_iters and args.alns_iters > 0:
            search_evaluator = evaluator_fast
            if no_seed and int(g) == 0:
                search_evaluator = evaluator_hard_partial
            print(f"[ALNS] Start: iters={args.alns_iters}, gamma={g}")
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
                stage1_eval_budget_total = 42000
                stage1_eval_budget_heavy = 25000
                if alns_iters_limit >= 2000:
                    # Long-run mode: scale eval budgets with requested iteration cap.
                    stage1_eval_budget_total = max(stage1_eval_budget_total, 20 * alns_iters_limit)
                    stage1_eval_budget_heavy = max(stage1_eval_budget_heavy, 12 * alns_iters_limit)
                # For long-run experiments, do not early-stop just because the seed is feasible.
                # Keep the <=1000 shortcut only for short iterations.
                if int(g) == 0:
                    stage1_target_obj = 1000.0 if int(args.alns_iters) <= 300 else None
                else:
                    stage1_target_obj = 1000.0

            stage1_restarts = 1 if no_seed else int(multistart_restarts_flag)
            stage1_restarts = max(1, min(int(stage1_restarts), int(alns_iters_limit)))
            # For gamma=0 on large-instance short runs, keep full budget in one trajectory.
            if int(g) == 0 and int(alns_iters_limit) <= 1200:
                stage1_restarts = 1

            start_pool: list[tuple[str, InitialSolution]] = []
            if forced_seed is not None:
                start_pool.append(("forced_seed", clone_initial_solution(forced_seed)))
            if feasible_seed is not None:
                start_pool.append(("history_seed", clone_initial_solution(feasible_seed)))
            if (not no_seed) and (not cross_gamma_independent_mode) and (prev_best_sol is not None):
                start_pool.append(("prev_gamma", clone_initial_solution(prev_best_sol)))
            start_pool.append(("init", clone_initial_solution(init_sol)))

            dedup_pool: list[tuple[str, InitialSolution]] = []
            dedup_keys: set[tuple] = set()
            for tag_seed, sol_seed in start_pool:
                sig = initial_solution_signature(sol_seed)
                if sig in dedup_keys:
                    continue
                dedup_keys.add(sig)
                dedup_pool.append((str(tag_seed), clone_initial_solution(sol_seed)))
            start_pool = dedup_pool

            best_run_sol: InitialSolution | None = None
            best_run_ms: float = float("inf")
            remaining_iters = int(alns_iters_limit)

            for restart_idx in range(int(stage1_restarts)):
                runs_left = max(1, int(stage1_restarts) - int(restart_idx))
                iters_this = max(1, int(remaining_iters // runs_left))
                remaining_iters = max(0, int(remaining_iters - iters_this))

                if restart_idx < len(start_pool):
                    src_tag, start_seed = start_pool[int(restart_idx)]
                    init_for_alns = clone_initial_solution(start_seed)
                elif best_run_sol is not None:
                    src_tag = "carry_best"
                    init_for_alns = clone_initial_solution(best_run_sol)
                else:
                    src_tag = "init"
                    init_for_alns = clone_initial_solution(init_sol)

                budget_this = None if (budget_stage1 is None) else (float(budget_stage1) / float(stage1_restarts))
                eval_total_this = stage1_eval_budget_total
                eval_heavy_this = stage1_eval_budget_heavy
                if int(stage1_restarts) > 1:
                    if eval_total_this is not None:
                        eval_total_this = max(1000, int(math.ceil(float(eval_total_this) / float(stage1_restarts))))
                    if eval_heavy_this is not None:
                        eval_heavy_this = max(500, int(math.ceil(float(eval_heavy_this) / float(stage1_restarts))))

                run_seed = int(args.seed) + int(restart_idx) * 7919
                print(
                    f"[ALNS] stage1 restart {restart_idx + 1}/{stage1_restarts} "
                    f"source={src_tag} iters={iters_this} seed={run_seed}"
                )
                prof_run = EvalProfiler(search_evaluator)
                run_sol = alns_minimize(
                    init=init_for_alns,
                    evaluator=prof_run,
                    evaluator_exact=evaluator_exact,
                    iters=iters_this,
                    start_T=1.0,
                    cool=0.995,
                    S_near_by_j=S_near_by_j,
                    seed=run_seed,
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=True,
                    enable_shelf_tune=True,
                    shelf_init=shelf_init_for_alns,
                    speed_profile=str(run_profile),
                    time_budget_sec=budget_this,
                    enable_strong_init=True,
                    strong_init_tries=(6 if no_seed else 4),
                    strong_init_time_budget_sec=(12.0 if no_seed else 0.0),
                    eval_budget_total=eval_total_this,
                    eval_budget_heavy=eval_heavy_this,
                    target_feasible_obj=stage1_target_obj,
                    eval_layering=eval_layering_flag,
                    max_exact_evals_per_iter=max_exact_evals_per_iter_flag,
                    use_eval_cache=use_eval_cache_flag,
                    use_shallow_copy=use_shallow_copy_flag,
                    verbose=verbose_flag,
                    enable_ejection_chain=enable_ejection_chain_flag,
                    ejection_chain_prob=ejection_prob_flag,
                    enable_ws_micro_reorder=enable_ws_micro_reorder_flag,
                    ws_micro_reorder_prob=ws_micro_prob_flag,
                )
                prof_run.report(tag=f"[Profiler gamma={g} restart {restart_idx + 1}/{stage1_restarts}]")
                ms_run, _ = evaluator_exact.evaluate(run_sol.routes, run_sol.shelf_seq, run_sol.place)
                if math.isfinite(float(ms_run)):
                    print(f"[ALNS] restart {restart_idx + 1} exact cmax={float(ms_run):.2f}")
                else:
                    print(f"[ALNS] restart {restart_idx + 1} exact cmax=inf")

                if (best_run_sol is None) or (
                    math.isfinite(float(ms_run))
                    and ((not math.isfinite(float(best_run_ms))) or (float(ms_run) < float(best_run_ms) - 1e-9))
                ):
                    best_run_sol = clone_initial_solution(run_sol)
                    best_run_ms = float(ms_run)

                if (not no_seed) and math.isfinite(float(ms_run)):
                    start_pool.append((f"restart_{restart_idx + 1}", clone_initial_solution(run_sol)))

            init_sol_best = clone_initial_solution(best_run_sol) if best_run_sol is not None else clone_initial_solution(init_sol)

            print(f"[ALNS] Done: best structure found for gamma={g}.")
        else:
            init_sol_best = init_sol
            budget_stage2 = 0.0

        # ========= 绗?0 灞傜粨鏋勬鏌ワ紙鐢?exact锛?========
        from alns_min import basic_feasibility_check_level0
        _ = basic_feasibility_check_level0(
            routes=init_sol_best.routes,
            shelf_seq=init_sol_best.shelf_seq,
            place=init_sol_best.place,
            evaluator=evaluator_exact,
            task_shelf_mapping=task_shelf_mapping,
            verbose=verbose_flag,
        )

        # ========= 璇勪及 makespan =========
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
                    evaluator_exact=evaluator_exact,
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
                    eval_layering=eval_layering_flag,
                    max_exact_evals_per_iter=max_exact_evals_per_iter_flag,
                    use_eval_cache=use_eval_cache_flag,
                    use_shallow_copy=use_shallow_copy_flag,
                    verbose=verbose_flag,
                    enable_ejection_chain=enable_ejection_chain_flag,
                    ejection_chain_prob=ejection_prob_flag,
                    enable_ws_micro_reorder=enable_ws_micro_reorder_flag,
                    ws_micro_reorder_prob=ws_micro_prob_flag,
                )
                prof_rescue.report(tag=f"[Rescue gamma={g}]")
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
                    evaluator_exact=evaluator_exact,
                    iters=alns_iters_limit,
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
                    eval_layering=eval_layering_flag,
                    max_exact_evals_per_iter=max_exact_evals_per_iter_flag,
                    use_eval_cache=use_eval_cache_flag,
                    use_shallow_copy=use_shallow_copy_flag,
                    verbose=verbose_flag,
                    enable_ejection_chain=enable_ejection_chain_flag,
                    ejection_chain_prob=ejection_prob_flag,
                    enable_ws_micro_reorder=enable_ws_micro_reorder_flag,
                    ws_micro_reorder_prob=ws_micro_prob_flag,
                )
                prof_emg.report(tag=f"[Emergency gamma={g} seed={s_try}]")
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
            else:
                print("[ALNS] no-seed polish stage: short fast ALNS refinement.")
                polish_evaluator = evaluator_hard_partial if int(g) == 0 else evaluator_fast
                prof_polish = EvalProfiler(polish_evaluator)
                polish_time_budget = 35.0
                polish_eval_total = 42000
                polish_eval_heavy = 26000
                if alns_iters_limit >= 2000:
                    polish_time_budget = 120.0
                    polish_eval_total = max(polish_eval_total, 14 * alns_iters_limit)
                    polish_eval_heavy = max(polish_eval_heavy, 9 * alns_iters_limit)
                polish_sol = alns_minimize(
                    init=init_sol_best,
                    evaluator=prof_polish,
                    evaluator_exact=evaluator_exact,
                    iters=min(alns_iters_limit, max(120, alns_iters_limit // 2)),
                    start_T=0.9,
                    cool=0.997,
                    S_near_by_j=S_near_by_j,
                    seed=int(args.seed) + 271828,
                    task_shelf_mapping=task_shelf_mapping,
                    enable_place_tune=True,
                    enable_shelf_tune=True,
                    shelf_init=shelf_init_for_alns,
                    speed_profile="balanced",
                    time_budget_sec=float(polish_time_budget),
                    relabel_interval=20,
                    enable_strong_init=False,
                    feasible_first=False,
                    eval_budget_total=int(polish_eval_total),
                    eval_budget_heavy=int(polish_eval_heavy),
                    eval_layering=eval_layering_flag,
                    max_exact_evals_per_iter=max_exact_evals_per_iter_flag,
                    use_eval_cache=use_eval_cache_flag,
                    use_shallow_copy=use_shallow_copy_flag,
                    verbose=verbose_flag,
                    enable_ejection_chain=enable_ejection_chain_flag,
                    ejection_chain_prob=ejection_prob_flag,
                    enable_ws_micro_reorder=enable_ws_micro_reorder_flag,
                    ws_micro_reorder_prob=ws_micro_prob_flag,
                )
                prof_polish.report(tag=f"[Polish gamma={g}]")
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
        print(f"[ALNS] Exact makespan: base={base_ms:.2f} -> best={best_ms:.2f} (delta={base_ms - best_ms:+.2f})")

        if verbose_flag:
            print_alns_solution(g, init_sol_best, evaluator_exact)

        if fast_post_mode:
            # turbo锛氳烦杩?WS-fix 浜屾閲嶈瘎浼颁笌 full diag锛岀洿鎺ヨ惤 bundle锛屼紭鍏堥€熷害
            ms_final, diag_final = float(best_ms), best_diag
        else:
            # === WS 鍧楀唴閲嶆帓涓€娆★紙璁╃粨鏋勬洿绋冲畾锛涗篃璁?bundle 鏇寸ǔ瀹氾級 ===
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

            # 鍐嶇畻涓€娆★紝纭繚 diag 涓庢渶缁?place 涓€鑷达紙鍙湪闇€瑕佸鍑?璇︾粏鏃舵墠鍋氾級
            if need_full_diag:
                ms_final, diag_final = evaluator_exact.evaluate(
                    init_sol_best.routes, init_sol_best.shelf_seq, init_sol_best.place
                )
            else:
                ms_final, diag_final = ms_wsfix, diag_wsfix

        prev_best_sol = init_sol_best

        # ===== 淇濆瓨 ALNS bundle锛坈ross-gamma 鐢級=====
        if fast_post_mode:
            cell_sigma = {}
        else:
            # 1) 鐢?event gate 璺戜竴娆★紝鎶?蟽锛堢珯浣嶉『搴忥級
            cell_sigma = build_cell_sigma_from_event_run(
                evaluator_factory=evaluator_factory_for_cross,
                eval_gamma=int(g),  # 鐢ㄢ€滃綋鍓?gamma 鐨?event 浠跨湡鈥濇娊椤哄簭
                routes=init_sol_best.routes,
                shelf_seq=init_sol_best.shelf_seq,
                place=init_sol_best.place,
                J_set=set(int(x) for x in J),
            )

            # 鍙€夛細浣犲叧蹇冪殑 cell=23 鎵撳嵃鍑烘潵鐪嬬湅
            if 23 in cell_sigma:
                print(f"[SIGMA] srcG={g} cell=23 order={cell_sigma[23]}")

        # 2) 鍐欏叆 bundle
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

        # ===== 鍙€夛細瀵煎嚭 evaluator diag锛圓LNS-only 鐗堬級=====
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
                print(f"[EXPORT-EVAL] ALNS diag exported for gamma={g}")
            except Exception as e:
                print(f"[EXPORT-EVAL] ALNS export failed: {type(e).__name__}: {e}")

# =========================
# Cross-gamma 妫€鏌ワ紙缁熶竴璺戯級
    # =========================
    # =========================
    # Cross-gamma 妫€鏌ワ紙鍙 ALNS锛?
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
                milp_opt_by_eval_gamma=None,   # 鉁?ALNS-only锛氫笉绠?regret
                outdir="solution_exports",
            )
        else:
            print("[CrossGamma] WARN: no ALNS bundles collected, skip.")

if __name__ == "__main__":
    main()


