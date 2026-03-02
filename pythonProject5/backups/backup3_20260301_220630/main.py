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
        print(f"[ALIGN-DETAIL] WARN: 鎵句笉鍒?MILP 鐨?p/q CSV锛堝皾璇曡繃锛歿cand_paths}锛?)
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
        print(f"[ALIGN-TEST] 鎵句笉鍒?bundle 鏂囦欢锛歿path}")
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
        raise ValueError("tasks_df 瀛樺湪 NaN锛岃鍏堟竻娲椼€?)

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
        help="ALNS 鎼滅储妗ｄ綅锛歜alanced(璐ㄩ噺浼樺厛) / turbo(閫熷害浼樺厛锛岄粯璁?",
    )
    ap.add_argument(
        "--alns-time-budget-sec",
        type=float,
        default=0.0,
        help="ALNS 杞椂闂撮绠楋紙绉掞級銆?=0 琛ㄧず涓嶉檺鍒躲€?,
    )
    ap.add_argument(
        "--eval-layering",
        type=int,
        choices=[0, 1],
        default=0,
        help="ALNS eval layering: 0=single evaluator, 1=fast+exact layering",
    )
    ap.add_argument(
        "--max-exact-evals-per-iter",
        type=int,
        default=1,
        help="ALNS layering mode: max exact evaluations per iteration",
    )
    ap.add_argument(
        "--alns-ignore-cache",
        action="store_true",
        help="蹇界暐 turbo cache 鏂囦欢锛堢敤浜庡喎鍚姩楠岃瘉锛夈€?,
    )
    ap.add_argument(
        "--alns-no-seed",
        action="store_true",
        help="涓ユ牸浠庡ご姹傝В锛氫笉璇诲彇浠讳綍鍘嗗彶鍙绉嶅瓙锛坈ache/bridge锛夈€?,
    )

    ap.add_argument(
        "--cross-gamma-check",
        action="store_true",
        help="瀵规瘡涓?gamma 鐨?ALNS 瑙ｅ仛 cross-gamma 璇勪及鐭╅樀锛歴ource_gamma 瑙ｆ斁鍒?eval_gamma 涓嬮噸鏂拌瘎浼?,
    )
    ap.add_argument(
        "--export-eval-diag",
        action="store_true",
        help="瀵煎嚭 ALNS 瑙ｇ殑 evaluator diag/timeline/pq/robust_segments/v_arcs 鍒?solution_exports/",
    )
    ap.add_argument(`r`n        "--verbose",`r`n        type=int,`r`n        choices=[0, 1],`r`n        default=0,`r`n        help="verbose output switch: 0=off, 1=on",`r`n    )
                print(f"[EXPORT-EVAL] ALNS diag exported for 纬={g}")
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

