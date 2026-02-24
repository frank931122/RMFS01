import argparse
import os
import json
from typing import Dict, List, Tuple

import pandas as pd

from map_generator import load_map_csv, create_map_from_components
from movement_manager import MovementManager
from time_manager import TimeManager
from utils import distance
from evaluator import RobustEvaluator

# 直接复用你 main.py 里这个函数（避免复制出错）
from main import build_task_structures

# 直接复用你 check_gamma_growth.py 里的解析函数（你已经验证能读）
from check_gamma_growth import find_pq_file, load_pq_wide_or_long


def _idx2rc(idx: int, W: int) -> Tuple[int, int]:
    return divmod(int(idx) - 1, W)


def _load_bundle(prefix: str, gamma: int, outdir: str = "solution_exports"):
    path = os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"bundle not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)

    routes = {int(r): [int(t) for t in (seq or [])] for r, seq in (b.get("routes") or {}).items()}
    place = {int(j): int(s) for j, s in (b.get("place") or {}).items()}
    shelf_seq = {int(c): [int(t) for t in (seq or [])] for c, seq in (b.get("shelf_seq") or {}).items()}
    cmax = float(b.get("cmax", float("nan")))

    return path, routes, place, shelf_seq, cmax


def _build_evaluator(
    *,
    gamma: int,
    J: set,
    R: dict,
    S: dict,
    pi: dict,
    D: dict,
    J0: dict,
    Jd: dict,
    J_I: dict,
    shelf_data: dict,
    agv_data: dict,
    d_s_pi: dict,
    d_pi_s: dict,
    d_s_s: dict,
    Delta_s_pi: dict,
    Delta_pi_s: dict,
    Delta_s_s: dict,
    ws_fixed_seq: dict,
):
    return RobustEvaluator(
        J=J, R=R, S=S,
        pi=pi,
        D=D,
        J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        gamma=gamma,
        ws_fixed_seq=ws_fixed_seq,
        ws_setup_rule="flat",
        lock_place=True,
        detach_on_mismatch=False
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--gamma", type=int, required=True)
    args = ap.parse_args()

    prefix = args.prefix
    gamma = args.gamma

    # ---------- load scenario inputs (same as main) ----------
    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"tasks.csv not found: {tasks_csv}")

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

    shelf_ids = sorted(shelf_data.keys())
    (tasks, bj, hj, task_shelf_mapping, J_I, J_E, J,
     J0, Jd, shelf_virtual_tasks, unused_shelves, J_I_SI) = \
        build_task_structures(tasks_df, agv_data, shelf_ids)

    # R/S/K
    R = map_obj.extract_AGVs()
    S = {int(sp): _idx2rc(int(sp), W) for sp in sp_indices}
    K = {i + 1: _idx2rc(ws_indices[i], W) for i in range(len(ws_indices))}

    # pi/D
    pi = {j: tasks[j][0] for j in J}
    D = {j: float(tasks[j][1]) for j in J}

    # distances
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

    Delta_s_pi = dict(d_s_pi)
    Delta_pi_s = dict(d_pi_s)
    Delta_s_s = dict(d_s_s)

    # ---------- load bundle & milp pq csv ----------
    bundle_path, routes, place, shelf_seq, milp_cmax = _load_bundle(prefix, gamma)
    print(f"[OK] bundle: {bundle_path}")
    print(f"[OK] milp_cmax(from bundle) = {milp_cmax:.2f}")

    pq_path = find_pq_file(prefix, gamma)
    if not pq_path:
        raise FileNotFoundError(f"p_q_times csv not found for prefix={prefix}, gamma={gamma}")
    q0_milp, q1_milp, mode = load_pq_wide_or_long(pq_path)
    print(f"[OK] pq csv: {pq_path} (mode={mode}) | q0={len(q0_milp)} q1={len(q1_milp)}")

    # ---------- evaluate with gamma=0 and gamma=gamma ----------
    eval0 = _build_evaluator(
        gamma=0, J=J, R=R, S=S, pi=pi, D=D, J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        ws_fixed_seq=ws_fixed_seq
    )
    ms0, diag0 = eval0.evaluate(routes, shelf_seq, place)

    evalg = _build_evaluator(
        gamma=gamma, J=J, R=R, S=S, pi=pi, D=D, J0=J0, Jd=Jd, J_I=J_I,
        shelf_data=shelf_data, agv_data=agv_data,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        ws_fixed_seq=ws_fixed_seq
    )
    msg, diagg = evalg.evaluate(routes, shelf_seq, place)

    print("\n========== evaluator makespan ==========")
    print(f"evaluator(gamma=0) = {ms0:.2f}")
    print(f"evaluator(gamma={gamma}) = {msg:.2f}")
    if msg + 1e-9 < ms0:
        print("[WARN] evaluator(gamma>0) < evaluator(gamma=0) 这在最坏情况鲁棒语义下不应发生！")

    # ---------- per-task compare ----------
    q0_eval = (diag0.get("q") or {})
    qg_eval = (diagg.get("q") or {})

    rows: List[Tuple[int, float, float, float, float]] = []
    for j in sorted(J):
        milp0 = float(q0_milp.get(j, float("nan")))
        milp1 = float(q1_milp.get(j, float("nan")))
        ev0 = float(q0_eval.get(j, float("nan")))
        ev1 = float(qg_eval.get(j, float("nan")))
        rows.append((j, milp0, ev0, milp1, ev1))

    # 计算差值
    diff_rows = []
    for (j, milp0, ev0, milp1, ev1) in rows:
        d0 = ev0 - milp0
        d1 = ev1 - milp1
        # 只看两边都有数的任务
        if not (pd.isna(d0) or pd.isna(d1)):
            diff_rows.append((j, milp0, ev0, d0, milp1, ev1, d1))

    diff_rows.sort(key=lambda x: max(abs(x[3]), abs(x[6])), reverse=True)

    print("\n========== Top-10 per-task mismatch (eval - milp) ==========")
    for j, milp0, ev0, d0, milp1, ev1, d1 in diff_rows[:10]:
        print(
            f"Task {j:>2} | "
            f"q0: milp={milp0:>7.2f} eval={ev0:>7.2f} diff={d0:+7.2f} | "
            f"q1: milp={milp1:>7.2f} eval={ev1:>7.2f} diff={d1:+7.2f}"
        )

    print("\n[Hint] 如果你看到 q0 对齐但 q1 不对齐，基本就锁定是 evaluator 的 γ 递推/预算逻辑。")
    print("[Hint] 如果 q0 就不对齐，优先查 V_arcs / end_shelf_final / 资源占用约束是否一致。")


if __name__ == "__main__":
    main()
