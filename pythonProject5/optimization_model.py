# optimization_model.py  —— 只改接口，不改任何约束/目标/变量定义
# ===============================================================
# 改动点（接口层，保持约束/目标/变量完全不变）：
#   1) WarmHint/MIPStart：只写 Start，不锁 LB/UB；v 弧只 Start；增加输入清洗；
#      **新增**：若 warm_start 是 InitialSolution（routes/place/shelf_seq），自动转为 warm_hint。
#   2) ws_fixed_seq：稳健解析为 int 并过滤非法任务号（不改约束形式）。
#   3) IIS 打印增强（仅日志）。
#   4) **新增**：完整写入 p/q 的 .Start，并把 **C_max_AGV.Start** 设置为 hint["cmax"] 或 max(q)。
#   5) **新增**：补全 j0/jd/vt 的 w/x 的 .Start；显式调用 model.addMIPStart() 提交起点。
# 其它均与原始模型一致，约束一条不动。
# ===============================================================

import os
import re
import gurobipy as gp
from gurobipy import GRB
from itertools import combinations
from collections import defaultdict

from utils import distance
import pandas as pd

# ---------------------------------------------------------------
# >>> IFACE-ONLY：WarmHint 输入清洗与转换（不改约束）
# ---------------------------------------------------------------
def _coerce_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _is_mapping(obj):
    try:
        return isinstance(obj, dict) or (hasattr(obj, 'items') and callable(obj.items))
    except Exception:
        return False

def _warm_start_to_hint(warm_start, J, R, S, J0, Jd, J_I):
    """
    把 InitialSolution 或同构对象 (routes/place/shelf_seq) 转为“可写 .Start”的 warm_hint。
    这一步 **只做接口转换**，不改变约束语义：
      - w 由 routes 推出
      - z 由 routes + j0/jd 串起来
      - x 来自 place
      - immediate: 由 shelf_seq + J_I 还原（vt_c -> first -> ...）
    """
    if warm_start is None:
        return None

    # 1) 尝试属性取值（InitialSolution）
    routes = getattr(warm_start, "routes", None)
    place  = getattr(warm_start, "place",  None)
    shelf_seq = getattr(warm_start, "shelf_seq", None)

    # 2) 若是字典也兼容：{routes:..., place:..., shelf_seq:...}
    if routes is None and _is_mapping(warm_start):
        routes = warm_start.get("routes")
        place  = warm_start.get("place")
        shelf_seq = warm_start.get("shelf_seq")

    # 3) 结构不完整则放弃转换
    if not isinstance(routes, dict) or not isinstance(place, dict):
        return None

    # 4) 清洗/限界
    Jset = set(int(j) for j in J)
    Rset = set(int(r) for r in R)
    Sset = set(int(s) for s in S)

    # w: 由 routes 映射
    w_map = {}
    routes_clean = {}
    for r_id, seq in (routes or {}).items():
        try:
            rr = int(r_id)
        except Exception:
            continue
        if rr not in Rset:
            continue
        seq_clean = []
        if isinstance(seq, (list, tuple)):
            for j in seq:
                try:
                    jj = int(j)
                except Exception:
                    continue
                if jj in Jset:
                    seq_clean.append(jj)
                    w_map[jj] = rr
        routes_clean[rr] = seq_clean

    # x: 回库位
    x_map = {}
    for j, s in (place or {}).items():
        try:
            jj = int(j); ss = int(s)
        except Exception:
            continue
        if jj in Jset and ss in Sset:
            x_map[jj] = ss

    # z：j0 -> seq -> jd
    z_edges = []
    for r_id, seq in routes_clean.items():
        j0 = J0.get(int(r_id))
        jd = Jd.get(int(r_id))
        prev = j0
        for j in seq:
            if prev is not None and (prev in Jset or prev in set(J0.values())):
                z_edges.append((int(prev), int(j), int(r_id)))
            prev = j
        if prev is not None and jd is not None:
            z_edges.append((int(prev), int(jd), int(r_id)))

    # immediate: 由 shelf_seq + J_I 还原
    imm_edges = []
    if isinstance(shelf_seq, dict):
        for c, seq in shelf_seq.items():
            try:
                cc = int(c)
            except Exception:
                continue
            vt_c = J_I.get(cc)
            if vt_c is None:
                continue
            # 只保留真实任务
            seq_clean = [int(j) for j in (seq or []) if int(j) in Jset]
            if not seq_clean:
                continue
            # vt_c -> first
            imm_edges.append((int(vt_c), int(seq_clean[0]), cc))
            # 连续任务对
            for a, b in zip(seq_clean[:-1], seq_clean[1:]):
                imm_edges.append((int(a), int(b), cc))

    warm_hint = {
        "w": w_map,
        "x": x_map,
        "z": z_edges,
        "routes": routes_clean,
    }
    if imm_edges:
        warm_hint["immediate"] = imm_edges
    return warm_hint


def _sanitize_warm_hint_data(warm_hint, J, R, S, J0, Jd):
    """
    仅对 warm_hint 做集合过滤与类型修正；不改变约束。返回“可写入 .Start 的”干净副本。
    支持字段：
      - w: {j: r}
      - x: {j: s}
      - z: [(i,j,r), ...]
      - v: [(i,j,s,sp), ...]
      - p: {j: val}
      - q: {j: val}
      - g/h（可选，tuple key 或嵌套 dict，写 .Start 用；不改变任何约束）
      - cmax（可选）：makespan 起点
      - 可选同义：end_shelf / end_shelf_by_task / place_final
      - 可选同义：agv_routes / task_seq / routes / task_seq_by_agv
    """
    if not warm_hint:
        return None

    Jset = set(int(j) for j in J)
    Rset = set(int(r) for r in R)
    Sset = set(int(s) for s in S)
    J0set = set(int(v) for v in J0.values())
    Jdset = set(int(v) for v in Jd.values())

    out = {}

    # w: {j: r}
    if isinstance(warm_hint.get("w"), dict):
        w_clean = {}
        dropped = 0
        for j, r in warm_hint["w"].items():
            jj = _coerce_int(j); rr = _coerce_int(r)
            if jj in Jset and rr in Rset:
                w_clean[jj] = rr
            else:
                dropped += 1
        if w_clean:
            out["w"] = w_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped w items = {dropped}")

    # x / end_shelf aliases
    x_src = (warm_hint.get("x") or
             warm_hint.get("end_shelf_by_task") or
             warm_hint.get("end_shelf") or
             warm_hint.get("place_final"))
    if isinstance(x_src, dict):
        x_clean = {}
        dropped = 0
        for j, s in x_src.items():
            jj = _coerce_int(j); ss = _coerce_int(s)
            if jj in Jset and ss in Sset:
                x_clean[jj] = ss
            else:
                dropped += 1
        if x_clean:
            out["x"] = x_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped x/end_shelf items = {dropped}")

    # z: list of (i,j,r)
    if isinstance(warm_hint.get("z"), (list, tuple)):
        z_clean = []
        dropped = 0
        for tup in warm_hint["z"]:
            if not isinstance(tup, (list, tuple)) or len(tup) != 3:
                dropped += 1; continue
            i, j, r = _coerce_int(tup[0]), _coerce_int(tup[1]), _coerce_int(tup[2])
            # i 可以是 J 或 J0；j 可以是 J 或 Jd；r 必须在 R
            if r in Rset and ((i in Jset) or (i in J0set)) and ((j in Jset) or (j in Jdset)):
                z_clean.append((i, j, r))
            else:
                dropped += 1
        if z_clean:
            out["z"] = z_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped z items = {dropped}")

    # v: list of (i,j,s,sp)
    if isinstance(warm_hint.get("v"), (list, tuple)):
        v_clean = []
        dropped = 0
        for tup in warm_hint["v"]:
            if not isinstance(tup, (list, tuple)) or len(tup) != 4:
                dropped += 1; continue
            i, j, s, sp = map(_coerce_int, tup)
            # i: J or J0；j: J or Jd；s,sp in S
            if ((i in Jset) or (i in J0set)) and ((j in Jset) or (j in Jdset)) and (s in Sset) and (sp in Sset):
                v_clean.append((i, j, s, sp))
            else:
                dropped += 1
        if v_clean:
            out["v"] = v_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped v items = {dropped}")
    # immediate: list of (i,j,c)
    if isinstance(warm_hint.get("immediate"), (list, tuple)):
        imm_clean = []
        dropped = 0
        for tup in warm_hint["immediate"]:
            if not isinstance(tup, (list, tuple)) or len(tup) != 3:
                dropped += 1
                continue
            i, j, c = tup
            try:
                i = int(i); j = int(j); c = int(c)
            except Exception:
                dropped += 1
                continue
            imm_clean.append((i, j, c))
        if imm_clean:
            out["immediate"] = imm_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped immediate items = {dropped}")

    # p/q：只对 J 写
    if isinstance(warm_hint.get("p"), dict):
        p_clean = {}
        dropped = 0
        for j, val in warm_hint["p"].items():
            jj = _coerce_int(j)
            if jj in Jset:
                try:
                    p_clean[jj] = float(val)
                except Exception:
                    dropped += 1
            else:
                dropped += 1
        if p_clean:
            out["p"] = p_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped p items = {dropped}")

    if isinstance(warm_hint.get("q"), dict):
        q_clean = {}
        dropped = 0
        for j, val in warm_hint["q"].items():
            jj = _coerce_int(j)
            if jj in Jset:
                try:
                    q_clean[jj] = float(val)
                except Exception:
                    dropped += 1
            else:
                dropped += 1
        if q_clean:
            out["q"] = q_clean
        if dropped:
            print(f"[WarmHint] sanitize: dropped q items = {dropped}")

    # routes 同义：字典 {agv_id: [task,...]}
    routes = (warm_hint.get("agv_routes") or warm_hint.get("task_seq") or
              warm_hint.get("routes") or warm_hint.get("task_seq_by_agv"))
    if isinstance(routes, dict):
        routes_clean = {}
        dropped_pairs = 0
        for r_id, seq in routes.items():
            rr = _coerce_int(r_id)
            if rr not in Rset or not isinstance(seq, (list, tuple)):
                continue
            seq_clean = []
            for j in seq:
                jj = _coerce_int(j)
                if jj in Jset:
                    seq_clean.append(jj)
                else:
                    dropped_pairs += 1
            routes_clean[rr] = seq_clean
        if routes_clean:
            out["routes"] = routes_clean
        if dropped_pairs:
            print(f"[WarmHint] sanitize: dropped routes bad tasks = {dropped_pairs}")

    # 透传 g/h（仅用于写 .Start，不改变任何约束）
    # 允许两种格式：
    #   1) {(j,γ,s): val}
    #   2) {j: {γ: {s: val}}}
    for key in ("g", "h"):
        if isinstance(warm_hint.get(key), dict):
            out[key] = warm_hint.get(key)

    # cmax
    if "cmax" in warm_hint:
        try:
            out["cmax"] = float(warm_hint["cmax"])
        except Exception:
            pass

    return out or None
# ---------------------------------------------------------------
# <<< IFACE-ONLY
# ---------------------------------------------------------------

# ======== NEW: 导出工具（v / immediate / 货架序列 / 从v反推EndShelf）========
import csv, json

def _build_shelf_sequences_from_immediate(J_I, rack_to_tasks, immediate_vars):
    """
    根据 immediate[(i,j,c)] = 1 把每条货架链的真实顺序还原出来。
    返回 dict: shelf_seq[c] = [j1, j2, ...]
    """
    shelf_seq = {}
    for c, tasks_c in rack_to_tasks.items():
        vt = J_I[c]
        ext = [vt] + tasks_c
        # 先收集 i->j
        succ = {}
        for i in ext:
            for j in tasks_c:
                if i == j:
                    continue
                key = (i, j, c)
                var = immediate_vars.get(key)
                if var is not None and var.X > 0.5:
                    succ[int(i)] = int(j)
        # 从 vt 开始沿后继走
        cur = int(vt)
        seq = []
        seen = set()
        while cur in succ and succ[cur] not in seen:
            nxt = succ[cur]
            seq.append(int(nxt))
            seen.add(int(nxt))
            cur = nxt
        shelf_seq[int(c)] = seq
    return shelf_seq

def export_v_arcs_csv(v_vars, J, J0, Jd, S, out_path):
    """
    把 v[i,j,s,sp] = 1 的弧导出到 CSV。
    只导出 j 为真实任务的弧（忽略 j 为终端虚点），i 可为 vt 或 j0 或真实任务。
    """
    rows = []
    for (i, j, s, sp), var in v_vars.items():
        if var.X > 0.5:
            # 仅保留 j 是真实任务（通常 >= 1 且不在 Jd）
            if (j in J):
                rows.append({"i": int(i), "j": int(j), "s_from": int(s), "s_to": int(sp)})
    if not rows:
        print("[EXPORT] (v_arcs) 没有 v=1 的真实任务入弧，跳过。"); return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["i","j","s_from","s_to"])
        w.writeheader()
        w.writerows(rows)
    print(f"[EXPORT] v 弧 → {out_path} (rows={len(rows)})")

def export_immediate_edges_csv(immediate_vars, J_I, rack_to_tasks, out_path):
    """
    导出 immediate 边 (i,j,c) = 1 的集合（含 vt->首任务）。
    CSV 列: shelf_id, i, j
    """
    rows = []
    for c, tasks_c in rack_to_tasks.items():
        vt = J_I[c]
        ext = [vt] + tasks_c
        for i in ext:
            for j in tasks_c:
                if i == j:
                    continue
                key = (i, j, c)
                var = immediate_vars.get(key)
                if var is not None and var.X > 0.5:
                    rows.append({"shelf_id": int(c), "i": int(i), "j": int(j)})
    if not rows:
        print("[EXPORT] (immediate_edges) 没有 immediate=1，跳过。"); return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["shelf_id","i","j"])
        w.writeheader()
        w.writerows(rows)
    print(f"[EXPORT] immediate 边 → {out_path} (rows={len(rows)})")

def export_shelf_seq_csv(shelf_seq_dict, out_path):
    """
    导出每条货架链的任务顺序，CSV 列: shelf_id, seq（JSON 数组）
    """
    rows = []
    for c, seq in sorted(shelf_seq_dict.items()):
        rows.append({"shelf_id": int(c), "seq": json.dumps(list(map(int, seq)))})
    if not rows:
        print("[EXPORT] (shelf_seq) 无内容，跳过。"); return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["shelf_id","seq"])
        w.writeheader()
        w.writerows(rows)
    print(f"[EXPORT] 货架链顺序 → {out_path} (rows={len(rows)})")

def export_taskEndShelf_fromV_csv(v_vars, J, out_path):
    """
    由 v 的“入弧 j 的 s_to”唯一性，反推出每个任务 j 的回库位（与 x 一致）。
    CSV 列: Task, EndShelf
    """
    end_by_v = {}
    for (i, j, s, sp), var in v_vars.items():
        if var.X > 0.5 and (j in J):
            end_by_v[int(j)] = int(sp)
    rows = [{"Task": j, "EndShelf": s_to} for j, s_to in sorted(end_by_v.items())]
    if not rows:
        print("[EXPORT] (EndShelf_fromV) 无条目，跳过。"); return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Task","EndShelf"])
        w.writeheader()
        w.writerows(rows)
    print(f"[EXPORT] EndShelf(from v) → {out_path} (rows={len(rows)})")
# ======== /NEW ===================================================


def optimize_warehouse(
        R, S, K,
        tasks,
        AGV_positions,
        agv_positions_map,
        bj, hj,
        map_obj,
        task_shelf_mapping,
        J_I, J_E, J, J0, Jd,
        J_I_SI,
        shelf_data, agv_data,
        shelf_virtual_tasks,
        unused_shelves,
        tmax=5000,
        file_prefix: str | None = None,
        gamma_budget: int = 0,
        ws_fixed_seq: dict[int, list[int]] | None = None,   # ← 固定顺序（只做解析保护）
        warm_start=None,                                     # ← 兼容 main：可以是 InitialSolution 或 dict
        warm_hint=None,                                      # ← 兼容 main：dict
        lock_hint=None,                                      # ← 兼容 main（接口保留，不锁 v）
        # ===== bridge extras (interface-only) =====
        d_s_pi_in=None,
        d_pi_s_in=None,
        d_s_s_in=None,
        time_limit: float = 300.0,
        no_improve_limit: float = 120.0,
        bridge_quiet: bool = False,
):

    """
    搭建并求解仓储调度模型的主要函数。
    返回 (task_assignments_result, model, p, q)：
      - task_assignments_result: { agv_id: [task_id按执行顺序排] }
      - model: Gurobi模型
      - p, q: 任务开始/结束时间 变量字典
    """

    def get_initial_shelf_position(i, shelf_data, J_I):
        shelf_id_for_i = None
        for sid, i_task in J_I.items():
            if i_task == i:
                shelf_id_for_i = sid
                break
        if shelf_id_for_i is None:
            return None
        s0 = shelf_data.get(shelf_id_for_i, None)
        return s0

    # ========== 1. 创建模型 ==========
    model = gp.Model("WarehouseOptimization")
    M = 1e4  # 大数
    D_setup = 2  # 相邻任务之间的准备时间(若需要)
    Gamma = {r: int(gamma_budget) for r in R.keys()}
    maxG = max(Gamma.values()) if Gamma else 0
    Γ_range = range(maxG + 1)

    # ========== 2. 定义主要索引集合 ==========
    all_tasks = set(J).union(set(J0.values())).union(set(Jd.values())).union(set(J_I.values()))
    pi = {}
    D = {}
    for j in J:
        wst, dur, pred_j, succ_j = tasks[j]
        pi[j] = wst
        D[j] = dur

    # ========== 3. 决策变量 ==========
    x = model.addVars(all_tasks, S.keys(), vtype=GRB.BINARY, name="x")
    w = model.addVars(set(J).union(J0.values()).union(Jd.values()), R.keys(), vtype=GRB.BINARY, name="w")
    z = model.addVars(set(J).union(J0.values()), set(J).union(Jd.values()), R.keys(), vtype=GRB.BINARY, name="z")
    v = model.addVars(set(J).union(J0.values()), set(J).union(Jd.values()), S.keys(), S.keys(),
                      vtype=GRB.BINARY, name="v")

    maxG = max(Gamma.values())
    Γ_range = range(maxG + 1)

    pair_keys = [(j, jp, s) for (j, jp) in combinations(J, 2) for s in S.keys()]
    # mu = model.addVars(pair_keys, vtype=GRB.BINARY, name="mu")
    # sigma = model.addVars(pair_keys, vtype=GRB.BINARY, name="sigma")
    # tau = model.addVars(pair_keys, vtype=GRB.BINARY, name="tau")
    #
    # # 车内先后顺序（原样保留）
    # y_r = model.addVars(J, J, R.keys(), vtype=GRB.BINARY, name="y_r")

    # 占位区间
    g = model.addVars(all_tasks, range(maxG + 1), S.keys(), vtype=GRB.CONTINUOUS, lb=0, name="g")
    h = model.addVars(all_tasks, range(maxG + 1), S.keys(), vtype=GRB.CONTINUOUS, lb=0, name="h")

    # 工作站开始/结束
    p = model.addVars(all_tasks, range(maxG + 1), vtype=GRB.CONTINUOUS, lb=0, name="p")
    q = model.addVars(all_tasks, range(maxG + 1), vtype=GRB.CONTINUOUS, lb=0, name="q")

    # >>> NEW：r-分层时间链（原代码已有，这里保持）
    p_r = model.addVars(J, R.keys(), Γ_range, vtype=GRB.CONTINUOUS, lb=0, name="p_r")
    q_r = model.addVars(J, R.keys(), Γ_range, vtype=GRB.CONTINUOUS, lb=0, name="q_r")
    g_r = model.addVars(J, R.keys(), Γ_range, S.keys(), vtype=GRB.CONTINUOUS, lb=0, name="g_r")
    h_r = model.addVars(J, R.keys(), Γ_range, S.keys(), vtype=GRB.CONTINUOUS, lb=0, name="h_r")
    # <<< NEW

    C_max_AGV = model.addVar(vtype=GRB.CONTINUOUS, name="C_max_AGV")

    # ========== 4. 距离 ==========
    # ========== 4. 距离 ==========
    # bridge：若外部给了 d_s_pi/d_pi_s/d_s_s，则直接用（不依赖 K）
    if isinstance(d_s_pi_in, dict) and isinstance(d_pi_s_in, dict) and isinstance(d_s_s_in, dict) and d_s_pi_in and d_pi_s_in and d_s_s_in:
        d_s_pi = {(int(s), int(j)): float(v) for (s, j), v in d_s_pi_in.items()}
        d_pi_s = {(int(j), int(s)): float(v) for (j, s), v in d_pi_s_in.items()}
        d_s_s  = {(int(a), int(b)): float(v) for (a, b), v in d_s_s_in.items()}
    else:
        # fallback：按你原来的方式重算（需要 K）
        d_s_pi = {}
        d_pi_s = {}
        d_s_s = {}

        for j in J:
            wst = pi[j]
            if wst is None:
                continue
            if wst not in K:
                continue
            for s in S.keys():
                d_s_pi[s, j] = distance(s, wst, S, K)
                d_pi_s[j, s] = distance(wst, s, K, S)

        for s in S.keys():
            for s_prime in S.keys():
                d_s_s[s, s_prime] = distance(s, s_prime, S, S)

    # 货架 -> 任务
    rack_to_tasks = defaultdict(list)
    for j_real in J:
        c = task_shelf_mapping.get(j_real, None)
        if c is not None:
            rack_to_tasks[c].append(j_real)

    print(rack_to_tasks.items())

    def needs_shelf(j):
        return (task_shelf_mapping.get(j, None) is not None)

    # ========== 5. 约束（保持不动） ==========
    # (1)
    model.addConstrs(
        (gp.quicksum(x[j, s] for s in S) == 1
         for j in set(J).union(J0.values()).union(Jd.values())),
        name="Constraint1"
    )

    # (2)
    for sid, vt_id in J_I.items():
        s0 = shelf_data[sid]
        model.addConstr(x[vt_id, s0] == 1, name=f"Constraint2_initShelf_{vt_id}")

    for agv_id, j0_id in J0.items():
        s0 = agv_data[agv_id]
        model.addConstr(x[j0_id, s0] == 1, name=f"Constraint2_initAGV_{j0_id}")
        model.addConstr(w[j0_id, agv_id] == 1, name=f"Constraint2_initAGV_{j0_id}_Robot")
        for r_any in R.keys():
            if r_any != agv_id:
                model.addConstr(w[j0_id, r_any] == 0, name=f"Constraint2_initAGV_{j0_id}_NoOtherRobot_{r_any}")

    for agv_id, jd_id in Jd.items():
        s0 = agv_data[agv_id]
        model.addConstr(x[jd_id, s0] == 1, name=f"Constraint2_endAGV_{jd_id}")
        model.addConstr(w[jd_id, agv_id] == 1, name=f"Constraint2_endAGV_{jd_id}_Robot")
        for r_any in R.keys():
            if r_any != agv_id:
                model.addConstr(w[jd_id, r_any] == 0, name=f"Constraint2_endAGV_{jd_id}_NoOtherRobot_{r_any}")

    # (3)
    model.addConstrs((gp.quicksum(w[j, r] for r in R) == 1 for j in J), name="Constraint3")

    # (4)
    model.addConstrs(
        (gp.quicksum(z[j, j_prime, r] for j_prime in J.union(set(Jd.values())) if j_prime != j) == w[j, r]
         for j in J.union(J0.values()) for r in R),
        name="Constraint4"
    )
    # (5)
    model.addConstrs(
        (gp.quicksum(z[j, j_prime, r] for j in J.union(J0.values()) if j != j_prime) == w[j_prime, r]
         for j_prime in J.union(Jd.values()) for r in R),
        name="Constraint5"
    )

    # link p/q/g/h 与 p_r/q_r/g_r/h_r
    model.addConstrs((p[j, γ] >= p_r[j, r, γ] - M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range), name="Link_p_lower")
    model.addConstrs((p[j, γ] <= p_r[j, r, γ] + M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range), name="Link_p_upper")

    model.addConstrs((q[j, γ] >= q_r[j, r, γ] - M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range), name="Link_q_lower")
    model.addConstrs((q[j, γ] <= q_r[j, r, γ] + M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range), name="Link_q_upper")

    model.addConstrs((g[j, γ, s] >= g_r[j, r, γ, s] - M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range for s in S), name="Link_g_lower")
    model.addConstrs((g[j, γ, s] <= g_r[j, r, γ, s] + M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range for s in S), name="Link_g_upper")

    model.addConstrs((h[j, γ, s] >= h_r[j, r, γ, s] - M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range for s in S), name="Link_h_lower")
    model.addConstrs((h[j, γ, s] <= h_r[j, r, γ, s] + M * (1 - w[j, r])
                      for j in J for r in R for γ in Γ_range for s in S), name="Link_h_upper")
    # ------------------------------------------------------------
    # NEW: 禁止任何真实(需要货架)任务把回库位选到“闲置货架初始占用”的 cell
    # ------------------------------------------------------------
    blocked_cells = set(int(shelf_data[c]) for c in unused_shelves)

    for s0 in blocked_cells:
        for j in J:
            if needs_shelf(j):
                model.addConstr(
                    x[j, s0] == 0,
                    name=f"BlockIdleInitCell_s{s0}_j{j}"
                )


    # (6a)
    model.addConstrs(
        (gp.quicksum(v[j, jp, s, sp]
                     for jp in J.union(set(Jd.values())) if jp != j
                     for sp in S) == x[j, s])
        for j in J
        for s in S
    )

    # (6b)
    for r_id, j0_r in J0.items():
        s0_r = agv_data[r_id]
        model.addConstr(
            gp.quicksum(v[j0_r, jp, s0_r, sp]
                        for jp in J.union(set(Jd.values())) if jp != j0_r
                        for sp in S) == 1,
            name=f"FlowFromAGV_r{r_id}_j0{j0_r}"
        )

    # (7) pos/immediate
    pos = {}
    immediate = {}
    BIG_M = 10000
    BIG_M_OCC = BIG_M + 10
    for c, tasks_c in rack_to_tasks.items():
        vt_c = J_I[c]
        ext_c = [vt_c] + tasks_c
        for j in ext_c:
            pos[(j, c)] = model.addVar(vtype=GRB.INTEGER, name=f"pos_{j}_c{c}")
        for i in ext_c:
            for j_task in tasks_c:
                if i != j_task:
                    immediate[(i, j_task, c)] = model.addVar(vtype=GRB.BINARY, name=f"imm_{i}_{j_task}_c{c}")

    for c, tasks_c in rack_to_tasks.items():
        vt_c = J_I[c]
        model.addConstr(pos[(vt_c, c)] == 0, name=f"posInit_{vt_c}_c{c}")

    for c, tasks_c in rack_to_tasks.items():
        vt_c = J_I[c]; ext_c = [vt_c] + tasks_c
        for j_task in tasks_c:
            model.addConstr(gp.quicksum(immediate[(i, j_task, c)] for i in ext_c if i != j_task) == 1,
                            name=f"imm_unique_pred_j{j_task}_c{c}")
        for i in ext_c:
            model.addConstr(gp.quicksum(immediate[(i, j_task, c)] for j_task in tasks_c if i != j_task) <= 1,
                            name=f"imm_outdeg_le1_i{i}_c{c}")

    for c, tasks_c in rack_to_tasks.items():
        vt_c = J_I[c]; ext_c = [vt_c] + tasks_c
        for j_task in tasks_c:
            for i in ext_c:
                if i == j_task:
                    continue
                model.addConstr(
                    pos[(j_task, c)] >= pos[(i, c)] + 1 - BIG_M * (1 - immediate[(i, j_task, c)]),
                    name=f"posLo_{i}_{j_task}_c{c}"
                )
                model.addConstr(
                    pos[(j_task, c)] <= pos[(i, c)] + 1 + BIG_M * (1 - immediate[(i, j_task, c)]),
                    name=f"posUp_{i}_{j_task}_c{c}"
                )

    # (7.6)
    immX = model.addVars(
        [
            (i, jprime, c, sp)
            for c, tasks_c in rack_to_tasks.items()
            for jprime in tasks_c
            for i in ([J_I[c]] + tasks_c) if i != jprime
            for sp in S
        ],
        vtype=GRB.BINARY, name="immX"
    )
    for c, tasks_c in rack_to_tasks.items():
        ext_c = [J_I[c]] + tasks_c
        for jprime in tasks_c:
            for i in ext_c:
                if i == jprime:
                    continue
                for sp in S:
                    model.addConstr(immX[i, jprime, c, sp] <= x[i, sp], name=f"immX_lin1_i{i}_j{jprime}_c{c}_s{sp}")
                    model.addConstr(immX[i, jprime, c, sp] <= immediate[(i, jprime, c)],
                                    name=f"immX_lin2_i{i}_j{jprime}_c{c}_s{sp}")
                    model.addConstr(immX[i, jprime, c, sp] >= x[i, sp] + immediate[(i, jprime, c)] - 1,
                                    name=f"immX_lin3_i{i}_j{jprime}_c{c}_s{sp}")

    for c, tasks_c in rack_to_tasks.items():
        ext_c = [J_I[c]] + tasks_c
        for jprime in tasks_c:
            for sp in S:
                lhs = gp.quicksum(
                    v[j, jprime, s, sp]
                    for j in J.union(set(J0.values())) if j != jprime
                    for s in S
                )
                rhs = gp.quicksum(
                    immX[i, jprime, c, sp]
                    for i in ext_c if i != jprime
                )
                model.addConstr(lhs == rhs, name=f"C7_6_mapIn_c{c}_j{jprime}_s{sp}")

    # 唯一入弧
    for jprime in J:
        model.addConstr(
            gp.quicksum(
                v[j, jprime, s, sp]
                for j in J.union(set(J0.values())) if j != jprime
                for s in S for sp in S
            ) == 1,
            name=f"C29a_UniqueIn_real_j{jprime}"
        )

    for jd_task in Jd.values():
        model.addConstr(
            gp.quicksum(
                v[j, jd_task, s, sp]
                for j in J.union(set(J0.values())) if j != jd_task
                for s in S for sp in S
            ) == 1,
            name=f"C29b_UniqueIn_terminal_{jd_task}"
        )
    # === NEW: 链尾任务的格子占用“常驻到 BIG_M”（只改这一点） ===
    # 逻辑：若 j 在货架链 c 上没有后继（outdeg=0），则对所有 γ、所有被选中的回库位 s，
    #      约束 h[j,γ,s] >= BIG_M * x[j,s] ；配合你已有的 h[j,γ,s] <= BIG_M * x[j,s]，
    #      于是 h[j,γ,s] 被夹成 BIG_M（只在 x[j,s]=1 时生效）。
    for c, tasks_c in rack_to_tasks.items():
        for j in tasks_c:
            outdeg = gp.quicksum(
                immediate[(j, jprime, c)] for jprime in tasks_c if jprime != j
            )
            for γ in Γ_range:
                for s in S:
                    model.addConstr(
                        h[j, γ, s] >= BIG_M * x[j, s] - BIG_M * outdeg,
                        name=f"LastStay_lb_c{int(c)}_j{int(j)}_g{int(γ)}_s{int(s)}"
                    )
    # (8)
    model.addConstrs(
        (gp.quicksum(v[j, j_prime, s, s_prime] for s in S for s_prime in S)
         == gp.quicksum(z[j, j_prime, r] for r in R))
        for j in J.union(set(J0.values()))
        for j_prime in J.union(set(Jd.values()))
        if j != j_prime
    )

    # (9)
    model.addConstrs(
        (g[j, γ, s] == 0
         for j in set(J_I.values()) | set(J0.values())
         for γ in Γ_range
         for s in S),
        name="C9_InitG"
    )

    # (10)（原版里前后各有一遍，这里保持不动）
    model.addConstrs(
        (g[j, γ, s] <= BIG_M * x[j, s]
         for j in J for γ in Γ_range for s in S),
        name="C10_g_le_Mx"
    )
    model.addConstrs(
        (h[j, γ, s] <= BIG_M * x[j, s]
         for j in J for γ in Γ_range for s in S),
        name="C10_h_le_Mx"
    )

    # (12)
    model.addConstrs(
        (h[j, γ, s] >= BIG_M * x[j, s] - 1
         for j in J_E if needs_shelf(j)
         for γ in Γ_range for s in S),
        name="C12_End_He"
    )

    # (13)～(16)
    # ---------- (13)~(16) 库位占用冲突（含 vt；不含 j0/jd） ----------
    # 参与占用的节点：真实且需要货架的任务 + 每条货架的 vt
    #VT = set(J_I.values())
    #OCC = sorted(list(VT.union({j for j in J if needs_shelf(j)})))
    # 参与占用的节点：只包含真实且需要货架的任务（剔除虚拟起点 vt）
    # 虚拟起点 vt 的初始占用已经由 C9_InitG 和 UnusedShelf 约束覆盖了，
    # 如果把它加入 OCC，它会跟自己链上的后续任务（如果回到同一点）产生逻辑死锁。
    #OCC = sorted(list({j for j in J if needs_shelf(j)}))
    # VT = set(J_I.values())
    # OCC = sorted(list(VT.union({j for j in J if needs_shelf(j)})))
    #
    # # 触发变量：当且仅当 (i,j) 都把“同一库位 s”选为 end_s 时，mu[i,j,s] = 1
    # mu = model.addVars(OCC, OCC, S.keys(), vtype=GRB.BINARY, name="mu")
    #
    # # 先后关系：sigma=1 表示 i 在 j 之前占用完同一库位 s；tau=1 表示 j 在 i 之前
    # sigma = model.addVars(OCC, OCC, S.keys(), Γ_range, Γ_range, vtype=GRB.BINARY, name="sigma")
    # tau = model.addVars(OCC, OCC, S.keys(), Γ_range, Γ_range, vtype=GRB.BINARY, name="tau")
    #
    # # AND 线性化：mu == (x[i,s] AND x[j,s])
    # model.addConstrs(
    #     (mu[i, j, s] <= x[i, s] for i in OCC for j in OCC if i != j for s in S),
    #     name="C13a_mu_le_x_i"
    # )
    # model.addConstrs(
    #     (mu[i, j, s] <= x[j, s] for i in OCC for j in OCC if i != j for s in S),
    #     name="C13b_mu_le_x_j"
    # )
    # model.addConstrs(
    #     (mu[i, j, s] >= x[i, s] + x[j, s] - 1 for i in OCC for j in OCC if i != j for s in S),
    #     name="C13c_mu_ge_and"
    # )
    #
    # # sigma + tau = mu
    # model.addConstrs(
    #     (sigma[i, j, s, γ, γp] + tau[i, j, s, γ, γp] == mu[i, j, s]
    #      for i in OCC for j in OCC if i != j
    #      for s in S for γ in Γ_range for γp in Γ_range),
    #     name="C14_sigma_plus_tau_eq_mu"
    # )
    #
    # # 若 sigma[i,j,s,γ,γ'] = 1，则 j 的占位必须在 i 的占位之后开始（同一库位 s）
    # # 注意：不再在左边减 "- mu"，保持标准 Big-M 形式，数值更稳健
    # model.addConstrs(
    #     (g[j, γp, s] - h[i, γ, s] >= mu[i, j, s]- BIG_M_OCC* (1 - sigma[i, j, s, γ, γp])
    #      for i in OCC for j in OCC if i != j
    #      for s in S for γ in Γ_range for γp in Γ_range),
    #     name="C15_order_sigma"
    # )
    #
    # # 若 tau[i,j,s,γ,γ'] = 1，则 i 的占位必须在 j 的占位之后开始（同一库位 s）
    # model.addConstrs(
    #     (g[i, γ, s] - h[j, γp, s] >= mu[i, j, s]- BIG_M_OCC* (1 - tau[i, j, s, γ, γp])
    #      for i in OCC for j in OCC if i != j
    #      for s in S for γ in Γ_range for γp in Γ_range),
    #     name="C16_order_tau"
    # )
    # ---------- (13)~(16) 库位占用冲突（含 vt；不含 j0/jd） ----------
    VT = set(J_I.values())
    OCC = sorted(list(VT.union({j for j in J if needs_shelf(j)})))

    BIG_M = 10000
    BIG_M_OCC = BIG_M + 10
    CELL_GAP = 1.0  # 如果你 evaluator 里 cell_gap 不是 1.0，请改成同一个值

    # mu[i,j,s] = 1 当且仅当 i 和 j 都把 cell s 作为“落位 cell”
    mu = model.addVars(OCC, OCC, S.keys(), vtype=GRB.BINARY, name="mu")

    # ✅ 关键修复：sigma/tau 不再带 γ 维度（全局排序）
    sigma = model.addVars(OCC, OCC, S.keys(), vtype=GRB.BINARY, name="sigma")
    tau = model.addVars(OCC, OCC, S.keys(), vtype=GRB.BINARY, name="tau")

    # AND 线性化：mu == (x[i,s] AND x[j,s])
    model.addConstrs(
        (mu[i, j, s] <= x[i, s]
         for i in OCC for j in OCC if i != j for s in S),
        name="C13a_mu_le_x_i"
    )
    model.addConstrs(
        (mu[i, j, s] <= x[j, s]
         for i in OCC for j in OCC if i != j for s in S),
        name="C13b_mu_le_x_j"
    )
    model.addConstrs(
        (mu[i, j, s] >= x[i, s] + x[j, s] - 1
         for i in OCC for j in OCC if i != j for s in S),
        name="C13c_mu_ge_and"
    )

    # ✅ sigma + tau = mu （全局排序，不依赖 γ）
    model.addConstrs(
        (sigma[i, j, s] + tau[i, j, s] == mu[i, j, s]
         for i in OCC for j in OCC if i != j for s in S),
        name="C14_sigma_plus_tau_eq_mu"
    )

    # ✅ 若 sigma=1：对所有 γ/γp 强制 g[j,γp,s] >= h[i,γ,s] + gap
    model.addConstrs(
        (g[j, γp, s] - h[i, γ, s] >= CELL_GAP * mu[i, j, s] - BIG_M_OCC * (1 - sigma[i, j, s])
         for i in OCC for j in OCC if i != j
         for s in S for γ in Γ_range for γp in Γ_range),
        name="C15_order_sigma_allGamma"
    )

    # ✅ 若 tau=1：对所有 γ/γp 强制 g[i,γ,s] >= h[j,γp,s] + gap
    model.addConstrs(
        (g[i, γ, s] - h[j, γp, s] >= CELL_GAP * mu[i, j, s] - BIG_M_OCC * (1 - tau[i, j, s])
         for i in OCC for j in OCC if i != j
         for s in S for γ in Γ_range for γp in Γ_range),
        name="C16_order_tau_allGamma"
    )

    # vt 也满足唯一格子
    model.addConstrs(
        (gp.quicksum(x[vt, s] for s in S) == 1 for vt in J_I.values()),
        name="Constraint1_vt_unique_place"
    )
    # (24) 固定工位顺序的相邻约束 —— 只做解析保护（不改约束形式）
    # ---------------------------------------------------------------
    # >>> IFACE-ONLY：解析 ws_fixed_seq 为合法的 {ws: [tasks...]}（任务需在 J 中）
    # ---------------------------------------------------------------
    if ws_fixed_seq is not None:
        workstation_to_seq = {}
        for ws_key, seq in ws_fixed_seq.items():
            try:
                ws_id = int(ws_key)
            except Exception:
                continue
            clean_seq = []
            for t in (seq or []):
                try:
                    tt = int(t)
                except Exception:
                    continue
                if tt in J:
                    clean_seq.append(tt)
            workstation_to_seq[ws_id] = clean_seq
    else:
        workstation_to_seq = {}
        for k_id in K:
            tasks_k = sorted([jt for jt in J if pi.get(jt, None) == k_id])
            workstation_to_seq[int(k_id)] = tasks_k
    # ---------------------------------------------------------------
    # <<< IFACE-ONLY
    # ---------------------------------------------------------------

    for k_id, seq in workstation_to_seq.items():
        if len(seq) <= 1:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if pi.get(a, None) != k_id or pi.get(b, None) != k_id:
                continue
            for γ1 in Γ_range:
                for γ2 in Γ_range:
                    model.addConstr(
                        p[b, γ1] >= q[a, γ2] + D_setup,
                        name=f"C24_fix_WS{k_id}_{a}_before_{b}_g{γ1, γ2}"
                    )

    # # (25)
    # for j_p in J:
    #     c = task_shelf_mapping.get(j_p, None)
    #     if c is None:
    #         continue
    #     vt = J_I.get(c, None)
    #     if vt is None:
    #         print(f"Warning: shelf {c} missing vt!")
    #         continue
    #
    #     predecessors = [vt] + rack_to_tasks[c]
    #     for i in predecessors:
    #         if i == j_p:
    #             continue
    #         for s in S:
    #             for γ in Γ_range:
    #                 model.addConstr(
    #                     p[j_p, γ] >=
    #                     d_s_pi[(s, j_p)] * x[i, s] + h[i, γ, s] + D_setup
    #                     - BIG_M * (1 - immediate[(i, j_p, c)]),
    #                     name=f"C25_noDev_{i}->{j_p}_s{s}_γ{γ}"
    #                 )
    #                 if γ > 0:
    #                     model.addConstr(
    #                         p[j_p, γ] >=
    #                         d_s_pi[(s, j_p)] * x[i, s] + h[i, γ-1, s] + d_s_pi[(s, j_p)] * x[i, s] + D_setup
    #                         - BIG_M * (1 - immediate[(i, j_p, c)]),
    #                         name=f"C25_dev_{i}->{j_p}_s{s}_γ{γ}"
    #                     )

    # (21)
    model.addConstrs(
        (q[j, γ] >= p[j, γ] + D[j]
         for j in J
         for γ in Γ_range),
        name="Constraint21"
    )

    # (22a)
    model.addConstrs(
        g[j, γ, s] >= d_pi_s[j, s] + q[j, γ] + BIG_M * (x[j, s] - 1)
        for j in J
        for s in S
        for γ in Γ_range
    )

    # (22b)
    model.addConstrs(
        g[j, γ, s] >= d_pi_s[j, s] + d_pi_s[j, s] + q[j, γ-1] + BIG_M * (x[j, s] - 1)
        for j in J
        for γ in Γ_range if γ > 0
        for s in S
    )

    # h ≥ g
    model.addConstrs(
        (h[j, γ1, s] >= g[j, γ2, s] + M * (x[j, s] - 1)
         for j in J
         for γ1 in Γ_range
         for γ2 in Γ_range
         for s in S),
        name="Constraint22_h"
    )

    # # (26)
    # for γ in Γ_range:
    #     for c, tasks_c in rack_to_tasks.items():
    #         vt_c = J_I[c]
    #         chain = [vt_c] + tasks_c
    #         for i in chain:
    #             for j_task in J.union(J0.values()):
    #                 if i == j_task:
    #                     continue
    #                 for s in S:
    #                     for s_p in S:
    #                         for j_t in J:
    #                             if (i, j_t, c) not in immediate:
    #                                 continue
    #                             model.addConstr(
    #                                 h[i, γ, s_p] >=
    #                                 g[j_task, γ, s] + d_s_s[s, s_p]
    #                                 - BIG_M * (1 - v[j_task, j_t, s, s_p])
    #                                 - BIG_M * (1 - immediate[(i, j_t, c)]),
    #                                 name=(
    #                                     f"Constraint26_γ{γ}_{i}->{j_t}_via_{j_task}_{s}->{s_p}_shelf{c}"
    #                                 )
    #                             )
    #                             if γ > 0:
    #                                 model.addConstr(
    #                                     h[i, γ, s_p] >=
    #                                     g[j_task, γ-1, s] + d_s_s[s, s_p] + d_s_s[s, s_p]
    #                                     - BIG_M * (1 - v[j_task, j_t, s, s_p])
    #                                     - BIG_M * (1 - immediate[(i, j_t, c)]),
    #                                     name=(
    #                                         f"Constraint26_γ{γ}_{i}->{j_t}_via_{j_task}_{s}->{s_p}_shelf{c}"
    #                                     )
    #                                 )
    # ============================================================
    # 更安全的 Big-M：专门用于 (v/immediate) 这类“关门”约束
    # 关键点：它必须明显大于 g/h 的上界（你的 BIG_M=10000）
    # ============================================================
    BIG_M = 10000

    max_d_s_pi = max(d_s_pi.values()) if len(d_s_pi) > 0 else 0.0
    max_d_pi_s = max(d_pi_s.values()) if len(d_pi_s) > 0 else 0.0
    max_d_s_s = max(d_s_s.values()) if len(d_s_s) > 0 else 0.0
    max_D = max(D.values()) if len(D) > 0 else 0.0

    # 这个 M_LINK 的思想是：即便 h/g 取到 10000，也能保证 immediate=0 或 v=0 时 RHS 变成足够负数
    M_LINK = BIG_M + 2 * max(max_d_s_pi, max_d_pi_s, max_d_s_s) + max_D + D_setup + 10

    # 你也可以更保守一点：M_LINK = 2 * BIG_M
    # M_LINK = 2 * BIG_M

    # ============================================================
    # (25) —— 用 M_LINK 替换 BIG_M（最关键）
    # ============================================================
    for j_p in J:
        c = task_shelf_mapping.get(j_p, None)
        if c is None:
            continue
        vt = J_I.get(c, None)
        if vt is None:
            print(f"Warning: shelf {c} missing vt!")
            continue

        predecessors = [vt] + rack_to_tasks[c]
        for i in predecessors:
            if i == j_p:
                continue
            for s in S:
                for γ in Γ_range:
                    model.addConstr(
                        p[j_p, γ] >=
                        d_s_pi[(s, j_p)] * x[i, s] + h[i, γ, s] + D_setup
                        - M_LINK * (1 - immediate[(i, j_p, c)]),
                        name=f"C25_noDev_{i}->{j_p}_s{s}_γ{γ}"
                    )
                    if γ > 0:
                        model.addConstr(
                            p[j_p, γ] >=
                            d_s_pi[(s, j_p)] * x[i, s] + h[i, γ - 1, s]
                            + d_s_pi[(s, j_p)] * x[i, s] + D_setup
                            - M_LINK * (1 - immediate[(i, j_p, c)]),
                            name=f"C25_dev_{i}->{j_p}_s{s}_γ{γ}"
                        )

    # ============================================================
    # (26) —— 同样把 BIG_M 换成 M_LINK，并建议合并“两个开关”
    #       用 (2 - v - immediate) 让关门更彻底
    # ============================================================
    for γ in Γ_range:
        for c, tasks_c in rack_to_tasks.items():
            vt_c = J_I[c]
            chain = [vt_c] + tasks_c
            for i in chain:
                for j_task in J.union(J0.values()):
                    for s in S:
                        for s_p in S:
                            for j_t in J:
                                if (i, j_t, c) not in immediate:
                                    continue

                                # 激活条件：v==1 且 immediate==1
                                act = 2 - v[j_task, j_t, s, s_p] - immediate[(i, j_t, c)]

                                model.addConstr(
                                    h[i, γ, s_p] >=
                                    g[j_task, γ, s] + d_s_s[s, s_p]
                                    - M_LINK * act,
                                    name=f"Constraint26_γ{γ}_{i}->{j_t}_via_{j_task}_{s}->{s_p}_shelf{c}"
                                )

                                if γ > 0:
                                    model.addConstr(
                                        h[i, γ, s_p] >=
                                        g[j_task, γ - 1, s] + d_s_s[s, s_p] + d_s_s[s, s_p]
                                        - M_LINK * act,
                                        name=f"Constraint26_dev_γ{γ}_{i}->{j_t}_via_{j_task}_{s}->{s_p}_shelf{c}"
                                    )

    # 未用货架边界
    for c in unused_shelves:
        vt = J_I[c]
        s0 = shelf_data[c]
        for γ in Γ_range:
            model.addConstr(g[vt, γ, s0] == 0, name=f"UnusedShelf_g0_vt{vt}_s{s0}_γ{γ}")
            model.addConstr(h[vt, γ, s0] == BIG_M, name=f"UnusedShelf_hM_vt{vt}_s{s0}_γ{γ}")

    # g,h ≤ M·x —— 把所有任务都包括进来（保持原样）
    model.addConstrs(
        (g[j, γ, s] <= BIG_M * x[j, s]
         for j in all_tasks
         for γ in Γ_range
         for s in S),
        name="C10_g_le_Mx"
    )
    model.addConstrs(
        (h[j, γ, s] <= BIG_M * x[j, s]
         for j in all_tasks
         for γ in Γ_range
         for s in S),
        name="C10_h_le_Mx"
    )

    # 唯一出弧 ≤ 1
    for i in J.union(J0.values()):
        model.addConstr(
            gp.quicksum(
                v[i, j, s, s2]
                for j in J.union(Jd.values()) if j != i
                for s in S for s2 in S
            ) <= 1,
            name=f"UniqueOut_from_{i}"
        )

    # 目标函数
    model.setObjective(C_max_AGV, GRB.MINIMIZE)
    for j in J:
        for γ in Γ_range:
            model.addConstr(C_max_AGV >= q[j, γ], name=f"Cmax_ge_q_{j}_γ{γ}")

    # ---------------------------------------------------------------
    # >>> IFACE-ONLY：WarmHint/MIPStart 写入（只 .Start，不锁）
    # ---------------------------------------------------------------
    def _apply_mipstart_from_hint(
            model,
            R, S, J, J0, Jd,
            x, w, z, v, immediate,
            p, q, g, h,
            C_max_AGV,
            warm_hint_clean,
            agv_data, shelf_data, J_I,
            Γ_range,
    ):
        """
        只写 .Start，不锁 LB/UB；兼容 w/x/z/v/immediate/p/q/g/h/cmax。
        注：
          - v 的含义是 “完成 j 后从 s 出发到 s' 去拿 j' 的货”，与 x[j',·] 无关；
          - immediate 是货架链内部的先后 (i,j,c)，这里只写 Start，不锁；
          - 会自动补全 j0/jd/vt 的 w/x 的 Start（即使 hint 未给）。
        """
        if not warm_hint_clean:
            # 即使没有 hint，也补齐 j0/jd/vt 的 w/x 的 Start，提升起点一致性
            warm_hint_clean = {}

        # 识别若干字段
        w_map = warm_hint_clean.get("w")  # {j: r}
        x_map = warm_hint_clean.get("x")  # {j: s}
        routes = (warm_hint_clean.get("routes")
                  or warm_hint_clean.get("agv_routes")
                  or warm_hint_clean.get("task_seq")
                  or warm_hint_clean.get("task_seq_by_agv"))
        v_list = warm_hint_clean.get("v")  # [(i,j,s,sp), ...]
        imm_list = warm_hint_clean.get("immediate")  # [(i,j,c), ...]
        p_map = warm_hint_clean.get("p")  # {j: val}
        q_map = warm_hint_clean.get("q")  # {j: val}
        g_map = warm_hint_clean.get("g")  # {(j,γ,s): val} 或嵌套
        h_map = warm_hint_clean.get("h")
        cmax_val = warm_hint_clean.get("cmax", None)

        # ---- 基础：补齐 j0/jd/vt 的 w/x 的 Start ----
        # w：j0/jd 固定归属本车
        for r in R.keys():
            j0_r = J0.get(r)
            jd_r = Jd.get(r)
            if j0_r is not None:
                try:
                    w[j0_r, r].Start = 1.0
                except Exception:
                    pass
            if jd_r is not None:
                try:
                    w[jd_r, r].Start = 1.0
                except Exception:
                    pass

        # x：j0/jd 在 agv 初始位；vt 在各自货架初始位
        for r in R.keys():
            j0_r = J0.get(r)
            jd_r = Jd.get(r)
            s0 = agv_data[r]
            for s in S.keys():
                val = 1.0 if int(s) == int(s0) else 0.0
                if j0_r is not None:
                    try:
                        x[j0_r, s].Start = val
                    except Exception:
                        pass
                if jd_r is not None:
                    try:
                        x[jd_r, s].Start = val
                    except Exception:
                        pass
        for sid, vt in J_I.items():
            s0 = shelf_data[sid]
            for s in S.keys():
                val = 1.0 if int(s) == int(s0) else 0.0
                try:
                    x[vt, s].Start = val
                except Exception:
                    pass

        # ---- w：任务归属（Start only）----
        if isinstance(w_map, dict):
            cnt1 = cnt0 = 0
            for j in J:
                jr = w_map.get(int(j))
                for r in R.keys():
                    val = 1.0 if (jr is not None and int(jr) == int(r)) else 0.0
                    try:
                        w[j, r].Start = val
                    except Exception:
                        pass
                    if val == 1.0:
                        cnt1 += 1
                    else:
                        cnt0 += 1
            print(f"[WarmHint] applied w: set-1={cnt1}, set-0~={cnt0}")

        # ---- x：任务完工后的回库位（EndShelf）（Start only）----
        if isinstance(x_map, dict):
            cnt1 = cnt0 = 0
            for j in J:
                sj = x_map.get(int(j))
                if sj is None:
                    continue
                for s in S.keys():
                    val = 1.0 if int(s) == int(sj) else 0.0
                    try:
                        x[j, s].Start = val
                    except Exception:
                        pass
                    if val == 1.0:
                        cnt1 += 1
                    else:
                        cnt0 += 1
            print(f"[WarmHint] applied x: set-1={cnt1}, set-0~={cnt0}")

        # ---- z：由 routes 串起 j0 -> ... -> jd（Start only）----
        if isinstance(routes, dict):
            cnt = 0
            for r_id, seq in routes.items():
                try:
                    r_id = int(r_id)
                except Exception:
                    continue
                if r_id not in R:
                    continue

                # 同步写 w 的 Start（该车包含的任务置 1，其它车置 0，仅作 Start）
                if isinstance(seq, (list, tuple)):
                    for j in seq:
                        jj = _coerce_int(j)
                        if jj is None or jj not in J:
                            continue
                        for rr in R.keys():
                            try:
                                w[jj, rr].Start = 1.0 if rr == r_id else 0.0
                            except Exception:
                                pass

                # 写 z 的 Start：j0 -> j1 -> ... -> jd
                if isinstance(seq, (list, tuple)) and len(seq) > 0:
                    prev = J0.get(r_id, None)
                    for j in seq:
                        jj = _coerce_int(j)
                        if jj is None:
                            continue
                        if prev is not None and (prev in J or prev in J0.values()):
                            if (prev, jj, r_id) in z:
                                try:
                                    z[prev, jj, r_id].Start = 1.0
                                    cnt += 1
                                except Exception:
                                    pass
                        prev = jj
                    last = seq[-1]
                    jd = Jd.get(r_id, None)
                    if jd is not None and (_coerce_int(last) in J):
                        if (last, jd, r_id) in z:
                            try:
                                z[last, jd, r_id].Start = 1.0
                                cnt += 1
                            except Exception:
                                pass
            print(f"[WarmHint] applied z: set-1={cnt}")

        # ---- v：按 hint 原样写 Start（永不锁死）----
        if isinstance(v_list, (list, tuple)):
            cnt = 0
            for tup in v_list:
                if not isinstance(tup, (list, tuple)) or len(tup) != 4:
                    continue
                i, j2, s_, sp_ = tup
                try:
                    v[int(i), int(j2), int(s_), int(sp_)].Start = 1.0
                    cnt += 1
                except Exception:
                    pass
            print(f"[WarmHint] applied v: set-1={cnt} (Start only, NOT locked)")

        # ---- immediate: 仅写 Start=1，不锁 ----
        if isinstance(imm_list, (list, tuple)) and immediate is not None:
            cnt = 0
            for tup in imm_list:
                if not isinstance(tup, (list, tuple)) or len(tup) != 3:
                    continue
                i, j2, c = tup
                try:
                    i = int(i);
                    j2 = int(j2);
                    c = int(c)
                except Exception:
                    continue
                key = (i, j2, c)
                if key in immediate:
                    try:
                        immediate[key].Start = 1.0
                        cnt += 1
                    except Exception:
                        pass
            print(f"[WarmHint] applied immediate: set-1={cnt} (Start only, NOT locked)")

        # ---- p/q：对所有 γ 赋同一 Start（若提供）----
        if isinstance(p_map, dict):
            cnt = 0
            for j, val in p_map.items():
                jj = _coerce_int(j)
                if jj not in J:
                    continue
                for γ in Γ_range:
                    try:
                        p[jj, γ].Start = float(val)
                        cnt += 1
                    except Exception:
                        pass
            print(f"[WarmHint] applied p.Start for |J|={len(p_map)} × |Γ|={len(list(Γ_range))} (set≈{cnt})")

        if isinstance(q_map, dict):
            cnt = 0
            for j, val in q_map.items():
                jj = _coerce_int(j)
                if jj not in J:
                    continue
                for γ in Γ_range:
                    try:
                        q[jj, γ].Start = float(val)
                        cnt += 1
                    except Exception:
                        pass
            print(f"[WarmHint] applied q.Start for |J|={len(q_map)} × |Γ|={len(list(Γ_range))} (set≈{cnt})")

        # ---- g/h：仅按提供的 (j,γ,s) 写 Start（若提供）；不改变约束 ----
        def _iter_j_g_s(maybe_map):
            if not isinstance(maybe_map, dict):
                return
            # 形式 1：{(j,γ,s): val}
            all_tuple_keys = True
            for k in maybe_map.keys():
                if not (isinstance(k, tuple) and len(k) == 3):
                    all_tuple_keys = False
                    break
            if all_tuple_keys:
                for (jj, gg, ss), val in maybe_map.items():
                    yield _coerce_int(jj), _coerce_int(gg), _coerce_int(ss), val
                return
            # 形式 2：{j: {γ: {s: val}}}
            for jj, sub1 in maybe_map.items():
                jj = _coerce_int(jj)
                if not isinstance(sub1, dict):
                    continue
                for gg, sub2 in sub1.items():
                    gg = _coerce_int(gg)
                    if not isinstance(sub2, dict):
                        continue
                    for ss, val in sub2.items():
                        ss = _coerce_int(ss)
                        yield jj, gg, ss, val

        if isinstance(g_map, dict):
            cnt = 0
            for jj, gg, ss, val in _iter_j_g_s(g_map):
                if jj in J and ss in S and gg in set(Γ_range):
                    try:
                        g[jj, gg, ss].Start = float(val)
                        cnt += 1
                    except Exception:
                        pass
            print(f"[WarmHint] applied g.Start count≈{cnt}")

        if isinstance(h_map, dict):
            cnt = 0
            for jj, gg, ss, val in _iter_j_g_s(h_map):
                if jj in J and ss in S and gg in set(Γ_range):
                    try:
                        h[jj, gg, ss].Start = float(val)
                        cnt += 1
                    except Exception:
                        pass
            print(f"[WarmHint] applied h.Start count≈{cnt}")

        # ---- C_max_AGV.Start ----
        if cmax_val is None and isinstance(q_map, dict) and q_map:
            try:
                cmax_val = max(float(vv) for vv in q_map.values())
            except Exception:
                cmax_val = None
        if cmax_val is not None:
            try:
                C_max_AGV.Start = float(cmax_val)
                print(f"[WarmHint] applied C_max_AGV.Start = {float(cmax_val):.2f}")
            except Exception:
                pass

    def _lock_structure_from_hint(model, R, S, J, J0, Jd, x, w, z,
                                  immediate, rack_to_tasks,
                                  warm_hint_clean, lock_hint):
        """
        根据 lock_hint 把 w/x/z/immediate 锁死为给定结构；
        v 永远不锁（最多写 Start），避免因为 v 不完整导致 infeasible。
        """
        if not (warm_hint_clean and lock_hint):
            return

        want_lock_w = bool(lock_hint.get("w"))
        want_lock_x = bool(lock_hint.get("x"))
        want_lock_z = bool(lock_hint.get("z"))
        want_lock_immediate = bool(lock_hint.get("immediate"))

        # 来源优先级：显式字段 > routes 推断
        w_map = warm_hint_clean.get("w") or {}
        routes = warm_hint_clean.get("routes") or {}
        z_list = warm_hint_clean.get("z") or []
        x_map = warm_hint_clean.get("x") or {}

        # 若没有 w_map ，则由 routes 反推 w
        if want_lock_w and (not w_map) and routes:
            for r_id, seq in routes.items():
                rr = int(r_id)
                for j in (seq or []):
                    jj = int(j)
                    if jj in J:
                        w_map[jj] = rr

        # 1) 锁 w：只给指定车的条目 LB=UB=1 （利用 sum_r w[j,r]=1 自动将其它置 0）
        if want_lock_w and w_map:
            for j, r in w_map.items():
                jj = int(j); rr = int(r)
                if (jj in J) and (rr in R):
                    try:
                        w[jj, rr].LB = 1.0
                        w[jj, rr].UB = 1.0
                    except Exception:
                        pass
            print(f"[LockHint] locked w for |J_locked|={len(w_map)}")

        # 2) 锁 x：只给指定库位 LB=UB=1 （利用 sum_s x[j,s]=1 自动将其它置 0）
        if want_lock_x and x_map:
            locked_cnt = 0
            for j, s_fix in x_map.items():
                jj = int(j); ss = int(s_fix)
                if (jj in J) and (ss in S):
                    try:
                        x[jj, ss].LB = 1.0
                        x[jj, ss].UB = 1.0
                        locked_cnt += 1
                    except Exception:
                        pass
            print(f"[LockHint] locked x for |J_locked|={locked_cnt}")

        # 3) 锁 z：优先用显式 z_list；若没有，则由 routes 串 j0→…→jd 生成
        if want_lock_z:
            edges = set()
            for (i, j2, r) in (z_list or []):
                edges.add((int(i), int(j2), int(r)))

            if (not edges) and routes:
                for r, seq in routes.items():
                    r = int(r)
                    seq = [int(t) for t in (seq or [])]
                    prev = J0.get(r, None)
                    for j2 in seq:
                        if prev is not None:
                            edges.add((int(prev), int(j2), r))
                        prev = j2
                    jd = Jd.get(r, None)
                    if prev is not None and jd is not None:
                        edges.add((int(prev), int(jd), r))

            locked_e = 0
            for (i, j2, r) in edges:
                if ((i in J) or (i in set(J0.values()))) and ((j2 in J) or (j2 in set(Jd.values()))) and (r in R):
                    try:
                        z[i, j2, r].LB = 1.0
                        z[i, j2, r].UB = 1.0
                        locked_e += 1
                    except Exception:
                        pass
            print(f"[LockHint] locked z edges = {locked_e}")

        # 4) 锁 immediate（这是你现在真正缺的）
        if want_lock_immediate:
            imm_list = warm_hint_clean.get("immediate") or []
            locked_set = set()
            for tup in imm_list:
                if not isinstance(tup, (list, tuple)) or len(tup) != 3:
                    continue
                i, j2, c = tup
                try:
                    locked_set.add((int(i), int(j2), int(c)))
                except Exception:
                    continue

            shelves_in_hint = {c for (_, _, c) in locked_set}

            # 一点点 sanity check：每条 shelf 的任务数应当 == immediate=1 的边数（vt->first + 内部边）
            for c, tasks_c in rack_to_tasks.items():
                if not tasks_c:
                    continue
                cc = int(c)
                if cc not in shelves_in_hint:
                    print(f"[LockHint] WARN: shelf {cc} has tasks {tasks_c} but NO immediate edges in hint; skip locking this shelf.")
                    continue
                need_edges = len(tasks_c)
                got_edges = sum(1 for (_, _, c2) in locked_set if int(c2) == cc)
                if got_edges != need_edges:
                    print(f"[LockHint] WARN: shelf {cc} expects {need_edges} immediate edges, but hint provides {got_edges}. (May become infeasible)")

            # 真正锁：对 hint 涉及的 shelf，把所有 immediate 变量都固定成 0/1
            lock1 = lock0 = 0
            for (i, j2, c), var in immediate.items():
                cc = int(c)
                if cc not in shelves_in_hint:
                    continue  # 未提供该 shelf 的结构，就不锁它
                key = (int(i), int(j2), cc)
                try:
                    if key in locked_set:
                        var.LB = 1.0
                        var.UB = 1.0
                        lock1 += 1
                    else:
                        var.LB = 0.0
                        var.UB = 0.0
                        lock0 += 1
                except Exception:
                    pass
            print(f"[LockHint] locked immediate: set-1={lock1}, set-0={lock0}")

    def _submit_mipstart(model, groups, name="ALNS-Start"):
        """把已设置 .Start 的变量收集并 addMIPStart（兼容 dict/list/单个 Var）。"""
        start_vars, start_vals = [], []

        def _push_var(var):
            try:
                st = var.Start
            except Exception:
                st = None
            if st is not None:
                start_vars.append(var)
                start_vals.append(st)

        for g in groups:
            if isinstance(g, dict):
                for v_ in g.values():
                    _push_var(v_)
            elif isinstance(g, (list, tuple, set)):
                for v_ in g:
                    _push_var(v_)
            else:
                _push_var(g)
        if start_vars:
            try:
                model.addMIPStart(start_vars, start_vals, name=name)
                print(f"[MIPStart] Submitted {len(start_vars)} var starts.")
            except Exception as e:
                print(f"[WARN] addMIPStart failed: {e}")

    # ==== 构造可用的 warm-hint 源 ====
    # 优先使用显式 warm_hint；否则尝试把 warm_start(InitialSolution) 转为 hint；二者都没有则为 None
    warm_from_ws = _warm_start_to_hint(warm_start, J, R, S, J0, Jd, J_I) if warm_start is not None else None

    hint_input = warm_hint if warm_hint else warm_from_ws
    warm_hint_clean = _sanitize_warm_hint_data(hint_input, J, R, S, J0, Jd)

    if lock_hint and (lock_hint.get("v", False)):
        print("[WarmHint] 注意：lock_hint['v']=True 已被忽略；v 仅写 Start。")

    # 写入 .Start（不锁）
    _apply_mipstart_from_hint(model, R, S, J, J0, Jd, x, w, z, v, immediate, p, q, g, h,
                              C_max_AGV, warm_hint_clean, agv_data, shelf_data, J_I, Γ_range)


    # 若需要，锁定结构（w/x/z）；v 永远不锁
    if lock_hint:
        _lock_structure_from_hint(model, R, S, J, J0, Jd, x, w, z,
                                  immediate, rack_to_tasks,
                                  warm_hint_clean, lock_hint)


    # 提交 MIPStart（显式）
    _submit_mipstart(model, [x, w, z, v, immediate, p, q, g, h, C_max_AGV], name="ALNS-Start")


    # ---------- 优化 ----------
    # ---------- 优化 ----------
    outdir = "solution_exports"
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, f"gurobi_{(file_prefix or 'run')}_milpG{gamma_budget}.log")

    model.setParam(GRB.Param.LogFile, log_path)

    # 1) 总时间上限：2000 秒
    model.setParam(GRB.Param.TimeLimit, 400)

    # 2) 1000 秒无 incumbent 改进就停3000500020001000800500
    NO_IMPROVE_LIMIT = 300.0

    # 用 model 上的属性保存“上次改进时间”和“最好 incumbent”
    model._last_improve_time = 0.0
    model._best_incumbent = float("inf")

    def _cb_stop_if_no_improve(m, where):
        # 在 MIP 过程回调里检查（会被频繁调用，但开销很小）
        if where == GRB.Callback.MIP:
            runtime = m.cbGet(GRB.Callback.RUNTIME)
            solcnt = m.cbGet(GRB.Callback.MIP_SOLCNT)

            # 只有在已经找到可行解（incumbent）后才触发“无改进停止”
            if solcnt > 0:
                objbst = m.cbGet(GRB.Callback.MIP_OBJBST)  # 当前最好可行解目标值（越小越好）

                # 发现改进：刷新 last_improve_time
                if objbst < m._best_incumbent - 1e-9:
                    m._best_incumbent = objbst
                    m._last_improve_time = runtime

                # 超过阈值：终止
                if (runtime - m._last_improve_time) >= NO_IMPROVE_LIMIT:
                    print(
                        f"[MILP] early stop: no incumbent improvement for {NO_IMPROVE_LIMIT:.0f}s "
                        f"(runtime={runtime:.1f}s, best={m._best_incumbent:.6g})"
                    )
                    m.terminate()

    # 记得把 callback 传给 optimize
    model.optimize(_cb_stop_if_no_improve)

    if model.status == GRB.INFEASIBLE:
        print("模型不可行，正在计算 IIS...")
        model.computeIIS()
        # >>> IFACE-ONLY：IIS 细化打印（仅日志，不改模型）
        try:
            for c in model.getConstrs():
                if c.IISConstr:
                    print(f"[IIS] Constraint: {c.ConstrName}")
            for vvar in model.getVars():
                if vvar.IISLB or vvar.IISUB:
                    flags = []
                    if vvar.IISLB: flags.append("LB")
                    if vvar.IISUB: flags.append("UB")
                    print(f"[IIS] Var: {vvar.VarName} ({'/'.join(flags)}) LB={vvar.LB} UB={vvar.UB}")
        except Exception:
            pass
        model.write("Infeasible.ilp")

    # ====== 无解保护 ======
    if model.SolCount == 0:
        print("[MILP] 无可行解或无 incumbent，跳过结果打印与导出。")
        return {}, model, p, q
    # ===== bridge_quiet：cross-gamma feasibility check 时不要刷屏/导出 =====
    # 只要模型有 incumbent，我们就直接返回（由 milp_solver.py 读取 model.Status / ObjVal）
    if bridge_quiet:
        task_assignments_result = {}
        for r_id in R.keys():
            assigned_tasks = [j_id for j_id in J if w[j_id, r_id].X > 0.5]
            assigned_tasks.sort(key=lambda t: p[t, 0].X)
            task_assignments_result[r_id] = assigned_tasks
        return task_assignments_result, model, p, q
    maxG_range = range(maxG + 1)

    # 下面与原版一致：打印与导出
    def quick_print_v(v_):
        print("\n=== v[i,j,s,s'] = 1 (所有被选中的搬架弧) ===")
        for (i_, j_, s1, s2), var in v_.items():
            if var.X > 0.5:
                print(f"  v[{i_},{j_},{s1},{s2}] = 1")

    def print_g_h_values(g_, h_, all_tasks_, S_, eps=1e-4):
        print("\n=== (g, h) 非零值 ===")
        for j_ in all_tasks_:
            for γ_ in maxG_range:
                for s_ in S_:
                    gv = g_[j_, γ_, s_].X
                    hv = h_[j_, γ_, s_].X
                    if abs(gv) > eps or abs(hv) > eps:
                        print(f"  g[{j_},{γ_},{s_}]={gv:.2f}, h[{j_},{γ_},{s_}]={hv:.2f}")
        print("=== (g, h) 打印完毕 ===\n")

    def print_task_and_shelf_sequences(J_, J_I_, rack_to_tasks_, pi_, D_, immediate_, C_max_AGV_=None):
        print("\n=== 任务信息 (Task → (Workstation, Duration)) ===")
        for j_ in sorted(J_):
            print(f" Task {j_}: Workstation={pi_[j_]}, Duration={D_[j_]}")
        print("\n=== 货架上的任务序列 (Shelf → [Task…]) ===")
        for c_ in sorted(rack_to_tasks_):
            seq = []
            cur = J_I_[c_]
            tasks_c = rack_to_tasks_[c_]
            while True:
                nxt = None
                for j_ in tasks_c:
                    key = (cur, j_, c_)
                    if key in immediate_ and immediate_[key].X > 0.5:
                        nxt = j_
                        break
                if nxt is None:
                    break
                seq.append(nxt)
                cur = nxt
            print(f" Shelf {c_}: {seq}")
        if C_max_AGV_ is not None:
            print(f"\n=== C_max_AGV = {C_max_AGV_.X:.2f} ===\n")

    def export_ws_plan_csv(p_vars, q_vars, pi_map, J_real,
                           file_prefix_: str | None,
                           gamma: int = 0,
                           D_setup_: float = 2.0,
                           outdir_: str = "solution_exports"):
        rows = []
        for j_ in sorted(J_real):
            ws = int(pi_map[j_]); p0 = float(p_vars[j_, gamma].X); q0 = float(q_vars[j_, gamma].X)
            rows.append({"Task": int(j_), "WS": ws, "p_opt": p0, "q_opt": q0, "D_setup": float(D_setup_)})
        os.makedirs(outdir_, exist_ok=True)
        if not rows:
            print("[EXPORT] (wsPlan) 无任务可导出。"); return
        df = pd.DataFrame(rows).sort_values(["WS", "p_opt", "Task"])
        if file_prefix_ is None:
            print("[EXPORT] (wsPlan 预览)"); print(df.to_string(index=False)); return
        path = os.path.join(outdir_, f"{file_prefix_}_wsPlan_gamma{gamma}.csv")
        df.to_csv(path, index=False)
        print(f"[EXPORT] ws 计划(γ={gamma}) → {path}")

    def export_task_sequence_csv(assignments: dict,
                                 file_prefix_: str | None,
                                 gamma: int = 0,
                                 outdir_: str = "solution_exports"):
        rows = []
        for agv_id, seq in sorted(assignments.items()):
            rows.append({"AGV_ID": int(agv_id), "seq": str(list(map(int, seq)))})
        if not rows:
            print("[EXPORT] (taskSeq) 无任务序列。"); return
        os.makedirs(outdir_, exist_ok=True)
        if file_prefix_ is None:
            print("[EXPORT] (taskSeq 预览)"); print(pd.DataFrame(rows).to_string(index=False)); return
        path = os.path.join(outdir_, f"{file_prefix_}_taskSeq_gamma{gamma}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[EXPORT] 任务序列(γ={gamma}) → {path}")

    def export_task_end_shelf_csv(x_vars, J_real, S_all,
                                  file_prefix_: str | None,
                                  gamma: int = 0,
                                  outdir_: str = "solution_exports"):
        rows = []
        for j_ in sorted(J_real):
            best_s, best_val = None, -1.0
            over_half = []
            for s_ in S_all:
                val = float(x_vars[j_, s_].X)
                if val > 0.5:
                    over_half.append((s_, val))
                if val > best_val:
                    best_s, best_val = s_, val
            end_s = int((over_half[0][0] if over_half else best_s)) if best_s is not None else None
            if end_s is not None:
                rows.append({"Task": int(j_), "EndShelf": end_s})
        if not rows:
            print("[EXPORT] (taskEndShelf) 无条目。"); return
        os.makedirs(outdir_, exist_ok=True)
        if file_prefix_ is None:
            print("[EXPORT] (taskEndShelf 预览)"); print(pd.DataFrame(rows).sort_values("Task").to_string(index=False)); return
        path = os.path.join(outdir_, f"{file_prefix_}_taskEndShelf_gamma{gamma}.csv")
        pd.DataFrame(rows).sort_values("Task").to_csv(path, index=False)
        print(f"[EXPORT] 写出 {path} 行数={len(rows)}")

    def print_detailed_agv_rack_flow(model_, J_, J0_, Jd_, R_, S_,
                                     w_, x_, v_, p_, q_, g_, h_,
                                     immediate_, gamma_show=0, eps=1e-4):
        if model_.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED]:
            print("无可行解，跳过详细打印。"); return
        all_tasks2 = set(J_).union(J0_.values()).union(Jd_.values())
        tasks_sorted = sorted(all_tasks2, key=lambda jj: p_[jj, gamma_show].X)
        print(f"\n=== 任务-货架-AGV 时序（γ={gamma_show}） ===")
        for j_ in tasks_sorted:
            p_j = p_[j_, gamma_show].X
            q_j = q_[j_, gamma_show].X
            agv_r = next((r for r in R_ if w_[j_, r].X > 0.5), None)
            s_end = next((s for s in S_ if x_[j_, s].X > 0.5), None)
            print(f"\n[Task {j_}]  p={p_j:.2f}, q={q_j:.2f}, AGV={agv_r}, EndShelf={s_end}")
            for (i_, jj_, s1, s2), var in v_.items():
                if jj_ == j_ and var.X > 0.5:
                    print(f"   ← 由 task {i_} : {s1} ➜ {s2}")
            for s_ in S_:
                gv = g_[j_, gamma_show, s_].X
                hv = h_[j_, gamma_show, s_].X
                if abs(gv) > eps or abs(hv) > eps:
                    print(f"      g[{s_}]={gv:.2f}, h[{s_}]={hv:.2f}")
        print("\n=== w[j,r] = 1 ===")
        for j_ in J_:
            for r_ in R_:
                if w_[j_, r_].X > 0.5:
                    print(f"  w[{j_},{r_}] = 1")
        print("\n=== x[j,s] = 1 ===")
        for j_ in J_:
            for s_ in S_:
                if x_[j_, s_].X > 0.5:
                    print(f"  x[{j_},{s_}] = 1")
        print("\n=== 时序打印完毕 ===")

    def quick_print_w_x_z(J_, R_, S_, w_, x_, z_):
        print("\n=== w[j,r] = 1 ===")
        for j_ in J_:
            for r_ in R_:
                if w_[j_, r_].X > .5:
                    print(f"  w[{j_},{r_}] = 1")
        print("\n=== x[j,s] = 1 ===")
        for j_ in J_:
            for s_ in S_:
                if x_[j_, s_].X > .5:
                    print(f"  x[{j_},{s_}] = 1")
        print("\n=== z[j,j’,r] = 1 ===")
        for (j1, j2, r_) in z_.keys():
            if z_[j1, j2, r_].X > .5:
                print(f"  z[{j1},{j2},{r_}] = 1")

    # ======== NEW (more exports) ===================================
    import json

    def build_routes_from_z(J, J0, Jd, z_vars):
        """从 z[i,j,r]=1 复原每辆 AGV 的任务序列（不含 j0/jd）。"""
        # 建邻接：每辆车、每个节点的唯一后继
        succ = {}
        for (i, j, r), var in z_vars.items():
            if var.X > 0.5:
                succ.setdefault(int(r), {})[int(i)] = int(j)
        routes = {}
        order = {}  # order[r][j] = 位置序号（从1开始）
        for r, j0 in J0.items():
            r = int(r);
            cur = int(j0);
            jd = int(Jd[r])
            seq, ordmap, seen = [], {}, set()
            while True:
                nxt = succ.get(r, {}).get(cur, None)
                if nxt is None:
                    break
                if nxt == jd:
                    # 到尾了
                    break
                if nxt in seen:
                    print(f"[WARN] z-route cycle on r={r} at {nxt}, truncating.")
                    break
                seen.add(nxt)
                if nxt in J:
                    seq.append(nxt)
                    ordmap[nxt] = len(seq)  # 1-based
                cur = nxt
            routes[r] = seq
            order[r] = ordmap
        return routes, order

    def export_routes_csv(routes_by_agv, out_path):
        rows = [{"AGV_ID": int(r), "seq": json.dumps(list(map(int, seq)))} for r, seq in sorted(routes_by_agv.items())]
        if not rows:
            print("[EXPORT] (routes_by_agv) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] routes_by_agv → {out_path}")

    def export_w_assignments_csv(w_vars, J, R, out_path):
        rows = []
        for (j, r), var in w_vars.items():
            if (j in J) and (r in R) and var.X > 0.5:
                rows.append({"Task": int(j), "AGV": int(r)})
        if not rows:
            print("[EXPORT] (w_assignments) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).sort_values(["AGV", "Task"]).to_csv(out_path, index=False)
        print(f"[EXPORT] w_assignments → {out_path}")

    def export_z_edges_csv(z_vars, J, J0, Jd, routes_order, out_path):
        """
        导出所有 z=1 的边，并给出到达 j 的序号 ord_in_route（如 j 为 jd 则为 len(route)+1）。
        """
        rows = []
        for (i, j, r), var in z_vars.items():
            if var.X > 0.5:
                r = int(r);
                i = int(i);
                j = int(j)
                if j in J:
                    ord_j = routes_order.get(r, {}).get(j, None)
                elif j in set(Jd.values()):
                    # jd 的序号 = len(route) + 1
                    ord_j = (max(routes_order.get(r, {}).values()) + 1) if routes_order.get(r, {}) else 1
                else:
                    ord_j = None
                rows.append({"i": i, "j": j, "AGV_ID": r, "ord_in_route": ord_j})
        if not rows:
            print("[EXPORT] (z_edges) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).sort_values(["AGV_ID", "ord_in_route"]).to_csv(out_path, index=False)
        print(f"[EXPORT] z_edges → {out_path}")

    def export_v_arcs_full_csv(v_vars, J, Jd, out_path):
        """导出完整 v（包括通向 jd 的弧），方便诊断。"""
        rows = []
        Jdset = set(int(v) for v in Jd.values())
        for (i, j, s, sp), var in v_vars.items():
            if var.X > 0.5:
                rows.append({
                    "i": int(i), "j": int(j),
                    "s_from": int(s), "s_to": int(sp),
                    "j_is_terminal": int(j in Jdset)
                })
        if not rows:
            print("[EXPORT] (v_arcs_full) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] v_arcs_full → {out_path}")

    def debug_print_inconsistent_cell_order(mu, sigma, tau, OCC, S, Γ_range, eps=0.5, limit=50):
        """
        打印：同一对 (i,j,s) 在不同 (γ,γp) 下出现 sigma/tau 选择不一致的案例。
        只要你看到同一 (i,j,s) 有的 (γ,γp) 选 sigma=1、有的选 tau=1，
        就基本坐实“跨γ排序可变”导致的 evaluator 不对齐。
        """
        shown = 0
        for i in OCC:
            for j in OCC:
                if i == j:
                    continue
                for s in S:
                    try:
                        if mu[i, j, s].X <= eps:
                            continue
                    except Exception:
                        continue

                    chosen = []
                    for γ in Γ_range:
                        for γp in Γ_range:
                            try:
                                sig = sigma[i, j, s, γ, γp].X
                                ta = tau[i, j, s, γ, γp].X
                            except Exception:
                                continue
                            if sig > eps:
                                chosen.append(("sigma", γ, γp))
                            elif ta > eps:
                                chosen.append(("tau", γ, γp))

                    # 看是否同时出现 sigma 和 tau
                    kinds = set(k for (k, _, _) in chosen)
                    if len(kinds) >= 2:
                        print(f"\n[InconsistentOrder] cell={s} pair(i={i}, j={j}) mu=1")
                        print("  selected:")
                        for (k, γ, γp) in chosen:
                            print(f"   - {k} at (γ={γ}, γp={γp})")
                        shown += 1
                        if shown >= limit:
                            print(f"\n[InconsistentOrder] reach limit={limit}, stop.")
                            return

    def export_pq_all_gamma_csv(p_vars, q_vars, all_tasks, Γ_range, out_path):
        rows = []
        for j in sorted(all_tasks):
            for γ in Γ_range:
                rows.append({"Task": int(j), "Gamma": int(γ),
                             "p": float(p_vars[j, γ].X), "q": float(q_vars[j, γ].X)})
        if not rows:
            print("[EXPORT] (p_q_times) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] p_q_times (all γ) → {out_path}")

    def export_gh_selected_shelves_csv(g_vars, h_vars, x_vars, J_all, Γ_range, S, eps, out_path):
        """
        只导出每个 j 在“被选中的库位(s 满足 x[j,s]>0.5)”上的 g/h（大幅降维，但可复现关键口径）。
        """
        rows = []
        for j in J_all:
            sel_s = [s for s in S if x_vars[j, s].X > 0.5]
            for s in sel_s:
                for γ in Γ_range:
                    gv = float(g_vars[j, γ, s].X)
                    hv = float(h_vars[j, γ, s].X)
                    if abs(gv) > eps or abs(hv) > eps or True:
                        rows.append({"Task": int(j), "Gamma": int(γ), "Shelf": int(s), "g": gv, "h": hv})
        if not rows:
            print("[EXPORT] (g_h_selected) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] g_h_at_selected_shelves → {out_path}")

    def export_pos_on_shelf_csv(pos_vars, rack_to_tasks, J_I, out_path):
        rows = []
        for c, tasks_c in rack_to_tasks.items():
            vt = J_I[c]
            ext = [vt] + tasks_c
            for j in ext:
                var = pos_vars.get((j, c))
                if var is not None:
                    rows.append({"shelf_id": int(c), "task": int(j), "pos": int(round(var.X))})
        if not rows:
            print("[EXPORT] (pos_on_shelf) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).sort_values(["shelf_id", "pos"]).to_csv(out_path, index=False)
        print(f"[EXPORT] pos_on_shelf → {out_path}")

    def export_immX_csv(immX_vars, out_path, threshold=0.5):
        rows = []
        for key, var in immX_vars.items():
            val = var.X
            if val > threshold:
                i, jprime, c, sp = key
                rows.append({"i": int(i), "j": int(jprime), "shelf_id": int(c), "sp": int(sp), "immX": float(val)})
        if not rows:
            print("[EXPORT] (immX) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] immX → {out_path}")

    def export_y_r_csv(y_r_vars, out_path, threshold=0.5):
        rows = []
        for (j, jp, r), var in y_r_vars.items():
            if var.X > threshold:
                rows.append({"Task": int(j), "Task2": int(jp), "AGV_ID": int(r), "y": float(var.X)})
        if not rows:
            print("[EXPORT] (y_r) 无内容，跳过。");
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"[EXPORT] y_r → {out_path}")

    def export_meta_json(model, C_max_AGV, gamma_budget, J, R, S, J0, Jd, J_I, out_path):
        meta = {
            "status": int(model.Status),
            "obj_val": float(getattr(model, "ObjVal", float("nan"))),
            "cmax": float(getattr(C_max_AGV, "X", float("nan"))),
            "runtime": float(getattr(model, "Runtime", float("nan"))),
            "mip_gap": float(getattr(model, "MIPGap", float("nan"))) if hasattr(model, "MIPGap") else None,
            "gamma_budget": int(gamma_budget),
            "sizes": {"|J|": len(J), "|R|": len(R), "|S|": len(S)},
            "sets": {
                "J": list(map(int, sorted(J))),
                "R": list(map(int, sorted(R))),
                "S": list(map(int, sorted(S))),
                "J0": {int(r): int(j0) for r, j0 in J0.items()},
                "Jd": {int(r): int(jd) for r, jd in Jd.items()},
                "J_I": {int(c): int(vt) for c, vt in J_I.items()},
            },
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[EXPORT] meta → {out_path}")

    def export_bundle_json(routes_by_agv, place_by_task, shelf_seq, p_vars, q_vars, Γ_range,
                           v_vars, z_vars, w_vars, C_max_AGV,
                           J, R, S, J0, Jd, J_I, out_path):
        """一个可供 ALNS 直接读取的完整包。"""
        p0 = {int(j): float(p_vars[j, 0].X) for j in J}
        q0 = {int(j): float(q_vars[j, 0].X) for j in J}
        # place_by_task 传入就用 x 的结果（更权威）
        V = []
        for (i, j, s, sp), var in v_vars.items():
            if var.X > 0.5:
                V.append([int(i), int(j), int(s), int(sp)])
        Z = []
        for (i, j, r), var in z_vars.items():
            if var.X > 0.5:
                Z.append([int(i), int(j), int(r)])
        W = []
        for (j, r), var in w_vars.items():
            if var.X > 0.5 and j in J and r in R:
                W.append([int(j), int(r)])
        bundle = {
            "routes": {int(r): list(map(int, seq)) for r, seq in routes_by_agv.items()},
            "place": {int(j): int(s) for j, s in place_by_task.items()},
            "shelf_seq": {int(c): list(map(int, seq)) for c, seq in shelf_seq.items()},
            "p": p0, "q": q0,
            "cmax": float(getattr(C_max_AGV, "X", float("nan"))),
            "v_arcs": V, "z_edges": Z, "w_assignments": W,
            "sets": {
                "J": list(map(int, sorted(J))),
                "R": list(map(int, sorted(R))),
                "S": list(map(int, sorted(S))),
                "J0": {int(r): int(j0) for r, j0 in J0.items()},
                "Jd": {int(r): int(jd) for r, jd in Jd.items()},
                "J_I": {int(c): int(vt) for c, vt in J_I.items()},
            },
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2)
        print(f"[EXPORT] bundle → {out_path}")

    # ======== /NEW ==================================================

    # 求解成功后的主打印（保持原样）
    if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED):
        all_tasks_set = set(J) | set(J0.values()) | set(Jd.values()) | set(J_I.values())
        print_g_h_values(g, h, all_tasks_set, S)
        print_detailed_agv_rack_flow(model, J, J0, Jd, R, S, w, x, v, p, q, g, h, immediate, gamma_show=0)
        quick_print_w_x_z(J, R, S, w, x, z)
        print_task_and_shelf_sequences(J, J_I, rack_to_tasks, pi, D, immediate, C_max_AGV)
        if 'C_max_AGV' in locals():
            print(f"\n=== C_max_AGV = {C_max_AGV.X:.2f} ===\n")
    else:
        print(f"求解状态 model.Status = {model.Status}，未做结果打印。")

    # 任务分配结果
    task_assignments_result = {}
    for r_id in R.keys():
        assigned_tasks = [j_id for j_id in J if w[j_id, r_id].X > 0.5]
        assigned_tasks.sort(key=lambda t: p[t, 0].X)
        task_assignments_result[r_id] = assigned_tasks

    print("\n==== 优化完成, 任务分配结果 ====")
    for r_id, tlist in task_assignments_result.items():
        print(f"AGV {r_id}: {tlist}")

    quick_print_v(v)
    for γ in Γ_range:
        print(f"\n=== 任务-货架-AGV 时序（γ={γ}） ===")
        print_detailed_agv_rack_flow(model, J, J0, Jd, R, S, w, x, v, p, q, g, h, immediate, gamma_show=γ)
    # ======== NEW: 导出 v / immediate / 货架链顺序 / 从v反推EndShelf ========
    try:
        v_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_v_arcs_gamma{gamma_budget}.csv")
        export_v_arcs_csv(v_vars=v, J=J, J0=J0, Jd=Jd, S=S, out_path=v_csv)
    except Exception as e:
        print(f"[WARN] 导出 v_arcs 失败：{e}")

    try:
        imm_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_immediate_edges_gamma{gamma_budget}.csv")
        export_immediate_edges_csv(immediate_vars=immediate, J_I=J_I, rack_to_tasks=rack_to_tasks, out_path=imm_csv)
    except Exception as e:
        print(f"[WARN] 导出 immediate_edges 失败：{e}")

    try:
        shelf_seq = _build_shelf_sequences_from_immediate(J_I=J_I, rack_to_tasks=rack_to_tasks, immediate_vars=immediate)
        seq_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_shelf_seq_gamma{gamma_budget}.csv")
        export_shelf_seq_csv(shelf_seq, seq_csv)
    except Exception as e:
        print(f"[WARN] 导出 shelf_seq 失败：{e}")

    try:
        end_from_v_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_taskEndShelf_fromV_gamma{gamma_budget}.csv")
        export_taskEndShelf_fromV_csv(v_vars=v, J=J, out_path=end_from_v_csv)
    except Exception as e:
        print(f"[WARN] 导出 EndShelf_fromV 失败：{e}")
    # ======== /NEW =================================================

    # 导出
    try:
        export_task_end_shelf_csv(x_vars=x, J_real=J, S_all=S.keys(),
                                  file_prefix_=file_prefix, gamma=gamma_budget)
    except Exception as e:
        print(f"[WARN] 导出 taskEndShelf 失败：{e}")
    try:
        export_ws_plan_csv(p_vars=p, q_vars=q, pi_map=pi, J_real=J,
                           file_prefix_=file_prefix, gamma=gamma_budget, D_setup_=2.0)
    except Exception as e:
        print(f"[WARN] 导出 wsPlan 失败：{e}")
    try:
        export_task_sequence_csv(task_assignments_result,
                                 file_prefix_=file_prefix, gamma=gamma_budget)
    except Exception as e:
        print(f"[WARN] 导出 taskSeq 失败：{e}")
    # === NEW: more exports for ALNS compatibility ===
    try:
        # 1) 从 z 复原 routes，并导出 z_edges（带序号）
        routes_by_agv, routes_order = build_routes_from_z(J=J, J0=J0, Jd=Jd, z_vars=z)
        routes_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_routes_by_agv_gamma{gamma_budget}.csv")
        export_routes_csv(routes_by_agv, routes_csv)

        z_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_z_edges_gamma{gamma_budget}.csv")
        export_z_edges_csv(z_vars=z, J=J, J0=J0, Jd=Jd, routes_order=routes_order, out_path=z_csv)
    except Exception as e:
        print(f"[WARN] 导出 routes/z_edges 失败：{e}")

    try:
        # 2) w 取 1 的条目
        w_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_w_assignments_gamma{gamma_budget}.csv")
        export_w_assignments_csv(w_vars=w, J=J, R=R, out_path=w_csv)
    except Exception as e:
        print(f"[WARN] 导出 w_assignments 失败：{e}")

    try:
        # 3) v 完整弧
        v_full_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_v_arcs_full_gamma{gamma_budget}.csv")
        export_v_arcs_full_csv(v_vars=v, J=J, Jd=Jd, out_path=v_full_csv)
    except Exception as e:
        print(f"[WARN] 导出 v_arcs_full 失败：{e}")

    try:
        # 4) 全 γ 的 p/q
        pq_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_p_q_times_allGamma_gamma{gamma_budget}.csv")
        all_tasks_for_pq = set(J) | set(J0.values()) | set(Jd.values())
        export_pq_all_gamma_csv(p_vars=p, q_vars=q, all_tasks=all_tasks_for_pq,
                                Γ_range=Γ_range, out_path=pq_csv)
    except Exception as e:
        print(f"[WARN] 导出 p_q_times(all γ) 失败：{e}")

    try:
        # 5) 每个任务在被选中库位上的 g/h
        gh_sel_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_g_h_at_selected_shelves_gamma{gamma_budget}.csv")
        all_tasks_for_gh = set(J) | set(J0.values()) | set(Jd.values()) | set(J_I.values())
        export_gh_selected_shelves_csv(g_vars=g, h_vars=h, x_vars=x,
                                       J_all=all_tasks_for_gh, Γ_range=Γ_range, S=S.keys(),
                                       eps=1e-4, out_path=gh_sel_csv)
    except Exception as e:
        print(f"[WARN] 导出 g/h(selected shelves) 失败：{e}")

    try:
        # 6) pos 与 immX（大多用于诊断）
        pos_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_pos_on_shelf_gamma{gamma_budget}.csv")
        export_pos_on_shelf_csv(pos_vars=pos, rack_to_tasks=rack_to_tasks, J_I=J_I, out_path=pos_csv)

        immX_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_immX_gamma{gamma_budget}.csv")
        export_immX_csv(immX_vars=immX, out_path=immX_csv)
    except Exception as e:
        print(f"[WARN] 导出 pos/immX 失败：{e}")

    # try:
    #     # 7) y_r 若有值>0.5就导出（可选诊断）
    #     y_csv = os.path.join(outdir, f"{(file_prefix or 'run')}_y_r_gamma{gamma_budget}.csv")
    #     export_y_r_csv(y_r_vars=y_r, out_path=y_csv)
    # except Exception as e:
    #     print(f"[WARN] 导出 y_r 失败：{e}")

    try:
        # 8) 元信息
        meta_json = os.path.join(outdir, f"{(file_prefix or 'run')}_result_meta_gamma{gamma_budget}.json")
        export_meta_json(model=model, C_max_AGV=C_max_AGV, gamma_budget=gamma_budget,
                         J=J, R=R, S=S, J0=J0, Jd=Jd, J_I=J_I, out_path=meta_json)
    except Exception as e:
        print(f"[WARN] 导出 meta 失败：{e}")
    debug_print_inconsistent_cell_order(mu, sigma, tau, OCC, S, Γ_range, eps=0.5, limit=50)
    try:
        # 9) 汇总包：routes/place/shelf_seq/p/q/v/cmax/集合基准
        #    place 采用 x 的多数决；shelf_seq 已在上面通过 immediate 构建
        place_from_x = {}
        for j_ in J:
            best_s, best_val = None, -1.0
            for s_ in S.keys():
                val = float(x[j_, s_].X)
                if val > best_val:
                    best_s, best_val = s_, val
            if best_s is not None:
                place_from_x[int(j_)] = int(best_s)

        bundle_json = os.path.join(outdir, f"{(file_prefix or 'run')}_bundle_gamma{gamma_budget}.json")
        export_bundle_json(routes_by_agv=routes_by_agv,
                           place_by_task=place_from_x,
                           shelf_seq=shelf_seq,
                           p_vars=p, q_vars=q, Γ_range=Γ_range,
                           v_vars=v, z_vars=z, w_vars=w,
                           C_max_AGV=C_max_AGV,
                           J=J, R=R, S=S, J0=J0, Jd=Jd, J_I=J_I,
                           out_path=bundle_json)
    except Exception as e:
        print(f"[WARN] 导出 bundle 失败：{e}")

    # 额外保留一份 .sol，方便重放
    try:
        model.write(os.path.join(outdir, f"{(file_prefix or 'run')}_solution_gamma{gamma_budget}.sol"))
    except Exception as e:
        print(f"[WARN] 写出 .sol 失败：{e}")

    return task_assignments_result, model, p, q
