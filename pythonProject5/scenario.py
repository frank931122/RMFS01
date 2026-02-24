# scenario.py
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Callable
from collections import defaultdict
import heapq
import pandas as pd
import os

DistanceFn = Callable[[int, int, dict, dict], float]

@dataclass(frozen=True)
class Scenario:
    # === 集合/索引 ===
    J: Set[int]                     # 真实任务
    R: Set[int]                     # AGV ID
    S: Dict[int, Tuple[int,int]]    # SP_ID -> (row,col)
    K: Dict[int, Tuple[int,int]]    # WS_ID -> (row,col)

    # === 任务属性 ===
    pi: Dict[int, int]              # j -> 工位
    D: Dict[int, float]             # j -> 加工时长
    task_shelf: Dict[int, int]      # j -> shelf_id
    rack_to_tasks: Dict[int, List[int]]  # shelf -> [tasks...]
    J_I: Dict[int, int]             # shelf -> vt_task（如需）
    ws_fixed_seq: Dict[int, List[int]]   # 工位固定链（相邻顺序）

    # === 初始位置（ALNS 评估会用到）===  << 新增
    shelf_init: Dict[int, int]      # shelf_id -> 初始SP_ID
    agv_init: Dict[int, int]        # agv_id   -> 初始SP_ID

    # === 距离缓存（O(1) 查询）===
    d_s_pi: Dict[Tuple[int, int], float] # (s, j) -> time
    d_pi_s: Dict[Tuple[int, int], float] # (j, s) -> time
    d_s_s : Dict[Tuple[int, int], float] # (s, s') -> time

    # === 候选储位（供启发式/MILP裁边用）===
    S_cand: Dict[int, List[int]]         # j -> [s...]

    # === 参数 ===
    D_setup: float
    gamma_budget: int
    width: int
    height: int

def _idx2rc(idx: int, W: int) -> Tuple[int,int]:
    return divmod(int(idx)-1, W)

def _build_distance_tables(S: Dict[int, Tuple[int,int]],
                           K: Dict[int, Tuple[int,int]],
                           pi: Dict[int,int],
                           J: Set[int],
                           distance: DistanceFn):
    d_s_pi, d_pi_s, d_s_s = {}, {}, {}
    S_ids = list(S.keys())

    # S x S
    for s in S_ids:
        for sp in S_ids:
            d_s_s[(s, sp)] = distance(s, sp, S, S)

    # S x J, J x S
    for j in J:
        ws = pi[j]
        for s in S_ids:
            d_s_pi[(s, j)] = distance(s, ws, S, K)
            d_pi_s[(j, s)] = distance(ws, s, K, S)
    return d_s_pi, d_pi_s, d_s_s

def _build_S_cand(J: Set[int],
                  task_shelf: Dict[int,int],
                  shelf_data: Dict[int,int],
                  d_s_pi: Dict[Tuple[int,int], float],
                  S_ids: List[int],
                  K_near: int = 12) -> Dict[int, List[int]]:
    S_cand = {}
    for j in J:
        near = heapq.nsmallest(K_near, S_ids, key=lambda s: d_s_pi[(s, j)])
        # 强制加入该任务货架的初始位置（若有）
        c = task_shelf.get(j, None)
        if c is not None:
            s0 = shelf_data.get(c, None)
            if s0 is not None and s0 not in near:
                near.append(int(s0))
        S_cand[j] = near
    return S_cand

def _derive_ws_fixed_seq(tasks_df: pd.DataFrame) -> Dict[int, List[int]]:
    if "WSOrder" not in tasks_df.columns:
        tasks_df = tasks_df.copy()
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1
    seq_map = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
                .groupby("Workstation")["Task"]
                .apply(lambda s: [int(x) for x in s.tolist()])
                .to_dict()
    )
    return {int(k): [int(t) for t in v] for k, v in seq_map.items()}

def _build_rack_to_tasks(J: Set[int], task_shelf: Dict[int,int]) -> Dict[int, List[int]]:
    d = defaultdict(list)
    for j in J:
        c = task_shelf.get(j, None)
        if c is not None:
            d[int(c)].append(int(j))
    for c in d:
        d[c].sort()
    return dict(d)

def sanity_check(sc: Scenario):
    # 任务在固定链中不应越界
    for k, seq in sc.ws_fixed_seq.items():
        for j in seq:
            if j not in sc.J or sc.pi.get(j) != k:
                raise ValueError(f"[Scenario] ws_fixed_seq[{k}] 含非法或越界任务 {j}")
    # 距离尺寸
    if len(sc.d_s_s) != len(sc.S)*len(sc.S):
        raise ValueError("[Scenario] d_s_s 尺寸不符")
    if len(sc.d_s_pi) != len(sc.S)*len(sc.J):
        raise ValueError("[Scenario] d_s_pi 尺寸不符")
    if len(sc.d_pi_s) != len(sc.S)*len(sc.J):
        raise ValueError("[Scenario] d_pi_s 尺寸不符")
    # 候选非空
    for j in sc.J:
        if not sc.S_cand.get(j):
            raise ValueError(f"[Scenario] S_cand[{j}] 为空")
    # 初始位置存在性（供评估器/构造器使用）
    if not sc.shelf_init:
        raise ValueError("[Scenario] shelf_init 为空")
    if not sc.agv_init:
        raise ValueError("[Scenario] agv_init 为空")

def print_scenario_brief(sc: Scenario, k: int = 3):
    print(f"[Scenario] |J|={len(sc.J)} |R|={len(sc.R)} |S|={len(sc.S)} |K|={len(sc.K)} Γ={sc.gamma_budget}")
    sample = sorted(list(sc.J))[:k]
    for j in sample:
        print(f"  j={j}: WS={sc.pi[j]}, D={sc.D[j]}, S_cand(top)={sc.S_cand[j][:min(5,len(sc.S_cand[j]))]}")

def build_scenario_from_prefix(
    prefix: str,
    *,
    distance: DistanceFn,
    D_setup: float = 2.0,
    gamma_budget: int = 0,
    K_near: int = 12,
    scen_dir: str = "scenario",
) -> Scenario:
    """
    直接读取 scenario/<prefix> 下的 CSV（你的生成器已经写好），
    一次性生成 Scenario 与所有缓存。
    """
    # 读地图
    grid_csv  = os.path.join(scen_dir, prefix, "grid.csv")
    ws_csv    = os.path.join(scen_dir, prefix, "workstations.csv")
    sp_csv    = os.path.join(scen_dir, prefix, "storage_points.csv")
    sh_csv    = os.path.join(scen_dir, prefix, "shelf_init.csv")
    agv_csv   = os.path.join(scen_dir, prefix, "agv_init.csv")
    tasks_csv = os.path.join(scen_dir, prefix, "tasks.csv")

    grid_df  = pd.read_csv(grid_csv)
    ws_df    = pd.read_csv(ws_csv)
    sp_df    = pd.read_csv(sp_csv)
    shelf_df = pd.read_csv(sh_csv)
    agv_df   = pd.read_csv(agv_csv)
    tasks_df = pd.read_csv(tasks_csv)

    W, H = int(grid_df.iloc[0]["width"]), int(grid_df.iloc[0]["height"])
    ws_indices = [int(x) for x in ws_df["idx"].tolist()]
    sp_indices = [int(x) for x in sp_df["SP_ID"].tolist()]
    shelf_data = {int(r.Shelf_ID): int(r.SP) for _, r in shelf_df.iterrows()}
    agv_data   = {int(r.AGV_ID):   int(r.SP) for _, r in agv_df.iterrows()}

    # 构 S/K 坐标（1-based index -> (r,c)）
    S: Dict[int, Tuple[int,int]] = {int(sp): _idx2rc(int(sp), W) for sp in sp_indices}
    K: Dict[int, Tuple[int,int]] = {int(i+1): _idx2rc(int(ws_indices[i]), W) for i in range(len(ws_indices))}
    R_ids: Set[int] = set(int(x) for x in agv_data.keys())

    # 从 tasks.csv 提取真实任务与属性
    need = {"Task","Shelf","Workstation","Duration"}
    if not need.issubset(tasks_df.columns):
        raise ValueError(f"[Scenario] tasks.csv 缺列：{need - set(tasks_df.columns)}")
    if tasks_df.isna().any().any():
        raise ValueError("[Scenario] tasks.csv 存在 NaN")

    J: Set[int] = set(int(t) for t in tasks_df["Task"].tolist())
    task_shelf: Dict[int,int] = {int(r.Task): int(r.Shelf) for _, r in tasks_df.iterrows()}
    pi: Dict[int,int] = {int(r.Task): int(r.Workstation) for _, r in tasks_df.iterrows()}
    D : Dict[int,float] = {int(r.Task): float(r.Duration) for _, r in tasks_df.iterrows()}
    ws_fixed_seq = _derive_ws_fixed_seq(tasks_df)

    # 构 rack_to_tasks, J_I
    rack_to_tasks = _build_rack_to_tasks(J, task_shelf)
    J_I = {int(sid): 3000 + int(sid) for sid in shelf_data.keys()}  # 若后续需要 vt_id，可用此映射

    # 距离缓存 + 候选储位
    d_s_pi, d_pi_s, d_s_s = _build_distance_tables(S, K, pi, J, distance)
    S_cand = _build_S_cand(J, task_shelf, shelf_data, d_s_pi, list(S.keys()), K_near=K_near)

    sc = Scenario(
        J=J, R=R_ids, S=S, K=K,
        pi=pi, D=D, task_shelf=task_shelf,
        rack_to_tasks=rack_to_tasks, J_I=J_I,
        ws_fixed_seq=ws_fixed_seq,
        # 新增：初始位置
        shelf_init=shelf_data,
        agv_init=agv_data,
        # 距离与候选
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        S_cand=S_cand,
        D_setup=D_setup, gamma_budget=gamma_budget,
        width=W, height=H
    )
    sanity_check(sc)
    return sc
