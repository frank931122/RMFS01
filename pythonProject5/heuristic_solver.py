# -*- coding: utf-8 -*-
"""
heuristic_solver.py
-------------------
目的：
  用“启发式 + 局部搜索（ALNS 的简化版）”替代 MILP/Gurobi，为 RMFS 任务分配与调度生成
  一份可与 run_phys_sim / animationRMFS 对接的三张 CSV（taskSeq / taskInfo / taskShelf）。

特点：
  1) 读取现有 CSV（若有），并对含 NaN 的“虚拟任务”/缺列/孤立任务做 **显式清洗**；
  2) 首选 MovementManager 提供的 A* 路径长度作为行驶时间，失败时 **回退到曼哈顿距离**；
  3) 先构造贪婪初解，再用“破坏/修复 + 模拟退火接受准则”的 ALNS 外圈迭代优化；
  4) 导出与 run_phys_sim 兼容的三张 CSV，便于后续动画与仿真联动；
  5) 日志可控（--verbose / --tracebacks），关键步骤、异常、数据差集、代价轨迹、缓存命中等
     都可打印出来，便于分析与定位问题。

使用：
  python heuristic_solver.py --prefix demo01 --seed 1 --iters 1500 --verbose
"""
import os
import math
import random
import argparse
import itertools
import traceback
from collections import defaultdict

import pandas as pd

# ========= 可选依赖：地图与路径管理（若失败会自动回退） =========
try:
    from map_generator import create_map
    from movement_manager import MovementManager
    HAS_MAP = True
except Exception:
    create_map = None
    MovementManager = None
    HAS_MAP = False

# ========= 简易日志器（支持 --verbose 与 --tracebacks） =========
class Logger:
    def __init__(self, verbose=False, tracebacks=False):
        self.verbose = verbose
        self.tracebacks = tracebacks

    def info(self, msg):
        print(msg)

    def debug(self, msg):
        if self.verbose:
            print(msg)

    def warn(self, msg):
        print(f"[WARN] {msg}")

    def error(self, msg, exc: Exception | None = None):
        print(f"[ERROR] {msg}")
        if exc is not None and self.tracebacks:
            traceback.print_exc()

LOG = Logger(verbose=False, tracebacks=False)  # 会在 main() 中根据命令行参数重置


# ======================================================
# 1) 输入与数据加载（含 NaN/虚拟任务清洗 + 详细日志）
# ======================================================
def load_problem(prefix: str, seed: int):
    """
    返回:
      shelf_data: {shelf_id -> storage_point_idx}
      agv_data:   {agv_id   -> storage_point_idx}
      tasks:      {task_id  -> (ws, dur)}     # 仅包含真实任务
      task_shelf: {task_id  -> shelf_id}
      workstation_ids: list
      storage_points:  list

    说明:
      - 若存在 solution_exports/<prefix>_taskInfo.csv 与 _taskShelf.csv，
        则读取并“严格清洗”：只保留 (Workstation, Duration) 非空的任务，且只保留两表交集；
      - 若不存在，随机生成一批小实例，并写出两张表供复用；
      - 每一步都打印详细统计，便于排查数据问题。
    """
    random.seed(seed)

    # 与工程一致的默认地图配置（用于 MovementManager 与默认 SP）
    shelf_data = {1: 23, 2: 24, 3: 25, 4: 41, 5: 42, 6: 43}
    agv_data   = {1: 14, 2: 15, 3: 16}

    info_csv  = f"solution_exports/{prefix}_taskInfo.csv"
    shelf_csv = f"solution_exports/{prefix}_taskShelf.csv"

    # 打印输入参数
    LOG.info(f"[LOAD] prefix={prefix}, seed={seed}")
    LOG.info(f"[LOAD] expect files: {info_csv}, {shelf_csv}")

    if os.path.exists(info_csv) and os.path.exists(shelf_csv):
        # 读取两张表
        info_df  = pd.read_csv(info_csv)
        shelf_df = pd.read_csv(shelf_csv)
        LOG.info(f"[LOAD] read taskInfo rows={len(info_df)}, taskShelf rows={len(shelf_df)}")

        # --- 基本字段检查 ---
        if not {"Task", "Workstation", "Duration"}.issubset(info_df.columns):
            raise ValueError(f"{info_csv} 缺少必须字段 Task/Workstation/Duration")
        if not {"Task", "Shelf"}.issubset(shelf_df.columns):
            raise ValueError(f"{shelf_csv} 缺少必须字段 Task/Shelf")

        # --- 清洗 1：仅保留真实任务（Workstation、Duration 都非空） ---
        mask_real = info_df["Workstation"].notna() & info_df["Duration"].notna()
        dropped_info = info_df.loc[~mask_real, "Task"].tolist()
        if dropped_info:
            LOG.warn(f"丢弃含 NaN 的 taskInfo 任务: {dropped_info[:50]}{' ...' if len(dropped_info)>50 else ''}")
        info_df_clean = info_df.loc[mask_real].copy()
        LOG.info(f"[CLEAN] taskInfo: before={len(info_df)}, after={len(info_df_clean)}")

        # --- 清洗 2：去掉 taskShelf 中 Shelf 为 NaN 的行（避免映射缺失）---
        mask_sh = shelf_df["Shelf"].notna()
        dropped_shelf = shelf_df.loc[~mask_sh, "Task"].tolist()
        if dropped_shelf:
            LOG.warn(f"丢弃 Shelf 为 NaN 的 taskShelf 任务: {dropped_shelf[:50]}{' ...' if len(dropped_shelf)>50 else ''}")
        shelf_df_clean = shelf_df.loc[mask_sh].copy()
        LOG.info(f"[CLEAN] taskShelf: before={len(shelf_df)}, after={len(shelf_df_clean)}")

        # --- 交集/差集：只保留两表共有的任务 ID ---
        info_ids  = set(info_df_clean["Task"].dropna().astype(int).tolist())
        shelf_ids = set(shelf_df_clean["Task"].dropna().astype(int).tolist())
        inter_ids = sorted(info_ids & shelf_ids)
        only_info = sorted(info_ids - shelf_ids)
        only_shelf= sorted(shelf_ids - info_ids)
        LOG.info(f"[CLEAN] intersect_ids={len(inter_ids)}, only_in_info={len(only_info)}, only_in_shelf={len(only_shelf)}")
        if only_info:
            LOG.warn(f"taskInfo 存在但 taskShelf 缺失: {only_info[:50]}{' ...' if len(only_info)>50 else ''}")
        if only_shelf:
            LOG.warn(f"taskShelf 存在但 taskInfo 缺失: {only_shelf[:50]}{' ...' if len(only_shelf)>50 else ''}")

        info_df_clean  = info_df_clean[info_df_clean["Task"].astype(int).isin(inter_ids)].copy()
        shelf_df_clean = shelf_df_clean[shelf_df_clean["Task"].astype(int).isin(inter_ids)].copy()

        # --- 类型转换（异常时给出上下文）---
        try:
            info_df_clean["Task"]        = info_df_clean["Task"].astype(int)
            info_df_clean["Workstation"] = info_df_clean["Workstation"].astype(int)
            info_df_clean["Duration"]    = info_df_clean["Duration"].astype(float)
            shelf_df_clean["Task"]       = shelf_df_clean["Task"].astype(int)
            shelf_df_clean["Shelf"]      = shelf_df_clean["Shelf"].astype(int)
        except Exception as e:
            LOG.error("类型转换失败（Task/Workstation/Duration/Shelf）", e)
            raise

        # --- 构建字典 ---
        tasks = {int(r.Task): (int(r.Workstation), float(r.Duration))
                 for _, r in info_df_clean.iterrows()}
        task_shelf = {int(r.Task): int(r.Shelf) for _, r in shelf_df_clean.iterrows()}
        workstation_ids = sorted(set(info_df_clean["Workstation"].tolist()))
        LOG.info(f"[CLEAN] 最终真实任务数={len(tasks)}, 工位ID={workstation_ids}")

        if not tasks:
            LOG.warn("清洗后没有任何真实任务，改为随机生成一个小实例。")
            return _random_tasks(prefix, shelf_data, agv_data, seed)

    else:
        LOG.warn("未发现现成 CSV，将随机生成一份任务集合并写出。")
        return _random_tasks(prefix, shelf_data, agv_data, seed)

    storage_points = [14, 15, 16, 23, 24, 25, 41, 42, 43, 50, 51, 52]
    return shelf_data, agv_data, tasks, task_shelf, workstation_ids, storage_points


def _random_tasks(prefix, shelf_data, agv_data, seed):
    """
    当没有现成 CSV 或清洗后为空时，随机生成少量任务并写出两张 CSV。
    """
    random.seed(seed)
    os.makedirs("solution_exports", exist_ok=True)
    LOG.info("[RAND] 生成随机任务集（仅用于占位/调试）")

    workstation_ids = [1, 2]
    real_tasks = list(range(1, 9))  # 默认 8 个
    tasks = {}
    task_shelf = {}
    for t in real_tasks:
        ws  = random.choice(workstation_ids)
        dur = random.randint(6, 12)
        sh  = random.choice(list(shelf_data.keys()))
        tasks[t] = (ws, float(dur))
        task_shelf[t] = sh

    info_df = pd.DataFrame([{"Task": t, "Workstation": tasks[t][0], "Duration": tasks[t][1]}
                            for t in sorted(tasks)])
    shelf_df = pd.DataFrame([{"Task": t, "Shelf": task_shelf[t]}
                             for t in sorted(task_shelf)])
    info_csv  = f"solution_exports/{prefix}_taskInfo.csv"
    shelf_csv = f"solution_exports/{prefix}_taskShelf.csv"
    info_df.to_csv(info_csv, index=False)
    shelf_df.to_csv(shelf_csv, index=False)
    LOG.info(f"[RAND] 已写出 {info_csv} 与 {shelf_csv}")

    storage_points = [14, 15, 16, 23, 24, 25, 41, 42, 43, 50, 51, 52]
    return shelf_data, agv_data, tasks, task_shelf, workstation_ids, storage_points


# ======================================================
# 2) 距离/时间估计（优先 MovementManager；失败回退曼哈顿）
# ======================================================
class TravelTime:
    """
    提供 3 段行驶时间估计：
      - time_move1(AGV起点 -> 货架储位)
      - time_move2(货架储位 -> 工作站)
      - time_move3(工作站 -> 货架储位)
    优先调用 MovementManager.get_path() 获取路径长度；
    如失败（模块缺失/路径报错），回退为曼哈顿距离。
    """

    def __init__(self, shelf_data, agv_data, logger: Logger):
        self.shelf_data = shelf_data
        self.agv_data   = agv_data
        self.log        = logger

        self._mm = None
        self._W = None
        self._H = None
        self.ws_cell = {}
        self._path_cache = {}  # (a_idx, b_idx) -> length

        if HAS_MAP:
            try:
                self.log.info("[MAP] 尝试创建地图与 MovementManager ...")
                map_obj = create_map(shelf_data, agv_data)
                self._mm = MovementManager(map_obj, None)
                self._W, self._H = map_obj.width, map_obj.height
                ws_coords = map_obj.extract_workstations()
                self.ws_cell = {i+1: r*self._W + c + 1 for i, (r, c) in enumerate(ws_coords)}
                self.log.info(f"[MAP] 成功：W={self._W}, H={self._H}, ws_cell={self.ws_cell}")
            except Exception as e:
                self.log.error("[MAP] 创建 MovementManager 失败，将回退到曼哈顿距离", e)
                self._mm = None
        else:
            self.log.warn("[MAP] 未能导入 map_generator / movement_manager；使用曼哈顿距离")
            self._W, self._H = 10, 10  # 回退时需要一个宽度以便坐标换算

    def _manhattan(self, a_idx, b_idx):
        ra, ca = divmod(a_idx-1, self._W)
        rb, cb = divmod(b_idx-1, self._W)
        return abs(ra - rb) + abs(ca - cb)

    def _path_len(self, a_idx, b_idx):
        """
        获取两格之间的路径步数（格子数-1）。
        优先 get_path()；失败则回退曼哈顿。带缓存与日志。
        """
        key = (a_idx, b_idx)
        if key in self._path_cache:
            self.log.debug(f"[PATH][HIT] {key} -> {self._path_cache[key]}")
            return self._path_cache[key]

        if self._mm is not None:
            try:
                path = self._mm.get_path(a_idx, b_idx)
                if not path:
                    self.log.warn(f"[PATH] get_path 返回空，回退曼哈顿: {key}")
                    dist = self._manhattan(a_idx, b_idx)
                else:
                    dist = max(1, len(path) - 1)
                    self.log.debug(f"[PATH][MAP] {key} -> len={dist}")
            except Exception as e:
                self.log.error(f"[PATH] get_path 异常，回退曼哈顿: {key}", e)
                dist = self._manhattan(a_idx, b_idx)
        else:
            dist = self._manhattan(a_idx, b_idx)
            self.log.debug(f"[PATH][MANHATTAN] {key} -> {dist}")

        self._path_cache[key] = dist
        return dist

    def time_move1(self, agv_sp, shelf_sp):
        return self._path_len(agv_sp, shelf_sp)

    def time_move2(self, shelf_sp, ws_id):
        ws_cell = self.ws_cell.get(ws_id)
        if ws_cell is None:
            # 没地图时给个可观测提示
            self.log.warn(f"[WS_CELL] 未知 ws_id={ws_id} 的 cell，使用默认10步")
            return 10
        return self._path_len(shelf_sp, ws_cell)

    def time_move3(self, ws_id, shelf_sp):
        ws_cell = self.ws_cell.get(ws_id)
        if ws_cell is None:
            self.log.warn(f"[WS_CELL] 未知 ws_id={ws_id} 的 cell，使用默认10步")
            return 10
        return self._path_len(ws_cell, shelf_sp)


# ======================================================
# 3) 评价函数（由序列计算时间线 + 可选鲁棒附加）
# ======================================================
def evaluate(solution, shelf_data, agv_data, tasks, task_shelf,
             travel: TravelTime, robust_gamma=0, delay_per_unit=0.0):
    """
    输入:
      solution: {agv -> [task_id, ...]}
    输出:
      obj_ws:   max ws_e
      obj_drop: max drop_e
      timeline: list of dict (供导出或调试)
    """
    # 工位可用时间（上一个任务完成时间）
    ws_ready = {ws: 0.0 for ws, _ in set(tasks.values())}
    # 车可用时间与位置（最后放架后回到 shelf_sp）
    agv_ready = {a: 0.0 for a in agv_data}
    agv_pos   = {a: agv_data[a] for a in agv_data}

    tl = []
    for agv in sorted(solution):
        for t in solution[agv]:
            ws, dur = tasks[t]
            sh      = task_shelf[t]
            sp      = shelf_data[sh]

            tm1 = travel.time_move1(agv_pos[agv], sp)
            tm2 = travel.time_move2(sp, ws)
            tm3 = travel.time_move3(ws, sp)

            # 简易鲁棒：Γ 段最长行驶加 delay
            extra = 0.0
            if robust_gamma > 0 and delay_per_unit > 0:
                legs = sorted([tm1, tm2, tm3], reverse=True)
                extra = sum(legs[:min(robust_gamma, 3)]) * delay_per_unit

            t0 = agv_ready[agv]
            move1_s = t0
            move1_e = move1_s + tm1

            pick_s  = move1_e
            pick_e  = pick_s  # 这里 pick 时间设 0，可自行扩展

            move2_s = pick_e
            move2_e = move2_s + tm2 + extra/2.0  # 把鲁棒附加均分到去/回也可

            ws_s    = max(move2_e, ws_ready[ws])
            ws_e    = ws_s + dur

            move3_s = ws_e
            move3_e = move3_s + tm3 + extra/2.0

            drop_s  = move3_e
            drop_e  = drop_s  # 放架时间设 0

            agv_ready[agv] = move3_e
            agv_pos[agv]   = sp
            ws_ready[ws]   = ws_e

            tl.append({
                "Task": t, "AGV": agv, "Shelf": sh, "WS": ws,
                "move1_s": move1_s, "move1_e": move1_e,
                "pick_s": pick_s,   "pick_e": pick_e,
                "move2_s": move2_s, "move2_e": move2_e,
                "ws_s": ws_s,       "ws_e": ws_e,
                "move3_s": move3_s, "move3_e": move3_e,
                "drop_s": drop_s,   "drop_e": drop_e
            })

    df = pd.DataFrame(tl)
    obj_ws   = float(df["ws_e"].max())   if not df.empty else 0.0
    obj_drop = float(df["drop_e"].max()) if not df.empty else 0.0
    return obj_ws, obj_drop, tl


# ======================================================
# 4) 初解：贪婪插入（打印每次插入的代价变化）
# ======================================================
def greedy_initial(tasks, task_shelf, shelf_data, agv_data, travel: TravelTime):
    """
    简单启发式初解：
      - 先按“最难服务”的任务（粗略估计：从任一 AGV 到 shelf 的最短 + shelf->ws + ws->shelf）降序排序；
      - 逐一把任务插入到“当前最优”的 (AGV, 位置)。
    """
    T = list(sorted(tasks))
    # 粗估难度评分
    score = {}
    for t in T:
        ws, _ = tasks[t]
        sh = task_shelf[t]; sp = shelf_data[sh]
        best_to_shelf = min(travel.time_move1(agv_data[a], sp) for a in agv_data)
        score[t] = best_to_shelf + travel.time_move2(sp, ws) + travel.time_move3(ws, sp)

    ordered = sorted(T, key=lambda x: score[x], reverse=True)
    LOG.info(f"[INIT] 贪婪初解插入顺序: {ordered}")

    sol = {a: [] for a in agv_data}
    for idx, t in enumerate(ordered, start=1):
        best = None
        best_tmp = None
        for a in agv_data:
            L = sol[a]
            for pos in range(len(L)+1):
                tmp = {k: v[:] for k, v in sol.items()}
                tmp[a].insert(pos, t)
                obj_ws, obj_drop, _ = evaluate(tmp, shelf_data, agv_data, tasks, task_shelf, travel)
                if (best is None) or (obj_ws < best[0]) or (obj_ws == best[0] and obj_drop < best[1]):
                    best = (obj_ws, obj_drop, a, pos)
                    best_tmp = tmp
        sol = best_tmp
        LOG.info(f"[INIT] 插入第{idx}个任务 t={t} -> "
                 f"best_agv={best[2]} pos={best[3]} "
                 f"obj_ws={best[0]:.1f} obj_drop={best[1]:.1f}")
    return sol


# ======================================================
# 5) 邻域算子（破坏/修复） & ALNS 外圈（含日志）
# ======================================================
def shaw_relatedness(t1, t2, tasks, task_shelf, shelf_data):
    """
    Shaw 算子相关性：同工位 + 同货架 + （货架 SP 曼哈顿距离越近越相关）
    """
    ws1, _ = tasks[t1]; ws2, _ = tasks[t2]
    sh1 = task_shelf[t1]; sh2 = task_shelf[t2]
    r = 0
    if ws1 == ws2: r += 1
    if sh1 == sh2: r += 1
    # 曼哈顿距离惩罚（越近 r 越大）
    W = 10
    sp1 = shelf_data[sh1]; sp2 = shelf_data[sh2]
    ra, ca = divmod(sp1-1, W); rb, cb = divmod(sp2-1, W)
    r -= 0.01 * (abs(ra-rb) + abs(ca-cb))
    return r

def destroy(solution, k, mode, tasks, task_shelf, shelf_data):
    """
    破坏算子：从当前解中移除 k 个任务。三种模式：
      - random: 随机挑 k 个任务；
      - worst:  按任务加工时长降序取 k 个；
      - shaw:   先随机选一个种子，再取与之“相关性”最高的 k-1 个任务。
    """
    all_tasks = list(itertools.chain.from_iterable(solution[a] for a in solution))
    if not all_tasks:
        return solution, []
    removed = []
    if mode == "random":
        removed = random.sample(all_tasks, min(k, len(all_tasks)))
    elif mode == "worst":
        srt = sorted(all_tasks, key=lambda t: tasks[t][1], reverse=True)
        removed = srt[:min(k, len(srt))]
    else:
        seed_t = random.choice(all_tasks)
        rel = []
        for t in all_tasks:
            if t == seed_t: continue
            rel.append((shaw_relatedness(seed_t, t, tasks, task_shelf, shelf_data), t))
        rel.sort(reverse=True)
        removed = [seed_t] + [t for _, t in rel[:max(0, k-1)]]

    new_sol = {a: [t for t in solution[a] if t not in set(removed)] for a in solution}
    return new_sol, removed

def repair_greedy(solution, removed, shelf_data, agv_data, tasks, task_shelf, travel):
    """
    修复算子：将被移除的任务逐个插回到全局最优的 (AGV, 位置) 处。
    """
    sol = {k: v[:] for k, v in solution.items()}
    for t in removed:
        best = None
        best_tmp = None
        for a in agv_data:
            L = sol[a]
            for pos in range(len(L)+1):
                tmp = {k: v[:] for k, v in sol.items()}
                tmp[a].insert(pos, t)
                obj_ws, obj_drop, _ = evaluate(tmp, shelf_data, agv_data, tasks, task_shelf, travel)
                if (best is None) or (obj_ws < best[0]) or (obj_ws == best[0] and obj_drop < best[1]):
                    best = (obj_ws, obj_drop, a, pos)
                    best_tmp = tmp
        sol = best_tmp
        LOG.debug(f"[REPAIR] 插回任务 t={t} -> agv={best[2]} pos={best[3]} obj_ws={best[0]:.1f}")
    return sol

def alns_search(init_sol, shelf_data, agv_data, tasks, task_shelf, travel,
                iters=1500, k_remove=3, seed=1, robust_gamma=0, delay_per_unit=0.0,
                log_every=200):
    """
    外圈：ALNS 简化版
      - 三种破坏算子随机选择（random/shaw/worst）；
      - 贪婪修复；
      - 模拟退火式接受；
      - 每 log_every 次打印一次当前/最优代价。
    """
    random.seed(seed)
    cur = init_sol
    cur_ws, cur_drop, _ = evaluate(cur, shelf_data, agv_data, tasks, task_shelf, travel,
                                   robust_gamma, delay_per_unit)
    best = cur
    best_ws, best_drop = cur_ws, cur_drop
    LOG.info(f"[ALNS] 初解: obj_ws={cur_ws:.1f}, obj_drop={cur_drop:.1f}")

    T0 = max(1.0, cur_ws * 0.05)  # 初始温度
    Tmin = 1e-3

    accept_cnt = 0
    improve_cnt = 0
    for it in range(1, iters+1):
        mode = random.choice(["random", "shaw", "worst"])
        new_sol, removed = destroy(cur, k_remove, mode, tasks, task_shelf, shelf_data)
        new_sol = repair_greedy(new_sol, removed, shelf_data, agv_data, tasks, task_shelf, travel)

        new_ws, new_drop, _ = evaluate(new_sol, shelf_data, agv_data, tasks, task_shelf, travel,
                                       robust_gamma, delay_per_unit)

        d = (new_ws - cur_ws)
        T = max(Tmin, T0 * (0.995 ** it))
        accept = False
        cause  = ""
        if new_ws < cur_ws or (new_ws == cur_ws and new_drop < cur_drop):
            accept = True
            cause = "improved"
        else:
            prob = math.exp(-d / max(1e-6, T))
            if prob > random.random():
                accept = True
                cause = f"sa-accept p={prob:.3f}"

        if accept:
            cur, cur_ws, cur_drop = new_sol, new_ws, new_drop
            accept_cnt += 1

        if (cur_ws < best_ws) or (cur_ws == best_ws and cur_drop < best_drop):
            best, best_ws, best_drop = cur, cur_ws, cur_drop
            improve_cnt += 1
            LOG.debug(f"[ALNS][BEST] it={it} mode={mode} -> obj_ws={best_ws:.1f}, obj_drop={best_drop:.1f}")

        if it % log_every == 0:
            LOG.info(f"[ALNS] iter={it:4d} mode={mode:6s} "
                     f"cur_ws={cur_ws:6.1f} best_ws={best_ws:6.1f} "
                     f"accept={accept_cnt} improve={improve_cnt} cause={cause}")

    LOG.info(f"[ALNS] 完成: best_ws={best_ws:.1f}, best_drop={best_drop:.1f}, "
             f"accepted={accept_cnt}, improved={improve_cnt}")
    return best, best_ws, best_drop


# ======================================================
# 6) 导出 CSV（与 run_phys_sim 兼容，带核验日志）
# ======================================================
def export_csv(prefix, solution, tasks, task_shelf):
    os.makedirs("solution_exports", exist_ok=True)

    # taskSeq.csv
    rows = [{"AGV_ID": agv, "seq": str(solution[agv])} for agv in sorted(solution)]
    seq_csv = f"solution_exports/{prefix}_taskSeq.csv"
    pd.DataFrame(rows).to_csv(seq_csv, index=False)

    # taskInfo.csv
    info_rows = [{"Task": t, "Workstation": tasks[t][0], "Duration": tasks[t][1]}
                 for t in sorted(tasks)]
    info_csv = f"solution_exports/{prefix}_taskInfo.csv"
    pd.DataFrame(info_rows).to_csv(info_csv, index=False)

    # taskShelf.csv
    shelf_rows = [{"Task": t, "Shelf": task_shelf[t]} for t in sorted(task_shelf)]
    shelf_csv = f"solution_exports/{prefix}_taskShelf.csv"
    pd.DataFrame(shelf_rows).to_csv(shelf_csv, index=False)

    # 导出日志
    LOG.info(f"[EXPORT] 写出 {seq_csv} 行数={len(rows)}")
    LOG.info(f"[EXPORT] 写出 {info_csv} 行数={len(info_rows)}")
    LOG.info(f"[EXPORT] 写出 {shelf_csv} 行数={len(shelf_rows)}")

    # 覆盖核验：三表任务 ID 交集/差集
    try:
        df_info  = pd.read_csv(info_csv)
        df_shelf = pd.read_csv(shelf_csv)
        ids_info  = set(df_info["Task"].astype(int).tolist())
        ids_shelf = set(df_shelf["Task"].astype(int).tolist())
        inter     = sorted(ids_info & ids_shelf)
        only_info = sorted(ids_info - ids_shelf)
        only_shelf= sorted(ids_shelf - ids_info)
        LOG.info(f"[CHECK] 导出后交集={len(inter)}, only_info={len(only_info)}, only_shelf={len(only_shelf)}")
        if only_info:
            LOG.warn(f"[CHECK] 导出后 taskInfo 孤立任务: {only_info[:50]}{' ...' if len(only_info)>50 else ''}")
        if only_shelf:
            LOG.warn(f"[CHECK] 导出后 taskShelf 孤立任务: {only_shelf[:50]}{' ...' if len(only_shelf)>50 else ''}")
    except Exception as e:
        LOG.error("[CHECK] 导出自检失败", e)

    LOG.info(f"[OK] 已导出到 solution_exports/{prefix}_*.csv  (run_phys_sim / animationRMFS 可直接使用)")


# ======================================================
# 7) 主程序（命令行参数 + 统一日志控制）
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="demo01", type=str,
                        help="文件前缀，对应 solution_exports/<prefix>_*.csv")
    parser.add_argument("--seed",   default=1, type=int, help="随机种子，保证可复现")
    parser.add_argument("--iters",  default=1500, type=int, help="ALNS 迭代次数")
    parser.add_argument("--gamma",  default=0, type=int, help="Bertsimas–Sim 预算 Γ（简易鲁棒）")
    parser.add_argument("--delay",  default=0.0, type=float, help="每单位距离附加延迟（简易鲁棒）")
    parser.add_argument("--verbose", action="store_true", help="打印 DEBUG 级别日志")
    parser.add_argument("--tracebacks", action="store_true", help="异常时打印堆栈")
    parser.add_argument("--log-every", default=200, type=int, help="ALNS 日志打印间隔")
    args = parser.parse_args()

    # 设置全局日志器
    global LOG
    LOG = Logger(verbose=args.verbose, tracebacks=args.tracebacks)

    # 统一种子
    random.seed(args.seed)

    # 读取/清洗任务
    shelf_data, agv_data, tasks, task_shelf, ws_ids, sp_list = load_problem(args.prefix, args.seed)
    LOG.info(f"[INFO] 真实任务数={len(tasks)}，AGV数={len(agv_data)}，工位={ws_ids}")

    # 构造行驶时间估计器（含地图尝试与回退）
    travel = TravelTime(shelf_data, agv_data, LOG)

    # 生成初解 + ALNS 优化
    init_sol = greedy_initial(tasks, task_shelf, shelf_data, agv_data, travel)
    best_sol, best_ws, best_drop = alns_search(
        init_sol, shelf_data, agv_data, tasks, task_shelf, travel,
        iters=args.iters, k_remove=3, seed=args.seed,
        robust_gamma=args.gamma, delay_per_unit=args.delay,
        log_every=args.log_every
    )

    LOG.info(f"[RESULT] best_ws={best_ws:.1f} best_drop={best_drop:.1f}")

    # 导出三张 CSV（若不想覆盖任务定义，可在此处只导出 seq）
    export_csv(args.prefix, best_sol, tasks, task_shelf)


if __name__ == "__main__":
    main()
