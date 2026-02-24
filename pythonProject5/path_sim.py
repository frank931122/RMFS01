"""
物理仿真 (不依赖 GuroBI p/q)：
-------------------------------------------------
1. 读取   task_seq / task_shelf / task_info
2. 通过 MovementManager.get_path(idx_from, idx_to)
   得到 “格子 index” 路径；用恒速 + 转弯耗时 估计行驶时间
3. GridLock  —— 每个可行走格子(cap=1) 的资源，按路径整体上锁
4. 工作站、货架均 Resource(cap=1)
5. 输出 makespan 及 timeline（记录各阶段开始/结束时间）
"""

from __future__ import annotations
import simpy
from collections import defaultdict
from typing import Dict, List, Tuple

from movement_manager import MovementManager

# ---------- 可调常数 (根据设备标定) ----------
AGV_SPEED       = 1.0   # 格 / 秒
AGV_ROTATE_TIME = 0   # 90° 转弯耗时
PICK_TIME       = 0   # 提升货架
DROP_TIME       = 0   # 放下货架
SETUP_TIME      = 0.0   # 工作站准备

# ===================================================
#                     GridLock
# ===================================================
class GridLock:
    """
    把地图每个 walkable 格子注册为 simpy.Resource(cap=1)。
    get_path() 返回 index 序列 -> 统一转成 (row,col) -> 请求/释放
    """
    def __init__(self, env: simpy.Environment, map_obj):
        self.env = env
        self.map = map_obj
        # {(row,col): Resource}
        self.res: Dict[Tuple[int, int], simpy.Resource] = {
            (r, c): simpy.Resource(env, capacity=1)
            for r in range(map_obj.height)
            for c in range(map_obj.width)
            if map_obj.grid[r, c] != 'obstacle'
        }

    def idx2coord(self, idx: int) -> Tuple[int, int]:
        r, c = divmod(idx - 1, self.map.width)
        return r, c

    def request_path(self, path_idx: List[int]):
        """
        path_idx: [i1, i2, ...]
        返回一系列 Resource.request()，调用方 yield 它们即可上锁。
        """
        reqs = []
        for idx in path_idx:
            coord = self.idx2coord(idx)
            reqs.append(self.res[coord].request())
        return reqs

    def release_path(self, reqs: List[simpy.events.Event]):
        for req in reqs:
            req.resource.release(req)


# ===================================================
#                 物理仿真核心
# ===================================================
class RMFSPathSim:
    """
    输入仅依赖 任务序列 & 地图；仿真自行推演 makespan
    -------------------------------------------------
    task_seq     : {AGV_ID : [task1, task2, …]}
    task_shelf   : {task : shelf_id}
    task_info    : {task : (workstation_id, pick_duration)}
    shelf_init_sp: {shelf_id: storage_point_idx}
    agv_init_sp  : {agv_id:   storage_point_idx}
    """

    def __init__(self,
                 map_obj,
                 task_seq: Dict[int, List[int]],
                 task_shelf: Dict[int, int],
                 task_info: Dict[int, Tuple[int, float]],
                 shelf_init_sp: Dict[int, int],
                 agv_init_sp: Dict[int, int]):

        self.env = simpy.Environment()
        self.map = map_obj
        self.mm  = MovementManager(map_obj, None)   # 只用 A* 路径搜索
        self.lock= GridLock(self.env, map_obj)

        # 输入
        self.seq      = task_seq
        self.shelf_of = task_shelf
        self.info     = task_info

        # 动态位置（index）
        self.shelf_pos = dict(shelf_init_sp)  # shelf_id → sp idx
        self.agv_pos   = dict(agv_init_sp)    # agv_id   → sp idx

        # 重建 “工作站 ID → 地图格子 index” 的映射
        # 假设 map_obj.extract_workstations() 返回 [(r1,c1), (r2,c2), …]
        ws_coords = map_obj.extract_workstations()
        self.ws_sp = {
            ws_id: (r * map_obj.width + c + 1)
            for ws_id, (r, c) in enumerate(ws_coords, start=1)
        }

        # 资源：每个工作站 & 每个货架占用互斥
        self.ws_res    = defaultdict(lambda: simpy.Resource(self.env, capacity=1))
        self.shelf_res = defaultdict(lambda: simpy.Resource(self.env, capacity=1))

        # KPI 记录
        self.timeline = []  # 每条记录为 dict，包含各阶段开始/结束时间

        # 启动每辆 AGV 的仿真进程
        for agv, tasks in self.seq.items():
            self.env.process(self._agv_proc(agv, tasks))

    # ---------- 时间估计 ----------
    def _travel_time(self, path_idx: List[int]) -> float:
        """恒速 + 转弯耗时"""
        if len(path_idx) <= 1:
            return 0.0
        coords = [divmod(idx - 1, self.map.width) for idx in path_idx]
        move_time = (len(coords) - 1) / AGV_SPEED
        turns = 0
        for (ax, ay), (bx, by), (cx, cy) in zip(coords, coords[1:], coords[2:]):
            if (bx-ax, by-ay) != (cx-bx, cy-by):
                turns += 1
        return move_time + turns * AGV_ROTATE_TIME

    # ---------- 单车流程 ----------
    def _agv_proc(self, agv: int, tasks: List[int]):
        for task in tasks:
            shelf    = self.shelf_of[task]
            ws_id, pick_dur = self.info[task]

            # === 1) 当前 → 货架 ===
            start_idx = self.agv_pos[agv]
            shelf_idx = self.shelf_pos[shelf]
            path1 = self.mm.get_path(start_idx, shelf_idx)
            t1    = self._travel_time(path1)
            reqs1 = self.lock.request_path(path1)
            for r in reqs1: yield r

            t_move1_s = self.env.now
            yield self.env.timeout(t1)
            t_move1_e = self.env.now

            self.lock.release_path(reqs1)
            self.agv_pos[agv] = shelf_idx

            # === 举架 ===
            t_pick_s = self.env.now
            yield self.env.timeout(PICK_TIME)
            t_pick_e = self.env.now

            # === 2) 货架 → 工作站 ===
            ws_cell = self.ws_sp[ws_id]                   # ← ID→地图格子
            path2   = self.mm.get_path(shelf_idx, ws_cell)
            t2      = self._travel_time(path2)
            reqs2   = self.lock.request_path(path2)
            for r in reqs2: yield r

            t_move2_s = self.env.now
            yield self.env.timeout(t2)
            t_move2_e = self.env.now

            self.lock.release_path(reqs2)
            self.agv_pos[agv] = ws_cell

            # === 占用工作站 ===
            with self.ws_res[ws_id].request() as ws_req:
                yield ws_req
                t_ws_s = self.env.now
                yield self.env.timeout(SETUP_TIME + pick_dur)
                t_ws_e = self.env.now

            # === 3) 工作站 → 返回原储位 ===
            path3 = self.mm.get_path(ws_cell, shelf_idx)
            t3    = self._travel_time(path3)
            reqs3 = self.lock.request_path(path3)
            for r in reqs3: yield r

            t_move3_s = self.env.now
            yield self.env.timeout(t3)
            t_move3_e = self.env.now

            self.lock.release_path(reqs3)
            self.agv_pos[agv] = shelf_idx

            # === 放架 ===
            t_drop_s = self.env.now
            yield self.env.timeout(DROP_TIME)
            t_drop_e = self.env.now

            # === 记录详细 timeline ===
            self.timeline.append({
                "Task":    task,
                "AGV":     agv,
                "Shelf":   shelf,
                "WS":      ws_id,
                "move1_s": t_move1_s, "move1_e": t_move1_e,
                "pick_s":  t_pick_s,  "pick_e":  t_pick_e,
                "move2_s": t_move2_s, "move2_e": t_move2_e,
                "ws_s":    t_ws_s,    "ws_e":    t_ws_e,
                "move3_s": t_move3_s, "move3_e": t_move3_e,
                "drop_s":  t_drop_s,  "drop_e":  t_drop_e,
            })

    # ---------- 运行 ----------
    def run(self) -> float:
        self.env.run()
        # 以“放架完成”时间作为 makespan
        return max(record["drop_e"] for record in self.timeline)
