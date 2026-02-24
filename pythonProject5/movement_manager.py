# movement_manager.py

from utils import distance as manhattan_distance
from time_manager import TimeManager
import math
class MovementManager:
    def __init__(self, map_obj, time_manager: TimeManager | None):
        self.map = map_obj
        self.time_manager = time_manager

        # 强制把 grid 转成 Python list‑of‑lists，彻底避免 numpy 索引问题
        try:
            self.grid = map_obj.grid.tolist()
        except Exception:
            # 如果不是 numpy，也直接用原来结构（若原来已是 list，也 OK）
            self.grid = map_obj.grid

    # =========================================================
    # ---------------  PATH SEARCH (A* 4‑邻) -----------------
    # =========================================================
    def _neighbors(self, idx: int) -> list[int]:
        """
        返回可行走邻格 index 列表（上下左右，不出界、不撞 'obstacle'），
        索引只用双中括号，安全兼容所有结构。
        """
        r, c = divmod(idx - 1, self.map.width)
        res = []
        for dr, dc in ((1,0), (-1,0), (0,1), (0,-1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.map.height and 0 <= nc < self.map.width:
                cell = self.grid[nr][nc]   # **只用 [nr][nc]，不用逗号索引**
                if cell != 'obstacle':
                    res.append(nr * self.map.width + nc + 1)
        return res

    def _heuristic(self, a: int, b: int) -> int:
        ra, ca = divmod(a - 1, self.map.width)
        rb, cb = divmod(b - 1, self.map.width)
        return abs(ra - rb) + abs(ca - cb)

    def _astar(self, start_idx: int, goal_idx: int) -> list[int]:
        if start_idx == goal_idx:
            return [start_idx]

        import heapq
        open_set = [(0, start_idx)]
        came_from = {}
        g_cost = {start_idx: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for nxt in self._neighbors(current):
                tentative = g_cost[current] + 1
                if nxt not in g_cost or tentative < g_cost[nxt]:
                    came_from[nxt] = current
                    g_cost[nxt] = tentative
                    f = tentative + self._heuristic(nxt, goal_idx)
                    heapq.heappush(open_set, (f, nxt))

        raise ValueError(f"No path from {start_idx} to {goal_idx}")

    # -------- 公共 API --------
    # === 这里是补丁 ===
    def get_path(self, idx_from, idx_to) -> list[int]:
        """
        入口强制转 int，避免 numpy.float64 导致索引错误。
        """
        start = int(idx_from)
        goal = int(idx_to)
        return self._astar(start, goal)

    # =========================================================
    # -------- 原有 move_robot / move_shelf 保留 ---------------
    # =========================================================
    def move_robot(self, agv_id, new_index):
        if agv_id not in self.map.agv_positions:
            print(f"[MovementManager] AGV {agv_id} not found!")
            return 0
        cur_r, cur_c = self.map.agv_positions[agv_id]
        new_r, new_c = divmod(new_index - 1, self.map.width)
        dist = abs(new_r - cur_r) + abs(new_c - cur_c)
        self.map.items[cur_r][cur_c] = 0
        self.map.items[new_r][new_c] = 1
        self.map.agv_positions[agv_id] = (new_r, new_c)
        if self.time_manager:
            self.time_manager.advance_time(dist)
        return dist

    def move_shelf(self, shelf_id, new_index):
        if shelf_id not in self.map.shelf_positions:
            print(f"[MovementManager] Shelf {shelf_id} not found!")
            return 0
        cur_r, cur_c = self.map.shelf_positions[shelf_id]
        new_r, new_c = divmod(new_index - 1, self.map.width)
        dist = abs(new_r - cur_r) + abs(new_c - cur_c)
        self.map.items[cur_r][cur_c] = 0
        self.map.items[new_r][new_c] = 2
        self.map.shelf_positions[shelf_id] = (new_r, new_c)
        if self.time_manager:
            self.time_manager.advance_time(dist)
        return dist
