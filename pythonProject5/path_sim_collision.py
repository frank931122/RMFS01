# path_sim_collision.py
import math
from collections import deque

class RMFSPathSimCollision:
    """
    动态绕路 + 工作站互斥 + 货架/AGV防碰撞
    run() 后:
        self.timeline            任务时间表
        self.agv_pos_history     {t: {agv: idx}}
        self.shelf_pos_history   {t: {shelf: idx或0(运输)}}
    """

    def __init__(self,
                 map_obj,
                 task_seq: dict,
                 task_shelf: dict,
                 task_info: dict,
                 shelf_init_sp: dict,
                 agv_init_sp: dict,
                 waiting_cells=None):
        self.map = map_obj
        self.W = map_obj.width
        self.H = map_obj.height

        self.task_seq = {int(a): list(v) for a, v in task_seq.items()}
        self.task_shelf = {int(k): int(v) for k, v in task_shelf.items()}

        # 过滤 NaN
        cleaned = {}
        valid = set()
        for k, v in task_info.items():
            ws, dur = v
            if (isinstance(ws, float) and math.isnan(ws)) or \
               (isinstance(dur, float) and math.isnan(dur)):
                print(f"[WARN] 忽略任务 {k}: Workstation 或 Duration 是 NaN")
                continue
            cleaned[int(k)] = (int(ws), float(dur))
            valid.add(int(k))
        self.task_info = cleaned
        for agv, seq in self.task_seq.items():
            self.task_seq[agv] = [int(t) for t in seq if int(t) in valid]

        self.shelf_home = {int(k): int(v) for k, v in shelf_init_sp.items()}
        self.agv_init = {int(k): int(v) for k, v in agv_init_sp.items()}
        self.waiting_cells = waiting_cells or {}

        # 货架状态
        self.shelf_state = {
            sh: {"state": "static", "agv": None, "home": sp}
            for sh, sp in self.shelf_home.items()
        }
        # AGV 状态
        self.agv_state = {}
        for agv, start in self.agv_init.items():
            self.agv_state[agv] = {
                "pos": start,
                "queue": list(self.task_seq.get(agv, [])),
                "phase": "idle",
                "task": None,
                "shelf": None,
                "ws_id": None,
                "svc_dur": 0.0,
                "timing": {},
                "goal": None,
                "ws_start": None
            }

        self.ws_busy_until = {}
        self.timeline = []
        self.agv_pos_history = {}      # t -> {agv: idx}
        self.shelf_pos_history = {}    # t -> {shelf: idx or 0}
    # ---------------- 工具 ----------------
    def _idx2rc(self, idx: int):
        return divmod(idx - 1, self.W)

    def _rc2idx(self, r: int, c: int):
        return r * self.W + c + 1

    def _neighbors(self, cell: int):
        r, c = self._idx2rc(cell)
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.H and 0 <= nc < self.W:
                yield self._rc2idx(nr, nc)

    def _static_shelf_cells(self):
        return {st["home"] for st in self.shelf_state.values()
                if st["state"] == "static"}

    def _ws_cell_index(self, ws_id: int):
        r, c = self.map.extract_workstations()[ws_id - 1]
        return self._rc2idx(r, c)

    def _bfs(self, start: int, goal: int, blocked: set):
        if start == goal:
            return [start]
        from collections import deque
        q = deque([start])
        prev = {start: None}
        while q:
            cur = q.popleft()
            for nb in self._neighbors(cur):
                if nb in prev:
                    continue
                if nb != goal and nb in blocked:
                    continue
                prev[nb] = cur
                if nb == goal:
                    path = [nb]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    path.reverse()
                    return path
                q.append(nb)
        return None

    # -------------- 启动新任务 --------------
    def _try_start_next(self, agv: int, t: int):
        st = self.agv_state[agv]
        if st["phase"] != "idle" or not st["queue"]:
            return
        task = int(st["queue"].pop(0))
        if task not in self.task_info:
            self._try_start_next(agv, t)
            return
        shelf = self.task_shelf[task]
        ws_id, dur = self.task_info[task]
        st.update({
            "task": task,
            "shelf": shelf,
            "ws_id": ws_id,
            "svc_dur": dur,
            "timing": {"Task": task, "AGV": agv, "Shelf": shelf, "WS": ws_id},
            "phase": "move1",
            "goal": self.shelf_home[shelf]
        })
        st["timing"]["move1_s"] = t

    # -------------- 主循环 ------------------
    def run(self, max_time=36000):
        t = 0
        for agv in self.agv_state:
            self._try_start_next(agv, 0)

        while t <= max_time:
            if all(st["phase"] == "idle" and not st["queue"]
                   for st in self.agv_state.values()):
                break

            # 阶段完成判定
            for agv, st in self.agv_state.items():
                phase = st["phase"]
                if phase == "move1" and st["pos"] == st["goal"]:
                    st["timing"]["move1_e"] = t
                    st["timing"]["pick_s"] = t
                    st["timing"]["pick_e"] = t
                    shelf = st["shelf"]
                    self.shelf_state[shelf]["state"] = "carried"
                    self.shelf_state[shelf]["agv"] = agv
                    st["phase"] = "move2"
                    st["goal"] = self._ws_cell_index(st["ws_id"])
                    st["timing"]["move2_s"] = t

                elif phase == "move2" and st["pos"] == st["goal"]:
                    st["timing"]["move2_e"] = t
                    ws_id = st["ws_id"]
                    ws_free = self.ws_busy_until.get(ws_id, 0)
                    ws_s = max(t, ws_free)
                    st["timing"]["ws_s"] = ws_s
                    st["ws_start"] = ws_s
                    st["phase"] = "wait_ws"

                elif phase == "wait_ws":
                    if t >= st["ws_start"]:
                        st["phase"] = "service"
                        st["timing"]["ws_e"] = st["timing"]["ws_s"] + st["svc_dur"]
                        self.ws_busy_until[st["ws_id"]] = st["timing"]["ws_e"]

                elif phase == "service":
                    if t >= st["timing"]["ws_e"]:
                        st["phase"] = "move3"
                        st["goal"] = self.shelf_home[st["shelf"]]
                        st["timing"]["move3_s"] = t

                elif phase == "move3" and st["pos"] == st["goal"]:
                    st["timing"]["move3_e"] = t
                    st["timing"]["drop_s"] = t
                    st["timing"]["drop_e"] = t
                    shelf = st["shelf"]
                    self.shelf_state[shelf]["state"] = "static"
                    self.shelf_state[shelf]["agv"] = None
                    self.timeline.append(dict(st["timing"]))
                    st.update({"phase": "idle", "task": None, "shelf": None,
                               "ws_id": None, "goal": None, "ws_start": None})
                    self._try_start_next(agv, t)

            # 计算移动意图
            reservations = set()
            new_pos = {}
            static_cells = self._static_shelf_cells()
            for agv in sorted(self.agv_state):
                st = self.agv_state[agv]
                cur = st["pos"]
                if st["phase"] not in ("move1", "move2", "move3"):
                    new_pos[agv] = cur
                    continue
                goal = st["goal"]
                blocked = set(static_cells)
                if st["phase"] == "move1":
                    blocked.discard(self.shelf_home[st["shelf"]])  # 允许钻进
                for other, ost in self.agv_state.items():
                    if other != agv:
                        blocked.add(ost["pos"])
                blocked |= reservations
                path = self._bfs(cur, goal, blocked)
                if path is None or len(path) == 1:
                    nxt = cur
                else:
                    nxt = path[1]
                # 防止穿过静止货架
                if nxt in static_cells and not (st["phase"] == "move1" and
                                                nxt == self.shelf_home[st["shelf"]]):
                    nxt = cur
                new_pos[agv] = nxt
                if nxt != cur:
                    reservations.add(nxt)

            # 执行移动
            for agv, p in new_pos.items():
                self.agv_state[agv]["pos"] = p

            # 记录逐秒位置
            self.agv_pos_history[t] = {agv: st["pos"]
                                       for agv, st in self.agv_state.items()}
            shelf_snapshot = {}
            for sh, sst in self.shelf_state.items():
                if sst["state"] == "carried":
                    agv = sst["agv"]
                    shelf_snapshot[sh] = self.agv_state[agv]["pos"]  # 跟车
                else:
                    shelf_snapshot[sh] = sst["home"]
            self.shelf_pos_history[t] = shelf_snapshot

            # 尝试启动新任务（有的刚 drop）
            for agv in self.agv_state:
                self._try_start_next(agv, t)

            t += 1
        return self.timeline
