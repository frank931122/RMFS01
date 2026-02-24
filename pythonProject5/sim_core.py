# sim/sim_core.py  —— 完整覆盖写入
import simpy
import pandas as pd
from collections import defaultdict

class RMFSSim:
    """RMFS 离散事件仿真（曼哈顿行驶略去；聚焦货架互斥 + 工作站互斥）"""

    def __init__(self,
                 env: simpy.Environment,
                 task_seq: dict[int, list[int]],
                 task_time: dict[int, tuple[float, float]],
                 task_shelf: dict[int, int | None],
                 task_info: dict[int, tuple[int, float]],   # task→(ws,dur)
                 map_obj=None):
        self.env = env
        self.task_seq  = task_seq
        self.task_time = task_time
        self.task_shelf= task_shelf
        self.task_info = task_info      # {task:(ws,dur)}
        self.map = map_obj

        # 资源：货架、工作站  capacity=1
        self.shelf_res: dict[int, simpy.Resource] = defaultdict(
            lambda: simpy.Resource(env, capacity=1))
        self.ws_res   : dict[int, simpy.Resource] = defaultdict(
            lambda: simpy.Resource(env, capacity=1))

        # KPI
        self.makespan   = 0.0
        self.delay_cnt  = 0
        self.delay_time = 0.0

        # 时间轴记录
        self.timeline: list[dict] = []

    # ---------------- 个别 AGV 过程 ----------------
    def agv_proc(self, agv_id: int, seq: list[int]):

        for task in seq:
            plan_p, plan_q = self.task_time[task]
            ws, dur = self.task_info[task]
            shelf   = self.task_shelf.get(task)

            # 计划开始前静待
            if self.env.now < plan_p:
                yield self.env.timeout(plan_p - self.env.now)

            # === 抢占工作站 / 货架 ===
            delay_start = self.env.now
            reqs = []
            # 1) 工作站
            if ws is not None:
                reqs.append(self.ws_res[ws].request())
            # 2) 货架（可能为空）
            if shelf is not None:
                reqs.append(self.shelf_res[shelf].request())

            for r in reqs:
                yield r               # 获得全部资源

            actual_start = self.env.now
            wait = actual_start - delay_start
            if wait > 1e-6:
                self.delay_cnt  += 1
                self.delay_time += wait

            # === 作业 ===
            yield self.env.timeout(dur)

            # 释放资源
            for r in reqs:
                r.resource.release(r)

            actual_end   = self.env.now
            self.makespan = max(self.makespan, actual_end)

            # 记录时间轴
            self.timeline.append(
                {"Task": task,
                 "AGV": agv_id,
                 "Shelf": shelf,
                 "Workstation": ws,
                 "PlanStart": plan_p,
                 "PlanEnd":   plan_q,
                 "RealStart": actual_start,
                 "RealEnd":   actual_end,
                 "Wait":      wait})

    # ---------------- 入口 ----------------
    def run(self):
        for agv, seq in self.task_seq.items():
            self.env.process(self.agv_proc(agv, seq))
        self.env.run()

    # ---------------- 报表 ----------------
    def report(self, save_csv: bool = True, prefix: str = "demo01"):
        df = pd.DataFrame(self.timeline).sort_values("RealStart")
        print("\n============== 任务时间轴明细 ==============")
        print(df.to_string(index=False))
        print("===========================================\n")
        if save_csv:
            df.to_csv(f"solution_exports/{prefix}_taskTimeline.csv", index=False)
            print(f"[report] 已写 timeline 到 solution_exports/{prefix}_taskTimeline.csv\n")
        print(f"Simulated  makespan : {self.makespan:.1f}")
        print(f"Total waits(count) : {self.delay_cnt}")
        print(f"Cumulative wait(t) : {self.delay_time:.1f}\n")
