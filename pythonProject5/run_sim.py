# run_sim.py  —— 只需改 2 行 + 调 report()

import ast, pandas as pd, simpy
from map_generator import create_map
from pythonProject5.sim_core import RMFSSim

PREFIX = "demo01"
seq_df   = pd.read_csv(f"solution_exports/{PREFIX}_taskSeq.csv")
time_df  = pd.read_csv(f"solution_exports/{PREFIX}_taskTiming.csv")
shelf_df = pd.read_csv(f"solution_exports/{PREFIX}_taskShelf.csv")
info_df  = pd.read_csv(f"solution_exports/{PREFIX}_taskInfo.csv")   # <‑‑ 新

task_seq  = {row.AGV_ID: ast.literal_eval(row.seq) for _, row in seq_df.iterrows()}
task_time = {row.Task: (row.p, row.q)             for _, row in time_df.iterrows()}
task_shelf= {row.Task: row.Shelf                  for _, row in shelf_df.iterrows()}
task_info = {row.Task: (row.Workstation, row.Duration)          # <‑‑ 新
             for _, row in info_df.iterrows()}

# 重新建 Map（若要用 distance()）
shelf_data = {1:23,2:24,3:25,4:41,5:42,6:43}
agv_data   = {1:14,2:15,3:16}
map_obj = create_map(shelf_data, agv_data)

env = simpy.Environment()
sim = RMFSSim(env, task_seq, task_time, task_shelf, task_info, map_obj)
sim.run()
sim.report(prefix=PREFIX)      # <‑‑ 打印 + 写 CSV
