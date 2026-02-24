import ast
import pandas as pd
from map_generator import create_map
from path_sim_collision import RMFSPathSimCollision   # ← 使用新的碰撞仿真

PREFIX = "demo01"
seq_df   = pd.read_csv(f"solution_exports/{PREFIX}_taskSeq.csv")
info_df  = pd.read_csv(f"solution_exports/{PREFIX}_taskInfo.csv")
shelf_df = pd.read_csv(f"solution_exports/{PREFIX}_taskShelf.csv")

task_seq   = {row.AGV_ID: ast.literal_eval(row.seq) for _, row in seq_df.iterrows()}
task_shelf = {row.Task: row.Shelf for _, row in shelf_df.iterrows()}
task_info  = {row.Task: (row.Workstation, row.Duration) for _, row in info_df.iterrows()}

shelf_data = {1:23,2:24,3:25,4:41,5:42,6:43}
agv_data   = {1:14,2:15,3:16}

map_obj = create_map(shelf_data, agv_data)

# 可选等待区（动画用，不写入 timeline，只供后续脚本展示）
waiting_cells = {
    1: [20, 29],
    2: [47, 56],
}

sim = RMFSPathSimCollision(map_obj,
                           task_seq,
                           task_shelf,
                           task_info,
                           shelf_init_sp=shelf_data,
                           agv_init_sp=agv_data,
                           waiting_cells=waiting_cells)

sim.run()

df = pd.DataFrame(sim.timeline)
# 各阶段用时
df["t_move1"] = df["move1_e"] - df["move1_s"]
df["t_pick"]  = df["pick_e"]  - df["pick_s"]
df["t_move2"] = df["move2_e"] - df["move2_s"]
df["t_ws"]    = df["ws_e"]    - df["ws_s"]
df["t_move3"] = df["move3_e"] - df["move3_s"]
df["t_drop"]  = df["drop_e"]  - df["drop_s"]

mk_ws   = df["ws_e"].max()
mk_drop = df["drop_e"].max()

print("\n=== 仿真详细时间表（防碰撞逐秒） ===")
print(df.to_string(index=False))
print(f"\n[Collision SIM] C_max_ws = {mk_ws:.1f} s")
print(f"[Collision SIM] C_max_drop = {mk_drop:.1f} s")

df.to_csv(f"solution_exports/{PREFIX}_timeline.csv", index=False)
# run_phys_sim.py 末尾追加 —— 在 sim.run() 之后
import pandas as pd

# 原有 timeline 输出保持不变
df = pd.DataFrame(sim.timeline)
...

# 新增：导出逐秒 AGV / Shelf 真实轨迹
agv_df = pd.DataFrame.from_dict(sim.agv_pos_history, orient='index').sort_index()
agv_df.index.name = 'time'
agv_df.to_csv("solution_exports/demo01_agv_pos.csv")

shelf_df = pd.DataFrame.from_dict(sim.shelf_pos_history, orient='index').sort_index()
shelf_df.index.name = 'time'
shelf_df.to_csv("solution_exports/demo01_shelf_pos.csv")
