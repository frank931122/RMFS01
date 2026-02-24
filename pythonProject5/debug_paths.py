# debug_paths.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from movement_manager import MovementManager
from map_generator import create_map

# —— 一、读 timeline.csv —— #
FOLDER, PREFIX = "solution_exports", "demo01"
df = pd.read_csv(f"{FOLDER}/{PREFIX}_timeline.csv")
time_cols = ["move1_s","move1_e","move2_s","move2_e",
             "ws_s","ws_e","move3_s","move3_e","drop_s","drop_e"]
for c in time_cols:
    df[c] = df[c].astype(float)

# —— 二、重建 map_obj + 初始点 —— #
# 必须和你 main.py / animationRMFS.py 保持一致
shelf_data   = {1:23,2:24,3:25,4:41,5:42,6:43}
agv_data     = {1:14,2:15,3:16}

map_obj      = create_map(shelf_data, agv_data)
agv_init_sp  = agv_data.copy()
shelf_init_sp= {s: shelf_data[s] for s in shelf_data}

# —— 三、插入我给的“预计算＋画图”代码 —— #
# （直接把那段 for agv_segments ... 到最后画图的代码拷过来）
# 记得把 mm = MovementManager(map_obj, None) 放到这里
import numpy as np
import pandas as pd
from movement_manager import MovementManager

# —— 1. 基本数据 —— #
# 已有：
#   df         : 仿真详细时间表，time_cols 已转换为 float
#   map_obj    : create_map(...) 返回的地图对象
#   agv_init_sp: {agv_id: storage_point_idx}
#   shelf_init_sp: {shelf_id: storage_point_idx}
#   AGV_SPEED, AGV_ROTATE_TIME 用于与 sim 保持一致
mm = MovementManager(map_obj, None)

# ———— 2. 准备时间轴（按秒取整） ———— #
T_max = int(np.ceil(df['drop_e'].max()))
times = np.arange(0, T_max+1)

# ———— 3. 对每辆 AGV，分段构建 path1/path2/path3 的格子路径及持续秒数 ———— #
#  travel_time = (len(path)-1)/AGV_SPEED + turns*AGV_ROTATE_TIME
#  假设 AGV_SPEED=1, AGV_ROTATE_TIME=0，则
#  每走一步花1秒，len(path)-1秒完成整个移动。

# 先给 df 排序，保证同一辆车任务按 move1_s 升序
df_sorted = df.sort_values(['AGV','move1_s'])

# 存每辆车每段路径
agv_segments = {r: [] for r in agv_init_sp}  # {agv_id: [ (t_start, path_idx_list), ... ]}

for agv_id, g in df_sorted.groupby('AGV'):
    for _, row in g.iterrows():
        # 1) move1: 当前 pos -> shelf_pos
        t1, t1e = row['move1_s'], row['move1_e']
        path1 = mm.get_path(int(row['Shelf_init_sp']), int(row['Shelf']))
        agv_segments[agv_id].append((t1, t1e, path1))
        # 2) move2: shelf_pos -> ws
        t2, t2e = row['move2_s'], row['move2_e']
        ws_cell = int(row['WS_cell'])
        path2 = mm.get_path(int(row['Shelf']), ws_cell)
        agv_segments[agv_id].append((t2, t2e, path2))
        # 3) move3: ws -> shelf_pos
        t3, t3e = row['move3_s'], row['move3_e']
        path3 = mm.get_path(ws_cell, int(row['Shelf']))
        agv_segments[agv_id].append((t3, t3e, path3))

# —— 4. 构建每秒的 “cell index” —— #
# 初始化：每秒都处在初始 SP
agv_cell_ts = {r: np.full(len(times), agv_init_sp[r], dtype=int)
               for r in agv_init_sp}

# 填写移动段：一旦 t 属于某个段，就根据 (t - t_start) 取路径中的第 k 步
for agv_id, segs in agv_segments.items():
    for t0, t1, path in segs:
        duration = t1 - t0
        # 对每个整数秒 t 从 ceil(t0) 到 floor(t1)-1
        for t in range(int(np.ceil(t0)), int(np.floor(t1))):
            k = int(t - t0)  # 向下取整
            if k < len(path):
                agv_cell_ts[agv_id][t] = path[k]
        # 在恰好 t1 秒，我们认为到了 path[-1]
        if int(round(t1)) <= T_max:
            agv_cell_ts[agv_id][int(round(t1))] = path[-1]

# —— 5. 货架：如果 t 在 move1_e—drop_s 区间，则随 AGV，其他时间在 home_sp —— #
shelf_cell_ts = {s: np.full(len(times), shelf_init_sp[s], dtype=int)
                 for s in shelf_init_sp}

# 为了快速查 AGV 在某秒在哪，反向映射一下：
#  cell_to_agv[t][cell_idx] = agv_id(s)
cell_to_agv = [{} for _ in times]
for r in agv_init_sp:
    for i, t in enumerate(times):
        ci = agv_cell_ts[r][i]
        cell_to_agv[i][ci] = r

for _, row in df_sorted.iterrows():
    sid = int(row['Shelf'])
    t_start = int(np.ceil(row['move1_e']))
    t_end   = int(np.floor(row['drop_s']))
    # 在 [t_start, t_end) 秒，货架在被运输：cell → AGV cell
    for t in range(t_start, t_end):
        if sid in shelf_cell_ts:
            # 找到这秒被哪个 AGV 搬
            for agv_id, mapping in enumerate(cell_to_agv[t], start=1):
                pass
        # 简单做：如果 AGV 在该格上，就认为货架也在该格
        for agv_id, arr in agv_cell_ts.items():
            if arr[t] in mm.idx2coord: continue
        # 其实直接：
        agv_positions = {r: agv_cell_ts[r][t] for r in agv_init_sp}
        # 找到承载它的 AGV：
        for r, cell in agv_positions.items():
            # 如果车的 path 中曾经从 shelf start 到 shelf end 包含这个 shelf id
            # 这里简化：只要当前格等于某车的格，就认定该车在搬这个货架
            if cell in mm.idx2coord:
                shelf_cell_ts[sid][t] = cell
                break

# —— 6. 把 idx → (row,col) 转换，画 step 图 —— #
def idx2rc(idx):
    return divmod(idx-1, map_obj.width)

import matplotlib.pyplot as plt

# AGV 画 row/time 图
for r in agv_cell_ts:
    rows = [idx2rc(idx)[0] for idx in agv_cell_ts[r]]
    plt.figure(figsize=(10,2))
    plt.step(times, rows, where='post')
    plt.title(f"AGV{r} 行驶行号时序")
    plt.xlabel("时间 (s)"); plt.ylabel("行号 (row)")
    plt.grid(True); plt.tight_layout(); plt.show()

    cols = [idx2rc(idx)[1] for idx in agv_cell_ts[r]]
    plt.figure(figsize=(10,2))
    plt.step(times, cols, where='post')
    plt.title(f"AGV{r} 行驶列号时序")
    plt.xlabel("时间 (s)"); plt.ylabel("列号 (col)")
    plt.grid(True); plt.tight_layout(); plt.show()

# 货架同理
for s in shelf_cell_ts:
    rows = [idx2rc(idx)[0] for idx in shelf_cell_ts[s]]
    plt.figure(figsize=(10,2))
    plt.step(times, rows, where='post')
    plt.title(f"货架{s} 行驶行号时序")
    plt.xlabel("时间 (s)"); plt.ylabel("行号 (row)")
    plt.grid(True); plt.tight_layout(); plt.show()

    cols = [idx2rc(idx)[1] for idx in shelf_cell_ts[s]]
    plt.figure(figsize=(10,2))
    plt.step(times, cols, where='post')
    plt.title(f"货架{s} 行驶列号时序")
    plt.xlabel("时间 (s)"); plt.ylabel("列号 (col)")
    plt.grid(True); plt.tight_layout(); plt.show()
