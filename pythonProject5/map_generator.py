# map_generator.py
import os
import numpy as np
import pandas as pd

class Map:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        # 默认全是通行点
        self.grid = np.full((height, width), 'waypoint', dtype=object)
        # items: 0 empty, 1 AGV, 2 Shelf
        self.items = np.zeros((height, width), dtype=int)
        self.shelf_positions = {}  # {shelf_id: (row,col)}
        self.agv_positions   = {}  # {agv_id: (row,col)}

    def set_attribute(self, indexes, attribute):
        """indexes 为 1-based cell 索引列表。"""
        for idx in indexes:
            r, c = divmod(int(idx)-1, self.width)
            self.grid[r, c] = attribute

    def place_shelf(self, shelf_id, index):
        r, c = divmod(int(index)-1, self.width)
        if self.items[r, c] == 0:
            self.items[r, c] = 2
            self.shelf_positions[int(shelf_id)] = (r, c)
        else:
            raise ValueError(f"格子 {index} 已被占用，无法放置货架 {shelf_id}.")

    def place_agv(self, agv_id, index):
        r, c = divmod(int(index)-1, self.width)
        if self.grid[r, c] == 'sp' and self.items[r, c] == 0:
            self.items[r, c] = 1
            self.agv_positions[int(agv_id)] = (r, c)
        else:
            raise ValueError(f"格子 {index} 非储位或已被占用，无法放置 AGV {agv_id}.")

    def extract_map_info(self):
        info = {}
        for i in range(self.height):
            for j in range(self.width):
                if self.items[i, j] == 1:
                    info[(i, j)] = 'AGV'
                elif self.items[i, j] == 2:
                    info[(i, j)] = 'Shelf'
                else:
                    info[(i, j)] = self.grid[i, j]
        return info

    def extract_storage_points(self):
        pts = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 'sp':
                    pts.append((i, j))
        return pts

    def extract_workstations(self):
        pts = []
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i, j] == 'ws':
                    pts.append((i, j))
        return pts

    def extract_shelves(self):
        return self.shelf_positions  # {shelf_id: (row,col)}

    def extract_AGVs(self):
        return self.agv_positions    # {agv_id: (row,col)}

def _idx2rc(idx: int, W: int):
    return divmod(idx-1, W)

def create_map_from_components(width: int,
                               height: int,
                               sp_indices: list[int],
                               ws_indices: list[int],
                               shelf_data: dict[int,int],
                               agv_data: dict[int,int]):
    """通用创建：用 CSV 读到的索引构造地图。"""
    m = Map(height, width)
    if sp_indices:
        m.set_attribute(sp_indices, 'sp')
    if ws_indices:
        m.set_attribute(ws_indices, 'ws')
    # 可选：示例充电/其他
    # m.set_attribute([9,63], 'cp')

    # 放置货架/AGV
    for sid, sp in shelf_data.items():
        m.place_shelf(sid, sp)
    for aid, sp in agv_data.items():
        m.place_agv(aid, sp)
    return m

# 兼容你原来的入口：若仍想快速创建默认 demo 布局
def create_map(shelf_data, agv_data):
    width, height = 9, 7
    m = Map(height, width)
    m.set_attribute([11, 38], 'ws')  # 两个工作站
    storage_indices = [14,15,16,23,24,25,41,42,43,50,51,52]
    m.set_attribute(storage_indices, 'sp')
    # 可选：
    m.set_attribute([9,63], 'cp')
    m.set_attribute([20,47], 'pp')
    for sid, idx in shelf_data.items():
        m.place_shelf(sid, idx)
    for aid, idx in agv_data.items():
        m.place_agv(aid, idx)
    return m

# ================= 场景 CSV 读/写接口 =================

def save_map_csv(prefix: str,
                 shelf_data: dict[int,int],
                 agv_data: dict[int,int],
                 ws_indices: list[int],
                 sp_indices: list[int],
                 width: int, height: int,
                 outdir: str = "scenario"):
    """
    写出：
      scenario/<prefix>/grid.csv             : width,height
      scenario/<prefix>/workstations.csv     : WS_ID, idx
      scenario/<prefix>/storage_points.csv   : SP_ID
      scenario/<prefix>/shelf_init.csv       : Shelf_ID, SP
      scenario/<prefix>/agv_init.csv         : AGV_ID, SP
    """
    scen_dir = os.path.join(outdir, prefix)
    os.makedirs(scen_dir, exist_ok=True)

    pd.DataFrame([(width, height)], columns=["width","height"])\
        .to_csv(os.path.join(scen_dir, "grid.csv"), index=False)

    ws_rows = [(i+1, ws_idx) for i, ws_idx in enumerate(ws_indices)]
    pd.DataFrame(ws_rows, columns=["WS_ID","idx"])\
        .to_csv(os.path.join(scen_dir, "workstations.csv"), index=False)

    pd.DataFrame([(int(x),) for x in sp_indices], columns=["SP_ID"])\
        .to_csv(os.path.join(scen_dir, "storage_points.csv"), index=False)

    pd.DataFrame([(int(k), int(v)) for k, v in shelf_data.items()],
                 columns=["Shelf_ID","SP"])\
        .to_csv(os.path.join(scen_dir, "shelf_init.csv"), index=False)

    pd.DataFrame([(int(k), int(v)) for k, v in agv_data.items()],
                 columns=["AGV_ID","SP"])\
        .to_csv(os.path.join(scen_dir, "agv_init.csv"), index=False)

    print(f"[MAP] 写出场景 CSV 至 {scen_dir}/")

def load_map_csv(prefix: str, indir: str = "scenario"):
    """
    读回场景：
      return shelf_data, agv_data, ws_indices, sp_indices, width, height
    """
    scen_dir = os.path.join(indir, prefix)
    grid_df  = pd.read_csv(os.path.join(scen_dir, "grid.csv"))
    ws_df    = pd.read_csv(os.path.join(scen_dir, "workstations.csv"))
    sp_df    = pd.read_csv(os.path.join(scen_dir, "storage_points.csv"))
    sh_df    = pd.read_csv(os.path.join(scen_dir, "shelf_init.csv"))
    agv_df   = pd.read_csv(os.path.join(scen_dir, "agv_init.csv"))

    width, height = int(grid_df.iloc[0]["width"]), int(grid_df.iloc[0]["height"])
    ws_indices    = [int(x) for x in ws_df["idx"].tolist()]
    sp_indices    = [int(x) for x in sp_df["SP_ID"].tolist()]
    shelf_data    = {int(r.Shelf_ID): int(r.SP) for _, r in sh_df.iterrows()}
    agv_data      = {int(r.AGV_ID):   int(r.SP) for _, r in agv_df.iterrows()}

    print(f"[MAP] 读取场景: W={width},H={height}, WS={ws_indices}, #SP={len(sp_indices)}, #Shelf={len(shelf_data)}, #AGV={len(agv_data)}")
    return shelf_data, agv_data, ws_indices, sp_indices, width, height

if __name__ == "__main__":
    """
    一次性把你当前默认 demo 布局固化到 CSV：
      scenario/demo01/(grid|workstations|storage_points|shelf_init|agv_init).csv
    """
    prefix = "demo01"
    # 与你项目里的默认一致
    shelf_data = {1:23,2:24,3:25,4:41,5:42,6:43}
    agv_data   = {1:14,2:15,3:16}
    ws_indices = [11, 38]
    sp_indices = [14,15,16,23,24,25,41,42,43,50,51,52]
    save_map_csv(prefix, shelf_data, agv_data, ws_indices, sp_indices, width=9, height=7)
