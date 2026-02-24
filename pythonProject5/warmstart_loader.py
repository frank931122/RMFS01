# warmstart_loader.py
from __future__ import annotations
import re
from typing import Dict, List, Tuple
from types import SimpleNamespace

AGV_LINE = re.compile(r"AGV\s+(\d+)\s*:\s*\[([^\]]*)\]")
X_LINE   = re.compile(r"x\[(\d+)\s*,\s*(\d+)\]\s*=\s*1")
SHELF_LINE = re.compile(r"Shelf\s+(\d+)\s*:\s*\[([^\]]*)\]")
W_LINE   = re.compile(r"w\[(\d+)\s*,\s*(\d+)\]\s*=\s*1")
Z_LINE   = re.compile(r"z\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]\s*=\s*1")

def _parse_list_of_ints(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def _reconstruct_routes_from_z(
    z_triplets: List[Tuple[int,int,int]],
    w_pairs: List[Tuple[int,int]],
    tasks_set: set[int]
) -> Dict[int, List[int]]:
    """
    仅当未捕获 'AGV k: [...]' 行时，用 z[j,j',r]=1 和 w[j,r]=1 重建各车的任务顺序。
    过滤掉 j>1000 的“虚拟节点”(如 1001/2001...)。
    """
    # 每台车的边: r -> {j: j'}，以及入度统计
    by_r_edges: Dict[int, Dict[int,int]] = {}
    by_r_indeg: Dict[int, Dict[int,int]] = {}
    tasks = set(t for t in tasks_set if t <= 1000)  # 只保留真实任务编号

    # w: 任务归属车（用于确定哪些任务该出现在该车）
    tasks_by_r: Dict[int, set[int]] = {}
    for j, r in w_pairs:
        if j <= 1000:
            tasks_by_r.setdefault(int(r), set()).add(int(j))

    for j, jp, r in z_triplets:
        if j <= 1000 and jp <= 1000:
            by_r_edges.setdefault(int(r), {})[int(j)] = int(jp)
            by_r_indeg.setdefault(int(r), {}).setdefault(int(jp), 0)
            by_r_indeg[int(r)][int(jp)] += 1
            by_r_indeg[int(r)].setdefault(int(j), 0)

    routes: Dict[int, List[int]] = {}
    for r, edges in by_r_edges.items():
        cand_nodes = tasks_by_r.get(r, set(tasks))  # 若 w 缺失，就在全部真实任务中找
        indeg = by_r_indeg.get(r, {})
        # 起点：入度为 0 并且在 cand_nodes 里
        starts = [j for j in edges.keys() if indeg.get(j, 0) == 0 and j in cand_nodes]
        seq: List[int] = []
        used = set()
        for s in starts:
            cur = s
            while cur in edges and cur not in used and cur in cand_nodes:
                seq.append(cur)
                used.add(cur)
                cur = edges[cur]
            # 尾巴
            if cur in cand_nodes and cur not in used and cur in edges.values():
                # 有可能最后一个节点也需要纳入（如果它没有后继）
                pass
        # 如果还没覆盖 cand_nodes，按入度排序补齐
        remain = [j for j in cand_nodes if j not in set(seq)]
        remain.sort(key=lambda x: indeg.get(x, 0))
        seq += remain
        routes[int(r)] = seq
    # 若没有任何 r，则用 w_pairs 粗略构建
    if not routes and tasks_by_r:
        for r, tset in tasks_by_r.items():
            routes[int(r)] = sorted(int(x) for x in tset)
    return routes

def parse_milp_log(log_path: str) -> SimpleNamespace:
    """
    解析 MILP 输出日志文本（你贴的那种）→ 返回一个带 routes/place/shelf_seq 的对象。
    兼容两种来源：
      - 直接解析 `AGV k: [..]`、`x[j,s]=1`、`Shelf c: [..]`
      - 若缺少 `AGV k: [...]`，则用 `z[j,j',r]=1` + `w[j,r]=1` 重建。
    """
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # place: x[j,s]=1
    place: Dict[int, int] = {}
    for m in X_LINE.finditer(text):
        j, s = int(m.group(1)), int(m.group(2))
        place[j] = s

    # shelf_seq: "Shelf c: [ ... ]"
    shelf_seq: Dict[int, List[int]] = {}
    for m in SHELF_LINE.finditer(text):
        c = int(m.group(1))
        seq = _parse_list_of_ints(m.group(2))
        shelf_seq[c] = seq

    # routes: 优先解析 "AGV k: [ ... ]"
    routes: Dict[int, List[int]] = {}
    for m in AGV_LINE.finditer(text):
        r = int(m.group(1))
        seq = _parse_list_of_ints(m.group(2))
        routes[r] = seq

    # 若没有 AGV 行，用 z + w 重建
    if not routes:
        z_triplets = []
        for m in Z_LINE.finditer(text):
            j, jp, r = int(m.group(1)), int(m.group(2)), int(m.group(3))
            z_triplets.append((j, jp, r))
        w_pairs = []
        for m in W_LINE.finditer(text):
            j, r = int(m.group(1)), int(m.group(2))
            w_pairs.append((j, r))
        tasks_set = set(place.keys()) if place else set(j for j, _ in w_pairs)
        routes = _reconstruct_routes_from_z(z_triplets, w_pairs, tasks_set)

    # 清洗 routes 中的“虚拟节点”
    for r in list(routes.keys()):
        routes[r] = [int(j) for j in routes[r] if int(j) <= 1000]

    return SimpleNamespace(routes=routes, place=place, shelf_seq=shelf_seq)
