# heuristic_init.py
from typing import Dict, List, Set
from solution_exports.solution_structures import InitialSolution

def build_initial_solution_basic(
    J: Set[int],
    R,                          # dict 或 list：AGV ID 集
    task_shelf_mapping: Dict[int, int],
    shelf_data: Dict[int, int], # shelf_id -> 初始储位 s0
    J_I: Dict[int, int],        # shelf_id -> vt 任务
    pi: Dict[int, int],         # 任务 -> 工位
) -> InitialSolution:
    # 规范化机器人ID列表
    robot_ids = list(R.keys()) if hasattr(R, "keys") else list(R)
    robot_ids = [int(r) for r in robot_ids]

    # 1) routes：按(工位, 任务ID)排序后轮转分配
    routes: Dict[int, List[int]] = {r: [] for r in robot_ids}
    J_sorted = sorted(list(J), key=lambda j: (int(pi.get(j, 10**9)), int(j)))
    for idx, j in enumerate(J_sorted):
        r = robot_ids[idx % len(robot_ids)]
        routes[r].append(int(j))

    # 2) shelf_seq：每个货架内按任务ID升序
    #   （更复杂的启发式可后续替换；当前与 C7.6 一致且稳妥）
    # 构造：c -> 该货架上的真实任务列表
    rack_to_tasks: Dict[int, List[int]] = {}
    for j in J:
        c = task_shelf_mapping.get(j, None)
        if c is not None:
            rack_to_tasks.setdefault(int(c), []).append(int(j))
    shelf_seq: Dict[int, List[int]] = {c: sorted(ts) for c, ts in rack_to_tasks.items()}

    # 3) place：全部放回到该货架的初始储位（与 C7.6 无缝一致）
    place: Dict[int, int] = {}
    for j in J:
        c = task_shelf_mapping.get(j, None)
        if c is None or c not in shelf_data:
            # 容错：若无映射或未知货架，放任意一个有效储位（通常不会发生）
            # 这里不抛错，避免首跑阻塞
            place[int(j)] = next(iter(shelf_data.values()))
        else:
            place[int(j)] = int(shelf_data[c])

    return InitialSolution(routes=routes, shelf_seq=shelf_seq, place=place)
