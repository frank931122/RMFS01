# warmstart.py
from typing import Dict, List, Set, Tuple
from pythonProject5.solution_exports.solution_structures import InitialSolution

def decode_v_arcs(
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    J: Set[int],
    R,                      # dict 或 list：AGV ID 集
    J0: Dict[int, int],
    Jd: Dict[int, int],
    J_I: Dict[int, int],    # shelf_id -> vt
    shelf_data: Dict[int, int],
    agv_data: Dict[int, int],
) -> Set[Tuple[int,int,int,int]]:
    """返回 {(j, j', s, s')}，表示 v[j,j',s,s']=1 的弧集合。"""
    robot_ids = list(R.keys()) if hasattr(R, "keys") else list(R)
    robot_ids = [int(r) for r in robot_ids]
    vt2c = {int(vt): int(c) for c, vt in J_I.items()}

    # 计算每个真实任务 j 的紧邻前驱 pred[j]
    pred: Dict[int, int] = {}
    for c, seq in shelf_seq.items():
        vt = J_I[int(c)]
        seq_int = [int(x) for x in seq]
        for idx, j in enumerate(seq_int):
            pred[int(j)] = int(vt) if idx == 0 else int(seq_int[idx - 1])

    v_set: Set[Tuple[int,int,int,int]] = set()
    for r in robot_ids:
        seq = [int(J0[r])] + [int(x) for x in routes.get(r, [])] + [int(Jd[r])]
        for a, b in zip(seq[:-1], seq[1:]):
            # 放置位 s
            s = int(agv_data[r]) if a == int(J0[r]) else int(place[a])
            # 取货位 s'
            if b in J:
                i = pred[b]
                if i in vt2c:  # i 是 vt_c
                    c = vt2c[i]
                    sp = int(shelf_data[c])
                else:          # i 是真实任务
                    sp = int(place[i])
            else:
                sp = int(agv_data[r])  # 终点回原点
            v_set.add((a, b, s, sp))
    return v_set

def apply_warm_start(
    model,
    vars_pack,  # {'x':x, 'w':w, 'z':z, 'v':v, 'immediate': immediate}
    init: InitialSolution,
    J: Set[int], R, S,
    J0: Dict[int, int], Jd: Dict[int, int], J_I: Dict[int, int],
    task_shelf_mapping: Dict[int, int],
    shelf_data: Dict[int, int], agv_data: Dict[int, int],
):
    x = vars_pack['x']; w = vars_pack['w']; z = vars_pack['z']
    v = vars_pack['v']; immediate = vars_pack['immediate']

    robot_ids = list(R.keys()) if hasattr(R, "keys") else list(R)
    robot_ids = [int(r) for r in robot_ids]

    # 1) w, z：由 routes 构造
    for r in robot_ids:
        seq = [int(J0[r])] + [int(j) for j in init.routes.get(r, [])] + [int(Jd[r])]
        # w 的 start
        try: w[seq[0], r].start = 1
        except: pass
        try: w[seq[-1], r].start = 1
        except: pass
        for j in init.routes.get(r, []):
            try: w[int(j), r].start = 1
            except: pass
        # z 的 start（相邻对）
        for a, b in zip(seq[:-1], seq[1:]):
            try: z[a, b, r].start = 1
            except: pass

    # 2) immediate：由 shelf_seq 转换（仅设 1 的位置）
    for c, seq in init.shelf_seq.items():
        vt = int(J_I[int(c)])
        prev = vt
        for j in seq:
            key = (prev, int(j), int(c))
            if key in immediate:  # immediate 是 dict/tupledict
                try: immediate[key].start = 1
                except: pass
            prev = int(j)

    # 3) x：放置位
    for j, s in init.place.items():
        try: x[int(j), int(s)].start = 1
        except: pass

    # 4) v：由解码器构造
    v_set = decode_v_arcs(init.routes, init.shelf_seq, init.place, J, R, J0, Jd, J_I, shelf_data, agv_data)
    for (a, b, s, sp) in v_set:
        try: v[a, b, s, sp].start = 1
        except: pass

    model.update()
