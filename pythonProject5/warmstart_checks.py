# warmstart_checks.py
from typing import Dict, List, Set, Tuple
from warmstart import decode_v_arcs

def check_v_consistency(
    routes, shelf_seq, place,
    J: Set[int], R, S,
    J0: Dict[int,int], Jd: Dict[int,int], J_I: Dict[int,int],
    shelf_data: Dict[int,int], agv_data: Dict[int,int],
):
    v_set = decode_v_arcs(routes, shelf_seq, place, J, R, J0, Jd, J_I, shelf_data, agv_data)
    vt2c = {vt: c for c, vt in J_I.items()}

    # 入/出度统计
    in_deg, out_deg = {}, {}
    for (j,jp,s,sp) in v_set:
        out_deg[j]  = out_deg.get(j,0) + 1
        in_deg[jp]  = in_deg.get(jp,0) + 1

    # C30: 每个 j ∈ J∪J0 出度 ≤ 1
    for j in set(J)|set(J0.values()):
        assert out_deg.get(j,0) <= 1, f"Out-degree violation at {j}"

    # C29: 每个 j' ∈ J∪Jd 入度 = 1
    for jp in set(J)|set(Jd.values()):
        assert in_deg.get(jp,0) == 1, f"In-degree violation at {jp}"

    # C6a: 对每个 (j,s) 若 place[j]=s，则存在从 (j,s) 出发的弧
    for j in J:
        s = place[j]
        ok = any((jj==j and ss==s) for (jj,jp,ss,sp) in v_set)
        assert ok, f"No v from ({j},{s}) despite x[j,s]=1"

    # C7.6: j' 的取货位 = 前驱放置位（或初始位）
    # 先求 pred
    pred = {}
    for c, seq in shelf_seq.items():
        vt = J_I[c]
        for t, j in enumerate(seq):
            pred[j] = vt if t==0 else seq[t-1]

    for (j,jp,s,sp) in v_set:
        if jp in J:
            i = pred[jp]
            exp_sp = shelf_data[vt2c[i]] if i in vt2c else place[i]
            assert sp == exp_sp, f"Pick loc mismatch for jp={jp}: sp={sp}, exp={exp_sp}"

    return True
