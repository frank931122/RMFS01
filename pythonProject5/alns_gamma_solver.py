# alns_gamma_solver.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, random, time, dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception:
    gp = None
    GRB = None

TaskId = int
ShelfId = int
WsId = int
AgvId = int
StoreId = int

@dataclass
class ProblemData:
    J: List[TaskId]
    R: List[AgvId]
    S: List[StoreId]
    K: List[WsId]
    shelf_of: Dict[TaskId, ShelfId]
    ws_of: Dict[TaskId, WsId]
    duration: Dict[TaskId, float]
    setup_ws: float
    shelf_start_store: Dict[ShelfId, StoreId]
    agv_home_store: Dict[AgvId, StoreId]
    WSOrder: Dict[WsId, List[TaskId]]
    d_pi_s: Dict[Tuple[TaskId, StoreId], float]
    d_s_pi: Dict[Tuple[StoreId, TaskId], float]
    d_s_s:  Dict[Tuple[StoreId, StoreId], float]
    delta_proc: Dict[TaskId, float]
    delta_pi_s: Dict[Tuple[TaskId, StoreId], float]
    delta_s_pi: Dict[Tuple[StoreId, TaskId], float]
    store_candidate: Dict[TaskId, List[StoreId]] = field(default_factory=dict)
    Gamma: int = 1

@dataclass
class State:
    x: Dict[TaskId, StoreId]
    w: Dict[TaskId, AgvId]
    seq_agv: Dict[AgvId, List[TaskId]]
    seq_shelf: Dict[ShelfId, List[TaskId]]
    score: float = float('inf')
    p0: Dict[TaskId, float] = field(default_factory=dict)
    q0: Dict[TaskId, float] = field(default_factory=dict)
    g0: Dict[TaskId, float] = field(default_factory=dict)
    h0: Dict[TaskId, float] = field(default_factory=dict)
    pG: Dict[TaskId, float] = field(default_factory=dict)
    qG: Dict[TaskId, float] = field(default_factory=dict)
    gG: Dict[TaskId, float] = field(default_factory=dict)
    hG: Dict[TaskId, float] = field(default_factory=dict)

def build_initial_solution(P: ProblemData, rnd: random.Random) -> State:
    # 1) 货架链：按任务 id 升序
    seq_shelf: Dict[ShelfId, List[TaskId]] = {}
    shelf_groups: Dict[ShelfId, List[TaskId]] = {}
    for j in P.J:
        shelf_groups.setdefault(P.shelf_of[j], []).append(j)
    for c, lst in shelf_groups.items():
        seq_shelf[c] = sorted(lst)

    # 2) 结束储位：工位最近
    x: Dict[TaskId, StoreId] = {}
    for j in P.J:
        cand = P.store_candidate.get(j, P.S)
        x[j] = min(cand, key=lambda s: P.d_pi_s[(j, s)])

    # 3) AGV 指派：各自 home → 工位 + 负载均衡
    w: Dict[TaskId, AgvId] = {}
    load = {r: 0.0 for r in P.R}
    seq_agv: Dict[AgvId, List[TaskId]] = {r: [] for r in P.R}
    for j in P.J:
        def score(r: AgvId):
            s0 = P.agv_home_store[r]
            return (load[r], P.d_s_pi[(s0, j)] + P.d_pi_s[(j, x[j])])
        r_best = min(P.R, key=score)
        w[j] = r_best
        seq_agv[r_best].append(j)
        load[r_best] += P.duration[j]

    # 4) 工位顺序稳定化
    for r in P.R:
        seq = seq_agv[r]
        seq_agv[r] = sorted(seq, key=lambda j: (P.ws_of[j], P.WSOrder[P.ws_of[j]].index(j)))

    return State(x=x, w=w, seq_agv=seq_agv, seq_shelf=seq_shelf)

def solve_robust_time_lp(P: ProblemData, S: State, time_limit: Optional[float]=None) -> Tuple[float, Dict]:
    assert gp is not None, "需要安装 gurobipy 才能进行精确评估"
    m = gp.Model("robust_time_lp")
    if time_limit:
        m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 0

    J = P.J
    G = [0, P.Gamma] if P.Gamma > 0 else [0]
    bigT = sum(P.duration.values()) + (max(P.d_pi_s.values()) if P.d_pi_s else 0.0) \
           + (max(P.d_s_pi.values()) if P.d_s_pi else 0.0) + 100.0

    # 变量
    p = {(j,k): m.addVar(lb=0.0, name=f"p_{j}_{k}") for j in J for k in G}
    q = {(j,k): m.addVar(lb=0.0, name=f"q_{j}_{k}") for j in J for k in G}
    g = {(j,k): m.addVar(lb=0.0, name=f"g_{j}_{k}") for j in J for k in G}
    h = {(j,k): m.addVar(lb=0.0, name=f"h_{j}_{k}") for j in J for k in G}
    Cmax = m.addVar(lb=0.0, name="Cmax")

    # 同储位的二元先后（仅对实际选同 s 的任务对建）
    pi_bin = {}
    store_to_tasks: Dict[StoreId, List[TaskId]] = {}
    for j in J:
        store_to_tasks.setdefault(S.x[j], []).append(j)
    for s, lst in store_to_tasks.items():
        lst = sorted(lst)
        for i in range(len(lst)):
            for j2 in range(i+1, len(lst)):
                a, b = lst[i], lst[j2]
                pi_bin[(a,b)] = m.addVar(vtype=GRB.BINARY, name=f"pi_{a}_{b}_s{s}")

    m.update()

    # (A) 加工
    for j in J:
        D = P.duration[j]
        m.addConstr(q[(j,0)] >= p[(j,0)] + D, name=f"proc_nom_{j}")
        if P.Gamma > 0:
            m.addConstr(q[(j,P.Gamma)] >= p[(j,P.Gamma)] + D, name=f"proc_rob_stay_{j}")
            delta = P.delta_proc.get(j, 0.0)
            if delta > 0:
                m.addConstr(q[(j,P.Gamma)] >= p[(j,0)] + D + delta, name=f"proc_rob_use_{j}")

    # (B) 工位固定顺序（左鲁棒/右名义）
    for k_ws, chain in P.WSOrder.items():
        for u, v in zip(chain[:-1], chain[1:]):
            m.addConstr(p[(v,0)] >= q[(u,P.Gamma)] + P.setup_ws, name=f"ws_left_rob_{u}_{v}")
            if P.Gamma > 0:
                m.addConstr(p[(v,P.Gamma)] >= q[(u,P.Gamma)] + P.setup_ws, name=f"ws_rob_prop_{u}_{v}")

    # (C) π(j)→s: q→g→h
    for j in J:
        s_end = S.x[j]
        d_nom = P.d_pi_s[(j, s_end)]
        m.addConstr(g[(j,0)] >= q[(j,0)] + d_nom, name=f"q2g_nom_{j}")
        if P.Gamma > 0:
            m.addConstr(g[(j,P.Gamma)] >= q[(j,P.Gamma)] + d_nom, name=f"q2g_rob_stay_{j}")
            delta = P.delta_pi_s.get((j, s_end), 0.0)
            if delta > 0:
                m.addConstr(g[(j,P.Gamma)] >= q[(j,0)] + d_nom + delta, name=f"q2g_rob_use_{j}")
        for kk in G:
            m.addConstr(h[(j,kk)] >= g[(j,kk)], name=f"h_ge_g_{j}_{kk}")

    # (D) 货架链 i→j
    for c, chain in S.seq_shelf.items():
        for i, j in zip(chain[:-1], chain[1:]):
            s_i = S.x[i]
            d_nom = P.d_s_pi[(s_i, j)]
            m.addConstr(p[(j,0)] >= h[(i,P.Gamma)] + d_nom + P.setup_ws, name=f"shelf_left_rob_{i}_{j}")
            if P.Gamma > 0:
                m.addConstr(p[(j,P.Gamma)] >= h[(i,P.Gamma)] + d_nom, name=f"shelf_rob_stay_{i}_{j}")
                delta = P.delta_s_pi.get((s_i, j), 0.0)
                if delta > 0:
                    m.addConstr(p[(j,P.Gamma)] >= h[(i,0)] + d_nom + delta, name=f"shelf_rob_use_{i}_{j}")

    # (E) AGV 串（可选：这里也做左鲁棒/右名义）
    for r, chain in S.seq_agv.items():
        for a, b in zip(chain[:-1], chain[1:]):
            s_a = S.x[a]
            d_nom = P.d_s_pi[(s_a, b)]
            m.addConstr(p[(b,0)] >= h[(a,P.Gamma)] + d_nom, name=f"agv_left_rob_{a}_{b}")
            if P.Gamma > 0:
                m.addConstr(p[(b,P.Gamma)] >= h[(a,P.Gamma)] + d_nom, name=f"agv_rob_stay_{a}_{b}")
                delta = P.delta_s_pi.get((s_a, b), 0.0)
                if delta > 0:
                    m.addConstr(p[(b,P.Gamma)] >= h[(a,0)] + d_nom + delta, name=f"agv_rob_use_{a}_{b}")

    # (F) 同储位不重叠（Indicator；rhs 必须是常数 => 挪到左边）
    for (a,b), zbin in pi_bin.items():
        # 情形1：a 在 b 前 → g[b,*] ≥ h[a,*]  ⇒  g[b,*] - h[a,*] ≥ 0
        m.addGenConstrIndicator(zbin, True,  g[(b,0)] - h[(a,P.Gamma)], GRB.GREATER_EQUAL, 0.0,
                                name=f"pi1_nom_{a}_{b}")
        for kk in G:
            m.addGenConstrIndicator(zbin, True,  g[(b,kk)] - h[(a,kk)], GRB.GREATER_EQUAL, 0.0,
                                    name=f"pi1_rob_{a}_{b}_{kk}")
        # 情形2：b 在 a 前 → g[a,*] ≥ h[b,*]  ⇒  g[a,*] - h[b,*] ≥ 0
        m.addGenConstrIndicator(zbin, False, g[(a,0)] - h[(b,P.Gamma)], GRB.GREATER_EQUAL, 0.0,
                                name=f"pi0_nom_{a}_{b}")
        for kk in G:
            m.addGenConstrIndicator(zbin, False, g[(a,kk)] - h[(b,kk)], GRB.GREATER_EQUAL, 0.0,
                                    name=f"pi0_rob_{a}_{b}_{kk}")

    # 目标（右名义）
    for j in J:
        m.addConstr(Cmax >= q[(j,0)], name=f"Cmax_ge_q0_{j}")
    m.setObjective(Cmax, GRB.MINIMIZE)

    m.optimize()
    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        return float('inf'), {}

    val = lambda d: {k: v.X for k, v in d.items()}
    p_val, q_val, g_val, h_val = val(p), val(q), val(g), val(h)
    score = float(Cmax.X)

    out = {
        'p0': {j: p_val[(j,0)] for j in J},
        'q0': {j: q_val[(j,0)] for j in J},
        'g0': {j: g_val[(j,0)] for j in J},
        'h0': {j: h_val[(j,0)] for j in J},
        'pG': {j: p_val[(j,P.Gamma)] for j in J} if P.Gamma>0 else {},
        'qG': {j: q_val[(j,P.Gamma)] for j in J} if P.Gamma>0 else {},
        'gG': {j: g_val[(j,P.Gamma)] for j in J} if P.Gamma>0 else {},
        'hG': {j: h_val[(j,P.Gamma)] for j in J} if P.Gamma>0 else {},
    }
    return score, out

def fast_approx_score(P: ProblemData, S: State) -> float:
    ready_ws = {k: 0.0 for k in P.K}
    t_q = {}
    for k_ws, chain in P.WSOrder.items():
        for j in chain:
            t_start = max(ready_ws[k_ws], 0.0)
            t_end = t_start + P.duration[j]
            ready_ws[k_ws] = t_end + P.setup_ws
            t_q[j] = t_end
    return max(t_q.values()) if t_q else 0.0

@dataclass
class OperatorWeight:
    name: str
    w: float = 1.0
    score: float = 0.0
    use: int = 0

class ALNS:
    def __init__(self, P: ProblemData, seed: int = 0):
        self.P = P
        self.rnd = random.Random(seed)
        self.destroy_ops = [OperatorWeight("shaw"), OperatorWeight("critpath"),
                            OperatorWeight("store_conflict"), OperatorWeight("agv_seg"), OperatorWeight("shelf_cut")]
        self.repair_ops  = [OperatorWeight("shelf_insert"), OperatorWeight("agv_insert"),
                            OperatorWeight("storage_assign"), OperatorWeight("two_opt")]
        self.reaction = 0.6
        self.seglen = 50
        self.alpha = 0.97

    def select(self, ops: List[OperatorWeight]) -> OperatorWeight:
        tot = sum(max(1e-9, op.w) for op in ops)
        r = self.rnd.random() * tot
        acc = 0.0
        for op in ops:
            acc += max(1e-9, op.w)
            if acc >= r:
                return op
        return ops[-1]

    def destroy(self, S: State, op: OperatorWeight) -> State:
        rnd = self.rnd
        n = len(self.P.J)
        m = max(1, int(0.1 * n))
        remove_set: Set[TaskId] = set()

        if op.name == "shaw":
            seed_j = rnd.choice(self.P.J)
            cand = sorted((j for j in self.P.J if j != seed_j),
                          key=lambda j: (self.P.ws_of[j]==self.P.ws_of[seed_j],
                                         self.P.d_s_pi[(S.x[seed_j], j)]))
            for j in cand[:m]:
                remove_set.add(j)

        elif op.name == "critpath":
            tail = []
            for k_ws, chain in self.P.WSOrder.items():
                if chain:
                    tail.append(chain[-1])
            remove_set.update(tail[:m])

        elif op.name == "store_conflict":
            freq = {}
            for j in self.P.J:
                s = S.x[j]; freq[s] = freq.get(s,0)+1
            s_star = max(freq, key=freq.get) if freq else None
            js = [j for j in self.P.J if S.x.get(j,None)==s_star] if s_star is not None else []
            remove_set.update(js[:m])

        elif op.name == "agv_seg":
            cand_agv = [r for r in self.P.R if S.seq_agv.get(r)]
            if cand_agv:
                r = self.rnd.choice(cand_agv)
                seg = S.seq_agv[r]
                i = rnd.randrange(len(seg))
                j = min(len(seg), i + m)
                remove_set.update(seg[i:j])

        elif op.name == "shelf_cut":
            c = self.rnd.choice(list(S.seq_shelf.keys()))
            chain = S.seq_shelf.get(c, [])
            for j in chain[:m]:
                remove_set.add(j)

        S2 = dataclasses.replace(S)
        S2 = dataclasses.replace(
            S2,
            x=S2.x.copy(), w=S2.w.copy(),
            seq_agv={r: [j for j in lst if j not in remove_set] for r,lst in S.seq_agv.items()},
            seq_shelf={c:[j for j in lst if j not in remove_set] for c,lst in S.seq_shelf.items()}
        )
        for j in remove_set:
            S2.x.pop(j, None)
            S2.w.pop(j, None)
        return S2

    def repair(self, S: State, op: OperatorWeight) -> State:
        rnd = self.rnd
        missing = [j for j in self.P.J if j not in S.x]
        S2 = dataclasses.replace(S)

        if op.name == "shelf_insert":
            for j in missing:
                c = self.P.shelf_of[j]
                chain = S2.seq_shelf.get(c, [])
                best_pos, best_cost = 0, float('inf')
                if not chain:
                    best_pos = 0
                else:
                    for pos in range(len(chain)+1):
                        s_prev = self.P.shelf_start_store[c] if pos==0 else S2.x.get(chain[pos-1], self.P.shelf_start_store[c])
                        cost = self.P.d_s_pi[(s_prev, j)]
                        if cost < best_cost:
                            best_cost, best_pos = cost, pos
                S2.seq_shelf.setdefault(c, [])
                S2.seq_shelf[c].insert(best_pos, j)

        elif op.name == "agv_insert":
            for j in missing:
                best = None
                for r in self.P.R:
                    seq = S2.seq_agv.setdefault(r, [])
                    for pos in range(len(seq)+1):
                        s_prev = self.P.agv_home_store[r] if pos==0 else S2.x.get(seq[pos-1], self.P.agv_home_store[r])
                        s_end_j = min(self.P.store_candidate.get(j, self.P.S), key=lambda s: self.P.d_pi_s[(j,s)])
                        cost = self.P.d_s_pi[(s_prev, j)] + self.P.d_pi_s[(j, s_end_j)]
                        best = min(best, (cost, r, pos, s_end_j)) if best else (cost, r, pos, s_end_j)
                _, r, pos, s_end_j = best
                S2.seq_agv[r].insert(pos, j)
                S2.w[j] = r
                S2.x[j] = s_end_j

        elif op.name == "storage_assign":
            for j in missing:
                cand = self.P.store_candidate.get(j, self.P.S)
                s_end = min(cand, key=lambda s: self.P.d_pi_s[(j, s)])
                S2.x[j] = s_end

        elif op.name == "two_opt":
            cand_agv = [r for r in self.P.R if len(S2.seq_agv.get(r, [])) >= 3]
            if cand_agv:
                r = rnd.choice(cand_agv)
                seq = S2.seq_agv[r]
                i, j = sorted(rnd.sample(range(len(seq)), 2))
                if j - i >= 2:
                    seq[i:j+1] = reversed(seq[i:j+1])
                    S2.seq_agv[r] = seq

        # 兜底补全
        for j in self.P.J:
            if j not in S2.w:
                r = min(self.P.R, key=lambda rr: len(S2.seq_agv.get(rr, [])))
                S2.w[j] = r; S2.seq_agv.setdefault(r, []).append(j)
            if j not in S2.x:
                cand = self.P.store_candidate.get(j, self.P.S)
                S2.x[j] = min(cand, key=lambda s: self.P.d_pi_s[(j, s)])
            if self.P.shelf_of[j] not in S2.seq_shelf:
                S2.seq_shelf[self.P.shelf_of[j]] = [j]
            elif j not in S2.seq_shelf[self.P.shelf_of[j]]:
                S2.seq_shelf[self.P.shelf_of[j]].append(j)

        return S2

    def solve(self, max_iter: int = 2000, time_budget: float = 120.0) -> State:
        P, rnd = self.P, self.rnd
        S = build_initial_solution(P, rnd)
        S.score, out = solve_robust_time_lp(P, S)
        S.p0, S.q0, S.g0, S.h0 = out.get('p0',{}), out.get('q0',{}), out.get('g0',{}), out.get('h0',{})
        S.pG, S.qG, S.gG, S.hG = out.get('pG',{}), out.get('qG',{}), out.get('gG',{}), out.get('hG',{})
        S_best = dataclasses.replace(S)

        T = max(1e-6, 0.03 * S.score)
        t_end = time.time() + time_budget
        seg_counter = 0

        for it in range(max_iter):
            if time.time() > t_end:
                break
            d_op = self.select(self.destroy_ops); r_op = self.select(self.repair_ops)
            S1 = self.destroy(S, d_op)
            S2 = self.repair(S1, r_op)

            use_exact = ((it % 15) == 0) or (S_best.score < S.score)
            if gp is not None and use_exact:
                score, out = solve_robust_time_lp(P, S2)
                S2.score = score
            else:
                S2.score = fast_approx_score(P, S2)

            delta = S2.score - S.score
            accept = (delta <= 0) or (self.rnd.random() < math.exp(-delta / max(1e-9, T)))
            if accept:
                S = S2
                if S.score < S_best.score:
                    if gp is not None and (not use_exact):
                        S.score, out = solve_robust_time_lp(P, S)
                    if gp is not None and out:
                        S.p0, S.q0, S.g0, S.h0 = out.get('p0',{}), out.get('q0',{}), out.get('g0',{}), out.get('h0',{})
                        S.pG, S.qG, S.gG, S.hG = out.get('pG',{}), out.get('qG',{}), out.get('gG',{}), out.get('hG',{})
                    S_best = dataclasses.replace(S)
                    T *= 1.05
                    d_op.score += 5.0; r_op.score += 5.0
                else:
                    d_op.score += 1.0; r_op.score += 1.0
            else:
                d_op.score -= 0.2; r_op.score -= 0.2

            T *= self.alpha
            seg_counter += 1
            if seg_counter >= self.seglen:
                for op in self.destroy_ops + self.repair_ops:
                    op.use += 1
                    avg = op.score / max(1, op.use)
                    op.w = (1 - self.reaction) * op.w + self.reaction * max(0.1, avg)
                    op.score = 0.0; op.use = 0
                seg_counter = 0

        return S_best

def export_schedule_csv(P: ProblemData, S: State, path: str):
    import csv
    header = ["AGV","任务","工作站","开始时间p","结束时间q","使用货架",
              "起始储位","目标储位","货架初始位置","货架最终位置","AGV离开时间g","货架到位时间h"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        agv_of = S.w
        for j in P.J:
            r = agv_of.get(j, -1)
            k = P.ws_of[j]
            c = P.shelf_of[j]
            s0 = P.shelf_start_store[c]
            s_end = S.x[j]
            wr.writerow([r, j, k, S.p0.get(j,""), S.q0.get(j,""), c,
                         s0, s_end, s0, s_end, S.g0.get(j,""), S.h0.get(j,"")])
