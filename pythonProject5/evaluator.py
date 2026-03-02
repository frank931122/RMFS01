
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
from collections import defaultdict
import heapq
import math

EPS = 1e-4
# ===== MILP alignment switch =====
# True : 更物理（把 300c / 初始占用区间算进 cell 冲突）
# False: 更贴近你当前的 MILP（忽略初始占用虚拟区间，避免 evaluator 把 MILP 解判 infeasible）
USE_INITIAL_OCCUPANCY_INTERVALS = True
# ===== Microscope debug (very targeted) =====
MICRO_DEBUG = {
    "enabled": False,
    "focus_cell": 41,
    "focus_tasks": [1, 5, 7],
    "print": False,
    "max_records": 800,
}
# ===== /Microscope debug =====



class RobustEvaluator:
    """
    事件驱动评估器（鲁棒 Γ DP 传播 + 与 main.py 的 timeline 字段兼容）。

    你现在的核心诉求：ALNS 内层评估要“便宜”，而最终对齐/MILP warm-start 要“精确”。
    所以这个类提供了“快/准”两种生效方式的开关：

    - enable_cell_repair / cell_repair_max_iters:
        * Exact：True + 较大迭代上限（如 30~80）
        * Fast：False 或 max_iters=0

    - cell_conflict_mode:
        * "hard"：检测到 cell 占用冲突直接 infeasible（返回 inf）
        * "penalty"：cell 冲突不返回 inf，而是返回 makespan + penalty（ALNS 推荐）

    - allow_incomplete:
        * True：允许 routes 未覆盖全部任务（repair 过程的中间态也能打分）
        * False：未覆盖任务直接 inf（Exact 推荐）

    - timeline_mode:
        * "full"：输出完整 timeline（用于打印/诊断/对齐）
        * "min" ：输出最小 timeline（用于粗诊断）
        * "off" ：不输出 timeline（Fast 推荐），但仍会在 details 中提供：
                 details["pick_start"] / ["arrive_cell_act"] 两个 dict，供 cell 冲突检测/repair 使用

    - record_cell_repair_log:
        * False：不记录 repair 迭代日志（Fast 推荐）
        * True：记录（debug 用）

    - collect_v_arcs:
        * False：不构造 V_arcs/Z_arcs 等诊断弧（Fast 推荐）
        * True：构造（Exact/对齐用）
    """

    def __init__(
        self,
        *,
        J: Iterable[int],
        R: Dict[int, Tuple[int, int]] | Iterable[int],
        S: Dict[int, Tuple[int, int]],
        pi: Dict[int, int],
        D: Dict[int, float],
        J0: Dict[int, int],
        Jd: Dict[int, int],
        J_I: Dict[int, int],
        shelf_data: Dict[int, int],
        agv_data: Dict[int, int],
        d_s_pi: Dict[Tuple[int, int], float],
        d_pi_s: Dict[Tuple[int, int], float],
        d_s_s: Dict[Tuple[int, int], float],
        Delta_s_pi: Optional[Dict[Tuple[int, int], float]] = None,
        Delta_pi_s: Optional[Dict[Tuple[int, int], float]] = None,
        Delta_s_s: Optional[Dict[Tuple[int, int], float]] = None,
        gamma: int = 0,
        # WS setup
        D_setup: float = 2.0,
        ws_setup_rule: str = "flat",        # "flat" | "split"
        D_setup_first: float = 2.0,
        D_setup_next: float = 2.0,
        ws_fixed_seq: Optional[Dict[int, List[int]]] = None,
        # place 缺失时才会用的候选回库位
        end_candidates_m: Optional[int] = None,
        detach_on_mismatch: bool = False,
        lock_place: bool = True,
        # 为了兼容旧构造参数保留（本实现不再用它们做“挑弧加Δ”）
        gamma_alloc: str = "length",
        gamma_critical_window: int = 2,
        # cell occupancy
        cell_gap: float = 1.0,
        BIG_M_time: float = 10000.0,
        # --- route-1 envelope on shared resources ---
        envelope_shared_resources: bool = False,
        # --- cell repair options ---
        enable_cell_repair: bool = True,
        cell_repair_max_iters: int = 50,
        cell_repair_verbose: bool = False,
        # --- chain repair options (NEW) ---
        enable_chain_repair: bool = False,
        chain_repair_max_iters: int = 30,
        chain_repair_verbose: bool = False,
        # --- speed / fitness options ---
        mode: str = "exact",                        # "exact" | "fast"
        timeline_mode: str = "full",                 # "full" | "min" | "off"
        record_cell_repair_log: bool = True,
        collect_v_arcs: bool = True,
        # --- feasibility handling for ALNS ---
        cell_conflict_mode: str = "hard",            # "hard" | "penalty"
        cell_conflict_weight: float = 1e4,
        allow_incomplete: bool = False,
        missing_task_weight: float = 1e6,
        cell_gate_mode: str = "legacy",
        cell_sigma: Optional[Dict[int, List[int]]] = None,
        auto_cell_sigma: bool = True,
        # ===== compat: init-lock args may still be passed by main.py =====
        # 当前 evaluator 版本不使用 init-lock，但为了不改 main.py，这里接住参数避免 TypeError

        enable_init_cell_lock: bool = False,
        init_lock_max_iters: int = 0,
        init_lock_tol: float = 1e-6,
        init_lock_verbose: bool = False,

        # 兜底：未来 main.py 再多传别的参数，也不会直接崩
        **_ignored_kwargs: Any,
    ):
        self.J: Set[int] = set(int(x) for x in J)

        if isinstance(R, dict):
            self.R = {int(k): v for k, v in R.items()}
            self.R_ids = sorted(int(k) for k in R.keys())
        else:
            self.R_ids = sorted(int(k) for k in R)
            self.R = {}

        self.S = {int(k): v for k, v in S.items()}
        self.pi = {int(k): int(v) for k, v in pi.items()}
        self.D = {int(k): float(v) for k, v in D.items()}

        self.J0 = {int(k): int(v) for k, v in J0.items()}
        self.Jd = {int(k): int(v) for k, v in Jd.items()}
        self.J_I = {int(k): int(v) for k, v in J_I.items()}

        self.shelf_init = {int(k): int(v) for k, v in shelf_data.items()}
        self.agv_init = {int(k): int(v) for k, v in agv_data.items()}

        if not self.R:
            fallback_cell = next(iter(self.S.keys())) if self.S else 1
            self.R = {int(r): int(self.agv_init.get(int(r), fallback_cell)) for r in self.R_ids}

        self.d_s_pi = {(int(s), int(j)): float(dt) for (s, j), dt in d_s_pi.items()}
        self.d_pi_s = {(int(j), int(s)): float(dt) for (j, s), dt in d_pi_s.items()}
        self.d_s_s = {(int(a), int(b)): float(dt) for (a, b), dt in d_s_s.items()}

        self.Delta_s_pi = defaultdict(lambda: 0.0, {} if Delta_s_pi is None else {
            (int(s), int(j)): float(v) for (s, j), v in Delta_s_pi.items()
        })
        self.Delta_pi_s = defaultdict(lambda: 0.0, {} if Delta_pi_s is None else {
            (int(j), int(s)): float(v) for (j, s), v in Delta_pi_s.items()
        })
        self.Delta_s_s = defaultdict(lambda: 0.0, {} if Delta_s_s is None else {
            (int(a), int(b)): float(v) for (a, b), v in Delta_s_s.items()
        })

        self.gamma = max(0, int(gamma))

        self.ws_setup_rule = str(ws_setup_rule).lower().strip()
        if self.ws_setup_rule not in ("split", "flat"):
            self.ws_setup_rule = "flat"
        self.D_setup = float(D_setup)
        self.D_setup_first = float(D_setup_first)
        self.D_setup_next = float(D_setup_next)

        self.end_candidates_m = None if end_candidates_m is None else int(end_candidates_m)
        self.detach_on_mismatch = bool(detach_on_mismatch)
        self.lock_place = bool(lock_place)

        if ws_fixed_seq is None:
            by_ws: Dict[int, List[int]] = defaultdict(list)
            for j in self.J:
                by_ws[self.pi[j]].append(j)
            self.ws_fixed_seq = {int(ws): sorted(lst) for ws, lst in by_ws.items()}
        else:
            self.ws_fixed_seq = {int(ws): [int(x) for x in seq] for ws, seq in ws_fixed_seq.items()}

        # 兼容字段（保留但不再使用）
        self.gamma_alloc = str(gamma_alloc).lower().strip()
        self.gamma_critical_window = max(1, int(gamma_critical_window))

        #self.cell_gap = float(cell_gap)#+ 1e-4  # <--- 偷偷加一点点，确保比 MILP 严
        # base_gap = float(cell_gap)
        # eps_gap = 0 if self.gamma == 0 else 1e-4
        #
        # self.cell_gap = base_gap + eps_gap
        # self.BIG_M_time = float(BIG_M_time)
        # cell occupancy gap
        self.cell_gap = float(cell_gap)
        self.BIG_M_time = float(BIG_M_time)
        # route-1 envelope switch
        self.envelope_shared_resources = bool(envelope_shared_resources)

        self.mode = str(mode or "exact").lower().strip()
        if self.mode not in ("exact", "fast"):
            self.mode = "exact"
        if self.mode == "fast":
            # Fast mode: favor throughput during ALNS inner-loop screening.
            enable_cell_repair = False
            cell_repair_max_iters = 0
            enable_chain_repair = False
            chain_repair_max_iters = 0
            timeline_mode = "off"
            record_cell_repair_log = False
            collect_v_arcs = False
            cell_conflict_mode = "penalty"
            allow_incomplete = True

        # cell repair
        self.enable_cell_repair = bool(enable_cell_repair)
        self.cell_repair_max_iters = max(0, int(cell_repair_max_iters))
        self.cell_repair_verbose = bool(cell_repair_verbose)
        # chain repair (NEW)
        self.enable_chain_repair = bool(enable_chain_repair)
        self.chain_repair_max_iters = max(0, int(chain_repair_max_iters))
        self.chain_repair_verbose = bool(chain_repair_verbose)
        # speed switches
        self.timeline_mode = str(timeline_mode).lower().strip()
        if self.timeline_mode not in ("full", "min", "off"):
            self.timeline_mode = "full"
        self.record_cell_repair_log = bool(record_cell_repair_log)
        self.collect_v_arcs = bool(collect_v_arcs)

        # feasibility handling
        self.cell_conflict_mode = str(cell_conflict_mode).lower().strip()
        if self.cell_conflict_mode not in ("hard", "penalty"):
            self.cell_conflict_mode = "hard"
        self.cell_conflict_weight = float(cell_conflict_weight)
        self.allow_incomplete = bool(allow_incomplete)
        self.missing_task_weight = float(missing_task_weight)
        self.cell_gate_mode = str(cell_gate_mode).lower().strip()
        if self.cell_gate_mode not in ("legacy", "event"):
            self.cell_gate_mode = "legacy"

        if cell_sigma is None:
            self.cell_sigma = None
        else:
            self.cell_sigma = {int(s): [int(x) for x in (seq or [])] for s, seq in cell_sigma.items()}
        self.auto_cell_sigma = bool(auto_cell_sigma)

        self._last_details: Optional[Dict[str, Any]] = None
        # ===== compat stash (ignored in this evaluator version) =====
        self.enable_init_cell_lock = bool(enable_init_cell_lock)
        self.init_lock_max_iters = int(init_lock_max_iters)
        self.init_lock_tol = float(init_lock_tol)
        self.init_lock_verbose = bool(init_lock_verbose)
        self._ignored_init_kwargs = dict(_ignored_kwargs)

        # ---------------- distance miss handling (CRITICAL) ----------------
        # policy: "raise" (exact/align) | "penalty" (ALNS) | "zero" (old behavior)
        self.distance_miss_policy: str = "raise"
        # when policy == "penalty", use this as the travel time returned
        self.distance_miss_penalty: float = float(self.BIG_M_time)
        # record only first N misses for debugging
        self.distance_miss_log_limit: int = 30

        # per-evaluate counters (reset at evaluate() start)
        self._dist_miss_total: int = 0
        self._dist_miss_by_kind: Dict[str, int] = defaultdict(int)
        self._dist_miss_samples: List[Dict[str, Any]] = []

    # ------------------------ Γ 向量工具 ------------------------

    def _t0(self) -> List[float]:
        return [0.0] * (self.gamma + 1)

    @staticmethod
    def _t_copy(t: List[float]) -> List[float]:
        return [float(x) for x in t]

    @staticmethod
    def _t_max(a: List[float], b: List[float]) -> List[float]:
        return [max(x, y) for x, y in zip(a, b)]

    @staticmethod
    def _t_add_det(a: List[float], d: float) -> List[float]:
        dd = float(d)
        return [x + dd for x in a]

    @staticmethod
    def _t_changed(new: List[float], old: List[float], eps: float = 1e-4) -> bool:
        return any(new[k] > old[k] + eps for k in range(len(new)))

    # ------------------------ Route-1 envelope helpers ------------------------

    def _lift(self, scalar: float) -> List[float]:
        """把标量提升成长度 (G+1) 的向量"""
        return [float(scalar)] * (self.gamma + 1)

    @staticmethod
    def _max_scalar(vec: List[float]) -> float:
        """向量的最大分量（包络最坏用）"""
        if not vec:
            return 0.0
        return float(max(vec))

    def _env_ready_vec(self, ready_vec: List[float]) -> List[float]:
        """
        路线1：共享资源采用“包络最坏”（max over all γ）。
        - envelope_shared_resources=False：退回原来的同层 ready_vec
        """
        if not self.envelope_shared_resources:
            return ready_vec
        return self._lift(self._max_scalar(ready_vec))
    def _attach_dist_miss(self, details: Dict[str, Any]) -> None:
        """Attach distance-miss diagnostics into details."""
        details["dist_miss_total"] = int(self._dist_miss_total)
        details["dist_miss_by_kind"] = dict(self._dist_miss_by_kind)
        details["dist_miss_samples"] = list(self._dist_miss_samples)

    def _dist_lookup(
        self,
        kind: str,
        key: Tuple[int, int],
        base: Dict[Tuple[int, int], float],
        delta: Dict[Tuple[int, int], float],
        ctx: str = "",
    ) -> Tuple[float, float]:
        """
        Distance/Delta lookup with strict miss handling.

        kind: "s_s" | "s_pi" | "pi_s"
        key:  (a,b) index tuple in that matrix
        ctx:  context string for debugging (move1/move2/move3 + ids)

        policy:
          - "raise": raise KeyError immediately (EXACT/ALIGN)
          - "penalty": return (distance_miss_penalty, delta) and record miss (ALNS)
          - "zero": return (0.0, delta) (legacy; NOT recommended)
        """
        if key in base:
            return float(base[key]), float(delta[key])

        # record miss
        self._dist_miss_total += 1
        self._dist_miss_by_kind[str(kind)] += 1

        if self.distance_miss_log_limit > 0 and len(self._dist_miss_samples) < self.distance_miss_log_limit:
            self._dist_miss_samples.append({
                "kind": str(kind),
                "key": (int(key[0]), int(key[1])),
                "ctx": str(ctx),
            })

        if self.distance_miss_policy == "raise":
            raise KeyError(
                f"[DIST_MISS] kind={kind} key={key} ctx={ctx} "
                f"(policy=raise). Usually id-mapping mismatch or incomplete distance matrix."
            )

        if self.distance_miss_policy == "penalty":
            return float(self.distance_miss_penalty), float(delta[key])

        # "zero" fallback
        return 0.0, float(delta[key])

    # ------------------------ 鲁棒传播 ------------------------

    def _t_advance_unc(self, tin: List[float], nom: float, delta: float) -> List[float]:
        """
        鲁棒 budget 传播（Γ层 DP）：
            out[0] = tin[0] + nom
            out[k] = max(tin[k] + nom, tin[k-1] + nom + delta)  (k=1..Γ)
        """
        nom = float(nom)
        delta = float(delta)
        G = self.gamma
        out = [0.0] * (G + 1)
        out[0] = tin[0] + nom
        for k in range(1, G + 1):
            out[k] = max(tin[k] + nom, tin[k - 1] + nom + delta)
        return out

    # ------------------------ home_before 计划 ------------------------

    def _build_home_before_plan(
        self,
        shelf_seq: Dict[int, List[int]],
        place: Dict[int, int],
        any_s: int,
    ) -> Dict[int, int]:
        """
        对每条 shelf chain:
          first task 的 home_before = shelf_init[cell]
          后续 task 的 home_before = 上一个任务的 end cell（即 place[prev]）
        """
        home: Dict[int, int] = {}
        for c, seq in shelf_seq.items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if not seq_clean:
                continue
            s_init = int(self.shelf_init.get(cc, any_s))
            home[seq_clean[0]] = s_init
            for prev_j, cur_j in zip(seq_clean[:-1], seq_clean[1:]):
                s_prev = place.get(int(prev_j), None)
                home[cur_j] = int(s_prev) if (s_prev is not None and int(s_prev) in self.S) else s_init
        return home

    # ------------------------ 一致性检查 ------------------------

    def _check_tail_conflict(
        self,
        shelf_seq: Dict[int, List[int]],
        end_shelf_final: Dict[int, int],
        verbose: bool = False,
    ) -> bool:
        tail_tasks: Dict[int, int] = {}
        for c, seq in shelf_seq.items():
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if seq_clean:
                tail_tasks[int(c)] = int(seq_clean[-1])

        used: Dict[int, int] = {}
        for c, tail_j in tail_tasks.items():
            s_tail = end_shelf_final.get(int(tail_j))
            if s_tail is None:
                continue
            s_tail = int(s_tail)
            if s_tail in used:
                if verbose:
                    print(f"[TailConflict] shelves {used[s_tail]} and {c} both tail at cell {s_tail}")
                return True
            used[s_tail] = int(c)
        return False

    def _check_chain_time_order(
        self,
        shelf_seq: Dict[int, List[int]],
        p_map: Dict[int, float],
        q_map: Dict[int, float],
        verbose: bool = False,
    ) -> bool:
        for c, seq in shelf_seq.items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in p_map and int(j) in q_map]
            if len(seq_clean) <= 1:
                continue
            for prev_j, cur_j in zip(seq_clean[:-1], seq_clean[1:]):
                if float(q_map[prev_j]) > float(p_map[cur_j]) + 1e-9:
                    if verbose:
                        print(
                            f"[ChainOrder] violation shelf {cc}: "
                            f"{prev_j} q={q_map[prev_j]:.2f} > {cur_j} p={p_map[cur_j]:.2f}"
                        )
                    return True
        return False

    def _check_ws_time_order(
        self,
        p_map: Dict[int, float],
        verbose: bool = False,
    ) -> bool:
        for ws, seq_ws in self.ws_fixed_seq.items():
            seq_clean = [int(j) for j in (seq_ws or []) if int(j) in p_map]
            if len(seq_clean) <= 1:
                continue
            seq_by_time = sorted(seq_clean, key=lambda j: float(p_map[j]))
            if seq_by_time != seq_clean:
                if verbose:
                    print(f"[WSOrder] violation WS {ws}: fixed={seq_clean}, by_time={seq_by_time}")
                return True
        return False

    # ------------------------ cell occupancy: conflict detection / measure ------------------------

    def _extract_pick_and_arrive_maps(
        self, details: Dict[str, Any]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        提取：
          - pick_start[j]（γ层标量）
          - arrive_cell_act[j]（γ层标量）

        优先使用 details["pick_start"] / ["arrive_cell_act"]（dict），
        若没有则从 timeline 解析（兼容旧数据结构）。
        """
        pick_start: Dict[int, float] = {}
        arrive_act: Dict[int, float] = {}

        pmap = details.get("pick_start")
        amap = details.get("arrive_cell_act")
        if isinstance(pmap, dict) and isinstance(amap, dict):
            for j, t in pmap.items():
                try:
                    pick_start[int(j)] = float(t)
                except Exception:
                    pass
            for j, t in amap.items():
                try:
                    arrive_act[int(j)] = float(t)
                except Exception:
                    pass
            return pick_start, arrive_act

        timeline = details.get("timeline", []) or []
        for rec in timeline:
            try:
                j = int(rec.get("Task"))
            except Exception:
                continue
            if j not in self.J:
                continue
            if "pick_start" in rec:
                pick_start[j] = float(rec["pick_start"])
            if "arrive_cell_act" in rec:
                arrive_act[j] = float(rec["arrive_cell_act"])
        return pick_start, arrive_act
    def _first_chain_pick_violation(
        self,
        shelf_seq: Dict[int, List[int]],
        details: Dict[str, Any],
        eps: float = 1e-9,
    ) -> Optional[Tuple[int, int, int, float, float]]:
        """
        找到第一个违反“同一货架链先后”的相邻任务对 (prev -> cur)：
            pick_start[cur] < arrive_cell_act[prev]

        返回:
            (chain_id, prev_task, cur_task, prev_drop_time, cur_pick_time)
        """
        pick_start, arrive_act = self._extract_pick_and_arrive_maps(details)

        for c, seq in (shelf_seq or {}).items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if len(seq_clean) <= 1:
                continue

            for prev_j, cur_j in zip(seq_clean[:-1], seq_clean[1:]):
                t_prev = arrive_act.get(int(prev_j), None)
                t_cur = pick_start.get(int(cur_j), None)
                if t_prev is None or t_cur is None:
                    continue

                t_prev = float(t_prev)
                t_cur = float(t_cur)
                if (math.isfinite(t_prev) and math.isfinite(t_cur)) and (t_cur < t_prev - eps):
                    return (cc, int(prev_j), int(cur_j), float(t_prev), float(t_cur))

        return None

    def _initial_shelf_intervals(
            self,
            shelf_seq: Dict[int, List[int]],
            pick_start: Dict[int, float],
    ) -> List[Tuple[int, float, float, int, int]]:
        """
        生成“货架初始占用”区间（对应 MILP 的 300x / J_I）：
          对每条货架链 c：
            cell = shelf_init[c]
            interval = [0, pick_start(first_task_on_chain_c)]
          若该链没有任务：interval = [0, BIG_M_time]

        返回列表元素: (cell, start, end, pseudo_task_id, chain_id)
        pseudo_task_id 用负数，避免和真实任务冲突。
        """
        intervals: List[Tuple[int, float, float, int, int]] = []
        for c, init_cell in (self.shelf_init or {}).items():
            cc = int(c)
            cell = int(init_cell)
            if cell not in self.S:
                continue

            seq = (shelf_seq or {}).get(cc, []) or []
            seq_clean = [int(j) for j in seq if int(j) in self.J]

            if seq_clean:
                j_first = int(seq_clean[0])
                leave = float(pick_start.get(j_first, self.BIG_M_time))
                if (not math.isfinite(leave)) or leave <= 0.0:
                    leave = float(self.BIG_M_time)
            else:
                leave = float(self.BIG_M_time)

            pseudo = -3000 - cc
            intervals.append((cell, 0.0, leave, pseudo, cc))

        return intervals

    def _cell_conflict_stats(
            self,
            shelf_seq: Dict[int, List[int]],
            details: Dict[str, Any],
            verbose: bool = False,
    ) -> Dict[str, float]:
        """
        计算所有 cell 占用冲突统计（用于 penalty fitness）：
          - conflict_pairs
          - overlap_time

        ★关键修复：加入“初始货架占用区间”（MILP 的 300x / J_I 语义）
        """
        end_final: Dict[int, int] = details.get("end_shelf_final", {}) or {}
        pick_start, arrive_act = self._extract_pick_and_arrive_maps(details)

        # 任务属于哪条链
        chain_of: Dict[int, int] = {}
        succ: Dict[int, int] = {}
        tail_set: Set[int] = set()
        for c, seq in (shelf_seq or {}).items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if not seq_clean:
                continue
            for a, b in zip(seq_clean[:-1], seq_clean[1:]):
                succ[int(a)] = int(b)
                chain_of[int(a)] = cc
                chain_of[int(b)] = cc
            tail_set.add(int(seq_clean[-1]))
            chain_of[int(seq_clean[-1])] = cc

        # by_cell[cell] = list of (start, end, task_id, chain_id)
        by_cell: Dict[int, List[Tuple[float, float, int, int]]] = defaultdict(list)

        # (A) 真实任务落位占用区间
        for j, s_end in end_final.items():
            jj = int(j)
            if jj not in self.J:
                continue
            s_end = int(s_end)
            if s_end not in self.S:
                continue

            start = float(arrive_act.get(jj, 0.0))
            if (jj not in arrive_act) or (not math.isfinite(start)):
                start = 0.0

            if jj in tail_set:
                end = float(self.BIG_M_time)
            else:
                j2 = succ.get(jj, None)
                ps = pick_start.get(int(j2), None) if j2 is not None else None
                if ps is None or (not math.isfinite(float(ps))):
                    end = float(self.BIG_M_time)
                else:
                    end = float(ps)

            by_cell[s_end].append((start, end, jj, int(chain_of.get(jj, -1))))

        # (B) 初始货架占用区间（虚拟）
        # (B) 初始货架占用区间（虚拟）
        if USE_INITIAL_OCCUPANCY_INTERVALS:
            for cell, a, b, pseudo, cc in self._initial_shelf_intervals(shelf_seq, pick_start):
                by_cell[int(cell)].append((float(a), float(b), int(pseudo), int(cc)))

        gap = float(self.cell_gap)
        conflict_pairs = 0.0
        overlap_time = 0.0

        # 扫描（小规模直接 O(n^2) 更稳）
        for s, intervals in by_cell.items():
            if len(intervals) <= 1:
                continue
            intervals.sort(key=lambda x: (x[0], x[1], x[2]))
            n = len(intervals)
            for i in range(n):
                si, ei, ji, ci = intervals[i]
                thr = ei + gap
                for j in range(i + 1, n):
                    sj, ej, jj, cj = intervals[j]
                    if sj >= thr - 1e-9:
                        break
                    # 同一条链的占用不算冲突（同一个货架）
                    if ci == cj and ci != -1:
                        continue
                    conflict_pairs += 1.0
                    overlap_time += float(thr - sj)
                    if verbose:
                        print(
                            f"[CellConflict] cell {s}: "
                            f"Task {ji}(chain {ci}) [{si:.2f},{ei:.2f}] overlaps "
                            f"Task {jj}(chain {cj}) [{sj:.2f},{ej:.2f}] gap={gap:.2f}"
                        )

        return {"conflict_pairs": conflict_pairs, "overlap_time": overlap_time}

    def _first_cell_occupancy_conflict(
            self,
            shelf_seq: Dict[int, List[int]],
            details: Dict[str, Any],
            verbose: bool = False,
    ) -> Optional[Tuple[int, int, Tuple[float, float, int], int, Tuple[float, float, int]]]:
        """
        返回第一个检测到的 cell 占用冲突（若无冲突返回 None）

        ★关键修复：加入“初始货架占用区间”（MILP 的 300x / J_I 语义）
        """
        end_final: Dict[int, int] = details.get("end_shelf_final", {}) or {}
        pick_start, arrive_act = self._extract_pick_and_arrive_maps(details)

        chain_of: Dict[int, int] = {}
        succ: Dict[int, int] = {}
        tail_set: Set[int] = set()
        for c, seq in (shelf_seq or {}).items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if not seq_clean:
                continue
            for a, b in zip(seq_clean[:-1], seq_clean[1:]):
                succ[int(a)] = int(b)
                chain_of[int(a)] = cc
                chain_of[int(b)] = cc
            tail_set.add(int(seq_clean[-1]))
            chain_of[int(seq_clean[-1])] = cc

        by_cell: Dict[int, List[Tuple[float, float, int, int]]] = defaultdict(list)

        # (A) 真实任务区间
        for j, s_end in end_final.items():
            jj = int(j)
            if jj not in self.J:
                continue
            s_end = int(s_end)
            if s_end not in self.S:
                continue

            start = float(arrive_act.get(jj, 0.0))
            if (jj not in arrive_act) or (not math.isfinite(start)):
                start = 0.0

            if jj in tail_set:
                end = float(self.BIG_M_time)
            else:
                j2 = succ.get(jj, None)
                ps = pick_start.get(int(j2), None) if j2 is not None else None
                if ps is None or (not math.isfinite(float(ps))):
                    end = float(self.BIG_M_time)
                else:
                    end = float(ps)

            by_cell[s_end].append((start, end, jj, int(chain_of.get(jj, -1))))

        # (B) 初始占用虚拟区间
        # (B) 初始占用虚拟区间
        if USE_INITIAL_OCCUPANCY_INTERVALS:
            for cell, a, b, pseudo, cc in self._initial_shelf_intervals(shelf_seq, pick_start):
                by_cell[int(cell)].append((float(a), float(b), int(pseudo), int(cc)))

        gap = float(self.cell_gap)

        for s in sorted(by_cell.keys()):
            intervals = by_cell[s]
            if len(intervals) <= 1:
                continue
            intervals.sort(key=lambda x: (x[0], x[1], x[2]))

            prev_s, prev_e, prev_j, prev_c = intervals[0]
            for cur_s, cur_e, cur_j, cur_c in intervals[1:]:
                # 同一条链（同一货架）不算冲突
                if prev_c == cur_c and prev_c != -1:
                    if cur_e > prev_e + 1e-9:
                        prev_s, prev_e, prev_j, prev_c = cur_s, cur_e, cur_j, cur_c
                    continue

                if cur_s < prev_e + gap - 1e-9:
                    if verbose:
                        print(
                            f"[CellConflict] cell {s}: "
                            f"Task {prev_j}(chain {prev_c}) [{prev_s:.2f},{prev_e:.2f}] overlaps "
                            f"Task {cur_j}(chain {cur_c}) [{cur_s:.2f},{cur_e:.2f}] gap={gap:.2f}"
                        )
                    return (
                        int(s),
                        int(prev_j),
                        (float(prev_s), float(prev_e), int(prev_c)),
                        int(cur_j),
                        (float(cur_s), float(cur_e), int(cur_c)),
                    )

                if cur_e > prev_e + 1e-9:
                    prev_s, prev_e, prev_j, prev_c = cur_s, cur_e, cur_j, cur_c

        return None

    def _check_cell_occupancy_conflicts(
        self,
        shelf_seq: Dict[int, List[int]],
        details: Dict[str, Any],
        verbose: bool = False,
    ) -> bool:
        """兼容旧接口：返回 bool。"""
        return self._first_cell_occupancy_conflict(shelf_seq, details, verbose=verbose) is not None

    # ------------------------ 对外接口 ------------------------

    def evaluate(
            self,
            routes: Dict[int, List[int]],
            shelf_seq: Dict[int, List[int]],
            place: Dict[int, int],
            verbose: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        返回：(fitness, details)

        关键升级（NEW）：
          - chain repair：当发现货架链反序（pick_start[cur] < arrive_cell_act[prev]）
            不再直接判 inf，而是对后继任务加 pick_lb 下界并重仿真，直到链顺序成立或达到上限。
          - cell repair：保持你已有逻辑（place_lb 延迟落位）；
            所有重仿真都同时携带 pick_lb + place_lb，避免互相打穿。

        - hard 模式：任何硬不可行（尤其是 cell 冲突）直接返回 inf
        - penalty 模式：cell 冲突不返回 inf，而是返回 makespan + penalty（用于 ALNS 内层）
        """
        # reset distance-miss counters for this evaluation (including repair re-simulations)
        self._dist_miss_total = 0
        self._dist_miss_by_kind = defaultdict(int)
        self._dist_miss_samples = []

        routed_tasks = {int(j) for r in routes for j in routes.get(r, []) if int(j) in self.J}
        missing_cover = len(self.J - routed_tasks)

        # ===== NEW: repair state =====
        place_lb: Dict[int, float] = {}
        pick_lb: Dict[int, float] = {}

        cell_repair_log: Optional[List[Dict[str, Any]]] = ([] if self.record_cell_repair_log else None)
        chain_repair_log: Optional[List[Dict[str, Any]]] = ([] if self.record_cell_repair_log else None)

        stalled_cell = False
        stalled_chain = False
        chain_added_lb_total = 0.0

        # 1) 先仿真一次（不带修复）
        details = self._simulate_once(routes, shelf_seq, place, place_lb=None, pick_lb=None)

        # baseline pick_start for "waiting cost" measurement
        base_pick_start: Dict[int, float] = {}
        if isinstance(details.get("pick_start"), dict):
            for k, v in (details.get("pick_start", {}) or {}).items():
                try:
                    base_pick_start[int(k)] = float(v)
                except Exception:
                    pass

        # ===== NEW: unified repair loop (chain first, then cell) =====
        max_chain = int(self.chain_repair_max_iters) if getattr(self, "enable_chain_repair", False) else 0
        max_cell = int(self.cell_repair_max_iters) if getattr(self, "enable_cell_repair", False) else 0

        chain_iters = 0
        cell_iters = 0

        # total guard: at most max_chain + max_cell successful applications
        for _guard in range(max(0, max_chain) + max(0, max_cell)):
            # ----- (A) chain repair: enforce shelf_seq by waiting at pick stage -----
            if (not stalled_chain) and (max_chain > 0) and (chain_iters < max_chain):
                viol = self._first_chain_pick_violation(shelf_seq, details, eps=1e-9)
                if viol is not None:
                    cc, prev_j, cur_j, prev_drop, cur_pick = viol
                    need = float(prev_drop)

                    old = float(pick_lb.get(int(cur_j), float("-inf")))
                    if need <= old + 1e-9:
                        stalled_chain = True
                        if chain_repair_log is not None:
                            chain_repair_log.append({
                                "iter": int(chain_iters),
                                "chain": int(cc),
                                "prev": int(prev_j),
                                "cur": int(cur_j),
                                "prev_drop": float(prev_drop),
                                "cur_pick": float(cur_pick),
                                "need_pick_lb": float(need),
                                "old_pick_lb": float(old),
                                "status": "stalled",
                            })
                        break

                    pick_lb[int(cur_j)] = need
                    chain_added_lb_total += max(0.0, need - (old if math.isfinite(old) else 0.0))

                    if chain_repair_log is not None:
                        chain_repair_log.append({
                            "iter": int(chain_iters),
                            "chain": int(cc),
                            "prev": int(prev_j),
                            "cur": int(cur_j),
                            "prev_drop": float(prev_drop),
                            "cur_pick": float(cur_pick),
                            "need_pick_lb": float(need),
                            "old_pick_lb": float(old),
                            "status": "applied",
                        })

                    if verbose or getattr(self, "chain_repair_verbose", False):
                        print(f"[ChainFix] iter={chain_iters} chain={cc} delay Task {cur_j}: pick_lb {old:.2f} -> {need:.2f}")

                    chain_iters += 1
                    details = self._simulate_once(routes, shelf_seq, place, place_lb=place_lb, pick_lb=pick_lb)
                    continue  # after changing times, restart from chain check

            # ----- (B) cell repair: your existing logic, but re-simulate with pick_lb too -----
            if (not stalled_cell) and (max_cell > 0) and (cell_iters < max_cell):
                conflict = self._first_cell_occupancy_conflict(
                    shelf_seq, details, verbose=(verbose or self.cell_repair_verbose)
                )
                if conflict is not None:
                    it = int(cell_iters)
                    cell_iters += 1

                    s, jA, (a_s, a_e, a_c), jB, (b_s, b_e, b_c) = conflict

                    jA_i = int(jA)
                    jB_i = int(jB)

                    # dummy vs dummy: nothing meaningful to repair
                    if jA_i < 0 and jB_i < 0:
                        stalled_cell = True
                        if cell_repair_log is not None:
                            cell_repair_log.append({
                                "iter": it,
                                "cell": int(s),
                                "conflict": {
                                    "A": {"task": int(jA), "chain": int(a_c), "interval": [float(a_s), float(a_e)]},
                                    "B": {"task": int(jB), "chain": int(b_c), "interval": [float(b_s), float(b_e)]},
                                },
                                "action": {"delay_task": None, "reason": "dummy_vs_dummy_no_repair"},
                                "status": "stalled",
                            })
                        break

                    # dummy vs real: delay the REAL task to be after dummy leaves (+ gap)
                    if jA_i < 0 or jB_i < 0:
                        if jA_i < 0:
                            dummy_end = float(a_e)
                            real_j = jB_i
                            dummy_task = jA_i
                        else:
                            dummy_end = float(b_e)
                            real_j = jA_i
                            dummy_task = jB_i

                        need = float(dummy_end + self.cell_gap)
                        old = float(place_lb.get(real_j, float("-inf")))

                        if need <= old + 1e-9:
                            stalled_cell = True
                            if cell_repair_log is not None:
                                cell_repair_log.append({
                                    "iter": it,
                                    "cell": int(s),
                                    "conflict": {
                                        "A": {"task": int(jA), "chain": int(a_c), "interval": [float(a_s), float(a_e)]},
                                        "B": {"task": int(jB), "chain": int(b_c), "interval": [float(b_s), float(b_e)]},
                                    },
                                    "action": {"delay_task": int(real_j), "need_lb": need, "old_lb": old,
                                               "reason": f"dummy_conflict_with_{dummy_task}"},
                                    "status": "stalled",
                                })
                            break

                        place_lb[real_j] = need
                        if cell_repair_log is not None:
                            cell_repair_log.append({
                                "iter": it,
                                "cell": int(s),
                                "conflict": {
                                    "A": {"task": int(jA), "chain": int(a_c), "interval": [float(a_s), float(a_e)]},
                                    "B": {"task": int(jB), "chain": int(b_c), "interval": [float(b_s), float(b_e)]},
                                },
                                "action": {"delay_task": int(real_j), "need_lb": need, "old_lb": old,
                                           "reason": f"dummy_conflict_with_{dummy_task}"},
                                "status": "applied",
                            })

                        if verbose or self.cell_repair_verbose:
                            print(f"[CellFix-DUMMY] iter={it} cell={s} delay Task {real_j}: lb {old:.2f} -> {need:.2f}")

                        details = self._simulate_once(routes, shelf_seq, place, place_lb=place_lb, pick_lb=pick_lb)
                        continue

                    # general real/dummy selection logic (your original)
                    isA_real = (int(jA) in self.J)
                    isB_real = (int(jB) in self.J)

                    if isA_real and (not isB_real):
                        delay_j = int(jA)
                        other_end = float(b_e)
                        other_j = int(jB)

                    elif isB_real and (not isA_real):
                        delay_j = int(jB)
                        other_end = float(a_e)
                        other_j = int(jA)

                    elif (not isA_real) and (not isB_real):
                        stalled_cell = True
                        if cell_repair_log is not None:
                            cell_repair_log.append({
                                "iter": it,
                                "cell": int(s),
                                "conflict": {
                                    "A": {"task": int(jA), "chain": int(a_c), "interval": [float(a_s), float(a_e)]},
                                    "B": {"task": int(jB), "chain": int(b_c), "interval": [float(b_s), float(b_e)]},
                                },
                                "action": {"delay_task": None, "reason": "both_dummy_initial_occupancy"},
                                "status": "stalled",
                            })
                        break

                    else:
                        if a_e > b_e + 1e-9:
                            delay_j = int(jA)
                            other_end = float(b_e)
                            other_j = int(jB)
                        elif b_e > a_e + 1e-9:
                            delay_j = int(jB)
                            other_end = float(a_e)
                            other_j = int(jA)
                        else:
                            if a_s >= b_s - 1e-9:
                                delay_j = int(jA)
                                other_end = float(b_e)
                                other_j = int(jB)
                            else:
                                delay_j = int(jB)
                                other_end = float(a_e)
                                other_j = int(jA)

                    need = float(other_end + self.cell_gap)
                    old = float(place_lb.get(delay_j, float("-inf")))
                    if need <= old + 1e-9:
                        stalled_cell = True
                        if cell_repair_log is not None:
                            cell_repair_log.append({
                                "iter": it,
                                "cell": int(s),
                                "conflict": {
                                    "A": {"task": int(jA), "chain": int(a_c), "interval": [a_s, a_e]},
                                    "B": {"task": int(jB), "chain": int(b_c), "interval": [b_s, b_e]},
                                },
                                "action": {"delay_task": int(delay_j), "other_task": int(other_j),
                                           "need_lb": need, "old_lb": old},
                                "status": "stalled",
                            })
                        break

                    place_lb[delay_j] = need
                    if cell_repair_log is not None:
                        cell_repair_log.append({
                            "iter": it,
                            "cell": int(s),
                            "conflict": {
                                "A": {"task": int(jA), "chain": int(a_c), "interval": [a_s, a_e]},
                                "B": {"task": int(jB), "chain": int(b_c), "interval": [b_s, b_e]},
                            },
                            "action": {"delay_task": int(delay_j), "other_task": int(other_j),
                                       "need_lb": need, "old_lb": old},
                            "status": "applied",
                        })

                    if verbose or self.cell_repair_verbose:
                        print(f"[CellFix] iter={it} cell={s} delay Task {delay_j}: lb {old:.2f} -> {need:.2f}")

                    details = self._simulate_once(routes, shelf_seq, place, place_lb=place_lb, pick_lb=pick_lb)
                    continue

            # neither chain nor cell can progress
            break

        # attach repair diagnostics
        details["chain_repair_enabled"] = bool(getattr(self, "enable_chain_repair", False) and max_chain > 0)
        details["chain_repair_iters"] = int(chain_iters) if chain_repair_log is None else len(chain_repair_log)
        details["chain_repair_pick_lb"] = dict(pick_lb)
        details["chain_repair_log"] = ([] if chain_repair_log is None else chain_repair_log)
        details["chain_repair_stalled"] = bool(stalled_chain)
        details["chain_repair_added_lb_total"] = float(chain_added_lb_total)

        # waiting cost estimate (relative to first simulation) for tasks that were constrained by pick_lb
        final_pick: Dict[int, float] = {}
        if isinstance(details.get("pick_start"), dict):
            for k, v in (details.get("pick_start", {}) or {}).items():
                try:
                    final_pick[int(k)] = float(v)
                except Exception:
                    pass

        wait_list: List[Tuple[int, float]] = []
        wait_total = 0.0
        for j in pick_lb.keys():
            if int(j) in final_pick and int(j) in base_pick_start:
                inc = max(0.0, float(final_pick[int(j)]) - float(base_pick_start[int(j)]))
                if inc > 1e-9:
                    wait_total += inc
                    wait_list.append((int(j), float(inc)))
        wait_list.sort(key=lambda x: -x[1])
        details["chain_wait_added_total"] = float(wait_total)
        details["chain_wait_added_top"] = wait_list[:20]

        details["cell_repair_enabled"] = bool(getattr(self, "enable_cell_repair", False) and max_cell > 0)
        details["cell_repair_iters"] = int(cell_iters) if cell_repair_log is None else len(cell_repair_log)
        details["cell_repair_place_lb"] = dict(place_lb)
        details["cell_repair_log"] = ([] if cell_repair_log is None else cell_repair_log)
        details["cell_repair_stalled"] = bool(stalled_cell)

        # 3) 常规一致性检查（基于最终 details）
        p_map = details.get("p", {}) or {}
        q_map = details.get("q", {}) or {}
        end_final = details.get("end_shelf_final", {}) or {}

        # chain / ws 顺序违反：一直 hard
        viol2 = self._first_chain_pick_violation(shelf_seq, details, eps=1e-9)
        if viol2 is not None:
            cc, prev_j, cur_j, prev_drop, cur_pick = viol2
            details["C_task_max"] = float("inf")
            details["feasible"] = False
            details.setdefault("penalties", {})
            details["penalties"]["chain_order_violation"] = 1.0
            details["chain_order_violation_detail"] = {
                "chain": int(cc),
                "prev": int(prev_j),
                "cur": int(cur_j),
                "prev_drop": float(prev_drop),
                "cur_pick": float(cur_pick),
            }
            self._attach_dist_miss(details)
            self._last_details = details
            return float("inf"), details

        if self._check_ws_time_order(p_map, verbose=verbose):
            details["C_task_max"] = float("inf")
            details["feasible"] = False
            details.setdefault("penalties", {})
            details["penalties"]["ws_order_violation"] = 1.0
            self._attach_dist_miss(details)
            self._last_details = details
            return float("inf"), details

        # 统计未调度任务
        scheduled = set(q_map.keys())
        unscheduled = sorted(list(self.J - scheduled))

        # tail 冲突
        tail_conf = self._check_tail_conflict(shelf_seq, end_final, verbose=verbose)

        # cell 冲突检测/统计
        if self.cell_conflict_mode == "penalty" or verbose:
            cell_stats = self._cell_conflict_stats(shelf_seq, details, verbose=verbose)
            conflict_pairs = float(cell_stats.get("conflict_pairs", 0.0))
            overlap_time = float(cell_stats.get("overlap_time", 0.0))
            has_cell_conf = (conflict_pairs > 0.0) or tail_conf
        else:
            first_conf = self._first_cell_occupancy_conflict(shelf_seq, details, verbose=False)
            conflict_pairs = 1.0 if (first_conf is not None) else 0.0
            overlap_time = 0.0
            has_cell_conf = (first_conf is not None) or tail_conf

        # ---------- penalties bookkeeping ----------
        details.setdefault("penalties", {})
        details["penalties"]["missing_tasks_in_routes"] = float(missing_cover)
        details["unscheduled_tasks"] = unscheduled
        details["penalties"]["cell_conflict_pairs"] = float(conflict_pairs)
        details["penalties"]["cell_conflict_overlap_time"] = float(overlap_time)
        details["penalties"]["tail_cell_conflict"] = (1.0 if tail_conf else 0.0)

        # base makespan
        C_task_max = float(details.get("C_task_max", float("inf")))
        base = C_task_max if math.isfinite(C_task_max) else 0.0

        # hard：不允许 incomplete
        if (not self.allow_incomplete) and unscheduled:
            details["C_task_max"] = float("inf")
            details["feasible"] = False
            self._attach_dist_miss(details)
            self._last_details = details
            return float("inf"), details

        # hard：cell 冲突直接 infeasible
        if self.cell_conflict_mode == "hard" and has_cell_conf:
            details["C_task_max"] = float("inf")
            details["feasible"] = False
            self._attach_dist_miss(details)
            self._last_details = details
            return float("inf"), details

        # penalty：计算罚
        penalty = 0.0

        if self.allow_incomplete and unscheduled:
            penalty += float(self.missing_task_weight) * float(len(unscheduled))

        if self.cell_conflict_mode == "penalty" and has_cell_conf:
            gap = max(float(self.cell_gap), 1e-6)
            measure = float(conflict_pairs) + float(overlap_time) / gap
            if tail_conf:
                measure += 10.0
            penalty += float(self.cell_conflict_weight) * measure

        # feasible 标记
        details["feasible"] = (missing_cover == 0 and (len(unscheduled) == 0) and (not has_cell_conf))
        details["robust_gamma"] = self.gamma
        details["envelope_shared_resources"] = self.envelope_shared_resources

        fitness = base + penalty
        details["fitness"] = float(fitness)
        details["fitness_base"] = float(base)
        details["fitness_penalty"] = float(penalty)

        self._attach_dist_miss(details)
        self._last_details = details
        return float(fitness), details


    # ------------------------ 仿真（核心） ------------------------

    def _simulate_once(
            self,
            routes: Dict[int, List[int]],
            shelf_seq: Dict[int, List[int]],
            place: Dict[int, int],
            place_lb: Optional[Dict[int, float]] = None,
            pick_lb: Optional[Dict[int, float]] = None,  # NEW
    ) -> Dict[str, Any]:
        """
        place_lb: 可选的“落位时间下界”，用于 cell repair。
          - 若提供：对每个任务 j，最终 arrive_cell_act_vec >= lift(place_lb[j])
        """
        place_lb = {} if place_lb is None else {int(k): float(v) for k, v in place_lb.items()}
        pick_lb = {} if pick_lb is None else {int(k): float(v) for k, v in pick_lb.items()}

        # ===== Microscope init (inside _simulate_once) =====
        _micro_cfg = MICRO_DEBUG
        micro_cell = int(_micro_cfg.get("focus_cell", -1))
        micro_tasks = set(int(x) for x in (_micro_cfg.get("focus_tasks", []) or []))
        micro_max = int(_micro_cfg.get("max_records", 300))
        # 关键：只在“full timeline + collect_v_arcs”时启用，避免 ALNS 内层几十万次 evaluate 刷爆
        micro_enabled = bool(_micro_cfg.get("enabled", False)) and (self.timeline_mode == "full") and bool(self.collect_v_arcs)
        micro_print = bool(_micro_cfg.get("print", False)) and micro_enabled

        micro_log: List[Dict[str, Any]] = []

        def _micro(event: str, **kw: Any) -> None:
            if not micro_enabled:
                return
            if len(micro_log) >= micro_max:
                return
            rec = {"event": str(event)}
            rec.update(kw)
            micro_log.append(rec)
            if micro_print:
                # 做成一行，方便你在控制台 grep/查找
                items = ", ".join(f"{k}={rec[k]}" for k in rec.keys() if k != "event")
                print(f"[MICRO] {event}: {items}")
        # ===== /Microscope init =====

        G = self.gamma

        shelf_cur: Dict[int, int] = {int(c): int(s0) for c, s0 in self.shelf_init.items()}
        shelf_ready: Dict[int, List[float]] = {int(c): self._t0() for c in self.shelf_init}

        cell_busy_until: Dict[int, List[float]] = {int(s): self._t0() for s in self.S.keys()}

        any_s = next(iter(self.S.keys())) if self.S else 0

        agv_cell: Dict[int, int] = {int(r): int(self.agv_init.get(r, any_s)) for r in self.R_ids}
        agv_clock: Dict[int, List[float]] = {int(r): self._t0() for r in self.R_ids}
        agv_from_ws: Dict[int, Optional[int]] = {int(r): None for r in self.R_ids}

        all_ws = set(self.ws_fixed_seq.keys()) | set(self.pi[j] for j in self.J)
        ws_free: Dict[int, List[float]] = {int(ws): self._t0() for ws in all_ws}
        ws_cursor: Dict[int, int] = {int(ws): 0 for ws in all_ws}
        ws_park: Dict[int, Dict[int, dict]] = {int(ws): {} for ws in all_ws}

        task_chain: Dict[int, int] = {}
        for c, seq in shelf_seq.items():
            for j in (seq or []):
                task_chain[int(j)] = int(c)
        # ===== chain predecessor gating (avoid "future blocks past") =====
        pred_on_chain: Dict[int, Optional[int]] = {}
        for c, seq in (shelf_seq or {}).items():
            seq_clean = [int(x) for x in (seq or []) if int(x) in self.J]
            if not seq_clean:
                continue
            prev = None
            for jj in seq_clean:
                pred_on_chain[int(jj)] = prev
                prev = int(jj)

        scheduled_on_chain: Set[int] = set()

        def _can_prewrite_home_busy(jj: int) -> bool:
            pred = pred_on_chain.get(int(jj), None)
            return (pred is None) or (pred in scheduled_on_chain)

        def _expected_open_task_id(chain: int, task: int) -> int:
            """
            对 task 的 pick 动作来说，它关闭的是“同一链上前驱落位后形成的 open interval”。
            因此 planned_end 必须写到那个 open interval 上，而不是写到当前 chain_open 指向的任意 interval。

            - 如果 task 是链上的首任务：它关闭 init_dummy interval（task_id = -3000 - chain）
            - 否则：它关闭 pred_on_chain[task] 那个任务落位后打开的 interval（task_id = pred）
            """
            pred = pred_on_chain.get(int(task), None)
            return (-3000 - int(chain)) if (pred is None) else int(pred)

        # ===== /chain predecessor gating =====

        # ======== ADD: blocked_cells for unused shelves ========
        # unused_shelves 的定义：这条 shelf chain 在 shelf_seq 里没有任何任务
        unused_shelves: Set[int] = set()
        for c in self.shelf_init.keys():
            seq = (shelf_seq or {}).get(int(c), []) or []
            seq_clean = [int(j) for j in seq if int(j) in self.J]
            if not seq_clean:
                unused_shelves.add(int(c))

        # 这些 unused shelves 的初始位置 cell 视为“永远被占用”，不能作为 end cell 候选
        blocked_cells: Set[int] = set(
            int(self.shelf_init[c]) for c in unused_shelves
            if int(self.shelf_init.get(c, -1)) in self.S
        )
        # ======== ADD END ========
        # =========================
        # MILP-aligned cell interval ledger (true [g,h] occupancy)
        # =========================
        # cell_occ[cell] holds intervals introduced by shelves (chains):
        #   - start: g_start (drop arrival time vec)
        #   - end:   h_end   (next pick time vec). If unknown, use planned_end; if still unknown, treat as BIG_M.
        #
        # We keep exactly one "open" interval per chain (a shelf can only sit on one cell at a time).
        # This kills the order-dependence bug: future intervals won't block earlier drops incorrectly.
        cell_occ: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        # chain_open[c] = (cell, idx_in_cell_list) for the currently open interval of chain c
        chain_open: Dict[int, Optional[Tuple[int, int]]] = {int(c): None for c in (self.shelf_init or {}).keys()}


        # pending planned_end keyed by "open interval task id"
        # pending_plan_end[chain][open_task_id] = planned_end_vec
        # 这样可以保证：未来任务只会把 planned_end 写到它真正要关闭的那段占用区间上，不会污染别的区间（尤其是 init_dummy）
        pending_plan_end: Dict[int, Dict[int, List[float]]] = defaultdict(dict)
        # =========================
        # NEW: cell-gate pending drops (to kill BIG_M from unknown release)
        # =========================
        cell_gate_mode = str(getattr(self, "cell_gate_mode", "legacy")).lower().strip()
        if cell_gate_mode not in ("legacy", "event"):
            cell_gate_mode = "legacy"

        # pending_drops[cell][task] = drop_record (wait at cell gate)
        pending_drops: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        event_gate_debug: Dict[str, Any] = {}

        # avoid recursive wake loops
        _wake_guard: Set[int] = set()

        # runtime sigma:
        #   1) explicit self.cell_sigma
        #   2) auto-derived from current (shelf_seq + place), with tail tasks ordered after non-tail
        runtime_cell_sigma: Optional[Dict[int, List[int]]] = None
        if isinstance(getattr(self, "cell_sigma", None), dict):
            runtime_cell_sigma = {
                int(s): [int(x) for x in (seq or [])]
                for s, seq in (getattr(self, "cell_sigma") or {}).items()
            }
        elif (cell_gate_mode == "event") and bool(getattr(self, "auto_cell_sigma", True)):
            ws_rank: Dict[int, int] = {}
            for _ws, seq_ws in (self.ws_fixed_seq or {}).items():
                for idx, j_raw in enumerate(seq_ws or []):
                    jj = int(j_raw)
                    if (jj in self.J) and (jj not in ws_rank):
                        ws_rank[jj] = int(idx)

            chain_of_task: Dict[int, int] = {}
            chain_pos: Dict[int, int] = {}
            chain_len: Dict[int, int] = {}
            for c_raw, seq_c in (shelf_seq or {}).items():
                cc = int(c_raw)
                seq_clean = [int(jj) for jj in (seq_c or []) if int(jj) in self.J]
                ln = len(seq_clean)
                for pos, jj in enumerate(seq_clean):
                    chain_of_task[int(jj)] = int(cc)
                    chain_pos[int(jj)] = int(pos)
                    chain_len[int(jj)] = int(ln)

            bucket: Dict[int, List[Tuple[int, int, int, int, int]]] = defaultdict(list)
            for j_raw, s_raw in (place or {}).items():
                try:
                    jj = int(j_raw)
                    ss = int(s_raw)
                except Exception:
                    continue
                if (jj not in self.J) or (ss not in self.S):
                    continue
                pos = int(chain_pos.get(jj, 10**9))
                ln = int(chain_len.get(jj, 0))
                is_tail = 1 if (ln > 0 and pos == (ln - 1)) else 0
                rank = int(ws_rank.get(jj, 10**9 + jj))
                cc = int(chain_of_task.get(jj, 10**9))
                bucket[int(ss)].append((is_tail, rank, cc, pos, jj))

            runtime_cell_sigma = {}
            for ss, rows in bucket.items():
                rows.sort(key=lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4])))
                runtime_cell_sigma[int(ss)] = [int(x[4]) for x in rows]

        # optional σ index: sigma_idx[cell][event_id] = order
        sigma_idx: Dict[int, Dict[int, int]] = {}
        if isinstance(runtime_cell_sigma, dict):
            for s, seq in (runtime_cell_sigma or {}).items():
                ss = int(s)
                sigma_idx[ss] = {int(tok): i for i, tok in enumerate(seq or [])}
        sigma_ptr: Dict[int, int] = defaultdict(int)  # 需要 from collections import defaultdict
        # =========================
        # HARD SIGMA STATE (NEW)
        # =========================
        served_in_cell: Dict[int, Set[int]] = defaultdict(set)

        def _sigma_next_expected(cell: int) -> Optional[int]:
            sigma = None
            if isinstance(runtime_cell_sigma, dict):
                sigma = (runtime_cell_sigma or {}).get(int(cell))
            if not isinstance(sigma, list) or (not sigma):
                return None

            k = int(sigma_ptr.get(int(cell), 0))

            # 跳过已经完成落位的任务（关键：防止“当场落位”导致指针卡死）
            while k < len(sigma) and int(sigma[k]) in served_in_cell[int(cell)]:
                k += 1

            sigma_ptr[int(cell)] = k
            return int(sigma[k]) if k < len(sigma) else None

        def _sigma_mark_served(cell: int, j: int) -> None:
            served_in_cell[int(cell)].add(int(j))
            _ = _sigma_next_expected(int(cell))  # 推进指针（跳过已完成）

        # tail task of each chain: if tail placed at a cell, it is truly infinite occupancy
        tail_task_of_chain: Dict[int, int] = {}
        chains_with_tasks: Set[int] = set()
        for c, seq in (shelf_seq or {}).items():
            cc = int(c)
            seq_clean = [int(j) for j in (seq or []) if int(j) in self.J]
            if seq_clean:
                chains_with_tasks.add(cc)
                tail_task_of_chain[cc] = int(seq_clean[-1])

        unused_chains: Set[int] = set(int(c) for c in (self.shelf_init or {}).keys()) - set(chains_with_tasks)

        def _is_true_infinite_interval(it: Dict[str, Any]) -> bool:
            """
            True infinite release means:
              - unused shelf init dummy, or
              - tail task interval (no successor pick ever happens)
            """
            try:
                ci = int(it.get("chain", -1))
                tj = int(it.get("task", -999999))
            except Exception:
                return False

            # unused chain: init dummy occupies forever
            if tj < 0:
                return ci in unused_chains

            # tail task occupies forever at its end cell
            if ci in tail_task_of_chain and tj == int(tail_task_of_chain[ci]):
                return True

            return False

        def _occ_open(chain: int, cell: int, start_vec: List[float], task_id: int, src: str = "") -> None:
            cell = int(cell)
            chain = int(chain)
            task_id = int(task_id)

            rec_it = {
                "start": self._t_copy(start_vec),
                "end": None,
                "planned_end": None,
                "chain": chain,
                "task": task_id,  # 这个 interval 是由哪个 task（或 init_dummy）打开的
                "src_open": str(src),
            }
            cell_occ[cell].append(rec_it)
            chain_open[chain] = (cell, len(cell_occ[cell]) - 1)

            # apply pending planned_end for THIS open interval (keyed by task_id)
            pe_map = pending_plan_end.get(chain, None)
            if isinstance(pe_map, dict):
                pe = pe_map.pop(task_id, None)
                if pe is not None:
                    rec_it["planned_end"] = self._t_copy(pe)
                    rec_it["src_plan"] = "pending_applied_on_open"
                    if not pe_map:
                        pending_plan_end.pop(chain, None)
                    if cell_gate_mode == "event":
                        _try_finalize_pending_for_cell(int(cell))

            if micro_enabled and cell == micro_cell:
                _micro("occ_open", cell=cell, chain=chain, task=int(task_id), startG=float(start_vec[G]), src=str(src))

        def _occ_set_planned_end(
                chain: int,
                plan_end_vec: List[float],
                src: str = "",
                expected_open_task: Optional[int] = None,
        ) -> None:
            """
            planned_end 只能写到“正确的 open interval”上：
              - 这个 open interval 的 task_id 必须等于 expected_open_task
            如果当前 open interval 不是这个 task_id，就先缓存到 pending_plan_end[chain][expected_open_task]，
            等那个 interval 真正被打开时（_occ_open task_id=expected_open_task）再自动套用。

            这样就杜绝了“j2 把 init_dummy 的 planned_end 顶到很晚”的污染。
            """
            chain = int(chain)
            exp_task = None if expected_open_task is None else int(expected_open_task)

            # no exp_task: do nothing (should not happen in our calls)
            if exp_task is None:
                return

            cur = chain_open.get(chain, None)

            # if no open interval now, cache by exp_task
            if cur is None:
                old = pending_plan_end[chain].get(exp_task)
                pending_plan_end[chain][exp_task] = self._t_max(old, plan_end_vec) if isinstance(old,
                                                                                                 list) else self._t_copy(
                    plan_end_vec)
                return

            cell, idx = cur
            if (cell not in cell_occ) or (idx >= len(cell_occ[cell])):
                old = pending_plan_end[chain].get(exp_task)
                pending_plan_end[chain][exp_task] = self._t_max(old, plan_end_vec) if isinstance(old,
                                                                                                 list) else self._t_copy(
                    plan_end_vec)
                return

            it = cell_occ[cell][idx]
            open_task = it.get("task", None)

            # mismatch -> cache to pending by exp_task (DO NOT pollute current interval)
            try:
                if int(open_task) != exp_task:
                    old = pending_plan_end[chain].get(exp_task)
                    pending_plan_end[chain][exp_task] = self._t_max(old, plan_end_vec) if isinstance(old,
                                                                                                     list) else self._t_copy(
                        plan_end_vec)
                    return
            except Exception:
                old = pending_plan_end[chain].get(exp_task)
                pending_plan_end[chain][exp_task] = self._t_max(old, plan_end_vec) if isinstance(old,
                                                                                                 list) else self._t_copy(
                    plan_end_vec)
                return

            # match -> update planned_end on current open interval
            old = it.get("planned_end")
            if isinstance(old, list):
                it["planned_end"] = self._t_max(old, plan_end_vec)
            else:
                it["planned_end"] = self._t_copy(plan_end_vec)

            it["src_plan"] = str(src)

            if cell_gate_mode == "event":
                _try_finalize_pending_for_cell(int(cell))

            if micro_enabled and int(cell) == micro_cell:
                _micro(
                    "occ_plan_end",
                    cell=int(cell),
                    chain=chain,
                    task=int(it.get("task", -999)),
                    planEndG=float(it["planned_end"][G]),
                    src=str(src),
                )

        def _occ_close(chain: int, end_vec: List[float], src: str = "") -> None:
            """
            Close the currently open interval of chain at the given pick time.
            """
            chain = int(chain)
            cur = chain_open.get(chain, None)
            if cur is None:
                return

            cell, idx = cur
            if (cell not in cell_occ) or (idx >= len(cell_occ[cell])):
                chain_open[chain] = None
                return

            it = cell_occ[cell][idx]
            old_end = it.get("end")
            new_end = self._t_copy(end_vec)
            # ensure end >= planned_end
            pe = it.get("planned_end")
            if isinstance(pe, list):
                new_end = self._t_max(new_end, pe)
            if isinstance(old_end, list):
                it["end"] = self._t_max(old_end, new_end)
            else:
                it["end"] = new_end

            it["src_close"] = str(src)
            chain_open[chain] = None
            if cell_gate_mode == "event":
                _try_finalize_pending_for_cell(int(cell))

            if micro_enabled and int(cell) == micro_cell:
                _micro("occ_close", cell=int(cell), chain=chain, task=int(it.get("task", -999)),
                       endG=float(it["end"][G]), src=str(src))

        def _cell_lb_for_place(cell: int, plan_vec: List[float], chain: int) -> List[float]:
            """
            Compute a MILP-aligned busy2 lower bound vector for dropping chain's shelf to `cell`.
            We enforce: start >= (end_of_other_interval + gap) for other chains.

            SPEED NOTE:
              - Do NOT sort cell_occ[cell]. Sorting was pure overhead because we do not early-break on order.
              - Fixed-point loop rounds are bounded by number of intervals (len(lst)+2), with a hard cap.
            """
            cell = int(cell)
            chain = int(chain)
            t = self._t_copy(plan_vec)
            gap = float(self.cell_gap)
            use_env = bool(self.envelope_shared_resources)

            # --- SPEED: no sorting ---
            lst = cell_occ.get(cell, [])

            # at most one "activation" per interval typically; keep a hard ceiling for safety
            max_rounds = min(50, len(lst) + 2)

            for _ in range(max_rounds):
                bumped = False
                tG = float(t[G])  # cache scalar for "future start" test

                for it in lst:
                    ci = int(it.get("chain", -1))
                    if ci == chain:
                        continue

                    st = it.get("start")
                    if not isinstance(st, list):
                        continue

                    # future occupancy must not block earlier drops
                    if float(st[G]) > tG + 1e-9:
                        continue

                    ed = it.get("end")
                    if not isinstance(ed, list):
                        ed = it.get("planned_end")
                    if not isinstance(ed, list):
                        ed = self._lift(self.BIG_M_time)

                    if use_env:
                        thr_scalar = float(max(ed)) + gap
                        thr = self._lift(thr_scalar)
                    else:
                        thr = [float(ed[k]) + gap for k in range(G + 1)]

                    # if any component violates, bump to max
                    if any(t[k] < thr[k] - 1e-9 for k in range(G + 1)):
                        t = self._t_max(t, thr)
                        bumped = True
                        tG = float(t[G])

                if not bumped:
                    break

            # enforce monotone
            for k in range(1, G + 1):
                if t[k] < t[k - 1] - 1e-9:
                    t[k] = t[k - 1]

            if micro_enabled and cell == micro_cell:
                _micro(
                    "cell_lb_for_place",
                    cell=int(cell),
                    chain=int(chain),
                    planG=float(plan_vec[G]),
                    lbG=float(t[G]),
                    n=int(len(lst)),
                )
            return t
        def _cell_lb_for_place_status(cell: int, plan_vec: List[float], chain: int) -> Tuple[Optional[List[float]], str]:
            """
            Return (lb_vec, status)
              status:
                - "ok"       : lb_vec valid
                - "unknown"  : blocked by an interval whose end/planned_end not known yet -> must pending
                - "infinite" : truly infinite occupancy blocks forever -> infeasible for this drop
            """
            cell = int(cell)
            chain = int(chain)
            t = self._t_copy(plan_vec)
            gap = float(self.cell_gap)
            use_env = bool(self.envelope_shared_resources)

            lst = cell_occ.get(cell, [])

            max_rounds = min(50, len(lst) + 2)
            for _ in range(max_rounds):
                bumped = False
                tG = float(t[G])

                for it in lst:
                    ci = int(it.get("chain", -1))
                    if ci == chain:
                        continue

                    st = it.get("start")
                    if not isinstance(st, list):
                        continue

                    # future occupancy must not block earlier drops
                    if float(st[G]) > tG + 1e-9:
                        continue

                    ed = it.get("end")
                    if not isinstance(ed, list):
                        ed = it.get("planned_end")

                    if not isinstance(ed, list):
                        # no end and no planned_end
                        if _is_true_infinite_interval(it):
                            return None, "infinite"
                        return None, "unknown"

                    if use_env:
                        thr_scalar = float(max(ed)) + gap
                        thr = self._lift(thr_scalar)
                    else:
                        thr = [float(ed[k]) + gap for k in range(G + 1)]

                    if any(t[k] < thr[k] - 1e-9 for k in range(G + 1)):
                        t = self._t_max(t, thr)
                        bumped = True
                        tG = float(t[G])

                if not bumped:
                    break

            for k in range(1, G + 1):
                if t[k] < t[k - 1] - 1e-9:
                    t[k] = t[k - 1]

            return t, "ok"

        # ---- init: open dummy occupancy for each shelf at its init cell: [0, first_pick] ----
        # ---- init: open dummy occupancy for each shelf at its init cell ----
        # IMPORTANT:
        #   - USE_INITIAL_OCCUPANCY_INTERVALS=True  : 所有链都开 init_dummy（更物理）
        #   - USE_INITIAL_OCCUPANCY_INTERVALS=False : 仅对 unused_chains 开 init_dummy（它们确实永远占着）
        for cc, init_cell in (self.shelf_init or {}).items():
            cc = int(cc)
            init_cell = int(init_cell)
            if init_cell not in self.S:
                continue
            if (not USE_INITIAL_OCCUPANCY_INTERVALS) and (cc not in unused_chains):
                continue
            _occ_open(cc, init_cell, self._t0(), task_id=-3000 - cc, src="init_dummy")

        home_before_plan: Dict[int, int] = self._build_home_before_plan(shelf_seq, place, any_s)

        # 输出：核心时间
        p: Dict[int, float] = {}
        q: Dict[int, float] = {}
        p0: Dict[int, float] = {}
        q0: Dict[int, float] = {}
        end_shelf_final: Dict[int, int] = {}
        task_home_before: Dict[int, int] = {}

        # 额外：用于 cell conflict 检测/repair（不依赖 timeline）
        pick_start_t: Dict[int, float] = {}
        arrive_cell_act_t: Dict[int, float] = {}

        # timeline（可选）
        timeline: List[Dict[str, Any]] = []

        # 诊断弧（可选）
        V_chain_arcs: List[Tuple[int, int, int, int, int]] = []
        last_task_on_chain: Dict[int, int] = {int(c): int(self.J_I.get(int(c), -int(c))) for c in shelf_seq.keys()}

        Z_arcs: List[Tuple[int, int, int]] = []
        if self.collect_v_arcs:
            for r in sorted(routes.keys()):
                seq = [int(x) for x in routes.get(r, [])]
                for a, b in zip(seq[:-1], seq[1:]):
                    if a in self.J and b in self.J:
                        Z_arcs.append((int(a), int(b), int(r)))

        heap: List[Tuple[float, int, int, dict]] = []
        cnt = 0
        meta_ptr: Dict[int, int] = {int(r): 0 for r in self.R_ids}

        # 距离/Δ
        def _get_s_s(s1: int, s2: int, ctx: str = "") -> Tuple[float, float]:
            key = (int(s1), int(s2))
            return self._dist_lookup("s_s", key, self.d_s_s, self.Delta_s_s, ctx=ctx)

        def _get_s_j(s: int, j: int, ctx: str = "") -> Tuple[float, float]:
            key = (int(s), int(j))
            return self._dist_lookup("s_pi", key, self.d_s_pi, self.Delta_s_pi, ctx=ctx)

        def _get_j_s(j: int, s: int, ctx: str = "") -> Tuple[float, float]:
            key = (int(j), int(s))
            return self._dist_lookup("pi_s", key, self.d_pi_s, self.Delta_pi_s, ctx=ctx)

        def _cand_end_cells(j: int, m: Optional[int]) -> List[int]:
            s_sorted = sorted(self.S.keys(), key=lambda s: self.d_pi_s.get((int(j), int(s)), 1e9))

            # ======== ADD: filter blocked cells ========
            if blocked_cells:
                filtered = [s for s in s_sorted if int(s) not in blocked_cells]
                if filtered:  # 防止全过滤空了
                    s_sorted = filtered
            # ======== ADD END ========

            if (m is None) or (m >= len(s_sorted)):
                return [int(x) for x in s_sorted]
            return [int(x) for x in s_sorted[:m]]

        def _ws_setup_start_vec(ws: int, arrival_ws_vec: List[float]) -> List[float]:
            # flat：所有任务到 WS 都 +D_setup
            if self.ws_setup_rule == "flat":
                base_vec = self._env_ready_vec(ws_free[ws])
                tmp = self._t_max(arrival_ws_vec, base_vec)
                return self._t_add_det(tmp, self.D_setup)

            # split：首/续两种 setup 规则（保留）
            out = [0.0] * (G + 1)
            if self.envelope_shared_resources:
                base_scalar = self._max_scalar(ws_free[ws])  # max over γ
                for k in range(G + 1):
                    if arrival_ws_vec[k] >= base_scalar - EPS:
                        out[k] = arrival_ws_vec[k] + self.D_setup_first
                    else:
                        out[k] = base_scalar + self.D_setup_next
            else:
                base = ws_free[ws]
                for k in range(G + 1):
                    if arrival_ws_vec[k] >= base[k] - EPS:
                        out[k] = arrival_ws_vec[k] + self.D_setup_first
                    else:
                        out[k] = base[k] + self.D_setup_next

            for k in range(1, G + 1):
                if out[k] < out[k - 1] - EPS:
                    out[k] = out[k - 1]
            return out

        def _update_cell_busy(cell: int, new_vec: List[float], src: str = "") -> None:
            """busy_until 单调不减（显微镜：只记录 focus_cell 的写入来源）"""
            cell = int(cell)
            prev = cell_busy_until.get(cell, self._t0())
            after = self._t_max(prev, new_vec)
            cell_busy_until[cell] = after

            if micro_enabled and cell == micro_cell:
                _micro(
                    "cell_busy_update",
                    cell=cell,
                    src=str(src),
                    oldG=float(prev[G]),
                    newG=float(new_vec[G]),
                    afterG=float(after[G]),
                    oldMax=float(max(prev)) if prev else 0.0,
                    newMax=float(max(new_vec)) if new_vec else 0.0,
                    afterMax=float(max(after)) if after else 0.0,
                )


        def _dispatch_next(r: int, push: bool):
            nonlocal cnt
            seq = [int(x) for x in routes.get(int(r), [])]
            ptr = int(meta_ptr.get(int(r), 0))
            if ptr >= len(seq):
                return
            j = int(seq[ptr])
            if j not in self.J:
                meta_ptr[int(r)] = ptr + 1
                _dispatch_next(r, push=push)
                return

            c = int(task_chain.get(j, -1))
            if c == -1:
                meta_ptr[int(r)] = ptr + 1
                _dispatch_next(r, push=push)
                return

            home_before = int(home_before_plan.get(j, shelf_cur.get(c, any_s)))
            ws = int(self.pi[j])

            t0_vec = agv_clock[int(r)]

            # move1：从 cell 或从 WS 出发
            if agv_from_ws[int(r)] is not None:
                j_prev_ws = int(agv_from_ws[int(r)])
                nom1, del1 = _get_j_s(
                    j_prev_ws, home_before,
                    ctx=f"move1 ws->shelf AGV={r} prevWS={j_prev_ws} Task={j} home_before={home_before}"
                )
                agv_from_ws[int(r)] = None
            else:
                nom1, del1 = _get_s_s(
                    int(agv_cell[int(r)]), home_before,
                    ctx=f"move1 cell->shelf AGV={r} Task={j} from_cell={agv_cell[int(r)]} home_before={home_before}"
                )

            arrive_shelf_vec = self._t_advance_unc(t0_vec, nom1, del1)

            # 关键：路线1包络（共享资源 shelf_ready 用 max over all γ）
            pick_start_vec = self._t_max(arrive_shelf_vec, self._env_ready_vec(shelf_ready[c]))

            # NEW: chain-wait lower bound at pick stage
            lb_pick = pick_lb.get(int(j), None)
            if lb_pick is not None and math.isfinite(float(lb_pick)):
                pick_start_vec = self._t_max(pick_start_vec, self._lift(float(lb_pick)))

            exp_open = _expected_open_task_id(c, j)
            _occ_set_planned_end(
                c,
                pick_start_vec,
                src=f"dispatch_next plan_pick j={j} c={c} r={r}",
                expected_open_task=exp_open,
            )

            # home_before cell 至 pick_start 期间占用该 cell（粗粒度）
            # home_before cell 至 pick_start 期间占用该 cell（粗粒度）
            if _can_prewrite_home_busy(j):
                _update_cell_busy(
                    home_before,
                    pick_start_vec,
                    src=f"dispatch_next home_before j={j} c={c} r={r}"
                )
            else:
                # 可选：显微镜记录“被闸门拦住了”，只在你盯 micro_cell 时才打印
                if micro_enabled and int(home_before) == micro_cell:
                    _micro(
                        "skip_prewrite_home_before",
                        j=int(j), c=int(c), r=int(r),
                        home_before=int(home_before),
                        pred=int(pred_on_chain.get(int(j)) or -1),
                        reason="pred_not_scheduled_yet"
                    )

            # move2：货架 -> WS
            nom2, del2 = _get_s_j(
                home_before, j,
                ctx=f"move2 shelf->WS AGV={r} Task={j} home_before={home_before}"
            )
            arrival_ws_vec = self._t_advance_unc(pick_start_vec, nom2, del2)

            rec = {
                "AGV": int(r), "Task": int(j), "Chain": int(c), "WS": int(ws),
                "home_before": int(home_before),

                "t0_vec": self._t_copy(t0_vec),

                "dt1_nom": float(nom1), "dt1_del": float(del1),
                "arrive_shelf_vec": self._t_copy(arrive_shelf_vec),
                "pick_start_vec": self._t_copy(pick_start_vec),

                "dt2_nom": float(nom2), "dt2_del": float(del2),
                "arrival_ws_vec": self._t_copy(arrival_ws_vec),

                "spawn_next": True,
            }

            if push:
                heapq.heappush(heap, (float(arrival_ws_vec[G]), cnt, int(r), rec))
                cnt += 1

        def _try_run_waiting(ws: int):
            order = self.ws_fixed_seq.get(ws, [])
            cur = int(ws_cursor.get(ws, 0))
            while cur < len(order):
                j_target = int(order[cur])
                rec = ws_park[ws].get(j_target)
                if rec is None:
                    break
                rec["spawn_next"] = (not self.detach_on_mismatch)
                _schedule_one(rec)
                ws_park[ws].pop(j_target, None)
                cur += 1
                ws_cursor[ws] = cur

        def _on_arrival(rec: dict):
            nonlocal cnt
            r = int(rec["AGV"])
            j = int(rec["Task"])
            c = int(rec["Chain"])
            ws = int(self.pi[j])
            home_before = int(rec["home_before"])

            # 关键：路线1包络（共享资源 shelf_ready 用 max over all γ）
            pick_true_vec = self._t_max(rec["arrive_shelf_vec"], self._env_ready_vec(shelf_ready[c]))
            # NEW: chain-wait lower bound at pick stage
            lb_pick = pick_lb.get(int(j), None)
            if lb_pick is not None and math.isfinite(float(lb_pick)):
                pick_true_vec = self._t_max(pick_true_vec, self._lift(float(lb_pick)))

            if _can_prewrite_home_busy(j):
                _update_cell_busy(
                    home_before,
                    pick_true_vec,
                    src=f"on_arrival home_before j={j} c={c} r={r}"
                )
            else:
                if micro_enabled and int(home_before) == micro_cell:
                    _micro(
                        "skip_prewrite_home_before",
                        j=int(j), c=int(c), r=int(r),
                        home_before=int(home_before),
                        pred=int(pred_on_chain.get(int(j)) or -1),
                        reason="pred_not_scheduled_yet_on_arrival"
                    )

            if self._t_changed(pick_true_vec, rec["pick_start_vec"], eps=1e-4):
                rec["pick_start_vec"] = self._t_copy(pick_true_vec)
                nom2 = float(rec["dt2_nom"]); del2 = float(rec["dt2_del"])
                rec["arrival_ws_vec"] = self._t_copy(self._t_advance_unc(pick_true_vec, nom2, del2))
                heapq.heappush(heap, (float(rec["arrival_ws_vec"][G]), cnt, r, rec))
                cnt += 1
                exp_open = _expected_open_task_id(c, j)
                _occ_set_planned_end(
                    c,
                    pick_true_vec,
                    src=f"on_arrival revise_pick j={j} c={c} r={r}",
                    expected_open_task=exp_open,
                )

                return
            if not rec.get("_occ_closed", False):
                _occ_close(c, pick_true_vec, src=f"on_arrival pick_close j={j} c={c} r={r} home_before={home_before}")
                rec["_occ_closed"] = True

            order = self.ws_fixed_seq.get(ws, [])
            cur = int(ws_cursor.get(ws, 0))
            nxt = int(order[cur]) if cur < len(order) else None

            if (nxt is None) or (j == nxt):
                rec["spawn_next"] = True
                _schedule_one(rec)
                if nxt is not None:
                    ws_cursor[ws] = cur + 1
                _try_run_waiting(ws)
            else:
                ws_park[ws][j] = rec
                if self.detach_on_mismatch:
                    agv_clock[r] = self._t_copy(rec["arrival_ws_vec"])
                    agv_from_ws[r] = j
                    meta_ptr[r] = int(meta_ptr.get(r, 0)) + 1
                    _dispatch_next(r, push=True)
                _try_run_waiting(ws)

        def _busy_vec_for_cell_end(s: int) -> List[float]:
            """
            end cell 的 busy 阻塞向量：
              - 路线1：用 max over all γ 的包络，再 +gap
              - 否则：同层 +gap
            """
            s = int(s)
            busy_vec = cell_busy_until.get(s, self._t0())

            if self.envelope_shared_resources:
                b = self._max_scalar(busy_vec)
                b2 = (b + self.cell_gap) if (b > EPS) else b
                out = self._lift(b2)
            else:
                out = [(busy_vec[k] + self.cell_gap) if (busy_vec[k] > EPS) else busy_vec[k] for k in range(G + 1)]

            if micro_enabled and s == micro_cell:
                _micro(
                    "busy_vec_for_cell_end",
                    cell=s,
                    busyG=float(busy_vec[G]),
                    busyMax=float(max(busy_vec)) if busy_vec else 0.0,
                    gap=float(self.cell_gap),
                    outG=float(out[G]),
                    outMax=float(max(out)) if out else 0.0,
                )
            return out

        def _choose_end_cell(j: int, ws_end_vec: List[float], chain: int) -> int:
            if self.lock_place and (j in place) and (int(place[j]) in self.S):
                return int(place[j])

            best_s: Optional[int] = None
            best_val: float = float("inf")

            for s in _cand_end_cells(j, self.end_candidates_m):
                nom3, del3 = _get_j_s(j, int(s), ctx=f"choose_end move3 WS->cell Task={j} cand_end_s={s}")
                plan_vec = self._t_advance_unc(ws_end_vec, nom3, del3)

                # busy2 from true interval ledger
                lb_vec = _cell_lb_for_place(int(s), plan_vec, chain=int(chain))
                act_vec = self._t_max(plan_vec, lb_vec)

                # repair lower bound
                lb = place_lb.get(int(j), None)
                if lb is not None and math.isfinite(lb):
                    act_vec = self._t_max(act_vec, self._lift(float(lb)))

                if act_vec[G] < best_val - EPS:
                    best_val = float(act_vec[G])
                    best_s = int(s)

            return int(best_s) if best_s is not None else int(any_s)

        def _finalize_one_drop_record(dr: Dict[str, Any]) -> bool:
            """
            Try finalize a pending drop record:
              - compute lb again (now should be known)
              - complete drop: update agv_clock/agv_cell/shelf_ready/_occ_open + dispatch next
            Return True if finalized, False if still blocked.
            """
            j = int(dr["Task"])
            r = int(dr["AGV"])
            c = int(dr["Chain"])
            ws = int(dr["WS"])
            home_before = int(dr["home_before"])
            end_s = int(dr["end_s"])

            ws_start_vec = dr["ws_start_vec"]
            ws_end_vec = dr["ws_end_vec"]
            arrival_ws_vec = dr["arrival_ws_vec"]
            pick_true_vec = dr["pick_true_vec"]
            arrive_cell_use_vec = dr["arrive_cell_use_vec"]
            nom3 = float(dr["dt3_nom"])
            del3 = float(dr["dt3_del"])
            rec_base = dr["rec_base"]

            lb_vec, st = _cell_lb_for_place_status(end_s, arrive_cell_use_vec, chain=c)
            if st == "unknown":
                return False
            if st == "infinite":
                dr["_infinite_blocked"] = True
                return False

            # base finalize (busy2 known now)
            arrive_cell_act_vec = self._t_max(arrive_cell_use_vec, lb_vec)

            # apply place_lb (cell repair)
            lb = place_lb.get(int(j), None)
            place_wait_due_to_lb = 0.0
            if lb is not None and math.isfinite(lb):
                before = float(arrive_cell_act_vec[G])
                arrive_cell_act_vec = self._t_max(arrive_cell_act_vec, self._lift(float(lb)))
                after = float(arrive_cell_act_vec[G])
                place_wait_due_to_lb = max(0.0, after - before)

            # ===== idempotent write-back: always safe =====
            p.setdefault(j, float(ws_start_vec[G]))
            q.setdefault(j, float(ws_end_vec[G]))
            p0.setdefault(j, float(ws_start_vec[0]))
            q0.setdefault(j, float(ws_end_vec[0]))
            end_shelf_final.setdefault(j, int(end_s))
            task_home_before.setdefault(j, int(home_before))
            pick_start_t.setdefault(j, float(pick_true_vec[G]))

            # WS must be free no later than ws_end (do NOT be blocked by drop waiting)
            ws_free[ws] = self._t_max(ws_free.get(ws, self._t0()), self._t_copy(ws_end_vec))

            # record for conflict detection
            arrive_cell_act_t[j] = float(arrive_cell_act_vec[G])

            # update resources
            agv_clock[r] = self._t_copy(arrive_cell_act_vec)
            agv_cell[r] = int(end_s)

            shelf_cur[c] = int(end_s)
            shelf_ready[c] = self._t_max(shelf_ready.get(c, self._t0()), self._t_copy(arrive_cell_act_vec))
            _occ_open(
                c, int(end_s), arrive_cell_act_vec,
                task_id=j,
                src=f"place_open(PENDING) j={j} c={c} r={r} end_s={end_s}"
            )

            scheduled_on_chain.add(j)

            if self.collect_v_arcs:
                pred = int(last_task_on_chain.get(c, int(self.J_I.get(c, -c))))
                V_chain_arcs.append((pred, j, int(home_before), int(end_s), int(c)))
                last_task_on_chain[c] = int(j)

            # timeline
            if self.timeline_mode != "off":
                if self.timeline_mode == "min":
                    timeline.append({
                        "AGV": int(r), "Task": int(j), "Chain": int(c), "WS": int(ws),
                        "home_before": int(home_before),
                        "pick_start": float(pick_true_vec[G]),
                        "ws_start": float(ws_start_vec[G]),
                        "ws_end": float(ws_end_vec[G]),
                        "end_s": int(end_s),
                        "arrive_cell_act": float(arrive_cell_act_vec[G]),
                    })
                else:
                    dt1_eff = float(rec_base["arrive_shelf_vec"][G] - rec_base["t0_vec"][G])
                    dt2_eff = float(arrival_ws_vec[G] - pick_true_vec[G])
                    dt3_eff = float(arrive_cell_use_vec[G] - ws_end_vec[G])
                    wait_busy2 = float(arrive_cell_act_vec[G] - arrive_cell_use_vec[G])

                    timeline.append({
                        "AGV": int(r), "Task": int(j), "Chain": int(c), "WS": int(ws),
                        "home_before": int(home_before),

                        "dt1": float(dt1_eff),
                        "arrive_shelf": float(rec_base["arrive_shelf_vec"][G]),
                        "pick_start": float(pick_true_vec[G]),

                        "dt2_eff": float(dt2_eff),
                        "dt2_nom": float(rec_base["dt2_nom"]),
                        "arrival_ws": float(arrival_ws_vec[G]),

                        "ws_start": float(ws_start_vec[G]),
                        "ws_end": float(ws_end_vec[G]),

                        "end_s": int(end_s),

                        "dt3_eff": float(dt3_eff),
                        "dt3_nom": float(nom3),

                        "arrive_cell_nom": float(ws_end_vec[G] + nom3),
                        "arrive_cell_act": float(arrive_cell_act_vec[G]),
                        "cell_wait_due_to_busy2": float(wait_busy2),

                        "dt1_nom": float(rec_base["dt1_nom"]),
                        "dt1_del": float(rec_base["dt1_del"]),
                        "arrive_shelf_0": float(rec_base["arrive_shelf_vec"][0]),
                        "arrive_shelf_G": float(rec_base["arrive_shelf_vec"][G]),
                        "pick_start_0": float(pick_true_vec[0]),
                        "pick_start_G": float(pick_true_vec[G]),

                        "dt2_del": float(rec_base["dt2_del"]),
                        "arrival_ws_0": float(arrival_ws_vec[0]),
                        "arrival_ws_G": float(arrival_ws_vec[G]),

                        "ws_start_0": float(ws_start_vec[0]),
                        "ws_end_0": float(ws_end_vec[0]),
                        "ws_start_G": float(ws_start_vec[G]),
                        "ws_end_G": float(ws_end_vec[G]),

                        "dt3_del": float(del3),
                        "arrive_cell_use_0": float(arrive_cell_use_vec[0]),
                        "arrive_cell_use_G": float(arrive_cell_use_vec[G]),
                        "arrive_cell_act_0": float(arrive_cell_act_vec[0]),
                        "arrive_cell_act_G": float(arrive_cell_act_vec[G]),

                        "place_lb": float(lb) if (lb is not None and math.isfinite(lb)) else None,
                        "place_wait_due_to_lb": float(place_wait_due_to_lb),
                    })

            # now dispatch next (the task pointer advances only when drop done)
            if bool(dr.get("spawn_next", False)):
                meta_ptr[r] = int(meta_ptr.get(r, 0)) + 1
                _dispatch_next(r, push=True)

            return True

        def _try_finalize_pending_for_cell(cell: int) -> None:
            cell = int(cell)
            if cell in _wake_guard:
                return
            if cell not in pending_drops or (not pending_drops[cell]):
                return

            _wake_guard.add(cell)
            try:
                pend = pending_drops[cell]

                # 清掉已经 served 的（安全）
                for jj in list(pend.keys()):
                    if int(jj) in served_in_cell[int(cell)]:
                        pend.pop(int(jj), None)

                if not pend:
                    pending_drops.pop(cell, None)
                    return

                sigma = None
                if isinstance(runtime_cell_sigma, dict):
                    sigma = (runtime_cell_sigma or {}).get(int(cell))

                # ========== STRICT σ ==========
                # 如果有 σ：只允许按 σ_ptr 指向的 expected 任务落位。
                # expected 没到 -> 空等（不允许用到达先后抢占）
                if isinstance(sigma, list) and len(sigma) > 0:
                    while True:
                        exp = _sigma_next_expected(int(cell))
                        if exp is None:
                            break  # σ 序列耗尽，转 FIFO
                        exp = int(exp)

                        if exp not in pend:
                            break  # expected 还没到，严格空等

                        dr = pend.get(exp)
                        if dr is None:
                            break

                        ok = _finalize_one_drop_record(dr)
                        if ok:
                            pend.pop(exp, None)
                            _sigma_mark_served(int(cell), exp)
                            continue

                        # expected 已到但仍因 unknown/infinite busy2 卡住 -> 停
                        break

                    # σ 完全完成后，剩下的任务用 FIFO（按 arrive_use_G）释放
                    if _sigma_next_expected(int(cell)) is None and pend:
                        while True:
                            items = list(pend.items())
                            if not items:
                                break
                            items.sort(key=lambda it: (float(it[1].get("arrive_use_G", 0.0)), int(it[0])))

                            j0, dr0 = items[0]
                            if _finalize_one_drop_record(dr0):
                                pend.pop(int(j0), None)
                                _sigma_mark_served(int(cell), int(j0))
                                continue
                            break

                    if not pend:
                        pending_drops.pop(cell, None)
                    return

                # ========== NO σ : FIFO ==========
                while True:
                    items = list(pend.items())
                    if not items:
                        break
                    items.sort(key=lambda it: (float(it[1].get("arrive_use_G", 0.0)), int(it[0])))

                    j0, dr0 = items[0]
                    if _finalize_one_drop_record(dr0):
                        pend.pop(int(j0), None)
                        _sigma_mark_served(int(cell), int(j0))
                        continue
                    break

                if not pend:
                    pending_drops.pop(cell, None)

            finally:
                _wake_guard.remove(cell)
        def _schedule_one(rec: dict):
            j = int(rec["Task"])
            r = int(rec["AGV"])
            c = int(rec["Chain"])
            ws = int(self.pi[j])
            home_before = int(rec["home_before"])
            if micro_enabled and (j in micro_tasks or home_before == micro_cell):
                _micro(
                    "schedule_enter",
                    j=j, r=r, c=c, ws=ws,
                    home_before=home_before,
                    t0G=float(rec.get("t0_vec", self._t0())[G]) if isinstance(rec.get("t0_vec"), list) else None,
                    arrive_shelfG=float(rec.get("arrive_shelf_vec", self._t0())[G]) if isinstance(rec.get("arrive_shelf_vec"), list) else None,
                    shelf_readyG=float(shelf_ready.get(c, self._t0())[G]),
                    shelf_readyMax=float(max(shelf_ready.get(c, self._t0()))) if shelf_ready.get(c) else 0.0,
                )

            # 关键：路线1包络（共享资源 shelf_ready 用 max over all γ）
            pick_true_vec = self._t_max(rec["arrive_shelf_vec"], self._env_ready_vec(shelf_ready[c]))
            # NEW: chain-wait lower bound at pick stage
            lb_pick = pick_lb.get(int(j), None)
            if lb_pick is not None and math.isfinite(float(lb_pick)):
                pick_true_vec = self._t_max(pick_true_vec, self._lift(float(lb_pick)))

            _update_cell_busy(home_before, pick_true_vec, src=f"schedule_one home_before j={j} c={c} r={r}")
            if micro_enabled and (j in micro_tasks or home_before == micro_cell):
                _micro(
                    "pick_true",
                    j=j, r=r, c=c,
                    home_before=home_before,
                    arrive_shelfG=float(rec["arrive_shelf_vec"][G]),
                    shelf_readyG=float(self._env_ready_vec(shelf_ready[c])[G]),
                    pickG=float(pick_true_vec[G]),
                )

            # move2（以 pick_true 为准）
            nom2 = float(rec["dt2_nom"]); del2 = float(rec["dt2_del"])
            arrival_ws_vec = self._t_advance_unc(pick_true_vec, nom2, del2)

            # WS
            ws_start_vec = _ws_setup_start_vec(ws, arrival_ws_vec)
            ws_end_vec = self._t_add_det(ws_start_vec, float(self.D[j]))

            # end cell
            end_s = _choose_end_cell(j, ws_end_vec, chain=c)
            # ===== record WS completion immediately (even if drop becomes pending) =====
            p[j] = float(ws_start_vec[G])
            q[j] = float(ws_end_vec[G])
            p0[j] = float(ws_start_vec[0])
            q0[j] = float(ws_end_vec[0])
            end_shelf_final[j] = int(end_s)
            task_home_before[j] = int(home_before)
            pick_start_t[j] = float(pick_true_vec[G])

            # WS resource must be released at ws_end regardless of drop waiting
            ws_free[ws] = self._t_copy(ws_end_vec)

            if micro_enabled and (j in micro_tasks or end_s == micro_cell or home_before == micro_cell):
                _micro(
                    "chosen_end_cell",
                    j=j, r=r, c=c,
                    end_s=int(end_s),
                    lock_place=bool(self.lock_place and (j in place)),
                    place_value=int(place.get(j, -1)) if j in place else None,
                )

            # move3
            nom3, del3 = _get_j_s(
                j, end_s,
                ctx=f"move3 WS->cell AGV={r} Task={j} end_s={end_s}"
            )
            arrive_cell_use_vec = self._t_advance_unc(ws_end_vec, nom3, del3)

            # 关键：路线1包络（共享资源 cell_busy 用 max over all γ）
            # =========================
            # cell gate: legacy vs event
            # =========================
            if cell_gate_mode == "event":
                # ========== STRICT σ gate ==========
                exp = _sigma_next_expected(int(end_s))
                if exp is not None and int(j) != int(exp):
                    # 不管队列空不空，只要不是 expected，就必须等待
                    pending_drops[int(end_s)][int(j)] = {
                        "Task": int(j),
                        "AGV": int(r),
                        "Chain": int(c),
                        "WS": int(ws),
                        "home_before": int(home_before),
                        "end_s": int(end_s),

                        "ws_start_vec": self._t_copy(ws_start_vec),
                        "ws_end_vec": self._t_copy(ws_end_vec),
                        "arrival_ws_vec": self._t_copy(arrival_ws_vec),
                        "pick_true_vec": self._t_copy(pick_true_vec),
                        "arrive_cell_use_vec": self._t_copy(arrive_cell_use_vec),

                        "dt3_nom": float(nom3),
                        "dt3_del": float(del3),

                        "rec_base": {
                            "t0_vec": self._t_copy(rec["t0_vec"]),
                            "arrive_shelf_vec": self._t_copy(rec["arrive_shelf_vec"]),
                            "dt1_nom": float(rec["dt1_nom"]),
                            "dt1_del": float(rec["dt1_del"]),
                            "dt2_nom": float(rec["dt2_nom"]),
                            "dt2_del": float(rec["dt2_del"]),
                        },

                        "spawn_next": bool(rec.get("spawn_next", False)),
                        "arrive_use_G": float(arrive_cell_use_vec[G]),
                        "_sigma_wait": True,
                    }
                    _try_finalize_pending_for_cell(int(end_s))
                    return

                # 到这里：要么没有 σ，要么 j 就是 expected（允许尝试落位）
                lb_vec, st = _cell_lb_for_place_status(int(end_s), arrive_cell_use_vec, chain=c)

                if st == "unknown":
                    pending_drops[int(end_s)][int(j)] = {
                        "Task": int(j),
                        "AGV": int(r),
                        "Chain": int(c),
                        "WS": int(ws),
                        "home_before": int(home_before),
                        "end_s": int(end_s),

                        "ws_start_vec": self._t_copy(ws_start_vec),
                        "ws_end_vec": self._t_copy(ws_end_vec),
                        "arrival_ws_vec": self._t_copy(arrival_ws_vec),
                        "pick_true_vec": self._t_copy(pick_true_vec),
                        "arrive_cell_use_vec": self._t_copy(arrive_cell_use_vec),

                        "dt3_nom": float(nom3),
                        "dt3_del": float(del3),

                        "rec_base": {
                            "t0_vec": self._t_copy(rec["t0_vec"]),
                            "arrive_shelf_vec": self._t_copy(rec["arrive_shelf_vec"]),
                            "dt1_nom": float(rec["dt1_nom"]),
                            "dt1_del": float(rec["dt1_del"]),
                            "dt2_nom": float(rec["dt2_nom"]),
                            "dt2_del": float(rec["dt2_del"]),
                        },

                        "spawn_next": bool(rec.get("spawn_next", False)),
                        "arrive_use_G": float(arrive_cell_use_vec[G]),
                        "_unknown_release": True,
                    }
                    if micro_enabled and int(end_s) == micro_cell:
                        _micro("drop_pending_unknown_release", j=int(j), r=int(r), c=int(c), cell=int(end_s),
                               arriveUseG=float(arrive_cell_use_vec[G]))
                    _try_finalize_pending_for_cell(int(end_s))
                    return

                if st == "infinite":
                    pending_drops[int(end_s)][int(j)] = {
                        "Task": int(j),
                        "AGV": int(r),
                        "Chain": int(c),
                        "WS": int(ws),
                        "home_before": int(home_before),
                        "end_s": int(end_s),

                        "ws_start_vec": self._t_copy(ws_start_vec),
                        "ws_end_vec": self._t_copy(ws_end_vec),
                        "arrival_ws_vec": self._t_copy(arrival_ws_vec),
                        "pick_true_vec": self._t_copy(pick_true_vec),
                        "arrive_cell_use_vec": self._t_copy(arrive_cell_use_vec),

                        "dt3_nom": float(nom3),
                        "dt3_del": float(del3),

                        "rec_base": {
                            "t0_vec": self._t_copy(rec["t0_vec"]),
                            "arrive_shelf_vec": self._t_copy(rec["arrive_shelf_vec"]),
                            "dt1_nom": float(rec["dt1_nom"]),
                            "dt1_del": float(rec["dt1_del"]),
                            "dt2_nom": float(rec["dt2_nom"]),
                            "dt2_del": float(rec["dt2_del"]),
                        },

                        "spawn_next": bool(rec.get("spawn_next", False)),
                        "arrive_use_G": float(arrive_cell_use_vec[G]),
                        "_infinite_blocked": True,
                    }
                    if micro_enabled and int(end_s) == micro_cell:
                        _micro("drop_blocked_infinite", j=int(j), r=int(r), c=int(c), cell=int(end_s))
                    _try_finalize_pending_for_cell(int(end_s))
                    return

                # st == "ok"
                busy2_vec = lb_vec

            else:
                # legacy: your old behavior
                busy2_vec = _cell_lb_for_place(int(end_s), arrive_cell_use_vec, chain=c)

            arrive_cell_act_vec = self._t_max(arrive_cell_use_vec, busy2_vec)


            if micro_enabled and (j in micro_tasks or end_s == micro_cell or home_before == micro_cell):
                _micro(
                    "arrive_cell_act",
                    j=j, r=r, c=c,
                    end_s=int(end_s),
                    arrive_useG=float(arrive_cell_use_vec[G]),
                    busy2G=float(busy2_vec[G]),
                    arrive_actG=float(arrive_cell_act_vec[G]),
                )

            # cell repair - “延迟落位”下界
            lb = place_lb.get(int(j), None)
            place_wait_due_to_lb = 0.0
            if lb is not None and math.isfinite(lb):
                lb_vec = self._lift(float(lb))
                before = float(arrive_cell_act_vec[G])
                arrive_cell_act_vec = self._t_max(arrive_cell_act_vec, lb_vec)
                after = float(arrive_cell_act_vec[G])
                place_wait_due_to_lb = max(0.0, after - before)

            # 记录：Γ层用于目标；0层用于对照
            p[j] = float(ws_start_vec[G])
            q[j] = float(ws_end_vec[G])
            p0[j] = float(ws_start_vec[0])
            q0[j] = float(ws_end_vec[0])
            end_shelf_final[j] = int(end_s)
            task_home_before[j] = int(home_before)

            # 记录供 cell 冲突检测/repair 使用的标量
            pick_start_t[j] = float(pick_true_vec[G])
            arrive_cell_act_t[j] = float(arrive_cell_act_vec[G])

            # 更新资源
            ws_free[ws] = self._t_copy(ws_end_vec)

            agv_clock[r] = self._t_copy(arrive_cell_act_vec)
            agv_cell[r] = int(end_s)

            # _update_cell_busy(int(end_s), arrive_cell_act_vec,
            #                   src=f"schedule_one end_s j={j} c={c} r={r} end_s={end_s}")

            shelf_cur[c] = int(end_s)
            shelf_ready[c] = self._t_max(shelf_ready.get(c, self._t0()), self._t_copy(arrive_cell_act_vec))
            _occ_open(c, int(end_s), arrive_cell_act_vec, task_id=j, src=f"place_open j={j} c={c} r={r} end_s={end_s}")


            if cell_gate_mode == "event":
                _sigma_mark_served(int(end_s), int(j))
                _try_finalize_pending_for_cell(int(end_s))

            scheduled_on_chain.add(j)
            if self.collect_v_arcs:
                pred = int(last_task_on_chain.get(c, int(self.J_I.get(c, -c))))
                V_chain_arcs.append((pred, j, int(home_before), int(end_s), int(c)))
                last_task_on_chain[c] = int(j)

            # timeline（可选）
            if self.timeline_mode != "off":
                if self.timeline_mode == "min":
                    timeline.append({
                        "AGV": int(r), "Task": int(j), "Chain": int(c), "WS": int(ws),
                        "home_before": int(home_before),
                        "pick_start": float(pick_true_vec[G]),
                        "ws_start": float(ws_start_vec[G]),
                        "ws_end": float(ws_end_vec[G]),
                        "end_s": int(end_s),
                        "arrive_cell_act": float(arrive_cell_act_vec[G]),
                    })
                else:
                    # full：保留旧接口字段 + 扩展字段（兼容 main.py 打印）
                    dt1_eff = float(rec["arrive_shelf_vec"][G] - rec["t0_vec"][G])
                    dt2_eff = float(arrival_ws_vec[G] - pick_true_vec[G])
                    dt3_eff = float(arrive_cell_use_vec[G] - ws_end_vec[G])

                    timeline.append({
                        "AGV": int(r), "Task": int(j), "Chain": int(c), "WS": int(ws),
                        "home_before": int(home_before),

                        # ===== main.py 旧接口字段 =====
                        "dt1": float(dt1_eff),
                        "arrive_shelf": float(rec["arrive_shelf_vec"][G]),
                        "pick_start": float(pick_true_vec[G]),

                        "dt2_eff": float(dt2_eff),
                        "dt2_nom": float(nom2),
                        "arrival_ws": float(arrival_ws_vec[G]),

                        "ws_start": float(ws_start_vec[G]),
                        "ws_end": float(ws_end_vec[G]),

                        "end_s": int(end_s),

                        "dt3_eff": float(dt3_eff),
                        "dt3_nom": float(nom3),

                        "arrive_cell_nom": float(ws_end_vec[G] + nom3),
                        "arrive_cell_act": float(arrive_cell_act_vec[G]),

                        # ===== 扩展字段 =====
                        "dt1_nom": float(rec["dt1_nom"]),
                        "dt1_del": float(rec["dt1_del"]),
                        "arrive_shelf_0": float(rec["arrive_shelf_vec"][0]),
                        "arrive_shelf_G": float(rec["arrive_shelf_vec"][G]),
                        "pick_start_0": float(pick_true_vec[0]),
                        "pick_start_G": float(pick_true_vec[G]),

                        "dt2_del": float(rec["dt2_del"]),
                        "arrival_ws_0": float(arrival_ws_vec[0]),
                        "arrival_ws_G": float(arrival_ws_vec[G]),

                        "ws_start_0": float(ws_start_vec[0]),
                        "ws_end_0": float(ws_end_vec[0]),
                        "ws_start_G": float(ws_start_vec[G]),
                        "ws_end_G": float(ws_end_vec[G]),

                        "dt3_del": float(del3),
                        "arrive_cell_use_0": float(arrive_cell_use_vec[0]),
                        "arrive_cell_use_G": float(arrive_cell_use_vec[G]),
                        "arrive_cell_act_0": float(arrive_cell_act_vec[0]),
                        "arrive_cell_act_G": float(arrive_cell_act_vec[G]),

                        # cell repair diagnostics
                        "place_lb": float(lb) if (lb is not None and math.isfinite(lb)) else None,
                        "place_wait_due_to_lb": float(place_wait_due_to_lb),
                    })

            if rec.get("spawn_next", False):
                meta_ptr[r] = int(meta_ptr.get(r, 0)) + 1
                _dispatch_next(r, push=True)

        # 初始派工
        for r in self.R_ids:
            _dispatch_next(int(r), push=True)

        # 事件循环
        while True:
            while heap:
                _, _, _, rec = heapq.heappop(heap)
                _on_arrival(rec)

            before_total = sum(len(ws_park[w]) for w in ws_park)
            for ws in list(all_ws):
                _try_run_waiting(int(ws))
            after_total = sum(len(ws_park[w]) for w in ws_park)

            if not heap and after_total == before_total:
                break
        # final attempt to resolve any pending drops
        if cell_gate_mode == "event" and pending_drops:
            for s in list(pending_drops.keys()):
                _try_finalize_pending_for_cell(int(s))

        # if still pending, mark infeasible (avoid fake-feasible BIG_M)
        if cell_gate_mode == "event":
            still_pending = sum(len(v) for v in pending_drops.values())
            if still_pending > 0:
                event_gate_debug["_still_pending_cnt"] = int(still_pending)
                event_gate_debug["_pending_by_cell_top"] = {
                    int(s): list(pending_drops[s].keys())[:10]
                    for s in pending_drops.keys()
                }
                event_gate_debug["_sigma_ptr_snapshot"] = {
                    int(s): int(sigma_ptr.get(int(s), 0))
                    for s in pending_drops.keys()
                }

                p.clear()
                q.clear()

        # 构建 V_arcs（可选）
        V_arcs: List[Tuple[int, int, int, int]] = []
        if self.collect_v_arcs:
            for r in self.R_ids:
                seq_raw = [int(x) for x in routes.get(int(r), [])]
                seq = [jj for jj in seq_raw if jj in self.J and (jj in task_home_before)]
                if not seq:
                    continue
                j0 = self.J0.get(int(r))
                jd = self.Jd.get(int(r))

                if j0 is not None:
                    s_from = int(self.agv_init.get(int(r), any_s))
                    s_to = int(task_home_before[seq[0]])
                    V_arcs.append((int(j0), int(seq[0]), int(s_from), int(s_to)))

                for i, j in zip(seq[:-1], seq[1:]):
                    if (i in end_shelf_final) and (j in task_home_before):
                        V_arcs.append((int(i), int(j), int(end_shelf_final[i]), int(task_home_before[j])))

                if jd is not None:
                    last = int(seq[-1])
                    if last in end_shelf_final:
                        s_from = int(end_shelf_final[last])
                        s_to = int(self.agv_init.get(int(r), s_from))
                        if s_to not in self.S:
                            s_to = int(s_from)
                        V_arcs.append((int(last), int(jd), int(s_from), int(s_to)))

        C_task_max = float(max(q.values())) if q else float("inf")
        C_agv_max = float(max(agv_clock[r][G] for r in agv_clock)) if agv_clock else 0.0

        details = {
            "C_task_max": C_task_max,
            "C_agv_max": C_agv_max,
            "p": p,
            "q": q,
            "p0": p0,
            "q0": q0,
            "end_shelf_final": end_shelf_final,
            "timeline": timeline,
            "feasible": True,
            "penalties": {},
            "envelope_shared_resources": self.envelope_shared_resources,
            # NEW: 给 cell 冲突检测/repair 用（不依赖 timeline）
            "pick_start": pick_start_t,
            "arrive_cell_act": arrive_cell_act_t,
        }
        if micro_enabled:
            try:
                ints = self._initial_shelf_intervals(shelf_seq, pick_start_t)
                focus_ints = [x for x in ints if int(x[0]) == micro_cell]
                _micro("initial_occupancy_intervals_at_end", intervals=focus_ints)
            except Exception as e:
                _micro("initial_occupancy_intervals_error", err=str(e))

        details["micro_debug"] = micro_log

        if self.collect_v_arcs:
            details["V_arcs"] = V_arcs
            details["V_chain_arcs"] = V_chain_arcs
            details["Z_arcs"] = Z_arcs
        else:
            details["V_arcs"] = []
            details["V_chain_arcs"] = []
            details["Z_arcs"] = []
        if event_gate_debug:
            details.update(event_gate_debug)

        return details
