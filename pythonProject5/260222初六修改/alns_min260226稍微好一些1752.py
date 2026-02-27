from __future__ import annotations

import copy
import math
import random
from itertools import permutations
from typing import Dict, List, Tuple, Set, Optional

from evaluator import RobustEvaluator
from collections import defaultdict

# =========================
#        Feasibility
# =========================
def basic_feasibility_check_level0(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    verbose: bool = True,
) -> bool:
    """
    第0层：只检查最基础的结构可行性（与 MILP 无关）
      1) 每个任务是否被 routes 覆盖一次且仅一次
      2) 每辆车的 route 是否不含非法任务 / 重复任务
      3) shelf_seq 是否与 task_shelf_mapping 一致（若提供）
      4) place 是否只使用合法的货架cell
    """
    ok = True
    all_tasks: Set[int] = set(int(j) for j in evaluator.J)

    # 1) routes 覆盖计数
    appear_cnt: Dict[int, int] = {j: 0 for j in all_tasks}
    for r, seq in routes.items():
        for j in seq:
            jj = int(j)
            if jj not in all_tasks:
                ok = False
                if verbose:
                    print(f"[Lv0-Check] ERROR: AGV {r} route 中出现非法任务 {jj}")
            else:
                appear_cnt[jj] += 1

    for j in sorted(all_tasks):
        if appear_cnt[j] == 0:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: 任务 {j} 没有出现在任何 AGV route 中")
        elif appear_cnt[j] > 1:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: 任务 {j} 在 routes 中出现了 {appear_cnt[j]} 次 (>1)")

    # 2) 车内重复
    for r, seq in routes.items():
        seen: Set[int] = set()
        for j in seq:
            jj = int(j)
            if jj in seen:
                ok = False
                if verbose:
                    print(f"[Lv0-Check] ERROR: AGV {r} 的 route 中任务 {jj} 重复出现")
            seen.add(jj)

    # 3) shelf_seq 与 mapping 一致性
    if task_shelf_mapping is not None:
        for c, seq in shelf_seq.items():
            cc = int(c)
            for j in seq:
                jj = int(j)
                c_expect = int(task_shelf_mapping.get(jj, -999))
                if c_expect != cc:
                    ok = False
                    if verbose:
                        print(
                            f"[Lv0-Check] ERROR: shelf_seq 中链 {cc} 含任务 {jj}, "
                            f"但 task_shelf_mapping[{jj}] = {c_expect}"
                        )

        shelf_cover: Dict[int, Set[int]] = {
            int(c): set(int(j) for j in seq) for c, seq in shelf_seq.items()
        }
        for j in all_tasks:
            c_expect = int(task_shelf_mapping.get(j, -1))
            if c_expect == -1:
                continue
            if c_expect not in shelf_cover or j not in shelf_cover[c_expect]:
                ok = False
                if verbose:
                    print(
                        f"[Lv0-Check] ERROR: 任务 {j} 期望在链 {c_expect}, "
                        f"但 shelf_seq[{c_expect}] 中未出现"
                    )

    # 4) place 合法性
    all_cells: Set[int] = set(int(s) for s in evaluator.S)
    for j, s in place.items():
        jj, ss = int(j), int(s)
        if jj not in all_tasks:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: place 中出现未知任务 {jj}")
        if ss not in all_cells:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: place[{jj}] = {ss} 不是合法货架cell")

    # 5) place 覆盖所有 routes 内任务
    for j in all_tasks:
        if appear_cnt[j] > 0 and (j not in place):
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: 任务 {j} 出现在 routes 中，但 place 中没有对应回库位")

    if verbose:
        if ok:
            print("[Lv0-Check] 基础结构检查通过（routes / shelf_seq / place 一致性良好）")
        else:
            print("[Lv0-Check] 基础结构检查存在问题（见上方 ERROR）")
    return ok


# =========================
#        Utilities
# =========================
def _dc_routes(routes: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {int(r): [int(j) for j in seq] for r, seq in routes.items()}


def _dc_place(place: Dict[int, int]) -> Dict[int, int]:
    return {int(j): int(s) for j, s in place.items()}


def _dc_shelf_seq(shelf_seq: Dict[int, List[int]]) -> Dict[int, List[int]]:
    return {int(c): [int(j) for j in seq] for c, seq in shelf_seq.items()}
from typing import Any

def _safe_evaluate(
    evaluator: RobustEvaluator,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    *,
    fail_cost: float = 1e30,
) -> Tuple[float, Dict[str, Any]]:
    """
    保护性 evaluate：任何异常（KeyError 等）都不会把 ALNS 整体炸掉。
    出错时返回一个“非常不可行”的 details，让 infeas-key 变大、自动被拒绝/淘汰。
    """
    try:
        obj, det = evaluator.evaluate(routes, shelf_seq, place)
        obj_f = float(obj)
        if not math.isfinite(obj_f):
            obj_f = float(fail_cost)
        det_d = dict(det) if isinstance(det, dict) else {}
        return obj_f, det_d
    except Exception as e:
        # 让 infeas_key 极大：unscheduled 全部任务 + 大 cell_conflict
        all_j = [int(x) for x in (getattr(evaluator, "J", []) or [])]
        det_d: Dict[str, Any] = {
            "unscheduled_tasks": all_j,
            "penalties": {
                "tail_cell_conflict": 1.0,
                "cell_conflict_pairs": 1e9,
                "cell_conflict_overlap_time": 1e9,
            },
            "error": repr(e),
        }
        return float(fail_cost), det_d


def _normalize_one_route_ws_and_shelf(
    seq: List[int],
    *,
    evaluator: RobustEvaluator,
    chain_of: Dict[int, int],
    shelf_idx: Dict[int, Dict[int, int]],
) -> List[int]:
    """
    单条 route 的统一规范化：
      1) 连续同 WS 的块内按 ws_fixed_seq 排序
      2) 同一 shelf 链的任务在 route 内 slot 内按 shelf_seq 顺序排序
    """
    seq2 = _normalize_one_route_ws_blocks([int(x) for x in (seq or [])], evaluator)
    seq2 = _normalize_one_route_shelf_slots(seq2, chain_of, shelf_idx)
    return seq2

def _sanitize_task_shelf_mapping(
    task_shelf_mapping: Optional[Dict[int, Any]],
    *,
    verbose: bool = True,
) -> Optional[Dict[int, int]]:
    """
    把 task_shelf_mapping 清洗成“task->chain(int)”的干净字典：
      - 丢弃 value 为 None / NaN / 无法转 int 的项
      - key 也做 int 化
    返回：
      - 清洗后非空 dict
      - 或 None（表示不可用）
    """
    if not task_shelf_mapping:
        return None

    out: Dict[int, int] = {}
    bad = 0

    for j, c in task_shelf_mapping.items():
        # 丢弃 None
        if c is None:
            bad += 1
            continue

        # 丢弃 NaN（常见于 pandas 读入）
        try:
            if isinstance(c, float) and math.isnan(c):
                bad += 1
                continue
        except Exception:
            pass

        try:
            out[int(j)] = int(c)
        except Exception:
            bad += 1
            continue

    if verbose and bad > 0:
        print(f"[WARN] task_shelf_mapping 中有 {bad} 条记录为 None/NaN/非法值，已忽略（这些任务将尝试从 shelf_seq 反推 chain）。")

    return out if out else None


def _build_chain_of_map(
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, Any]] = None,
) -> Dict[int, int]:
    """
    task -> chain(c)

    优先用 task_shelf_mapping（但要容错：None/NaN/非法会被忽略）
    对于 mapping 缺失的任务，再从 shelf_seq 反推补齐。
    """
    mp: Dict[int, int] = {}

    # 1) 先用清洗后的 mapping
    clean = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    if clean:
        # clean 已经是 int->int
        mp.update(clean)

    # 2) 再用 shelf_seq 反推补齐（只补 mp 中没有的任务）
    for c, seq in (shelf_seq or {}).items():
        try:
            cc = int(c)
        except Exception:
            continue
        for j in (seq or []):
            try:
                jj = int(j)
            except Exception:
                continue
            if jj not in mp:
                mp[jj] = cc

    return mp



def _shelf_index(shelf_seq: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    """
    shelf_idx[c][task] = position in shelf_seq[c]
    """
    idx: Dict[int, Dict[int, int]] = {}
    for c, seq in (shelf_seq or {}).items():
        idx[int(c)] = {int(t): p for p, t in enumerate(seq or [])}
    return idx


def _normalize_one_route_shelf_slots(
    seq: List[int],
    chain_of: Dict[int, int],
    shelf_idx: Dict[int, Dict[int, int]],
) -> List[int]:
    """
    ✅ 核心：让 route 内同一条 shelf 链的任务相对顺序 == shelf_seq[c]

    做法（slot 重排，不跨 slot 改动）：
      - 对每个 chain c：收集其任务在 seq 中出现的位置 slots
      - 把这些 slots 上的任务按 shelf_seq[c] 的顺序排序后放回去
    """
    seq = [int(x) for x in (seq or [])]
    if len(seq) <= 1:
        return seq

    positions: Dict[int, List[int]] = defaultdict(list)  # c -> indices
    tasks: Dict[int, List[int]] = defaultdict(list)      # c -> tasks

    for i, j in enumerate(seq):
        c = chain_of.get(int(j), None)
        if c is None:
            continue
        cc = int(c)
        positions[cc].append(int(i))
        tasks[cc].append(int(j))

    for c, pos_list in positions.items():
        order = shelf_idx.get(int(c), {})
        tlist = tasks.get(int(c), [])

        # 不在 shelf_seq 里的任务（理论上不该出现）放最后
        tlist.sort(key=lambda t: order.get(int(t), 10**9))

        for idx, t in zip(pos_list, tlist):
            seq[int(idx)] = int(t)

    return seq


def normalize_routes_by_shelf_seq_order(
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
) -> Dict[int, List[int]]:
    """
    ✅ 对所有 AGV routes 做 “shelf_seq ↔ v(由routes隐含) 顺序一致性” 修复
    """
    routes = _dc_routes(routes)
    shelf_seq = _dc_shelf_seq(shelf_seq)

    chain_of = _build_chain_of_map(shelf_seq, task_shelf_mapping)
    shelf_idx = _shelf_index(shelf_seq)

    for r in list(routes.keys()):
        routes[int(r)] = _normalize_one_route_shelf_slots(routes[int(r)], chain_of, shelf_idx)

    return routes


def _least_loaded_agv(routes: Dict[int, List[int]], R_ids: List[int]) -> int:
    return min((int(r) for r in R_ids), key=lambda rr: len(routes.get(rr, [])))

def _missing_tasks_in_routes(routes: Dict[int, List[int]], evaluator: RobustEvaluator) -> Set[int]:
    all_tasks = set(int(j) for j in evaluator.J)
    present: Set[int] = set()
    for seq in (routes or {}).values():
        present.update(int(x) for x in (seq or []))
    return all_tasks - present
def _ws_index(ws_fixed_seq: Dict[int, List[int]]) -> Dict[int, Dict[int, int]]:
    idx: Dict[int, Dict[int, int]] = {}
    for ws, seq in (ws_fixed_seq or {}).items():
        idx[int(ws)] = {int(t): p for p, t in enumerate(seq)}
    return idx

def _infeas_components(details, evaluator):
    penalties = (details.get("penalties", {}) if isinstance(details, dict) else {}) or {}

    unscheduled = len((details.get("unscheduled_tasks", []) if isinstance(details, dict) else []) or [])
    tail = 1 if float(penalties.get("tail_cell_conflict", 0.0) or 0.0) > 0.0 else 0

    cell_pairs = float(penalties.get("cell_conflict_pairs", 0.0) or 0.0)
    overlap = float(penalties.get("cell_conflict_overlap_time", 0.0) or 0.0)
    gap = max(float(getattr(evaluator, "cell_gap", 1.0)), 1e-6)

    cell_measure = cell_pairs + overlap / gap
    return int(unscheduled), int(tail), float(cell_measure)

def _infeas_key(details, evaluator):
    unscheduled, tail, cell_measure = _infeas_components(details, evaluator)
    # 量化到 0.01 精度，避免 float 比较抖动
    cell_score = int(round(cell_measure * 100))
    return (unscheduled, tail, cell_score)

def _infeas_scalar(details, evaluator):
    # 给“偶尔接受更差不可行度”用：尺度不要像 1e4 那么夸张
    unscheduled, tail, cell_measure = _infeas_components(details, evaluator)
    return float(unscheduled) * 1e4 + float(tail) * 1e2 + float(cell_measure)

def _normalize_ws_blocks(routes: Dict[int, List[int]], evaluator: RobustEvaluator) -> Dict[int, List[int]]:
    """
    对每条车路由，把“连续同一 WS 的块”按 fixed 顺序重排（只在相邻同 WS 内部排序，不跨 WS 调整）。
    """
    routes = _dc_routes(routes)
    pi = evaluator.pi
    ws_idx = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})

    def _reorder_one(seq: List[int]) -> List[int]:
        res: List[int] = []
        i = 0
        n = len(seq)
        while i < n:
            j = i
            ws_i = pi.get(int(seq[i]))
            block: List[int] = []
            while j < n and pi.get(int(seq[j])) == ws_i:
                block.append(int(seq[j]))
                j += 1
            if ws_i in ws_idx and len(block) > 1:
                order = ws_idx[ws_i]
                block.sort(key=lambda t: order.get(int(t), 10**9))
            res.extend(block)
            i = j
        return res

    for r in list(routes.keys()):
        routes[r] = _reorder_one(routes[r])
    return routes
def _ws_block_boundary_positions(seq: List[int], pi: Dict[int, int]) -> List[int]:
    """
    返回 route 上“WS 块边界”插入点：0、每次 WS 变化的位置、len(seq)
    """
    seq = [int(x) for x in seq]
    if not seq:
        return [0]
    pos = {0, len(seq)}
    for i in range(1, len(seq)):
        if pi.get(int(seq[i])) != pi.get(int(seq[i - 1])):
            pos.add(i)
    return sorted(pos)


def _normalize_one_route_ws_blocks(seq: List[int], evaluator: RobustEvaluator) -> List[int]:
    """
    只对单条 route 做“连续同一 WS 块内排序”，避免每次 normalize 全 routes。
    """
    seq = [int(x) for x in seq]
    pi = evaluator.pi
    ws_idx = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})

    res: List[int] = []
    i = 0
    n = len(seq)
    while i < n:
        ws_i = pi.get(int(seq[i]))
        j = i
        block: List[int] = []
        while j < n and pi.get(int(seq[j])) == ws_i:
            block.append(int(seq[j]))
            j += 1
        if ws_i in ws_idx and len(block) > 1:
            order = ws_idx[int(ws_i)]
            block.sort(key=lambda t: order.get(int(t), 10**9))
        res.extend(block)
        i = j
    return res
# ===== Rule feasibility prechecker (safe, no simulation) =====
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Any


def _as_int_set(obj: Any) -> Set[int]:
    """
    Robustly convert evaluator.S / evaluator.J / evaluator.R into Set[int].
    Supports list/set/range/dict(keys)/numpy arrays, etc.
    """
    if obj is None:
        return set()
    if isinstance(obj, dict):
        it = obj.keys()
    else:
        it = obj

    out: Set[int] = set()
    try:
        for x in it:
            out.add(int(x))
    except TypeError:
        return set()
    return out


def _as_int_int_dict(obj: Any) -> Dict[int, int]:
    """Convert mapping-like object into Dict[int, int]."""
    if obj is None:
        return {}
    try:
        items = obj.items()
    except Exception:
        return {}
    out: Dict[int, int] = {}
    for k, v in items:
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def _has_key(map_like: Any, key: Tuple[int, int]) -> bool:
    """Safe key existence check for dict/defaultdict/custom map-like objects."""
    if map_like is None:
        return False
    try:
        return key in map_like
    except Exception:
        try:
            _ = map_like[key]
            return True
        except Exception:
            return False


@dataclass
class RulePrechecker:
    """
    Necessary-condition pruning BEFORE calling evaluator.evaluate().
    Goal: remove candidates that are definitely invalid / will definitely crash due to missing distance keys.

    Notes:
    - This does NOT attempt time-feasibility (needs simulation).
    - This focuses on structure-only necessary conditions.
    """
    any_s: int
    J_set: Set[int]
    S_set: Set[int]

    d_s_s: Any
    d_s_pi: Any
    d_pi_s: Any

    agv_init: Dict[int, int]
    shelf_init: Dict[int, int]

    chain_of: Dict[int, int]                 # task -> chain
    pred_on_chain: Dict[int, Optional[int]]  # task -> prev task on same chain
    tail_task_of_chain: Dict[int, int]       # chain -> tail task

    blocked_cells: Set[int]

    def home_before(self, j: int, place: Dict[int, int]) -> Optional[int]:
        """
        Structure home_before:
          - if chain predecessor exists and has place -> place[prev]
          - else -> shelf_init[chain]
        If we cannot determine safely -> return None (then we skip hb-based pruning).
        """
        j = int(j)
        c = self.chain_of.get(j, None)
        if c is None:
            return None

        prev = self.pred_on_chain.get(j, None)
        if prev is not None:
            s_prev = place.get(int(prev), None)
            if s_prev is not None:
                s_prev = int(s_prev)
                if s_prev in self.S_set:
                    return s_prev

        hb = self.shelf_init.get(int(c), None)
        if hb is None:
            return None
        hb = int(hb)
        return hb if hb in self.S_set else None

    def agv_cell_before(self, r: int, base_seq: List[int], pos: int, place: Dict[int, int]) -> Optional[int]:
        """
        Structure AGV cell before inserting at position pos:
          - pos==0 -> agv_init[r]
          - pos>0  -> place[ base_seq[pos-1] ]
        """
        r = int(r)
        pos = int(pos)

        if pos <= 0:
            s0 = int(self.agv_init.get(r, self.any_s))
            return s0 if s0 in self.S_set else None

        if pos - 1 >= len(base_seq):
            return None

        prev_task = int(base_seq[pos - 1])
        s = place.get(prev_task, None)
        if s is None:
            return None
        s = int(s)
        return s if s in self.S_set else None

    def build_used_tail_cells(
        self,
        *,
        place: Dict[int, int],
        ignore_tasks: Optional[Set[int]] = None,
    ) -> Dict[int, int]:
        """
        used_tail_cells[cell] = chain
        ignore_tasks: exclude "not fixed yet" tasks to avoid false pruning in repair.
        """
        ignore_tasks = set(int(x) for x in (ignore_tasks or set()))
        used: Dict[int, int] = {}
        for c, tail in (self.tail_task_of_chain or {}).items():
            t = int(tail)
            if t in ignore_tasks:
                continue
            s = place.get(t, None)
            if s is None:
                continue
            s = int(s)
            if s in self.S_set:
                used[s] = int(c)
        return used

    def check_insert_candidate(
        self,
        *,
        j: int,
        r: int,
        pos: int,
        end_s: int,
        routes: Dict[int, List[int]],
        place: Dict[int, int],
        used_tail_cells: Optional[Dict[int, int]] = None,
        strict_move1: bool = False,
    ) -> Tuple[bool, str]:
        """
        Return (ok, reason). ok=True means candidate passes necessary checks.
        strict_move1 default False to avoid mis-pruning when normalize changes local order.
        """
        j = int(j); r = int(r); pos = int(pos); end_s = int(end_s)

        base_seq = routes.get(r, []) or []
        if pos < 0 or pos > len(base_seq):
            return False, "pos_out_of_range"
        if j in base_seq:
            return False, "task_already_in_route"

        if end_s not in self.S_set:
            return False, "end_s_not_in_S"
        if end_s in self.blocked_cells:
            return False, "end_s_blocked"

        # Tail uniqueness: only if j is the tail task of its chain
        c = self.chain_of.get(j, None)
        if c is not None:
            tail_j = self.tail_task_of_chain.get(int(c), None)
            if tail_j is not None and int(tail_j) == j and used_tail_cells is not None:
                owner = used_tail_cells.get(int(end_s), None)
                if owner is not None and int(owner) != int(c):
                    return False, "tail_cell_conflict"

        # Distance key j->end_s is always required by evaluator (move3)
        if not _has_key(self.d_pi_s, (int(j), int(end_s))):
            return False, "missing_d_pi_s"

        hb = self.home_before(j, place)
        if hb is not None:
            # hb->j is required (move2)
            if not _has_key(self.d_s_pi, (int(hb), int(j))):
                return False, "missing_d_s_pi"

            # Optional move1 key check: from_cell->hb
            if strict_move1:
                s_from = self.agv_cell_before(r, base_seq, pos, place)
                if s_from is None:
                    return False, "move1_from_cell_unavailable"
                if not _has_key(self.d_s_s, (int(s_from), int(hb))):
                    return False, "missing_d_s_s"
        else:
            # hb unknown: skip hb-based pruning to avoid false negatives
            pass

        return True, ""


def build_rule_prechecker(
    *,
    evaluator: Any,
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
) -> RulePrechecker:
    """
    Build RulePrechecker from evaluator + shelf_seq. Read-only, no simulation.
    """
    J_set = _as_int_set(getattr(evaluator, "J", None))
    S_set = _as_int_set(getattr(evaluator, "S", None))
    any_s = min(S_set) if S_set else 0

    d_s_s = getattr(evaluator, "d_s_s", None)
    d_s_pi = getattr(evaluator, "d_s_pi", None)
    d_pi_s = getattr(evaluator, "d_pi_s", None)

    agv_init = _as_int_int_dict(getattr(evaluator, "agv_init", {}) or {})

    shelf_init_src = shelf_init_override if shelf_init_override is not None else (getattr(evaluator, "shelf_init", {}) or {})
    shelf_init = _as_int_int_dict(shelf_init_src)

    # chain structure from shelf_seq
    chain_of: Dict[int, int] = {}
    pred_on_chain: Dict[int, Optional[int]] = {}
    tail_task_of_chain: Dict[int, int] = {}

    used_chains: Set[int] = set()

    for c, seq in (shelf_seq or {}).items():
        cc = int(c)
        prev = None
        cleaned: List[int] = []
        for x in (seq or []):
            jj = int(x)
            if J_set and (jj not in J_set):
                continue
            cleaned.append(jj)
            chain_of[jj] = cc
            pred_on_chain[jj] = prev
            prev = jj
        if cleaned:
            tail_task_of_chain[cc] = int(cleaned[-1])
            used_chains.add(cc)

    # fallback chain info from task_shelf_mapping
    if task_shelf_mapping:
        for j, c in task_shelf_mapping.items():
            if c is None:
                continue
            jj = int(j); cc = int(c)
            if J_set and (jj not in J_set):
                continue
            used_chains.add(cc)
            if jj not in chain_of:
                chain_of[jj] = cc
            if jj not in pred_on_chain:
                pred_on_chain[jj] = None

    # blocked_cells: shelves with NO tasks at all occupy init cell forever
    blocked_cells: Set[int] = set()
    for c, init_cell in (shelf_init or {}).items():
        cc = int(c)
        if cc in used_chains:
            continue
        s0 = int(init_cell)
        if s0 in S_set:
            blocked_cells.add(s0)

    return RulePrechecker(
        any_s=int(any_s),
        J_set=J_set,
        S_set=S_set,
        d_s_s=d_s_s,
        d_s_pi=d_s_pi,
        d_pi_s=d_pi_s,
        agv_init=agv_init,
        shelf_init=shelf_init,
        chain_of=chain_of,
        pred_on_chain=pred_on_chain,
        tail_task_of_chain=tail_task_of_chain,
        blocked_cells=blocked_cells,
    )
class FastProxyEvaluatorLB:
    """
    Level-1 代理：计算“无等待/无资源冲突/无鲁棒膨胀”的名义完成时间下界(LB)。
    用途：
      - 安全剪枝：若 LB >= 基准 obj，则该候选不可能改进（0 误杀）
      - 候选排序：LB 越小越优先进入精评
    注意：为保证“剪枝安全”，缺失 key 不允许当大数参与剪枝；缺失 key 应由 RulePrechecker 拦截。
    这里缺失 key 统一当 0（更乐观 → 更安全的下界），最多会让排序弱一点，不会误剪。
    """
    def __init__(self, evaluator: RobustEvaluator, pre: RulePrechecker):
        self.pre = pre

        self.d_s_s = getattr(evaluator, "d_s_s", None)
        self.d_s_pi = getattr(evaluator, "d_s_pi", None)
        self.d_pi_s = getattr(evaluator, "d_pi_s", None)

        self.D = getattr(evaluator, "D", {}) or {}
        self.D_setup = float(getattr(evaluator, "D_setup", 0.0))

    def _get0(self, mp: Any, key: Tuple[int, int]) -> float:
        if mp is None:
            return 0.0
        try:
            return float(mp.get((int(key[0]), int(key[1])), 0.0))
        except Exception:
            try:
                return float(mp[(int(key[0]), int(key[1]))])
            except Exception:
                return 0.0

    def _proc(self, j: int) -> float:
        j = int(j)
        try:
            return float(self.D_setup) + float(self.D.get(j, 0.0))
        except Exception:
            return float(self.D_setup)

    def route_lb(
        self,
        *,
        r: int,
        seq: List[int],
        place: Dict[int, int],
        stop_at: Optional[float] = None,
    ) -> float:
        """
        单车名义完成时间下界（无等待）。
        stop_at：若超过阈值可提前返回（用于剪枝加速）
        """
        r = int(r)
        seq = [int(x) for x in (seq or [])]
        place_i = {int(k): int(v) for k, v in (place or {}).items()}

        cur_cell = int(self.pre.agv_init.get(r, self.pre.any_s))
        t = 0.0
        bound = None if stop_at is None else float(stop_at)

        for j in seq:
            j = int(j)
            end_s = place_i.get(j, None)
            if end_s is None:
                # 下界取 0 更安全（不剪枝误杀）
                return 0.0
            end_s = int(end_s)

            hb = self.pre.home_before(j, place_i)
            hb = int(hb) if hb is not None else int(self.pre.any_s)

            t += self._get0(self.d_s_s, (cur_cell, hb))
            t += self._get0(self.d_s_pi, (hb, j))
            t += self._proc(j)
            t += self._get0(self.d_pi_s, (j, end_s))

            cur_cell = end_s
            if bound is not None and t >= bound - 1e-9:
                return float(t)

        return float(t)

    def solution_lb(
        self,
        *,
        routes: Dict[int, List[int]],
        place: Dict[int, int],
        stop_at: Optional[float] = None,
    ) -> float:
        """
        解的下界：max_r route_lb(r)
        """
        mx = 0.0
        bound = None if stop_at is None else float(stop_at)
        for r, seq in (routes or {}).items():
            v = self.route_lb(r=int(r), seq=seq, place=place, stop_at=bound)
            if v > mx:
                mx = float(v)
                if bound is not None and mx >= bound - 1e-9:
                    return mx
        return float(mx)

def normalize_solution_by_ws_rank_once(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    rng: random.Random,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
    global_try_tasks: int = 2,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    同步重排（破局用）：
      - 对每个任务 j 定义 ws 归一化排名 rank = pos_in_ws / len(ws_seq)
      - routes 内按 rank 排序（稳定排序）
      - shelf_seq 内按 rank 排序（稳定排序）
      - 然后跑一轮 place_tune 适配新顺序
    返回：(routes_new, shelf_seq_new, place_new, improved, obj_new)
    """
    routes0 = _dc_routes(routes)
    shelf0 = _dc_shelf_seq(shelf_seq)
    place0 = _dc_place(place)

    routes0 = _normalize_ws_blocks(routes0, evaluator)
    base_obj, _ = evaluator.evaluate(routes0, shelf0, place0)
    base_obj = float(base_obj)

    ws_fixed = getattr(evaluator, "ws_fixed_seq", {}) or {}
    ws_pos = _ws_index(ws_fixed)
    ws_len = {int(ws): max(1, len(seq)) for ws, seq in ws_fixed.items()}

    def rank(j: int) -> float:
        j = int(j)
        ws = evaluator.pi.get(j, None)
        if ws is None:
            return 1e9
        ws = int(ws)
        pos = ws_pos.get(ws, {}).get(j, 10**9)
        den = ws_len.get(ws, 1)
        return float(pos) / float(max(1, den))

    # --- 1) routes 同步重排 ---
    routes_new = _dc_routes(routes0)
    for r, seq in routes_new.items():
        seq2 = [int(x) for x in seq]
        # 稳定：加原索引做 tie-break
        tagged = [(idx, j) for idx, j in enumerate(seq2)]
        tagged.sort(key=lambda it: (rank(it[1]), int(evaluator.pi.get(it[1], -1)), it[0]))
        routes_new[int(r)] = [j for _, j in tagged]
    routes_new = _normalize_ws_blocks(routes_new, evaluator)

    # --- 2) shelf_seq 同步重排 ---
    shelf_new = _dc_shelf_seq(shelf0)
    for c, seq in shelf_new.items():
        seq2 = [int(x) for x in seq]
        tagged = [(idx, j) for idx, j in enumerate(seq2)]
        tagged.sort(key=lambda it: (rank(it[1]), int(evaluator.pi.get(it[1], -1)), it[0]))
        shelf_new[int(c)] = [j for _, j in tagged]

    # --- 3) 轻量 place tune 适配 ---
    place_new, _ = local_place_tune_once(
        routes=routes_new,
        shelf_seq=shelf_new,
        place=place0,
        evaluator=evaluator,
        S_near_by_j=S_near_by_j,
        rng=rng,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
        top_k_try=6,
        global_try_tasks=int(global_try_tasks),
    )

    obj_new, _ = evaluator.evaluate(routes_new, shelf_new, place_new)
    obj_new = float(obj_new)

    improved = (obj_new < base_obj - 1e-9)
    return routes_new, shelf_new, place_new, improved, obj_new


def _relabel_best_routes(
    routes: Dict[int, List[int]],
    *,
    evaluator: RobustEvaluator,
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    max_exact_agv: int = 6,
) -> Tuple[Dict[int, List[int]], float]:
    """
    不改变各条“路线内任务顺序”、place 与 shelf_seq 的前提下，
    枚举 AGV 身份的全排列，选择 makespan 最小的映射（把整条路线换给更合适的车）。
    """
    routes = _dc_routes(routes)
    R_ids = sorted(int(r) for r in evaluator.R)
    seqs = [routes.get(r, []) for r in R_ids]

    if len(R_ids) > max_exact_agv:
        obj, _ = evaluator.evaluate(routes, shelf_seq, place)
        return routes, float(obj)

    best_obj = float("inf")
    best_routes = routes
    for perm in permutations(R_ids):
        cand_routes = {int(perm[i]): list(seqs[i]) for i in range(len(R_ids))}
        cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
        obj, _ = evaluator.evaluate(cand_routes, shelf_seq, place)
        if obj < best_obj - 1e-9:
            best_obj = float(obj)
            best_routes = cand_routes
    return best_routes, float(best_obj)


def _prev_on_chain(
    j: int,
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
) -> Optional[int]:
    """返回 j 在其货架链上的前驱任务（若 j 在链首或缺少映射，则返回 None）"""
    if not task_shelf_mapping:
        return None
    c = int(task_shelf_mapping.get(int(j), -1))
    if c not in shelf_seq:
        return None
    seq = shelf_seq[c]
    try:
        idx = seq.index(int(j))
    except ValueError:
        return None
    if idx <= 0:
        return None
    return int(seq[idx - 1])


def _cand_end_shelves_for_task(
    j: int,
    *,
    place: Dict[int, int],
    S_near_by_j: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]] = None,
    must_include_prev_or_init: bool = True,
    max_keep: Optional[int] = None,
) -> List[int]:
    """
    构造 j 的回库位候选：最近 m 个 + {链前驱 end_s 或货架初始位} + {现有 place[j]（若有）}
    按 d_pi_s(j,s) 升序去重；若 max_keep 给出则截断。
    """
    j = int(j)
    cands: Set[int] = set(int(s) for s in S_near_by_j.get(j, []))

    if must_include_prev_or_init and task_shelf_mapping:
        c = int(task_shelf_mapping.get(j, -1))
        shelf_init_map = shelf_init_override if shelf_init_override is not None else evaluator.shelf_init
        if c in shelf_init_map:
            j_prev = _prev_on_chain(j, shelf_seq, task_shelf_mapping)
            if j_prev is not None and int(j_prev) in place:
                cands.add(int(place[int(j_prev)]))
            else:
                cands.add(int(shelf_init_map[c]))

    if j in place:
        cands.add(int(place[j]))

    if not cands:
        cands.add(min(int(s) for s in evaluator.S))

    def _dist(ss: int) -> float:
        return float(evaluator.d_pi_s.get((int(j), int(ss)), 1e9))

    cands_sorted = sorted(cands, key=_dist)
    if max_keep is not None and len(cands_sorted) > max_keep:
        return cands_sorted[:max_keep]
    return cands_sorted


def _augment_shelf_candidates(
    j: int,
    base_cands: List[int],
    *,
    evaluator: RobustEvaluator,
    rng: random.Random,
    force_all_shelves: bool = False,
    extra_random_shelves: int = 0,
    cap_total: int = 40,   # ★关键：总候选封顶
) -> List[int]:
    """
    在 base_cands 上补充一些“远点货位”以增强探索，但永远封顶 cap_total，避免 |S| 大时爆炸。
    """
    cand_set: Set[int] = set(int(s) for s in base_cands)
    all_s = [int(s) for s in evaluator.S]

    if force_all_shelves:
        # 不再直接使用全部 S，而是“补足到 cap_total”
        if len(cand_set) < cap_total:
            pool = [s for s in all_s if s not in cand_set]
            need = cap_total - len(cand_set)
            if pool and need > 0:
                cand_set.update(int(x) for x in rng.sample(pool, min(need, len(pool))))
    else:
        if extra_random_shelves > 0 and len(cand_set) < cap_total:
            pool = [s for s in all_s if s not in cand_set]
            need = min(int(extra_random_shelves), cap_total - len(cand_set))
            if pool and need > 0:
                cand_set.update(int(x) for x in rng.sample(pool, min(need, len(pool))))

    def _dist(ss: int) -> float:
        return float(evaluator.d_pi_s.get((int(j), int(ss)), 1e9))

    return sorted(cand_set, key=_dist)



def _topk_keep(
    best_list: List[Tuple[float, int, int, int]],
    cand: Tuple[float, int, int, int],
    k: int,
) -> None:
    """维护一个容量为 k 的“当前最优候选集”（小顶意义），用替换最差的方式。"""
    k = max(1, int(k))
    if len(best_list) < k:
        best_list.append(cand)
        return
    worst_i = max(range(len(best_list)), key=lambda i: best_list[i][0])
    if cand[0] < best_list[worst_i][0] - 1e-9:
        best_list[worst_i] = cand


def _choose_from_topk(
    best_list: List[Tuple[float, int, int, int]],
    rng: random.Random,
    k: int,
) -> Optional[Tuple[float, int, int, int]]:
    if not best_list:
        return None
    best_list.sort(key=lambda x: x[0])
    kk = min(max(1, int(k)), len(best_list))
    return rng.choice(best_list[:kk])


# =========================
#      Destroy operators
# =========================
def destroy_ws_gap_route_bundle(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    min_k: int = 2,
    max_k: int = 4,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    先找“最大 WS idle gap”的后一个任务 cur_t，
    再在 cur_t 所在 AGV 的 route 上取一个偏向前缀的小段（把拖它变晚的前置任务一起删掉）。
    返回 removed_order 时保证 cur_t 排第一（先插回，利于抢占前面位置/换车）。
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    ws_fixed = getattr(evaluator, "ws_fixed_seq", None)
    if not ws_fixed:
        return [], set(), routes

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    _, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    p = (diag.get("p", {}) if isinstance(diag, dict) else {}) or {}
    q = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}

    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return [], set(), routes

    # 1) 找最大 idle gap 的后一个任务 best_cur
    best_gap = -1.0
    best_cur = None
    for ws, full_seq in ws_fixed.items():
        seq = [int(t) for t in full_seq if int(t) in present and int(t) in p and int(t) in q]
        if len(seq) <= 1:
            continue
        for i in range(1, len(seq)):
            prev_t = seq[i - 1]
            cur_t = seq[i]
            gap = float(p[cur_t]) - float(q[prev_t])
            if gap > best_gap + 1e-9:
                best_gap = gap
                best_cur = int(cur_t)

    if best_cur is None:
        return [], set(), routes

    # 2) 找 best_cur 所在车辆与位置
    r_hit, idx = None, None
    for r, seq in routes.items():
        if best_cur in seq:
            r_hit = int(r)
            idx = seq.index(best_cur)
            break
    if r_hit is None or idx is None:
        return [], set(), routes

    seq_hit = routes[r_hit]
    L = len(seq_hit)
    if L <= 0:
        return [], set(), routes

    max_k_eff = min(int(max_k), L)
    k = max(int(min_k), 1)
    k = min(k, max_k_eff)

    # 段尽量“以 best_cur 结尾”，把其前置任务带上（例如把 Task1 一起删掉）
    start = max(0, int(idx) - (k - 1))
    start = min(start, L - k)
    segment = [int(x) for x in seq_hit[start : start + k]]

    removed_order = [int(best_cur)] + [int(x) for x in segment if int(x) != int(best_cur)]
    removed = set(int(x) for x in removed_order)

    for r in list(routes.keys()):
        routes[r] = [j for j in routes[r] if int(j) not in removed]

    return removed_order, removed, routes


def intensify_close_largest_ws_gap_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    rng: random.Random,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
    early_pos_max: int = 2,
    max_place_try_each: int = 8,
    extra_random_shelves: int = 2,
    force_all_shelves: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    针对“当前最大 WS idle gap”的后一个任务 cur_t：
    尝试把它换车/前移到各车 route 的靠前位置（pos=0..early_pos_max），找到就收（first-improvement 的加强版）。
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    ws_fixed = getattr(evaluator, "ws_fixed_seq", None)
    if not ws_fixed:
        obj, _ = evaluator.evaluate(_normalize_ws_blocks(routes, evaluator), shelf_seq, place)
        return routes, place, False, float(obj)

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    p = (diag.get("p", {}) if isinstance(diag, dict) else {}) or {}
    q = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}

    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return routes, place, False, float(base_obj)

    best_gap = -1.0
    best_cur = None
    for ws, full_seq in ws_fixed.items():
        seq = [int(t) for t in full_seq if int(t) in present and int(t) in p and int(t) in q]
        if len(seq) <= 1:
            continue
        for i in range(1, len(seq)):
            prev_t = seq[i - 1]
            cur_t = seq[i]
            gap = float(p[cur_t]) - float(q[prev_t])
            if gap > best_gap + 1e-9:
                best_gap = gap
                best_cur = int(cur_t)

    if best_cur is None:
        return routes, place, False, float(base_obj)

    j_move = int(best_cur)

    # 从原车上移除 j_move
    routes_removed = _dc_routes(routes)
    for r in list(routes_removed.keys()):
        routes_removed[r] = [x for x in routes_removed[r] if int(x) != j_move]

    # 候选回库位
    cand_s = _cand_end_shelves_for_task(
        j_move,
        place=place,
        S_near_by_j=S_near_by_j,
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
        must_include_prev_or_init=True,
        max_keep=max_place_try_each,
    )
    cand_s = _augment_shelf_candidates(
        j_move,
        cand_s,
        evaluator=evaluator,
        rng=rng,
        force_all_shelves=force_all_shelves,
        extra_random_shelves=extra_random_shelves,
    )

    best_obj = float(base_obj)
    best_routes, best_place = routes, place
    improved = False

    R_ids = sorted(int(r) for r in evaluator.R)
    for r_to in R_ids:
        seq_to = routes_removed.get(int(r_to), [])
        max_pos = min(int(early_pos_max), len(seq_to))
        for pos in range(max_pos + 1):
            for s in cand_s:
                cand_routes = _dc_routes(routes_removed)
                cand_place = _dc_place(place)
                cand_routes.setdefault(int(r_to), [])
                cand_routes[int(r_to)].insert(int(pos), j_move)
                cand_place[j_move] = int(s)

                cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
                obj, _ = evaluator.evaluate(cand_routes, shelf_seq, cand_place)
                obj = float(obj)
                if obj < best_obj - 1e-9:
                    best_obj = obj
                    best_routes = cand_routes
                    best_place = cand_place
                    improved = True

    return best_routes, best_place, improved, float(best_obj)

def destroy_random(
    routes: Dict[int, List[int]],
    remove_frac: float,
    rng: random.Random,
) -> Tuple[Set[int], Dict[int, List[int]]]:
    """随机移除一定比例任务（不改 shelf_seq / place）"""
    routes = _dc_routes(routes)
    all_tasks: List[int] = []
    for seq in routes.values():
        all_tasks.extend(seq)
    all_tasks = sorted(set(all_tasks))
    if not all_tasks:
        return set(), routes

    k = max(1, int(len(all_tasks) * float(remove_frac)))
    k = min(k, len(all_tasks))
    removed = set(rng.sample(all_tasks, k))

    for r in list(routes.keys()):
        routes[r] = [j for j in routes[r] if j not in removed]
    return removed, routes

def destroy_ws_largest_idle_gap(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    min_k: int = 2,
    max_k: int = 5,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    找到“某个 WS 上相邻两任务之间最大的 idle gap”，删掉包含这对任务的窗口。
    目标：专打像你这次 WS2: Task2 -> Task4 之间的巨大空档。
    返回：(removed_order, removed_set, routes_removed)
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    ws_fixed = getattr(evaluator, "ws_fixed_seq", None)
    if not ws_fixed:
        return [], set(), routes

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    _, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    p = (diag.get("p", {}) if isinstance(diag, dict) else {}) or {}
    q = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}

    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return [], set(), routes

    best_ws = None
    best_idx = None
    best_gap = -1.0
    best_seq = None

    for ws, full_seq in ws_fixed.items():
        seq = [int(t) for t in full_seq if int(t) in present and int(t) in p and int(t) in q]
        if len(seq) <= 1:
            continue
        for i in range(1, len(seq)):
            prev_t = seq[i - 1]
            cur_t = seq[i]
            gap = float(p[cur_t]) - float(q[prev_t])
            if gap > best_gap + 1e-9:
                best_gap = gap
                best_ws = int(ws)
                best_idx = i
                best_seq = seq

    if best_ws is None or best_seq is None or best_idx is None:
        return [], set(), routes

    L = len(best_seq)
    max_k_eff = min(int(max_k), L)
    k = int(rng.randint(int(min_k), int(max_k_eff)))

    # 窗口必须覆盖 (best_idx-1, best_idx)
    left = max(0, best_idx - 1 - (k // 2))
    left = min(left, L - k)
    removed_order = best_seq[left : left + k]

    removed = set(int(x) for x in removed_order)
    for r in list(routes.keys()):
        routes[r] = [j for j in routes[r] if int(j) not in removed]

    return removed_order, removed, routes
def destroy_robust_sensitive_bundle(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    # 同车前置段长度（删 seed 前面的任务，专打“拖慢它”的前缀）
    pre_k: int = 2,
    # 同链邻居半径（左右各取 chain_nei 个）
    chain_nei: int = 1,
    # 同 WS 邻居半径（左右各取 ws_nei 个）
    ws_nei: int = 1,
    # 总删除规模控制
    min_k: int = 2,
    max_k: int = 5,
    # 从“鲁棒增量最大”的前 top_m 个任务里挑 seed（带随机性）
    top_m: int = 8,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    鲁棒敏感 destroy（只加一个算子就能让 ALNS 更“懂鲁棒性”）：

    信号：对每个任务 j 计算 robust_slack[j] = q[j] - q0[j] （即 q[G] - q[0]）
    直觉：robust_slack 大的任务，是“最坏情形膨胀最厉害/最敏感”的任务。

    删除策略（围绕 seed）：
      - seed 本身
      - seed 所在 AGV 路线的前置段（pre_k 个），专打“拖慢 seed 的前缀”
      - seed 同货架链的邻居（前/后）
      - seed 同工位 WS 的邻居（固定序列中前/后）
    然后截断/补齐到 [min_k, max_k]。

    返回：
      removed_order: 保证 seed 在第一位（repair 时先插回 seed，利于“抢前”）
      removed_set
      routes_removed（只从 routes 中删任务，不改 shelf_seq/place）
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    # present tasks
    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return [], set(), routes

    # 为了拿 q/q0（鲁棒增量），我们只做一次 evaluate（成本很低）
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    _, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    q_raw = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    q0_raw = (diag.get("q0", {}) if isinstance(diag, dict) else {}) or {}

    # 统一 key=int
    try:
        q_map = {int(k): float(v) for k, v in q_raw.items()}
    except Exception:
        q_map = {}
    try:
        q0_map = {int(k): float(v) for k, v in q0_raw.items()}
    except Exception:
        q0_map = {}

    # 计算鲁棒增量 slack = q - q0
    slack: Dict[int, float] = {}
    for j in present:
        if j in q_map:
            base0 = q0_map.get(j, q_map[j])  # 若缺 q0，则 slack=0
            slack[j] = float(q_map[j] - float(base0))

    # 如果 slack 全是 0（极端情况），就退化为“按 q 最大挑瓶颈”
    if (not slack) or (max(slack.values()) <= 1e-9):
        slack = {j: float(q_map.get(j, 0.0)) for j in present if j in q_map}

    # 兜底：如果连 q 都没有（通常是 allow_incomplete 造成），就随机挑 seed
    if not slack:
        seed = int(rng.choice(sorted(present)))
        slack = {seed: 0.0}
    else:
        ranked = sorted(slack.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: max(1, int(top_m))]
        seed = int(rng.choice([j for j, _ in top]))

    # ========== 收集邻居 ==========
    prio: List[int] = [seed]

    # (1) 同车前置段：删 seed 前面 pre_k 个任务
    r_hit = None
    idx_hit = None
    seq_hit: List[int] = []
    for r, seq in routes_norm.items():
        if seed in seq:
            r_hit = int(r)
            idx_hit = int(seq.index(seed))
            seq_hit = [int(x) for x in seq]
            break
    if r_hit is not None and idx_hit is not None and seq_hit:
        start = max(0, idx_hit - max(0, int(pre_k)))
        prefix = [int(x) for x in seq_hit[start:idx_hit]]
        # 离 seed 最近的前置任务优先
        prefix = list(reversed(prefix))
        for x in prefix:
            if x in present:
                prio.append(int(x))

    # (2) 同链邻居：左右各 chain_nei 个
    if task_shelf_mapping is not None:
        c = task_shelf_mapping.get(int(seed), None)
        if c is not None:
            cc = int(c)
            seq_c = [int(x) for x in (shelf_seq.get(cc, []) or [])]
            try:
                pos = int(seq_c.index(int(seed)))
                left = seq_c[max(0, pos - int(chain_nei)):pos]
                right = seq_c[pos + 1: pos + 1 + int(chain_nei)]
                # 近的优先：先左边从近到远，再右边从近到远
                left = list(reversed(left))
                for x in left + right:
                    if int(x) in present:
                        prio.append(int(x))
            except ValueError:
                pass

    # (3) 同 WS 邻居：固定序列里左右各 ws_nei 个
    ws_fixed = getattr(evaluator, "ws_fixed_seq", {}) or {}
    ws = evaluator.pi.get(int(seed), None)
    if ws is not None:
        ws = int(ws)
        seq_ws = [int(x) for x in (ws_fixed.get(ws, []) or [])]
        try:
            pos = int(seq_ws.index(int(seed)))
            left = seq_ws[max(0, pos - int(ws_nei)):pos]
            right = seq_ws[pos + 1: pos + 1 + int(ws_nei)]
            left = list(reversed(left))
            for x in left + right:
                if int(x) in present:
                    prio.append(int(x))
        except ValueError:
            pass

    # ========== 组装 removed_set（截断/补齐） ==========
    min_k = max(1, int(min_k))
    max_k = max(min_k, int(max_k))

    removed: List[int] = []
    seen: Set[int] = set()
    for x in prio:
        xx = int(x)
        if xx in present and xx not in seen:
            removed.append(xx)
            seen.add(xx)
        if len(removed) >= max_k:
            break

    # 不够 min_k：用 slack 排名补齐
    if len(removed) < min_k:
        ranked_fill = sorted(slack.items(), key=lambda kv: kv[1], reverse=True)
        for j, _ in ranked_fill:
            jj = int(j)
            if jj in present and jj not in seen:
                removed.append(jj)
                seen.add(jj)
            if len(removed) >= min_k:
                break

    removed_set = set(int(x) for x in removed)

    # removed_order：保证 seed 第一，其余按 prio 顺序补齐，再补漏（按 slack 降序）
    removed_order: List[int] = [int(seed)]
    for x in prio:
        xx = int(x)
        if xx in removed_set and xx not in removed_order:
            removed_order.append(xx)
    rest = [x for x in removed_set if x not in removed_order]
    rest.sort(key=lambda j: float(slack.get(int(j), 0.0)), reverse=True)
    removed_order.extend(rest)

    # 从 routes 里删掉 removed_set
    for r in list(routes.keys()):
        routes[int(r)] = [int(j) for j in routes[int(r)] if int(j) not in removed_set]

    return removed_order, removed_set, routes

def _relatedness(
    a: int,
    b: int,
    *,
    evaluator: RobustEvaluator,
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
) -> float:
    """Shaw相关度：同链/同工位/WS顺序邻近/回库位接近（越大越相关）"""
    a = int(a)
    b = int(b)
    same_chain = (
        1.0
        if (
            task_shelf_mapping
            and int(task_shelf_mapping.get(a, -999)) == int(task_shelf_mapping.get(b, -998))
        )
        else 0.0
    )
    same_ws = 1.0 if int(evaluator.pi.get(a)) == int(evaluator.pi.get(b)) else 0.0

    idx_ws: Dict[int, Dict[int, int]] = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})
    ord_a = idx_ws.get(int(evaluator.pi.get(a, -1)), {}).get(a, None)
    ord_b = idx_ws.get(int(evaluator.pi.get(b, -1)), {}).get(b, None)
    near_ws = 0.0
    if ord_a is not None and ord_b is not None:
        near_ws = 1.0 / (1.0 + abs(ord_a - ord_b))

    s_a = place.get(a, None)
    s_b = place.get(b, None)
    s_near = 0.0
    if s_a is not None and s_b is not None:
        dist = float(evaluator.d_s_s.get((int(s_a), int(s_b)), 0.0))
        s_near = 1.0 / (1.0 + dist)

    return 1.0 * same_chain + 0.6 * same_ws + 0.5 * near_ws + 0.4 * s_near


def destroy_shaw_related(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    rng: random.Random,
    remove_frac: float,
) -> Tuple[Set[int], Dict[int, List[int]]]:
    """Shaw/Related destroy：以“相关度”成簇删除"""
    routes = _dc_routes(routes)
    place = _dc_place(place)

    pool: List[int] = []
    for seq in routes.values():
        pool.extend(int(x) for x in seq)
    pool = sorted(set(pool))
    if not pool:
        return set(), routes

    k = max(1, int(len(pool) * float(remove_frac)))
    k = min(k, len(pool))

    seed = int(rng.choice(pool))
    scores: List[Tuple[float, int]] = []
    for j in pool:
        if int(j) == seed:
            continue  # 避免 seed 重复
        s = _relatedness(
            seed,
            j,
            evaluator=evaluator,
            place=place,
            shelf_seq=shelf_seq,
            task_shelf_mapping=task_shelf_mapping,
        )
        scores.append((float(s), int(j)))

    scores.sort(reverse=True)
    removed_list = [seed] + [j for _, j in scores[: max(0, k - 1)]]
    removed = set(int(x) for x in removed_list)

    for r in list(routes.keys()):
        routes[r] = [j for j in routes[r] if int(j) not in removed]
    return removed, routes


def destroy_chain(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    rng: random.Random,
) -> Tuple[Set[int], Dict[int, List[int]]]:
    """随机抽一个货架链，把该链上的任务从所有车路由里删除"""
    routes = _dc_routes(routes)
    if not shelf_seq:
        return set(), routes
    c = int(rng.choice(list(shelf_seq.keys())))
    chain_set = set(int(j) for j in shelf_seq.get(int(c), []))

    found: Set[int] = set()
    for r in list(routes.keys()):
        keep: List[int] = []
        for j in routes[r]:
            if int(j) in chain_set:
                found.add(int(j))
            else:
                keep.append(int(j))
        routes[r] = keep
    return found, routes


def destroy_ws_critical_window(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    min_k: int = 2,
    max_k: int = 5,  # ★默认加大到 5
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    WS-focused destroy：
      1) 找当前 makespan 对应的关键任务 j*
      2) 在其工位 ws 的 fixed 序列中抽一个“包含 j* 的窗口”(长度k)
      3) 删除窗口内任务
    返回：(removed_order, removed_set, routes_removed)
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return [], set(), routes

    ws_fixed = getattr(evaluator, "ws_fixed_seq", None)
    if not ws_fixed:
        return [], set(), routes

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    _, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    q = diag.get("q", {}) if isinstance(diag, dict) else {}

    # ★只在 present 内选关键任务，避免 dummy / 缺 pi 的任务导致失败
    q_present = {int(j): float(v) for j, v in (q or {}).items() if int(j) in present}
    if q_present:
        j_star = int(max(q_present.keys(), key=lambda jj: q_present[jj]))
    else:
        j_star = int(rng.choice(sorted(present)))

    ws_star = int(evaluator.pi.get(j_star, -1))
    ws_seq_full = [int(t) for t in ws_fixed.get(ws_star, [])]
    ws_seq = [t for t in ws_seq_full if t in present]
    if not ws_seq:
        return [], set(), routes

    L = len(ws_seq)
    if L <= min_k:
        removed_order = ws_seq[:]  # 全删
    else:
        max_k_eff = min(int(max_k), L)
        k = int(rng.randint(int(min_k), int(max_k_eff)))

        idx = ws_seq.index(j_star) if j_star in ws_seq else (L - 1)
        start_low = max(0, idx - (k - 1))
        start_high = min(idx, L - k)
        start = int(rng.randint(start_low, start_high))
        removed_order = ws_seq[start : start + k]

    removed = set(int(x) for x in removed_order)
    for r in list(routes.keys()):
        routes[r] = [j for j in routes[r] if int(j) not in removed]
    return removed_order, removed, routes


# =========================
#          Repair
# =========================
def repair_greedy_insert_with_place(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    removed: Set[int],
    rng: random.Random,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
    max_place_try_each: int = 6,
    top_k_random: int = 1,
    extra_random_shelves: int = 0,
    force_all_shelves: bool = False,
    max_agv_candidates: Optional[int] = None,
    max_pos_per_route: Optional[int] = None,

    # ✅ 评估预算
    max_evals_per_task: Optional[int] = None,
    max_evals_total: Optional[int] = None,
    copy_inputs: bool = True,

    # ✅ cheap 预筛选（核心）
    preselect_m: int = 12,
    preselect_per_pos_shelves: int = 2,

    # ✅ 新增：如果提供 removed_order，则严格按该顺序插回（seed-first）
    removed_order: Optional[List[int]] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:

    if copy_inputs:
        routes = _dc_routes(routes)
        place = _dc_place(place)
    else:
        routes = {int(r): [int(j) for j in (seq or [])] for r, seq in routes.items()}
        place = {int(j): int(s) for j, s in place.items()}

    R_ids = sorted(int(r) for r in evaluator.R)
    for r in R_ids:
        routes.setdefault(int(r), [])

    removed_set = set(int(x) for x in (removed or set()))

    ws_fixed = getattr(evaluator, "ws_fixed_seq", {}) or {}
    ws_pos = _ws_index(ws_fixed)
    pi = getattr(evaluator, "pi", {}) or {}

    # ✅ shelf_seq ↔ v 一致性：一次性构建（后面复用）
    chain_of = _build_chain_of_map(shelf_seq, task_shelf_mapping)
    shelf_idx = _shelf_index(shelf_seq)

    def _ws_rank(j: int) -> int:
        ws = int(pi.get(int(j), -1))
        return int(ws_pos.get(ws, {}).get(int(j), 10**9))

    # --------------------------
    # 1) 生成插回顺序 removed_list
    # --------------------------
    if removed_order is not None:
        # 严格按给定顺序，但要去重 & 与 removed_set 对齐
        base_set = removed_set if removed_set else set(int(x) for x in removed_order)
        seen = set()
        ordered = []
        for x in removed_order:
            xx = int(x)
            if xx in base_set and xx not in seen:
                ordered.append(xx)
                seen.add(xx)

        # 如果 removed_set 有额外任务（理论上不该发生），补到末尾（按 ws 排）
        rest = [int(x) for x in base_set if int(x) not in seen]
        rng.shuffle(rest)
        rest.sort(key=lambda j: (_ws_rank(j), rng.random()))
        removed_list = ordered + rest
    else:
        removed_list = [int(x) for x in removed_set]
        rng.shuffle(removed_list)
        removed_list.sort(key=lambda j: (_ws_rank(j), rng.random()))

    if not removed_list:
        return routes, place

    # ===== prechecker（一次构建）=====
    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )

    # succ_on_chain：cheap 里可轻微考虑链后继
    succ_on_chain: Dict[int, int] = {}
    for t, prev in (pre.pred_on_chain or {}).items():
        if prev is not None:
            succ_on_chain[int(prev)] = int(t)

    # remaining_unfixed：当前 routes 中缺失的任务集合（用于 tail 冲突忽略）
    all_tasks: Set[int] = set(int(x) for x in evaluator.J)
    present_tasks: Set[int] = set()
    for seq in routes.values():
        present_tasks.update(int(x) for x in (seq or []))
    remaining_unfixed: Set[int] = (all_tasks - present_tasks) | set(int(x) for x in removed_list)

    # 距离引用（局部变量更快）
    d_s_s = getattr(evaluator, "d_s_s", {}) or {}
    d_s_pi = getattr(evaluator, "d_s_pi", {}) or {}
    d_pi_s = getattr(evaluator, "d_pi_s", {}) or {}

    def _dist(map_like: Any, key: Tuple[int, int], default: float = 1e9) -> float:
        try:
            return float(map_like.get((int(key[0]), int(key[1])), default))
        except Exception:
            try:
                return float(map_like[(int(key[0]), int(key[1]))])
            except Exception:
                return float(default)

    def _pick_agv_candidates() -> List[int]:
        if max_agv_candidates is None:
            return R_ids[:]
        k = max(1, int(max_agv_candidates))
        if k >= len(R_ids):
            return R_ids[:]
        by_len = sorted(R_ids, key=lambda rr: len(routes.get(rr, [])))
        cand = by_len[:k]
        pool = [r for r in R_ids if r not in cand]
        if pool:
            cand.append(int(rng.choice(pool)))
        out, seen = [], set()
        for r in cand:
            if r not in seen:
                out.append(r)
                seen.add(r)
        return out

    def _cheap_insert_score(
        *,
        j: int,
        r: int,
        pos: int,
        end_s: int,
        base_seq: List[int],
    ) -> float:
        """
        cheap 分数：越小越好。只用局部距离结构估计增量，不做仿真。
        """
        j = int(j); r = int(r); pos = int(pos); end_s = int(end_s)

        hb = pre.home_before(j, place)
        hb = int(hb) if hb is not None else int(pre.any_s)

        from_cell = pre.agv_cell_before(r, base_seq, pos, place)
        from_cell = int(from_cell) if from_cell is not None else int(pre.any_s)

        hb_next = None
        if 0 <= pos < len(base_seq):
            nxt = int(base_seq[pos])
            hb_next = pre.home_before(nxt, place)

        score = 0.0
        score += _dist(d_s_s, (from_cell, hb), 1e9)
        score += _dist(d_s_pi, (hb, j), 1e9)
        score += _dist(d_pi_s, (j, end_s), 1e9)

        if hb_next is not None:
            hb_next = int(hb_next)
            score += _dist(d_s_s, (end_s, hb_next), 1e9) - _dist(d_s_s, (from_cell, hb_next), 1e9)

        succ = succ_on_chain.get(int(j), None)
        if succ is not None:
            score += 0.20 * _dist(d_s_pi, (end_s, int(succ)), 1e9)

        score += 0.05 * float(len(base_seq))
        return float(score)

    eval_used_total = 0

    for j in removed_list:
        j = int(j)
        eval_used_task = 0

        # 每个 j 重算一次“已固定 tail 占用”
        used_tail = pre.build_used_tail_cells(place=place, ignore_tasks=remaining_unfixed)

        cand_shelves = _cand_end_shelves_for_task(
            j,
            place=place,
            S_near_by_j=S_near_by_j,
            evaluator=evaluator,
            shelf_seq=shelf_seq,
            task_shelf_mapping=task_shelf_mapping,
            shelf_init_override=shelf_init_override,
            must_include_prev_or_init=True,
            max_keep=max_place_try_each,
        )
        cand_shelves = _augment_shelf_candidates(
            j,
            cand_shelves,
            evaluator=evaluator,
            rng=rng,
            force_all_shelves=force_all_shelves,
            extra_random_shelves=extra_random_shelves,
            cap_total=40,
        )

        agv_cands = _pick_agv_candidates()

        # ===== 1) 规则剪枝 + cheap 预筛选 =====
        # cheap_pool: (cheap, r, pos, s, seq_ins)
        cheap_pool: List[Tuple[float, int, int, int, List[int]]] = []

        for r in agv_cands:
            r = int(r)
            base_seq = routes[r]
            pos_list = _ws_block_boundary_positions(base_seq, evaluator.pi)

            if max_pos_per_route is not None and len(pos_list) > int(max_pos_per_route):
                must = [0, len(base_seq)]
                mid = [p for p in pos_list if p not in must]
                need = max(0, int(max_pos_per_route) - len(must))
                pick = rng.sample(mid, min(need, len(mid))) if (need > 0 and mid) else []
                pos_list = sorted(set(must + pick))

            seen_route_keys: Set[Tuple[int, Tuple[int, ...]]] = set()

            for pos in pos_list:
                pos = int(pos)

                seq_ins = base_seq[:]
                seq_ins.insert(pos, j)

                # ✅ route 内统一规范化（WS块 + shelf slot）
                seq_ins = _normalize_one_route_ws_and_shelf(
                    seq_ins,
                    evaluator=evaluator,
                    chain_of=chain_of,
                    shelf_idx=shelf_idx,
                )

                key = (r, tuple(seq_ins))
                if key in seen_route_keys:
                    continue
                seen_route_keys.add(key)

                # 规则剪枝：过滤 end_s
                ok_s: List[int] = []
                for s in cand_shelves:
                    ss = int(s)
                    ok, _ = pre.check_insert_candidate(
                        j=j,
                        r=r,
                        pos=pos,
                        end_s=ss,
                        routes=routes,      # base routes（不含 j）
                        place=place,
                        used_tail_cells=used_tail,
                        strict_move1=False, # 仍保持保守
                    )
                    if ok:
                        ok_s.append(ss)

                if not ok_s:
                    continue

                scored = [(_cheap_insert_score(j=j, r=r, pos=pos, end_s=ss, base_seq=base_seq), ss) for ss in ok_s]
                scored.sort(key=lambda x: x[0])

                keep_k = max(1, int(preselect_per_pos_shelves))
                for c, ss in scored[:keep_k]:
                    cheap_pool.append((float(c), int(r), int(pos), int(ss), list(seq_ins)))

        if not cheap_pool:
            # 兜底：最短车尾插
            r_fb = _least_loaded_agv(routes, R_ids)
            routes[int(r_fb)].append(j)
            routes[int(r_fb)] = _normalize_one_route_ws_and_shelf(
                routes[int(r_fb)],
                evaluator=evaluator,
                chain_of=chain_of,
                shelf_idx=shelf_idx,
            )
            place[j] = int(cand_shelves[0]) if cand_shelves else min(int(ss) for ss in evaluator.S)
            remaining_unfixed.discard(int(j))
            continue

        # 全局保留 cheap 最好的前 preselect_m 个
        cheap_pool.sort(key=lambda x: x[0])
        cheap_pool = cheap_pool[: max(1, int(preselect_m))]

        # ===== 2) 精评少量候选（safe evaluate）=====
        scored_cands: List[Tuple[Tuple[int, int, int], float, int, int, int, List[int]]] = []
        # (infeas_key, obj, r, pos, s, seq_ins)

        for _, rr, pp, ss, seq_ins in cheap_pool:
            if (max_evals_total is not None) and (eval_used_total >= int(max_evals_total)):
                break
            if (max_evals_per_task is not None) and (eval_used_task >= int(max_evals_per_task)):
                break

            old_seq = routes[rr]
            old_place = place.get(j, None)

            routes[rr] = seq_ins
            place[j] = int(ss)
            try:
                obj_f, det = _safe_evaluate(evaluator, routes, shelf_seq, place)
                key = _infeas_key(det, evaluator)
                scored_cands.append((tuple(key), float(obj_f), int(rr), int(pp), int(ss), list(seq_ins)))

                eval_used_task += 1
                eval_used_total += 1
            finally:
                routes[rr] = old_seq
                if old_place is None:
                    place.pop(j, None)
                else:
                    place[j] = int(old_place)

        if not scored_cands:
            # 没有精评候选：直接用 cheap 最好者落地（不再额外 evaluate）
            _, rr, pp, ss, seq_ins = cheap_pool[0]
            routes[int(rr)] = list(seq_ins)
            place[j] = int(ss)
        else:
            # ✅ infeas_key 优先，再按 obj
            scored_cands.sort(key=lambda x: (x[0], x[1]))
            kpick = min(max(1, int(top_k_random)), len(scored_cands))
            _, _, rr, pp, ss, seq_ins = rng.choice(scored_cands[:kpick])

            routes[int(rr)] = list(seq_ins)
            place[j] = int(ss)

        remaining_unfixed.discard(int(j))

    return routes, place

def repair_greedy_insert_with_place_ordered(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    removed_order: List[int],
    rng: random.Random,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
    max_place_try_each: int = 6,
    top_k_random: int = 1,
    extra_random_shelves: int = 0,
    force_all_shelves: bool = False,
    max_agv_candidates: Optional[int] = None,
    max_pos_per_route: Optional[int] = None,

    # ✅ 评估预算（跨任务共享）
    max_evals_per_task: Optional[int] = None,
    max_evals_total: Optional[int] = None,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Ordered repair：严格按 removed_order 插回（seed-first）。
    现在改成“单次 repair”，共享缓存与总预算，速度会明显提升。
    """
    removed_order = [int(x) for x in (removed_order or [])]
    removed_set = set(int(x) for x in removed_order)

    return repair_greedy_insert_with_place(
        routes=routes,
        shelf_seq=shelf_seq,
        place=place,
        evaluator=evaluator,
        removed=removed_set,
        removed_order=removed_order,  # ✅ 核心：顺序由这里控制
        rng=rng,
        S_near_by_j=S_near_by_j,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
        max_place_try_each=max_place_try_each,
        top_k_random=top_k_random,
        extra_random_shelves=extra_random_shelves,
        force_all_shelves=force_all_shelves,
        max_agv_candidates=max_agv_candidates,
        max_pos_per_route=max_pos_per_route,
        max_evals_per_task=max_evals_per_task,
        max_evals_total=max_evals_total,
        copy_inputs=True,
    )

# =========================
#        Local moves
# =========================
def local_place_tune_once(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    rng: random.Random,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
    top_k_try: int = 4,
    global_try_tasks: int = 0,

    # ✅ 新增：总预算 + 候选封顶（防爆）
    max_evals: Optional[int] = 200,
    cap_total: int = 60,
) -> Tuple[Dict[int, int], bool]:
    """
    place 微调（防爆版）：
      - 常规任务：只试近邻 top_k_try
      - 关键任务：不再“全 S”，而是补足采样到 cap_total
      - 总评估次数受 max_evals 控制
    """
    improved = False
    place = _dc_place(place)

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    eval_used = 0
    budget = None if max_evals is None else max(0, int(max_evals))

    keys = [int(j) for j in place.keys()]

    # --- 选关键任务：按 q 最大 ---
    global_set: Set[int] = set()
    q_map: Dict[int, float] = {}
    if isinstance(diag, dict):
        q_raw = diag.get("q", {}) or {}
        for j in keys:
            if int(j) in q_raw:
                q_map[int(j)] = float(q_raw[int(j)])

    if global_try_tasks and keys:
        if q_map:
            ranked = sorted(q_map.items(), key=lambda kv: kv[1], reverse=True)
            k = min(int(global_try_tasks), len(ranked))
            global_set = set(int(j) for j, _ in ranked[:k])
        else:
            global_set = set(rng.sample(keys, min(int(global_try_tasks), len(keys))))

    # 遍历顺序：关键任务优先，其余随机
    order: List[int] = []
    if global_set:
        if q_map:
            ranked_keys = [j for j, _ in sorted(q_map.items(), key=lambda kv: kv[1], reverse=True)]
            order.extend([j for j in ranked_keys if j in global_set])
        else:
            order.extend(sorted(global_set))

    rest = [j for j in keys if j not in global_set]
    rng.shuffle(rest)
    order.extend(rest)

    for j in order:
        j = int(j)
        s_now = int(place[j])

        if budget is not None and budget <= 0:
            break

        # 构造候选
        if j in global_set:
            # 关键任务：采样补足到 cap_total（不全 S）
            cands_base = _cand_end_shelves_for_task(
                int(j),
                place=place,
                S_near_by_j=S_near_by_j,
                evaluator=evaluator,
                shelf_seq=shelf_seq,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init_override,
                must_include_prev_or_init=True,
                max_keep=max(8, int(top_k_try) * 2),
            )
            cands = _augment_shelf_candidates(
                int(j),
                cands_base,
                evaluator=evaluator,
                rng=rng,
                force_all_shelves=True,
                extra_random_shelves=0,
                cap_total=int(cap_total),
            )
        else:
            cands_full = _cand_end_shelves_for_task(
                int(j),
                place=place,
                S_near_by_j=S_near_by_j,
                evaluator=evaluator,
                shelf_seq=shelf_seq,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init_override,
                must_include_prev_or_init=True,
            )
            cands = cands_full[: max(1, int(top_k_try))]

        if s_now not in cands:
            cands.append(s_now)

        best_s = s_now
        best_obj = float(base_obj)

        for s in cands:
            s = int(s)
            if s == s_now:
                continue

            if budget is not None and budget <= 0:
                break

            cand_place = _dc_place(place)
            cand_place[j] = s
            obj, _ = evaluator.evaluate(routes_norm, shelf_seq, cand_place)
            eval_used += 1
            if budget is not None:
                budget -= 1

            obj = float(obj)
            if obj < best_obj - 1e-9:
                best_obj = obj
                best_s = s

        if best_s != s_now:
            place[j] = best_s
            base_obj = best_obj
            improved = True

    return place, improved


def intensify_shelf_seq_promote_critical_ws_once(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 12,
) -> Tuple[Dict[int, List[int]], bool, float]:
    """
    强化算子：识别当前解的“关键任务” j*（q 最大），取其工位 ws* 作为关键工位。
    然后尝试在每条货架链内，把 ws* 上的任务整体往前推（并用 ws_fixed_seq 的顺序作为二级排序）。
    若能改善 makespan（甚至从 inf -> finite），就接受。

    返回：(new_shelf_seq, improved, new_obj)
    """
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    q_map = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    if not q_map:
        return shelf_seq, False, base_obj

    # 关键任务（makespan 任务）
    j_star = int(max(q_map.keys(), key=lambda jj: float(q_map[jj])))
    ws_star = evaluator.pi.get(j_star, None)
    if ws_star is None:
        return shelf_seq, False, base_obj
    ws_star = int(ws_star)

    ws_idx = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})

    def _key(j: int, orig_pos: int) -> Tuple[int, int, int]:
        j = int(j)
        ws = evaluator.pi.get(j, None)
        if ws is None:
            return (2, 10**9, orig_pos)
        ws = int(ws)
        # 关键工位优先
        pri = 0 if ws == ws_star else 1
        # 在各自工位内部按固定顺序
        ord_in_ws = ws_idx.get(ws, {}).get(j, 10**9)
        return (pri, int(ord_in_ws), orig_pos)

    # 方案 A：对每条链做一次“关键工位优先”的稳定排序
    candA = _dc_shelf_seq(shelf_seq)
    for c, seq in candA.items():
        seq2 = list(seq)
        tagged = [(idx, int(j)) for idx, j in enumerate(seq2)]
        tagged.sort(key=lambda it: _key(it[1], it[0]))
        candA[int(c)] = [j for _, j in tagged]

    objA, _ = evaluator.evaluate(routes_norm, candA, place)
    objA = float(objA)
    if objA < base_obj - 1e-9:
        return candA, True, objA

    # 方案 B：如果整体排序没用，再做若干次“单点前移”试探（更温柔）
    best_seq = shelf_seq
    best_obj = base_obj

    # 收集“包含关键工位任务、且该任务不在链首”的链
    shelves = []
    for c, seq in shelf_seq.items():
        seq2 = [int(x) for x in seq]
        ok = False
        for idx, j in enumerate(seq2):
            if idx > 0 and int(evaluator.pi.get(j, -1)) == ws_star:
                ok = True
                break
        if ok:
            shelves.append(int(c))

    if not shelves:
        return shelf_seq, False, base_obj

    for _ in range(max(1, int(max_trials))):
        c = int(rng.choice(shelves))
        seq = [int(x) for x in shelf_seq[int(c)]]
        # 找一个关键工位任务（不在首位）
        cand_pos = [i for i, j in enumerate(seq) if i > 0 and int(evaluator.pi.get(j, -1)) == ws_star]
        if not cand_pos:
            continue
        i = int(rng.choice(cand_pos))
        j = int(seq[i])
        # 把它前移到更靠前的位置（0..i-1）
        new_pos = int(rng.randrange(0, i))
        if new_pos == i:
            continue
        seq_new = seq[:]
        seq_new.pop(i)
        seq_new.insert(new_pos, j)

        candB = _dc_shelf_seq(shelf_seq)
        candB[int(c)] = seq_new

        objB, _ = evaluator.evaluate(routes_norm, candB, place)
        objB = float(objB)
        if objB < best_obj - 1e-9:
            best_obj = objB
            best_seq = candB

    if best_obj < base_obj - 1e-9:
        return best_seq, True, best_obj
    return shelf_seq, False, base_obj


def local_shelf_seq_relocate_once(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
    task_shelf_mapping: Dict[int, int],
    rng: random.Random,
) -> Tuple[Dict[int, List[int]], bool]:
    """
    同一货架链上的单点重定位（first-improvement）：
    固定 routes / place，只在 shelf_seq 上做邻域搜索。

    SPEED FIX:
      - routes 不变：routes_norm 只算一次并复用（原版本在内层重复 normalize 是纯开销）
      - cand_shelf_seq 用 shallow copy（dict(...)）即可，避免每次深拷贝整个 shelf_seq
    """
    shelf_seq = _dc_shelf_seq(shelf_seq)

    # ✅ routes 不变：只 normalize 一次
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    shelves = list(shelf_seq.keys())
    rng.shuffle(shelves)

    for c in shelves:
        c = int(c)
        seq = list(shelf_seq[c])
        L = len(seq)
        if L <= 1:
            continue

        idx_order = list(range(L))
        rng.shuffle(idx_order)

        for idx in idx_order:
            j = int(seq[idx])
            if int(task_shelf_mapping.get(j, -1)) != int(c):
                continue

            for pos in range(L):
                if pos == idx:
                    continue

                cand = seq[:]
                cand.pop(idx)
                cand.insert(pos, j)

                # ✅ shallow copy 足够：只替换一条链
                cand_shelf_seq = dict(shelf_seq)
                cand_shelf_seq[int(c)] = cand

                obj, _ = evaluator.evaluate(routes_norm, cand_shelf_seq, place)
                if float(obj) < base_obj - 1e-9:
                    shelf_seq[int(c)] = cand
                    return shelf_seq, True

    return shelf_seq, False



# =========================
#    Cross / Intra moves
# =========================
def cross_vehicle_move_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]],
    rng: random.Random,
    try_swap: bool = True,
    max_trials: int = 3,
    max_pos_samples: int = 8,
    max_s_samples: int = 5,
    max_evals: int = 10,  # ★最多精评多少个候选（建议 8~12）
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    Funnel cross-vehicle:
      Generate candidates (no evaluate)
        -> RulePrecheck
        -> Proxy-LB 排序/剪枝
        -> 只 evaluate Top-K
    """

    routes = _dc_routes(routes)
    place = _dc_place(place)

    R_ids = sorted(int(r) for r in evaluator.R)
    for r in R_ids:
        routes.setdefault(int(r), [])

    # 基准：规范化一次
    routes = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes, shelf_seq, place)
    base_obj = float(base_obj)

    nonempty = [r for r in R_ids if routes.get(r)]
    if len(R_ids) < 2 or not nonempty:
        return routes, place, False, base_obj

    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )
    proxy = FastProxyEvaluatorLB(evaluator, pre)

    def _sample_positions(seq: List[int]) -> List[int]:
        L = len(seq)
        all_pos = list(range(L + 1))
        k = max(2, int(max_pos_samples))
        if k >= len(all_pos):
            return all_pos

        must = set(_ws_block_boundary_positions(seq, evaluator.pi))
        must.add(0); must.add(L)
        must_list = sorted(must)

        if len(must_list) >= k:
            ends = [0, L] if L != 0 else [0]
            mids = [p for p in must_list if p not in set(ends)]
            need = max(0, k - len(ends))
            pick = rng.sample(mids, min(need, len(mids))) if (need > 0 and mids) else []
            return sorted(set(ends + pick))

        pool = [p for p in all_pos if p not in must]
        need = k - len(must_list)
        pick = rng.sample(pool, min(need, len(pool))) if (need > 0 and pool) else []
        return sorted(set(must_list + pick))

    def _top_shelves(j: int) -> List[int]:
        cands = _cand_end_shelves_for_task(
            int(j),
            place=place,
            S_near_by_j=S_near_by_j,
            evaluator=evaluator,
            shelf_seq=shelf_seq,
            task_shelf_mapping=task_shelf_mapping,
            shelf_init_override=shelf_init_override,
            must_include_prev_or_init=True,
            max_keep=max(1, int(max_s_samples) * 2),
        )
        cands = _augment_shelf_candidates(
            int(j),
            cands,
            evaluator=evaluator,
            rng=rng,
            force_all_shelves=False,
            extra_random_shelves=0,
            cap_total=max(8, int(max_s_samples) + 2),
        )
        if not cands:
            cands = [min(int(s) for s in evaluator.S)]
        # 再截断
        if len(cands) > int(max_s_samples):
            head = cands[: int(max_s_samples)]
            tail = cands[int(max_s_samples):]
            if tail:
                head.append(int(rng.choice(tail)))
            cands = head
        # 去重
        out, seen = [], set()
        for s in cands:
            s = int(s)
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    # 候选池： (lb, cand_routes, cand_place)
    cand_pool: List[Tuple[float, Dict[int, List[int]], Dict[int, int]]] = []

    # ========== 1) relocate 候选 ==========
    for _ in range(max(1, int(max_trials))):
        r_from = int(rng.choice(nonempty))
        r_to = int(rng.choice([r for r in R_ids if r != r_from]))
        if not routes.get(r_from):
            continue

        j = int(rng.choice(routes[r_from]))
        s_cands = _top_shelves(j)
        pos_cands = _sample_positions(routes[r_to])

        seq_from_base = [x for x in routes[r_from] if int(x) != int(j)]
        seq_from_base = _normalize_one_route_ws_blocks(seq_from_base, evaluator)

        used_tail = pre.build_used_tail_cells(place=place, ignore_tasks={j})

        for pos in pos_cands:
            pos = int(pos)
            seq_to_new = list(routes[r_to])
            seq_to_new.insert(pos, int(j))
            seq_to_new = _normalize_one_route_ws_blocks(seq_to_new, evaluator)

            for s in s_cands:
                s = int(s)

                ok, _ = pre.check_insert_candidate(
                    j=j,
                    r=r_to,
                    pos=pos,
                    end_s=s,
                    routes=routes,   # base routes（不含 j）
                    place=place,
                    used_tail_cells=used_tail,
                    strict_move1=False,
                )
                if not ok:
                    continue

                cand_routes = _dc_routes(routes)
                cand_routes[r_from] = seq_from_base
                cand_routes[r_to] = seq_to_new

                cand_place = _dc_place(place)
                cand_place[j] = s

                # 安全剪枝：LB >= base_obj => 不可能改进
                lb = proxy.solution_lb(routes=cand_routes, place=cand_place, stop_at=base_obj if math.isfinite(base_obj) else None)
                if math.isfinite(base_obj) and lb >= base_obj - 1e-9:
                    continue

                cand_pool.append((float(lb), cand_routes, cand_place))

    # ========== 2) swap 候选 ==========
    if try_swap and len(nonempty) >= 2:
        for _ in range(max(1, int(max_trials))):
            r1, r2 = rng.sample(nonempty, 2)
            r1 = int(r1); r2 = int(r2)
            if not routes[r1] or not routes[r2]:
                continue

            a = int(rng.choice(routes[r1]))
            b = int(rng.choice(routes[r2]))
            pos_a = int(routes[r1].index(a))
            pos_b = int(routes[r2].index(b))

            seq1 = list(routes[r1]); seq2 = list(routes[r2])
            seq1[pos_a] = int(b)
            seq2[pos_b] = int(a)
            seq1 = _normalize_one_route_ws_blocks(seq1, evaluator)
            seq2 = _normalize_one_route_ws_blocks(seq2, evaluator)

            sA = _top_shelves(a)
            sB = _top_shelves(b)
            used_tail = pre.build_used_tail_cells(place=place, ignore_tasks={a, b})

            for sa in sA:
                for sb in sB:
                    sa = int(sa); sb = int(sb)

                    okA, _ = pre.check_insert_candidate(
                        j=a, r=r1, pos=pos_a, end_s=sa,
                        routes=routes, place=place, used_tail_cells=used_tail,
                        strict_move1=False,
                    )
                    okB, _ = pre.check_insert_candidate(
                        j=b, r=r2, pos=pos_b, end_s=sb,
                        routes=routes, place=place, used_tail_cells=used_tail,
                        strict_move1=False,
                    )
                    if not (okA and okB):
                        continue

                    cand_routes = _dc_routes(routes)
                    cand_routes[r1] = seq1
                    cand_routes[r2] = seq2
                    cand_place = _dc_place(place)
                    cand_place[a] = sa
                    cand_place[b] = sb

                    lb = proxy.solution_lb(routes=cand_routes, place=cand_place, stop_at=base_obj if math.isfinite(base_obj) else None)
                    if math.isfinite(base_obj) and lb >= base_obj - 1e-9:
                        continue

                    cand_pool.append((float(lb), cand_routes, cand_place))

    if not cand_pool:
        return routes, place, False, base_obj

    cand_pool.sort(key=lambda x: x[0])

    # ★强制封顶，防止你外面忘记调参
    eval_k = min(max(1, int(max_evals)), 12, len(cand_pool))

    for i in range(eval_k):
        _, cand_routes, cand_place = cand_pool[i]
        # cand_routes 已保持 route-level normalize，无需全量 normalize
        obj, _ = evaluator.evaluate(cand_routes, shelf_seq, cand_place)
        obj = float(obj)
        if obj < base_obj - 1e-9:
            return cand_routes, cand_place, True, obj

    return routes, place, False, base_obj


def cross_vehicle_block_move_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]],
    rng: random.Random,
    max_block: int = 2,
    max_trials: int = 2,
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """跨车段块移动（长度 1~2）"""
    routes = _dc_routes(routes)
    place = _dc_place(place)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)

    R_ids = sorted(int(r) for r in evaluator.R)
    nonempty = [r for r in R_ids if routes.get(r)]
    if len(R_ids) < 2 or not nonempty:
        return routes, place, False, float(base_obj)

    for _ in range(max_trials):
        r_from = int(rng.choice(nonempty))
        seq_from = routes[r_from]
        if not seq_from:
            continue
        r_to = int(rng.choice([r for r in R_ids if r != r_from]))
        for k in [2, 1]:
            if k > int(max_block) or len(seq_from) < k:
                continue
            i = int(rng.randrange(0, len(seq_from) - k + 1))
            block = seq_from[i : i + k]
            rest_from = seq_from[:i] + seq_from[i + k :]

            for pos in range(len(routes[r_to]) + 1):
                cand_routes = _dc_routes(routes)
                cand_place = _dc_place(place)
                cand_routes[r_from] = rest_from
                cand_routes[r_to] = cand_routes[r_to][:pos] + block + cand_routes[r_to][pos:]

                # 给 block 内任务一个合理初值回库位
                for j in block:
                    cands = _cand_end_shelves_for_task(
                        int(j),
                        place=cand_place,
                        S_near_by_j=S_near_by_j,
                        evaluator=evaluator,
                        shelf_seq=shelf_seq,
                        task_shelf_mapping=task_shelf_mapping,
                        shelf_init_override=shelf_init_override,
                        must_include_prev_or_init=True,
                        max_keep=6,
                    )
                    if cands:
                        cand_place[int(j)] = int(cands[0])

                cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
                obj, _ = evaluator.evaluate(cand_routes, shelf_seq, cand_place)
                if obj < base_obj - 1e-9:
                    return cand_routes, cand_place, True, float(obj)

    return routes, place, False, float(base_obj)


def intra_two_opt_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 2,
) -> Tuple[Dict[int, List[int]], bool, float]:
    """车内 2-opt：随机选一车，反转一段，若改善则返回"""
    routes = _dc_routes(routes)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)

    R_ids = sorted(int(r) for r in evaluator.R)
    cand_rs = [r for r in R_ids if len(routes.get(r, [])) >= 4]
    if not cand_rs:
        return routes, False, float(base_obj)

    for _ in range(max_trials):
        r = int(rng.choice(cand_rs))
        seq = routes[r]
        L = len(seq)
        i = int(rng.randrange(0, L - 2))
        j = int(rng.randrange(i + 2, L))
        cand_routes = _dc_routes(routes)
        cand_routes[r] = seq[:i] + list(reversed(seq[i : j + 1])) + seq[j + 1 :]
        cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
        obj, _ = evaluator.evaluate(cand_routes, shelf_seq, place)
        if obj < base_obj - 1e-9:
            return cand_routes, True, float(obj)

    return routes, False, float(base_obj)


def intra_or_opt_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 2,
    max_block: int = 2,
) -> Tuple[Dict[int, List[int]], bool, float]:
    """车内 Or-opt：取长度 k(1/2) 的连续块在同车内重插，若改善则返回"""
    routes = _dc_routes(routes)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)

    R_ids = sorted(int(r) for r in evaluator.R)
    cand_rs = [r for r in R_ids if len(routes.get(r, [])) >= 3]
    if not cand_rs:
        return routes, False, float(base_obj)

    for _ in range(max_trials):
        r = int(rng.choice(cand_rs))
        seq = routes[r]
        L = len(seq)
        k = 1 if int(max_block) < 2 else int(rng.choice([1, min(2, L - 1)]))
        if L <= k:
            continue
        i = int(rng.randrange(0, L - k + 1))
        block = seq[i : i + k]
        rest = seq[:i] + seq[i + k :]
        for pos in range(len(rest) + 1):
            if pos == i:
                continue
            cand_routes = _dc_routes(routes)
            cand_routes[r] = rest[:pos] + block + rest[pos:]
            cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
            obj, _ = evaluator.evaluate(cand_routes, shelf_seq, place)
            if obj < base_obj - 1e-9:
                return cand_routes, True, float(obj)

    return routes, False, float(base_obj)


def intensify_critical_tail_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]],
    tail_k: int = 2,
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """瓶颈车末尾任务尝试移走，改善 Cmax"""
    routes = _dc_routes(routes)
    place = _dc_place(place)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    q = diag.get("q", {}) if isinstance(diag, dict) else {}
    best_r = None
    best_q = -1.0
    for r, seq in routes.items():
        end_q = max((float(q.get(int(j), 0.0)) for j in seq), default=0.0)
        if end_q > best_q:
            best_q = float(end_q)
            best_r = int(r)

    if best_r is None or not routes.get(best_r):
        return routes, place, False, float(base_obj)

    seq = routes[best_r]
    tail = seq[-min(int(tail_k), len(seq)) :]
    R_ids = sorted(int(r) for r in evaluator.R)

    improved_any = False
    best_routes = routes
    best_place = place
    best_obj = float(base_obj)

    for j in reversed(tail):
        for r_to in R_ids:
            if int(r_to) == int(best_r):
                continue
            for pos in range(len(routes[r_to]) + 1):
                cand_routes = _dc_routes(routes)
                cand_place = _dc_place(place)
                cand_routes[best_r] = [x for x in cand_routes[best_r] if int(x) != int(j)]
                cand_routes[r_to].insert(pos, int(j))

                # 给 j 一个合理回库位
                cands = _cand_end_shelves_for_task(
                    int(j),
                    place=cand_place,
                    S_near_by_j=S_near_by_j,
                    evaluator=evaluator,
                    shelf_seq=shelf_seq,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init_override,
                    must_include_prev_or_init=True,
                    max_keep=8,
                )
                if cands:
                    cand_place[int(j)] = int(cands[0])

                cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
                obj, _ = evaluator.evaluate(cand_routes, shelf_seq, cand_place)
                if obj < best_obj - 1e-9:
                    best_obj = float(obj)
                    best_routes = cand_routes
                    best_place = cand_place
                    improved_any = True

    return best_routes, best_place, improved_any, float(best_obj)


# =========================
#        2-opt*
# =========================
def cross_vehicle_2opt_star_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 12,
    allow_non_improving: bool = False,  # ★shake 模式：允许非改进解
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    跨两车 2-opt*：随机选两辆车，在各自路线随机切一刀，交换“尾段”。
    - allow_non_improving=False：只返回改进解（否则返回原解）
    - allow_non_improving=True ：返回试验中目标最小的那条（即便不比原来好，让 SA 决定）
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)

    R_ids = sorted(int(r) for r in evaluator.R)
    cand_rs = [r for r in R_ids if len(routes.get(r, [])) >= 1]
    if len(cand_rs) < 2:
        return routes, place, False, float(base_obj)

    if allow_non_improving:
        best_obj = float("inf")
        best_routes = None
    else:
        best_obj = float(base_obj)
        best_routes = routes

    for _ in range(max(1, int(max_trials))):
        r1, r2 = rng.sample(cand_rs, 2)
        r1, r2 = int(r1), int(r2)
        seq1 = routes[r1]
        seq2 = routes[r2]

        i = int(rng.randrange(0, len(seq1) + 1))
        j = int(rng.randrange(0, len(seq2) + 1))
        if i == len(seq1) and j == len(seq2):
            continue

        cand_routes = _dc_routes(routes)
        cand_routes[r1] = seq1[:i] + seq2[j:]
        cand_routes[r2] = seq2[:j] + seq1[i:]

        cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
        obj, _ = evaluator.evaluate(cand_routes, shelf_seq, place)

        if float(obj) < best_obj - 1e-9:
            best_obj = float(obj)
            best_routes = cand_routes

    if best_routes is None:
        return routes, place, False, float(base_obj)

    improved = (best_obj < float(base_obj) - 1e-9)
    if (not allow_non_improving) and (not improved):
        return routes, place, False, float(base_obj)
    return best_routes, place, improved, float(best_obj)

class AdaptiveOpPool:
    """
    ALNS 自适应算子权重池：
      - roulette wheel 选择
      - segment-based 权重更新：w = (1-r)*w + r*(avg_score)
    """
    def __init__(
        self,
        names: List[str],
        init_w: Optional[Dict[str, float]] = None,
        *,
        reaction: float = 0.2,
        segment_len: int = 50,
        w_min: float = 0.05,
        w_max: float = 50.0,
    ):
        self.names = [str(x) for x in names]
        self.w = {n: float((init_w or {}).get(n, 1.0)) for n in self.names}
        self.reaction = float(reaction)
        self.segment_len = int(segment_len)
        self.w_min = float(w_min)
        self.w_max = float(w_max)

        self._seg_score = defaultdict(float)
        self._seg_cnt = defaultdict(int)

    def pick(self, rng: random.Random, candidates: List[str]) -> str:
        cand = [c for c in candidates if c in self.w]
        if not cand:
            # 防御：不该发生
            return str(rng.choice(self.names))

        total = 0.0
        weights = []
        for n in cand:
            ww = max(self.w_min, float(self.w.get(n, 1.0)))
            weights.append((n, ww))
            total += ww

        u = rng.random() * total
        acc = 0.0
        for n, ww in weights:
            acc += ww
            if acc >= u:
                return n
        return weights[-1][0]

    def record(self, name: str, score: float) -> None:
        if name not in self.w:
            return
        self._seg_score[name] += float(score)
        self._seg_cnt[name] += 1

    def maybe_update(self, it: int) -> None:
        if (int(it) + 1) % int(self.segment_len) != 0:
            return

        r = float(self.reaction)
        for n in self.names:
            cnt = int(self._seg_cnt.get(n, 0))
            if cnt <= 0:
                continue
            avg = float(self._seg_score.get(n, 0.0)) / float(cnt)
            new_w = (1.0 - r) * float(self.w[n]) + r * avg
            new_w = max(self.w_min, min(self.w_max, float(new_w)))
            self.w[n] = float(new_w)

        self._seg_score = defaultdict(float)
        self._seg_cnt = defaultdict(int)

    def topk(self, k: int = 5) -> List[Tuple[str, float]]:
        items = sorted(((n, float(self.w[n])) for n in self.names), key=lambda x: x[1], reverse=True)
        return items[: max(1, int(k))]

# =========================
#         ALNS main
# =========================
def alns_minimize(
    init,
    evaluator: RobustEvaluator,
    iters: int = 400,
    start_T: float = 6.0,
    cool: float = 0.999,
    S_near_by_j: Dict[int, List[int]] | None = None,
    seed: int = 0,
    task_shelf_mapping: Dict[int, int] | None = None,
    enable_place_tune: bool = True,
    enable_shelf_tune: bool = True,
    shelf_init: Optional[Dict[int, int]] = None,
    ws_order_idx: Optional[Dict[int, Dict[int, int]]] = None,

    # ✅ 新增：是否启用“鲁棒敏感 destroy”
    enable_robust_destroy: bool = True,
    adaptive_reaction: float = 0.20,
    adaptive_segment_len: int = 50,
    adaptive_w_min: float = 0.05,
    adaptive_w_max: float = 50.0,
    reward_best: float = 33.0,
    reward_improve: float = 9.0,
    reward_accept: float = 3.0,
    reward_reject: float = 0.0,
):
    assert S_near_by_j is not None, "需要提供 S_near_by_j 作为回库位候选集"
    rng = random.Random(seed)
    task_shelf_mapping = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=True)

    # =========================
    # 规模自适应控参（提速关键）
    # =========================
    n_tasks = len(list(evaluator.J))
    n_s = len(list(evaluator.S))

    if n_tasks <= 30:
        max_agv_candidates = None
        max_pos_per_route = None
        max_place_try_each_base = 8
    else:
        max_agv_candidates = 3
        max_pos_per_route = 15
        max_place_try_each_base = 5
    if n_s >= 300:
        max_place_try_each_base = min(max_place_try_each_base, 4)

    best = copy.deepcopy(init)
    best.routes = _dc_routes(best.routes)
    best.place = _dc_place(best.place)
    best.shelf_seq = _dc_shelf_seq(best.shelf_seq)

    best.routes = _normalize_ws_blocks(best.routes, evaluator)
    best.routes = normalize_routes_by_shelf_seq_order(best.routes, best.shelf_seq, task_shelf_mapping)
    best_obj, best_details = evaluator.evaluate(best.routes, best.shelf_seq, best.place)

    best_obj = float(best_obj)
    best_key = _infeas_key(best_details, evaluator)

    cur = copy.deepcopy(best)
    cur_obj = float(best_obj)
    cur_details = dict(best_details) if isinstance(best_details, dict) else {}
    cur_key = tuple(best_key)

    T = float(start_T)

    print(f"[ALNS-inner] init_obj = {best_obj:.2f} | init_infeas={best_key}")

    # =========================================================
    # ALNS Adaptive core: destroy/repair weights (the "A")
    # =========================================================
    # destroy 算子池（名字只是标签）
    destroy_names = [
        "rand_small",
        "shaw_related",
        "chain",
        "ws_gap_bundle",
        "ws_idle_gap",
        "ws_critical",
        "robust_sensitive",
        "rand_big",
    ]
    # 初始权重：可以用你原来的概率当“先验”
    init_destroy_w = {
        "rand_small": 1.00,
        "shaw_related": 0.90,
        "chain": 0.70,
        "ws_gap_bundle": 0.95,
        "ws_idle_gap": 0.90,
        "ws_critical": 0.75,
        "robust_sensitive": 0.85,
        "rand_big": 0.60,
    }
    destroy_pool = AdaptiveOpPool(
        destroy_names,
        init_destroy_w,
        reaction=adaptive_reaction,
        segment_len=adaptive_segment_len,
        w_min=adaptive_w_min,
        w_max=adaptive_w_max,
    )

    # repair 算子池（至少两个，才算“有 repair 的自适应”）
    repair_names = ["repair_unordered", "repair_ordered"]
    init_repair_w = {"repair_unordered": 1.0, "repair_ordered": 1.0}
    repair_pool = AdaptiveOpPool(
        repair_names,
        init_repair_w,
        reaction=adaptive_reaction,
        segment_len=adaptive_segment_len,
        w_min=adaptive_w_min,
        w_max=adaptive_w_max,
    )

    # 哪些 destroy 必须保留 removed_order 的语义（seed-first / gap-first）
    ordered_required = {"ws_gap_bundle", "ws_idle_gap", "ws_critical", "robust_sensitive"}

    # reward 设计（标准 ALNS）
    SCORE_BEST = float(reward_best)
    SCORE_IMPROVE = float(reward_improve)
    SCORE_ACCEPT = float(reward_accept)
    SCORE_REJECT = float(reward_reject)
    PAIR_BIAS = 0.80
    PAIR_REACTION = 0.15
    PAIR_EPS = 0.08
    PAIR_DECAY = 0.995
    pair_quality: Dict[Tuple[str, str], float] = defaultdict(float)

    # destroy：正常阶段（你原来的）
    P_DESTROY_RANDOM_SMALL = 0.40
    P_DESTROY_RELATED = 0.35
    P_DESTROY_CHAIN = 0.25

    # 邻域概率（你原来的）
    P_USE_CROSS_VEHICLE = 0.55
    P_USE_CROSS_BLOCK = 0.35
    P_USE_INTRA_2OPT = 0.35
    P_USE_INTRA_OROPT = 0.35

    # 停滞控制（你原来的）
    STAG_WS = 40
    STAG_LIMIT = 120
    P_WS_FOCUSED = 0.70

    # 回温（你原来的）
    REHEAT_SOFT = 2.50
    REHEAT_STRONG = 6.00

    POST_SHAKE_ITERS = 25
    post_shake = 0
    stall = 0

    # ✅ 鲁棒 destroy 只对 gamma>0 有意义；概率做一个温和值（不破坏你现有主力算子）
    def _robust_destroy_prob(stagnating: bool) -> float:
        if (not enable_robust_destroy) or (int(getattr(evaluator, "gamma", 0)) <= 0):
            return 0.0
        return 0.20 if not stagnating else 0.30

    for it in range(max(1, int(iters))):
        improved_best_this_iter = False

        # ✅ 真停滞只由 stall 决定；post_shake 只是“回温窗口”
        stagnating = (stall >= STAG_WS)
        post_mode = (post_shake > 0)
        strong_shake = (stall >= STAG_LIMIT)

        # 温度：停滞或 post_mode 都允许回温（帮助跳局优）
        if stagnating or post_mode:
            T = max(T, REHEAT_SOFT)
        if strong_shake:
            T = max(T, REHEAT_STRONG)

        if strong_shake:
            cur = copy.deepcopy(best)
            cur_obj = float(best_obj)
            cur_details = dict(best_details) if isinstance(best_details, dict) else {}
            cur_key = tuple(best_key)

        # topk/extra：post_mode 只轻微加力，不要等同 stagnating
        if strong_shake:
            topk_insert = 3
            extra_shelves = 2
        elif stagnating:
            topk_insert = 3
            extra_shelves = 2
        elif post_mode:
            topk_insert = 2
            extra_shelves = 1
        else:
            topk_insert = 1
            extra_shelves = 0

        force_all_shelves = bool(strong_shake)

        # ===== 评估预算：以 stall 为主控；post_mode 不要升级成“停滞级别” =====
        if strong_shake:
            repair_evals_per_task = 55
            repair_evals_total = 320
        elif stagnating:
            repair_evals_per_task = 35
            repair_evals_total = 200
        elif post_mode:
            repair_evals_per_task = 25
            repair_evals_total = 140
        else:
            repair_evals_per_task = 20
            repair_evals_total = 120

        # =====================================================
        # -------------- 1) destroy + repair --------------
        # =====================================================
        # ========== 记录本迭代选中的算子（用于自适应更新）==========
        chosen_destroy = None
        chosen_repair = None

        # ========== 工具：把 removed_set 变成一个“可控插回顺序” ==========
        ws_fixed = getattr(evaluator, "ws_fixed_seq", {}) or {}
        ws_pos = _ws_index(ws_fixed)
        pi = getattr(evaluator, "pi", {}) or {}

        def _ws_rank_local(jj: int) -> int:
            ws = int(pi.get(int(jj), -1))
            return int(ws_pos.get(ws, {}).get(int(jj), 10 ** 9))

        def _order_from_set(rem_set: Set[int]) -> List[int]:
            lst = [int(x) for x in rem_set]
            rng.shuffle(lst)
            lst.sort(key=lambda x: (_ws_rank_local(x), rng.random()))
            return lst

        # ========== 自适应 destroy+repair 一步走 ==========
        def _adaptive_destroy_repair() -> Tuple[Dict[int, List[int]], Dict[int, int], str, str]:
            nonlocal stagnating, strong_shake, post_mode

            # --------- 1) 选 destroy 候选集（随阶段变）---------
            cand_destroy: List[str] = []
            if strong_shake:
                cand_destroy = ["ws_gap_bundle", "ws_idle_gap", "rand_big"]
            elif stagnating:
                cand_destroy = ["ws_gap_bundle", "ws_idle_gap", "ws_critical", "rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            elif post_mode:
                cand_destroy = ["ws_gap_bundle", "ws_critical", "rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            else:
                cand_destroy = ["rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")

            dname = destroy_pool.pick(rng, cand_destroy)

            # --------- 2) 执行 destroy ---------
            if dname == "rand_small":
                remove_frac = rng.uniform(0.15, 0.35)
                removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed)

            elif dname == "rand_big":
                remove_frac = rng.uniform(0.35, 0.65)
                removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed)

            elif dname == "shaw_related":
                remove_frac = rng.uniform(0.18, 0.38)
                removed, routes_half = destroy_shaw_related(
                    routes=cur.routes,
                    place=cur.place,
                    evaluator=evaluator,
                    shelf_seq=cur.shelf_seq,
                    task_shelf_mapping=task_shelf_mapping,
                    rng=rng,
                    remove_frac=remove_frac,
                )
                removed_order = _order_from_set(removed)

            elif dname == "chain":
                removed, routes_half = destroy_chain(routes=cur.routes, shelf_seq=cur.shelf_seq, rng=rng)
                removed_order = _order_from_set(removed)

            elif dname == "ws_gap_bundle":
                removed_order, removed, routes_half = destroy_ws_gap_route_bundle(
                    routes=cur.routes,
                    place=cur.place,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    min_k=3 if (stagnating or strong_shake) else 2,
                    max_k=4,
                )

            elif dname == "ws_idle_gap":
                removed_order, removed, routes_half = destroy_ws_largest_idle_gap(
                    routes=cur.routes,
                    place=cur.place,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    min_k=2,
                    max_k=5,
                )

            elif dname == "ws_critical":
                removed_order, removed, routes_half = destroy_ws_critical_window(
                    routes=cur.routes,
                    place=cur.place,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    min_k=2,
                    max_k=5,
                )
                if not removed_order:
                    # 兜底：退化随机小删
                    remove_frac = rng.uniform(0.15, 0.35)
                    removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                    removed_order = _order_from_set(removed)

            elif dname == "robust_sensitive":
                removed_order, removed, routes_half = destroy_robust_sensitive_bundle(
                    routes=cur.routes,
                    place=cur.place,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    task_shelf_mapping=task_shelf_mapping,
                    pre_k=2 if not stagnating else 3,
                    chain_nei=1 if not stagnating else 2,
                    ws_nei=1,
                    min_k=2,
                    max_k=5,
                    top_m=8 if not stagnating else 12,
                )

            else:
                # 防御：未知 name
                remove_frac = rng.uniform(0.15, 0.35)
                removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed)

            removed_set = set(int(x) for x in (removed_order or [])) if removed_order else set()
            # ✅ 强制修复：把当前解里缺失的任务也加入本轮 repair
            missing = _missing_tasks_in_routes(cur.routes, evaluator)
            if missing:
                miss_list = sorted((int(x) for x in missing), key=_ws_rank_local)
                # ordered destroy 要保留 seed-first 的语义，所以只追加
                if removed_order is None:
                    removed_order = []
                for x in miss_list:
                    if int(x) not in removed_set:
                        removed_order.append(int(x))
                        removed_set.add(int(x))
            if not removed_set:
                # 极端兜底
                remove_frac = rng.uniform(0.15, 0.35)
                removed_set, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed_set)
                removed_set = set(int(x) for x in removed_order)

            # --------- 3) 选 repair（有些 destroy 必须 ordered）---------
            if dname in ordered_required:
                cand_repair = ["repair_ordered"]
            else:
                cand_repair = ["repair_unordered", "repair_ordered"]

            # lightweight pair credit: bias repair roulette by (destroy,repair) historical quality
            cand_r = [c for c in cand_repair if c in repair_pool.w]
            use_pair_credit = bool(stagnating or post_mode or strong_shake)
            if (not cand_r) or (not use_pair_credit):
                rname = repair_pool.pick(rng, cand_repair)
            elif rng.random() < PAIR_EPS:
                rname = str(rng.choice(cand_r))
            else:
                weighted_r: List[Tuple[str, float]] = []
                total_w = 0.0
                for rr in cand_r:
                    base_w = max(repair_pool.w_min, float(repair_pool.w.get(rr, 1.0)))
                    pair_q = max(0.0, float(pair_quality.get((str(dname), str(rr)), 0.0)))
                    ww = base_w * math.exp(PAIR_BIAS * pair_q)
                    weighted_r.append((str(rr), float(ww)))
                    total_w += float(ww)

                u = rng.random() * max(1e-12, float(total_w))
                acc = 0.0
                rname = weighted_r[-1][0]
                for rr, ww in weighted_r:
                    acc += float(ww)
                    if acc >= u:
                        rname = str(rr)
                        break

            # --------- 4) 执行 repair ---------
            if rname == "repair_ordered":
                routes_new_, place_new_ = repair_greedy_insert_with_place_ordered(
                    routes=routes_half,
                    shelf_seq=cur.shelf_seq,
                    place=cur.place,
                    evaluator=evaluator,
                    removed_order=[int(x) for x in removed_order],
                    rng=rng,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    max_place_try_each=max_place_try_each_base,
                    top_k_random=topk_insert,
                    extra_random_shelves=extra_shelves,
                    force_all_shelves=force_all_shelves,
                    max_agv_candidates=max_agv_candidates,
                    max_pos_per_route=max_pos_per_route,
                    max_evals_per_task=repair_evals_per_task,
                    max_evals_total=repair_evals_total,
                )
            else:
                routes_new_, place_new_ = repair_greedy_insert_with_place(
                    routes=routes_half,
                    shelf_seq=cur.shelf_seq,
                    place=cur.place,
                    evaluator=evaluator,
                    removed=set(int(x) for x in removed_set),
                    rng=rng,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    max_place_try_each=max_place_try_each_base,
                    top_k_random=topk_insert,
                    extra_random_shelves=extra_shelves,
                    force_all_shelves=force_all_shelves,
                    max_agv_candidates=max_agv_candidates,
                    max_pos_per_route=max_pos_per_route,
                    max_evals_per_task=repair_evals_per_task,
                    max_evals_total=repair_evals_total,
                )

            return routes_new_, place_new_, dname, rname

        # ========== 执行自适应 destroy+repair ==========
        routes_new, place_new, chosen_destroy, chosen_repair = _adaptive_destroy_repair()

        # strong_shake 的“重置 + 回温窗口”仍保留
        if strong_shake:
            stall = 0
            post_shake = int(POST_SHAKE_ITERS)


        # =====================================================
        # =====================================================
        #  Gate：先快评估一次，决定是否值得做 heavy local search
        # =====================================================
        # =====================================================
        #  Gate：先快评估一次，决定是否值得做 heavy local search
        # =====================================================
        # =====================================================
        #  Gate：先评估一次（同时拿到 details），决定是否做 heavy local search
        # =====================================================
        routes_new = _normalize_ws_blocks(routes_new, evaluator)

        # 默认：不做任何 heavy local，直接用 destroy+repair 的结果参与 SA
        shelf_seq_new = cur.shelf_seq

        # ✅ v(由routes隐含) 与 shelf_seq 一致性兜底：只动 routes，不动 ws_fixed_seq
        routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)

        cand_obj_fast, cand_details_fast = evaluator.evaluate(routes_new, shelf_seq_new, place_new)

        cand_obj_fast = float(cand_obj_fast)
        cand_details_fast = dict(cand_details_fast) if isinstance(cand_details_fast, dict) else {}

        delta0 = cand_obj_fast - float(cur_obj)

        p_min = 0.05
        gate_delta = -math.log(p_min) * float(T)  # ≈ 3*T

        # ✅ 关键：当前不可行时，别让 Gate 把修复动作跳过
        infeasible_cur = (cur_key != (0, 0, 0))

        do_heavy_local = bool(strong_shake or stagnating or post_mode or infeasible_cur or (delta0 <= gate_delta))

        # cand_obj / cand_details 先用 fast 的；若做 heavy local，后面会覆盖
        cand_obj = float(cand_obj_fast)
        cand_details = dict(cand_details_fast)


        # =====================================================
        # -------------- 2) 邻域搜索（你原来的） --------------
        # =====================================================
        if do_heavy_local:
            if rng.random() < P_USE_CROSS_VEHICLE:
                routes_new_cv, place_new_cv, improved_cv, _ = cross_vehicle_move_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    rng=rng,
                    try_swap=bool(stagnating or strong_shake),  # ✅ 只有停滞/强震荡才开 swap
                    max_trials=3,
                    max_pos_samples=8,
                    max_s_samples=5,
                    max_evals=10,  # ✅ 这里一定要小（8~12）
                )

                if improved_cv:
                    routes_new, place_new = routes_new_cv, place_new_cv

            if rng.random() < P_USE_CROSS_BLOCK:
                routes_blk, place_blk, improved_blk, _ = cross_vehicle_block_move_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    rng=rng,
                    max_block=2,
                    max_trials=2,
                )
                if improved_blk:
                    routes_new, place_new = routes_blk, place_blk

            p_star = 0.25 if not stagnating else 0.65
            trials_star = 10 if not stagnating else 40
            if rng.random() < p_star:
                routes_star, place_star, improved_star, _ = cross_vehicle_2opt_star_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    max_trials=trials_star,
                    allow_non_improving=bool(stagnating),
                )
                if (not stagnating and improved_star) or stagnating:
                    routes_new, place_new = routes_star, place_star

            if rng.random() < P_USE_INTRA_2OPT:
                routes_new2, improved_2opt, _ = intra_two_opt_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    max_trials=2,
                )
                if improved_2opt:
                    routes_new = routes_new2

            if rng.random() < P_USE_INTRA_OROPT:
                routes_new3, improved_or, _ = intra_or_opt_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    max_trials=2,
                    max_block=2,
                )
                if improved_or:
                    routes_new = routes_new3

            if stagnating and ((it % 4) == 0):
                routes_gap, place_gap, improved_gap, _ = intensify_close_largest_ws_gap_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    S_near_by_j=S_near_by_j,
                    rng=rng,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    early_pos_max=2,
                    max_place_try_each=max_place_try_each_base,
                    extra_random_shelves=2,
                    force_all_shelves=False,
                )
                if improved_gap:
                    routes_new, place_new = routes_gap, place_gap

            # =====================================================
            # -------------- 3) place tune / shelf tune --------------
            # =====================================================
            if enable_place_tune:
                if strong_shake:
                    do_tune = ((it % 2) == 0)
                    global_try = 3
                elif stagnating:
                    do_tune = ((it % 3) == 0)
                    global_try = 2
                else:
                    do_tune = ((it % 10) == 0)
                    global_try = 0

                if do_tune:
                    place_new, _ = local_place_tune_once(
                        routes=routes_new,
                        shelf_seq=cur.shelf_seq,
                        place=place_new,
                        evaluator=evaluator,
                        S_near_by_j=S_near_by_j,
                        rng=rng,
                        task_shelf_mapping=task_shelf_mapping,
                        shelf_init_override=shelf_init,
                        top_k_try=5,
                        global_try_tasks=global_try,
                    )

            # 默认 shelf_seq 不变；若做 shelf_tune 再覆盖
            shelf_seq_new = cur.shelf_seq

            if enable_shelf_tune and task_shelf_mapping:
                do_reloc = strong_shake or (stagnating and ((it % 2) == 0)) or ((it % 8) == 0)
                if do_reloc:
                    shelf_seq_new, _ = local_shelf_seq_relocate_once(
                        routes=routes_new,
                        shelf_seq=cur.shelf_seq,
                        place=place_new,
                        evaluator=evaluator,
                        task_shelf_mapping=task_shelf_mapping,
                        rng=rng,
                    )

                do_promote = strong_shake or stagnating or ((it % 12) == 0)
                if do_promote:
                    shelf_seq_new, _, _ = intensify_shelf_seq_promote_critical_ws_once(
                        routes=routes_new,
                        shelf_seq=shelf_seq_new,
                        place=place_new,
                        evaluator=evaluator,
                        rng=rng,
                        max_trials=14 if stagnating else 8,
                    )

            if (it % 15) == 0:
                routes_int, place_int, improved_int, _ = intensify_critical_tail_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=shelf_seq_new,
                    evaluator=evaluator,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    tail_k=2,
                )
                if improved_int:
                    routes_new, place_new = routes_int, place_int

            if (stall >= STAG_WS) and ((it % 10) == 0):
                r_sync, s_sync, p_sync, imp_sync, _ = normalize_solution_by_ws_rank_once(
                    routes=routes_new,
                    shelf_seq=shelf_seq_new,
                    place=place_new,
                    evaluator=evaluator,
                    S_near_by_j=S_near_by_j,
                    rng=rng,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    global_try_tasks=2,
                )
                if imp_sync:
                    routes_new, shelf_seq_new, place_new = r_sync, s_sync, p_sync

            # =====================================================
            # -------------- 4) evaluate + relabel --------------
            # =====================================================
            # routes_new = _normalize_ws_blocks(routes_new, evaluator)
            # cand_obj, cand_details = evaluator.evaluate(routes_new, shelf_seq_new, place_new)
            #
            #
            # cand_obj = float(cand_obj)
            #
            # do_relabel = (stagnating or strong_shake or ((it % 10) == 0))
            # if do_relabel:
            #     routes_rl, obj_rl = _relabel_best_routes(
            #         routes_new,
            #         evaluator=evaluator,
            #         shelf_seq=shelf_seq_new,
            #         place=place_new,
            #     )
            #     if obj_rl < float(cand_obj) - 1e-9:
            #         routes_new = routes_rl
            #         cand_obj = float(obj_rl)

        # （如果 do_heavy_local=False）
        # 此时：routes_new/ place_new 来自 destroy+repair，shelf_seq_new=cur.shelf_seq，
        # cand_obj 直接用 cand_obj_fast，马上进入 SA accept（你下面那段不需要改）

        # =====================================================
        # -------------- 4) evaluate + relabel --------------
        # =====================================================
        # =====================================================
        # -------------- 4) evaluate + relabel --------------
        # =====================================================
        if do_heavy_local:
            # heavy local 已经改过 routes_new / shelf_seq_new / place_new，需要最终评估
            routes_new = _normalize_ws_blocks(routes_new, evaluator)
            routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)
            cand_obj, cand_details = evaluator.evaluate(routes_new, shelf_seq_new, place_new)

            cand_obj = float(cand_obj)
            cand_details = dict(cand_details) if isinstance(cand_details, dict) else {}
        else:
            # do_heavy_local=False：Gate 时已经 evaluate 过，直接复用 cand_obj/cand_details
            cand_obj = float(cand_obj)
            cand_details = dict(cand_details) if isinstance(cand_details, dict) else {}

        do_relabel = (stagnating or strong_shake or ((it % 10) == 0))
        if do_relabel:
            routes_rl, obj_rl = _relabel_best_routes(
                routes_new,
                evaluator=evaluator,
                shelf_seq=shelf_seq_new,
                place=place_new,
            )
            if obj_rl < float(cand_obj) - 1e-9:
                routes_new = routes_rl

                # ✅ relabel 后再兜底一次：保证 v 与 shelf_seq 一致
                routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)

                # ✅ 重要：relabel 改了 routes，必须同步更新 details（否则 SA 用错 cand_key）
                cand_obj, cand_details = evaluator.evaluate(routes_new, shelf_seq_new, place_new)
                cand_obj = float(cand_obj)
                cand_details = dict(cand_details) if isinstance(cand_details, dict) else {}

        # =====================================================
        # -------------- 5) SA accept --------------
        # =====================================================
        # =====================================================
        # -------------- 5) SA accept (feasibility-first) -----
        # =====================================================
        # =====================================================
        # -------------- 5) SA accept (feasibility-first) -----
        # =====================================================
        # ✅ 关键：缓存“上一轮”的 cur/best，用于 reward 判定（避免 accept 后 cur 被覆盖）
        prev_cur_obj = float(cur_obj)
        prev_cur_key = tuple(cur_key)
        prev_best_obj = float(best_obj)
        prev_best_key = tuple(best_key)

        cand_key = _infeas_key(cand_details, evaluator)

        accept = False

        # 1) 不可行度变好：无条件接受（feasibility-first）
        if cand_key < prev_cur_key:
            accept = True

        # 2) 不可行度一样：对 objective 用 SA
        elif cand_key == prev_cur_key:
            if cand_obj < prev_cur_obj - 1e-9:
                accept = True
            else:
                delta = float(cand_obj) - float(prev_cur_obj)
                prob = math.exp(-delta / max(1e-9, float(T)))
                if rng.random() < prob:
                    accept = True

        # 3) 不可行度变差：允许少量以 infeas-SA 接受（避免卡死）
        else:
            delta_inf = _infeas_scalar(cand_details, evaluator) - _infeas_scalar(cur_details, evaluator)
            T_inf = max(1.0, 2.0 * float(T))
            x = -float(delta_inf) / max(1e-9, float(T_inf))
            if x >= 700.0:
                prob = 1.0
            elif x <= -700.0:
                prob = 0.0
            else:
                prob = math.exp(x)
            if rng.random() < prob:
                accept = True

        if accept:
            cur.routes = routes_new
            cur.place = place_new
            cur.shelf_seq = shelf_seq_new
            cur_obj = float(cand_obj)

            cur_details = dict(cand_details) if isinstance(cand_details, dict) else {}
            cur_key = tuple(cand_key)

            # best 更新：也要基于 prev_best（可行性优先）
            if (cand_key < prev_best_key) or (cand_key == prev_best_key and cand_obj < prev_best_obj - 1e-9):
                best = copy.deepcopy(cur)
                best_obj = float(cand_obj)
                best_key = tuple(cand_key)
                best_details = dict(cur_details) if isinstance(cur_details, dict) else {}
                improved_best_this_iter = True

        # =====================================================
        # -------------- 5.5) Adaptive weight update ----------
        # =====================================================
        # ✅ 标准 ALNS：reward 用 cand 对比 prev_cur / prev_best
        if accept:
            is_new_best = (cand_key < prev_best_key) or (cand_key == prev_best_key and cand_obj < prev_best_obj - 1e-9)
            is_improve_cur = (cand_key < prev_cur_key) or (cand_key == prev_cur_key and cand_obj < prev_cur_obj - 1e-9)

            if is_new_best:
                reward = SCORE_BEST
            elif is_improve_cur:
                reward = SCORE_IMPROVE
            else:
                reward = SCORE_ACCEPT
        else:
            reward = SCORE_REJECT

        if chosen_destroy is not None:
            destroy_pool.record(chosen_destroy, reward)
        if chosen_repair is not None:
            repair_pool.record(chosen_repair, reward)
        if (chosen_destroy is not None) and (chosen_repair is not None):
            pair_key = (str(chosen_destroy), str(chosen_repair))
            target_q = max(0.0, min(1.0, float(reward) / max(1.0, float(SCORE_BEST))))
            if accept and (cand_key == prev_cur_key) and math.isfinite(float(prev_cur_obj)) and math.isfinite(float(cand_obj)):
                rel_gain = max(0.0, (float(prev_cur_obj) - float(cand_obj)) / max(1.0, abs(float(prev_cur_obj))))
                target_q = min(1.0, target_q + min(0.40, 6.0 * rel_gain))
            if accept and ((cand_key < prev_best_key) or (cand_key == prev_best_key and cand_obj < prev_best_obj - 1e-9)):
                target_q = min(1.0, target_q + 0.20)
            prev_q = float(pair_quality.get(pair_key, 0.0))
            pair_quality[pair_key] = (1.0 - PAIR_REACTION) * prev_q + PAIR_REACTION * target_q
        if ((it + 1) % max(1, int(adaptive_segment_len)) == 0) and pair_quality:
            for pk in list(pair_quality.keys()):
                qv = float(pair_quality.get(pk, 0.0)) * PAIR_DECAY
                if qv < 1e-6:
                    pair_quality.pop(pk, None)
                else:
                    pair_quality[pk] = qv

        destroy_pool.maybe_update(it)
        repair_pool.maybe_update(it)
        # =====================================================
        # -------------- 6) cool & stall --------------
        # =====================================================
        T *= float(cool)
        stall = 0 if improved_best_this_iter else (stall + 1)

        if stagnating or post_mode:
            T = max(float(T), REHEAT_SOFT)

        if post_shake > 0:
            post_shake -= 1

        if (it + 1) % 100 == 0:
            print(
                f"[ALNS-inner] iter {it + 1}/{iters} | cur={cur_obj:.2f} | "
                f"best={best_obj:.2f} | stall={stall} | T={T:.4f} | post={post_shake}"

            )
            print(f"[ALNS-adapt] destroy top = {destroy_pool.topk(4)}")
            print(f"[ALNS-adapt] repair  top = {repair_pool.topk(2)}")


    print(f"[ALNS-inner] best_obj after SA = {best_obj:.2f}")

    best.routes = _normalize_ws_blocks(best.routes, evaluator)
    best_rl, best_rl_obj = _relabel_best_routes(
        best.routes,
        evaluator=evaluator,
        shelf_seq=best.shelf_seq,
        place=best.place,
    )
    if best_rl_obj < float(best_obj) - 1e-9:
        best.routes = best_rl
        best_obj = float(best_rl_obj)

    for _ in range(3):
        routes_int, place_int, improved_int, obj_int = intensify_critical_tail_once(
            routes=best.routes,
            place=best.place,
            shelf_seq=best.shelf_seq,
            evaluator=evaluator,
            S_near_by_j=S_near_by_j,
            task_shelf_mapping=task_shelf_mapping,
            shelf_init_override=shelf_init,
            tail_k=2,
        )
        if improved_int and float(obj_int) < float(best_obj) - 1e-9:
            best.routes, best.place = routes_int, place_int
            best_obj = float(obj_int)
        else:
            break

    print(f"[ALNS-inner] final_best_obj = {best_obj:.2f}")
    # ✅ 返回前兜底：保证最终输出的 routes 与 shelf_seq 一致（不动 ws_fixed_seq）
    best.routes = _normalize_ws_blocks(best.routes, evaluator)
    best.routes = normalize_routes_by_shelf_seq_order(best.routes, best.shelf_seq, task_shelf_mapping)
    _ = basic_feasibility_check_level0(
        routes=best.routes,
        shelf_seq=best.shelf_seq,
        place=best.place,
        evaluator=evaluator,
        task_shelf_mapping=task_shelf_mapping,
        verbose=True,
    )
    return best
