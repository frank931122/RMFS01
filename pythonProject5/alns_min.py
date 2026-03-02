from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from itertools import permutations
from typing import Dict, List, Tuple, Set, Optional

from evaluator import RobustEvaluator
from collections import defaultdict, OrderedDict

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
    Lightweight structural feasibility check before running expensive MILP/evaluator logic.
      1) Every task must appear in exactly one AGV route.
      2) Each route must not contain duplicate tasks.
      3) shelf_seq must be consistent with task_shelf_mapping when provided.
      4) place must map every task to a valid shelf/cell id.
    """
    ok = True
    all_tasks: Set[int] = set(int(j) for j in evaluator.J)

    # 1) Validate route coverage and uniqueness across AGVs
    appear_cnt: Dict[int, int] = {j: 0 for j in all_tasks}
    for r, seq in routes.items():
        for j in seq:
            jj = int(j)
            if jj not in all_tasks:
                ok = False
                if verbose:
                    print(f"[Lv0-Check] ERROR: AGV {r} route [text_corrupted]{jj}")
            else:
                appear_cnt[jj] += 1

    for j in sorted(all_tasks):
        if appear_cnt[j] == 0:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: task {j} does not appear in any AGV route")
        elif appear_cnt[j] > 1:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: [text_corrupted] {j} [text_corrupted]routes [text_corrupted] {appear_cnt[j]} [text_corrupted](>1)")

    # 2) Validate no duplicates within each route
    for r, seq in routes.items():
        seen: Set[int] = set()
        for j in seq:
            jj = int(j)
            if jj in seen:
                ok = False
                if verbose:
                    print(f"[Lv0-Check] ERROR: AGV {r} [text_corrupted]route [text_corrupted]{jj} [text_corrupted]")
            seen.add(jj)

    # 3) Validate shelf_seq against mapping
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
                            f"[Lv0-Check] ERROR: shelf_seq [text_corrupted] {cc} [text_corrupted]{jj}, "
                            f"[text_corrupted]task_shelf_mapping[{jj}] = {c_expect}"
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
                        f"[Lv0-Check] ERROR: [text_corrupted] {j} [text_corrupted] {c_expect}, "
                        f"[text_corrupted]shelf_seq[{c_expect}] [text_corrupted]"
                    )

    # 4) Validate placement map
    all_cells: Set[int] = set(int(s) for s in evaluator.S)
    for j, s in place.items():
        jj, ss = int(j), int(s)
        if jj not in all_tasks:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: place [text_corrupted]{jj}")
        if ss not in all_cells:
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: place[{jj}] = {ss} [text_corrupted]cell")

    # 5) Every routed task must have a placement
    for j in all_tasks:
        if appear_cnt[j] > 0 and (j not in place):
            ok = False
            if verbose:
                print(f"[Lv0-Check] ERROR: [text_corrupted] {j} [text_corrupted]routes [text_corrupted]place [text_corrupted]")

    if verbose:
        if ok:
            print("[Lv0-Check] structural check passed for routes/shelf_seq/place")
        else:
            print("[Lv0-Check] structural check failed; see ERROR lines above")
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

def _make_fail_details(evaluator: RobustEvaluator, *, reason: str = "", base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    all_j = [int(x) for x in (getattr(evaluator, "J", []) or [])]
    out: Dict[str, Any] = dict(base) if isinstance(base, dict) else {}
    penalties = dict((out.get("penalties", {}) if isinstance(out, dict) else {}) or {})
    penalties["tail_cell_conflict"] = max(float(penalties.get("tail_cell_conflict", 0.0) or 0.0), 1.0)
    penalties["cell_conflict_pairs"] = max(float(penalties.get("cell_conflict_pairs", 0.0) or 0.0), 1e9)
    penalties["cell_conflict_overlap_time"] = max(float(penalties.get("cell_conflict_overlap_time", 0.0) or 0.0), 1e9)
    uns = out.get("unscheduled_tasks", None)
    if isinstance(uns, (list, tuple, set)):
        out["unscheduled_tasks"] = [int(x) for x in uns]
    else:
        out["unscheduled_tasks"] = all_j
    out["penalties"] = penalties
    if reason:
        out["error"] = str(reason)
    return out

def _safe_evaluate(
    evaluator: RobustEvaluator,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    *,
    fail_cost: float = 1e30,
) -> Tuple[float, Dict[str, Any]]:
    """
    Safe wrapper around evaluator.evaluate().
    Any exception or non-finite objective is converted into a large fail_cost and
    a standardized diagnostic payload for ALNS to continue robustly.
    """
    try:
        obj, det = evaluator.evaluate(routes, shelf_seq, place)
        obj_f = float(obj)
        det_d = dict(det) if isinstance(det, dict) else {}
        if not math.isfinite(obj_f):
            return float(fail_cost), _make_fail_details(
                evaluator,
                reason=f"non_finite_obj:{obj_f}",
                base=det_d,
            )
        return obj_f, det_d
    except Exception as e:
        return float(fail_cost), _make_fail_details(evaluator, reason=repr(e))


def _normalize_one_route_ws_and_shelf(
    seq: List[int],
    *,
    evaluator: RobustEvaluator,
    chain_of: Dict[int, int],
    shelf_idx: Dict[int, Dict[int, int]],
) -> List[int]:
    """
    [text_corrupted] route [text_corrupted]
      1) [text_corrupted]WS [text_corrupted] ws_fixed_seq [text_corrupted]
      2) [text_corrupted] shelf [text_corrupted]route [text_corrupted]slot [text_corrupted] shelf_seq [text_corrupted]
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
    [text_corrupted]task_shelf_mapping [text_corrupted]task->chain(int)[text_corrupted]
      - [text_corrupted] value [text_corrupted]None / NaN / [text_corrupted]int [text_corrupted]
      - key [text_corrupted] int [text_corrupted]
    [text_corrupted]
      - [text_corrupted]dict
      - [text_corrupted]None[text_corrupted]
    """
    if not task_shelf_mapping:
        return None

    out: Dict[int, int] = {}
    bad = 0

    for j, c in task_shelf_mapping.items():
        # [text_corrupted] None
        if c is None:
            bad += 1
            continue

        # [text_corrupted] NaN[text_corrupted] pandas [text_corrupted]
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
        print(f"[WARN] task_shelf_mapping dropped {bad} invalid entries (None/NaN/non-int)")

    return out if out else None


def _build_chain_of_map(
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, Any]] = None,
) -> Dict[int, int]:
    """
    task -> chain(c)

    [text_corrupted]task_shelf_mapping[text_corrupted]None/NaN/[text_corrupted]
    [text_corrupted] mapping [text_corrupted] shelf_seq [text_corrupted]
    """
    mp: Dict[int, int] = {}

    # 1) [text_corrupted] mapping
    clean = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    if clean:
        # clean [text_corrupted]int->int
        mp.update(clean)

    # 2) [text_corrupted] shelf_seq [text_corrupted]mp [text_corrupted]
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


def _ensure_shelf_seq_covers_all_tasks(
    shelf_seq: Dict[int, List[int]],
    *,
    evaluator: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
) -> Dict[int, List[int]]:
    """
    Ensure every task in evaluator.J appears in exactly one shelf chain sequence.
    """
    out = _dc_shelf_seq(shelf_seq)
    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)

    all_tasks = [int(j) for j in evaluator.J]
    all_set = set(all_tasks)
    if not out:
        if clean_map:
            c0 = int(next(iter(clean_map.values())))
        else:
            c0 = 1
        out = {int(c0): []}

    # remove duplicates/unknowns in existing shelf_seq
    seen: Set[int] = set()
    for c in sorted(list(out.keys())):
        seq = []
        for j in out.get(c, []) or []:
            jj = int(j)
            if (jj in all_set) and (jj not in seen):
                seen.add(jj)
                seq.append(jj)
        out[int(c)] = seq

    chain_ids = sorted(int(c) for c in out.keys())
    pi = getattr(evaluator, "pi", {}) or {}

    # infer a representative ws for each chain
    chain_ws: Dict[int, int] = {}
    for c in chain_ids:
        cnt: Dict[int, int] = {}
        for j in out.get(c, []) or []:
            ws = int(pi.get(int(j), -1))
            cnt[ws] = int(cnt.get(ws, 0)) + 1
        if cnt:
            chain_ws[c] = max(cnt.items(), key=lambda kv: kv[1])[0]
        else:
            chain_ws[c] = -1

    for j in all_tasks:
        jj = int(j)
        if jj in seen:
            continue

        c_pick: Optional[int] = None
        if clean_map and (jj in clean_map):
            c_try = int(clean_map[jj])
            if c_try in out:
                c_pick = c_try
        if c_pick is None:
            wsj = int(pi.get(jj, -1))
            same = [c for c in chain_ids if int(chain_ws.get(c, -999)) == wsj]
            if same:
                c_pick = min(same, key=lambda cc: len(out.get(int(cc), [])))
            else:
                c_pick = min(chain_ids, key=lambda cc: len(out.get(int(cc), [])))

        out[int(c_pick)].append(jj)
        seen.add(jj)

    return out



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
    [text_corrupted] route [text_corrupted]shelf [text_corrupted] == shelf_seq[c]

    [text_corrupted]slot [text_corrupted]slot [text_corrupted]
      - [text_corrupted]chain c[text_corrupted]seq [text_corrupted] slots
      - [text_corrupted]slots [text_corrupted]shelf_seq[c] [text_corrupted]
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

        # [text_corrupted] shelf_seq [text_corrupted]
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
    [text_corrupted]AGV routes [text_corrupted]shelf_seq [text_corrupted]v([text_corrupted]routes[text_corrupted]) [text_corrupted]
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
    # [text_corrupted]0.01 [text_corrupted]float [text_corrupted]
    cell_score = int(round(cell_measure * 100))
    return (unscheduled, tail, cell_score)

def _infeas_scalar(details, evaluator):
    # [text_corrupted] 1e4 [text_corrupted]
    unscheduled, tail, cell_measure = _infeas_components(details, evaluator)
    return float(unscheduled) * 1e4 + float(tail) * 1e2 + float(cell_measure)

def _safe_infeas_key(
    details,
    evaluator,
    *,
    obj: Optional[float] = None,
    fail_cost: float = 1e30,
) -> Tuple[int, int, int]:
    if obj is not None:
        try:
            obj_f = float(obj)
            if (not math.isfinite(obj_f)) or (obj_f >= float(fail_cost) * 0.999999):
                return (10**9, 1, 10**12)
        except Exception:
            return (10**9, 1, 10**12)
    try:
        return tuple(_infeas_key(details, evaluator))
    except Exception:
        return (10**9, 1, 10**12)

def _safe_infeas_scalar(
    details,
    evaluator,
    *,
    obj: Optional[float] = None,
    fail_cost: float = 1e30,
) -> float:
    if obj is not None:
        try:
            obj_f = float(obj)
            if (not math.isfinite(obj_f)) or (obj_f >= float(fail_cost) * 0.999999):
                return 1e30
        except Exception:
            return 1e30
    try:
        v = float(_infeas_scalar(details, evaluator))
        return v if math.isfinite(v) else 1e30
    except Exception:
        return 1e30

def _normalize_ws_blocks(routes: Dict[int, List[int]], evaluator: RobustEvaluator) -> Dict[int, List[int]]:
    """
    [text_corrupted] WS [text_corrupted] fixed [text_corrupted] WS [text_corrupted]WS [text_corrupted]
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
    [text_corrupted] route [text_corrupted]WS [text_corrupted]WS [text_corrupted]len(seq)
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
    [text_corrupted] route [text_corrupted] WS [text_corrupted] normalize [text_corrupted]routes[text_corrupted]
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
    Level-1 [text_corrupted]/[text_corrupted](LB)[text_corrupted]
    [text_corrupted]
      - [text_corrupted] LB >= [text_corrupted] obj[text_corrupted] [text_corrupted]
      - [text_corrupted]LB [text_corrupted]
    [text_corrupted] key [text_corrupted]key [text_corrupted] RulePrechecker [text_corrupted]
    [text_corrupted] key [text_corrupted]0[text_corrupted] [text_corrupted]
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
        [text_corrupted]
        stop_at[text_corrupted]
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
                # [text_corrupted]0 [text_corrupted]
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
        [text_corrupted]max_r route_lb(r)
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
    [text_corrupted]
      - [text_corrupted]j [text_corrupted] ws [text_corrupted]rank = pos_in_ws / len(ws_seq)
      - routes [text_corrupted] rank [text_corrupted]
      - shelf_seq [text_corrupted] rank [text_corrupted]
      - [text_corrupted]place_tune [text_corrupted]
    [text_corrupted]routes_new, shelf_seq_new, place_new, improved, obj_new)
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

    # --- 1) routes [text_corrupted] ---
    routes_new = _dc_routes(routes0)
    for r, seq in routes_new.items():
        seq2 = [int(x) for x in seq]
        # [text_corrupted] tie-break
        tagged = [(idx, j) for idx, j in enumerate(seq2)]
        tagged.sort(key=lambda it: (rank(it[1]), int(evaluator.pi.get(it[1], -1)), it[0]))
        routes_new[int(r)] = [j for _, j in tagged]
    routes_new = _normalize_ws_blocks(routes_new, evaluator)

    # --- 2) shelf_seq [text_corrupted] ---
    shelf_new = _dc_shelf_seq(shelf0)
    for c, seq in shelf_new.items():
        seq2 = [int(x) for x in seq]
        tagged = [(idx, j) for idx, j in enumerate(seq2)]
        tagged.sort(key=lambda it: (rank(it[1]), int(evaluator.pi.get(it[1], -1)), it[0]))
        shelf_new[int(c)] = [j for _, j in tagged]

    # --- 3) [text_corrupted] place tune [text_corrupted] ---
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
    eval_top_k: int = 12,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init_override: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, List[int]], float]:
    """
    Funnel relabel:
      1) [text_corrupted]proxy LB [text_corrupted]AGV [text_corrupted]
      2) [text_corrupted] LB [text_corrupted] Top-K [text_corrupted]evaluate [text_corrupted]    [text_corrupted] place [text_corrupted]shelf_seq [text_corrupted]    """
    routes = _dc_routes(routes)
    R_ids = sorted(int(r) for r in evaluator.R)
    seqs = [routes.get(r, []) for r in R_ids]
    n = len(R_ids)

    base_obj, _ = _safe_evaluate(evaluator, routes, shelf_seq, place)
    base_obj = float(base_obj)

    if n <= 1 or n > int(max_exact_agv):
        return routes, base_obj

    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )
    proxy = FastProxyEvaluatorLB(evaluator, pre)

    stop_at = base_obj if math.isfinite(base_obj) else None
    cost: Dict[int, List[float]] = {}
    for r in R_ids:
        rr = int(r)
        row: List[float] = []
        for i in range(n):
            lb_i = proxy.route_lb(r=rr, seq=seqs[i], place=place, stop_at=stop_at)
            row.append(float(lb_i))
        cost[rr] = row

    pool: List[Tuple[float, Tuple[int, ...]]] = []
    for perm in permutations(R_ids):
        lb = 0.0
        for i in range(n):
            vv = float(cost[int(perm[i])][i])
            if vv > lb:
                lb = vv
        if math.isfinite(base_obj) and lb >= base_obj - 1e-9:
            continue
        pool.append((float(lb), tuple(int(x) for x in perm)))

    if not pool:
        return routes, base_obj

    pool.sort(key=lambda x: x[0])
    kk = max(1, int(eval_top_k))
    pool = pool[:kk]

    best_obj = float(base_obj)
    best_routes = routes
    for _, perm in pool:
        cand_routes = {int(perm[i]): list(seqs[i]) for i in range(n)}
        cand_routes = _normalize_ws_blocks(cand_routes, evaluator)
        obj, _ = _safe_evaluate(evaluator, cand_routes, shelf_seq, place)
        obj = float(obj)
        if obj < best_obj - 1e-9:
            best_obj = obj
            best_routes = cand_routes
    return best_routes, float(best_obj)


def _prev_on_chain(
    j: int,
    shelf_seq: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
) -> Optional[int]:
    """[text_corrupted] j [text_corrupted]j [text_corrupted] None[text_corrupted]"""
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
    [text_corrupted]j [text_corrupted]m [text_corrupted]+ {[text_corrupted]end_s [text_corrupted]} + {[text_corrupted] place[j][text_corrupted]}
    [text_corrupted]d_pi_s(j,s) [text_corrupted] max_keep [text_corrupted]
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
    cap_total: int = 40,   # [text_corrupted]
) -> List[int]:
    """
    [text_corrupted]base_cands [text_corrupted] cap_total[text_corrupted]|S| [text_corrupted]
    """
    cand_set: Set[int] = set(int(s) for s in base_cands)
    all_s = [int(s) for s in evaluator.S]

    if force_all_shelves:
        # [text_corrupted] S[text_corrupted] cap_total[text_corrupted]
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
    """[text_corrupted] k [text_corrupted]"""
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
    [text_corrupted]WS idle gap[text_corrupted]cur_t[text_corrupted]
    [text_corrupted] cur_t [text_corrupted]AGV [text_corrupted]route [text_corrupted]
    [text_corrupted] removed_order [text_corrupted]cur_t [text_corrupted]
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

    # 1) [text_corrupted]idle gap [text_corrupted]best_cur
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

    # 2) [text_corrupted]best_cur [text_corrupted]
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

    # [text_corrupted] best_cur [text_corrupted] Task1 [text_corrupted]
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
    [text_corrupted]WS idle gap[text_corrupted]cur_t[text_corrupted]
    [text_corrupted]/[text_corrupted]route [text_corrupted]pos=0..early_pos_max[text_corrupted]first-improvement [text_corrupted]
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

    # [text_corrupted] j_move
    routes_removed = _dc_routes(routes)
    for r in list(routes_removed.keys()):
        routes_removed[r] = [x for x in routes_removed[r] if int(x) != j_move]

    # [text_corrupted]
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
    """Random destroy: remove a fraction of tasks from routes."""
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
    [text_corrupted]WS [text_corrupted] idle gap[text_corrupted]
    [text_corrupted]WS2: Task2 -> Task4 [text_corrupted]
    [text_corrupted]removed_order, removed_set, routes_removed)
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

    # [text_corrupted] (best_idx-1, best_idx)
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
    # [text_corrupted]seed [text_corrupted]
    pre_k: int = 2,
    # [text_corrupted]chain_nei [text_corrupted]
    chain_nei: int = 1,
    # [text_corrupted]WS [text_corrupted]ws_nei [text_corrupted]
    ws_nei: int = 1,
    # [text_corrupted]
    min_k: int = 2,
    max_k: int = 5,
    # [text_corrupted]top_m [text_corrupted]seed[text_corrupted]
    top_m: int = 8,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    [text_corrupted] destroy[text_corrupted] ALNS [text_corrupted]

    [text_corrupted] j [text_corrupted] robust_slack[j] = q[j] - q0[j] [text_corrupted] q[G] - q[0][text_corrupted]
    [text_corrupted]robust_slack [text_corrupted]/[text_corrupted]

    [text_corrupted]seed[text_corrupted]
      - seed [text_corrupted]
      - seed [text_corrupted]AGV [text_corrupted]pre_k [text_corrupted]seed [text_corrupted]
      - seed [text_corrupted]
      - seed [text_corrupted]WS [text_corrupted]/[text_corrupted]
    [text_corrupted]/[text_corrupted][min_k, max_k][text_corrupted]

    [text_corrupted]
      removed_order: [text_corrupted] seed [text_corrupted]repair [text_corrupted] seed[text_corrupted]
      removed_set
      routes_removed[text_corrupted]routes [text_corrupted]shelf_seq/place[text_corrupted]
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)

    # present tasks
    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in seq)
    if not present:
        return [], set(), routes

    # [text_corrupted]q/q0[text_corrupted]evaluate[text_corrupted]
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    _, diag = evaluator.evaluate(routes_norm, shelf_seq, place)

    q_raw = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    q0_raw = (diag.get("q0", {}) if isinstance(diag, dict) else {}) or {}

    # [text_corrupted] key=int
    try:
        q_map = {int(k): float(v) for k, v in q_raw.items()}
    except Exception:
        q_map = {}
    try:
        q0_map = {int(k): float(v) for k, v in q0_raw.items()}
    except Exception:
        q0_map = {}

    # [text_corrupted] slack = q - q0
    slack: Dict[int, float] = {}
    for j in present:
        if j in q_map:
            base0 = q0_map.get(j, q_map[j])  # [text_corrupted] q0[text_corrupted] slack=0
            slack[j] = float(q_map[j] - float(base0))

    # [text_corrupted] slack [text_corrupted] 0[text_corrupted] q [text_corrupted]
    if (not slack) or (max(slack.values()) <= 1e-9):
        slack = {j: float(q_map.get(j, 0.0)) for j in present if j in q_map}

    # [text_corrupted] q [text_corrupted]allow_incomplete [text_corrupted] seed
    if not slack:
        seed = int(rng.choice(sorted(present)))
        slack = {seed: 0.0}
    else:
        ranked = sorted(slack.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[: max(1, int(top_m))]
        seed = int(rng.choice([j for j, _ in top]))

    # ========== [text_corrupted] ==========
    prio: List[int] = [seed]

    # (1) [text_corrupted]seed [text_corrupted] pre_k [text_corrupted]
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
        # [text_corrupted]seed [text_corrupted]
        prefix = list(reversed(prefix))
        for x in prefix:
            if x in present:
                prio.append(int(x))

    # (2) [text_corrupted] chain_nei [text_corrupted]
    if task_shelf_mapping is not None:
        c = task_shelf_mapping.get(int(seed), None)
        if c is not None:
            cc = int(c)
            seq_c = [int(x) for x in (shelf_seq.get(cc, []) or [])]
            try:
                pos = int(seq_c.index(int(seed)))
                left = seq_c[max(0, pos - int(chain_nei)):pos]
                right = seq_c[pos + 1: pos + 1 + int(chain_nei)]
                # [text_corrupted]
                left = list(reversed(left))
                for x in left + right:
                    if int(x) in present:
                        prio.append(int(x))
            except ValueError:
                pass

    # (3) [text_corrupted]WS [text_corrupted]ws_nei [text_corrupted]
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

    # ========== [text_corrupted] removed_set[text_corrupted]==========
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

    # [text_corrupted] min_k[text_corrupted] slack [text_corrupted]
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

    # removed_order[text_corrupted]seed [text_corrupted] prio [text_corrupted] slack [text_corrupted]
    removed_order: List[int] = [int(seed)]
    for x in prio:
        xx = int(x)
        if xx in removed_set and xx not in removed_order:
            removed_order.append(xx)
    rest = [x for x in removed_set if x not in removed_order]
    rest.sort(key=lambda j: float(slack.get(int(j), 0.0)), reverse=True)
    removed_order.extend(rest)

    # [text_corrupted]routes [text_corrupted]removed_set
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
    """Shaw[text_corrupted]/[text_corrupted]WS[text_corrupted]/[text_corrupted]"""
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
    """Shaw/Related destroy[text_corrupted]"""
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
            continue  # [text_corrupted] seed [text_corrupted]
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
    """[text_corrupted]"""
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
    max_k: int = 5,  # [text_corrupted] 5
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    WS-focused destroy[text_corrupted]
      1) [text_corrupted]makespan [text_corrupted]j*
      2) [text_corrupted] ws [text_corrupted]fixed [text_corrupted]j* [text_corrupted]k)
      3) [text_corrupted]
    [text_corrupted]removed_order, removed_set, routes_removed)
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

    # [text_corrupted]present [text_corrupted] dummy / [text_corrupted]pi [text_corrupted]
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
        removed_order = ws_seq[:]  # [text_corrupted]
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


def destroy_critical_single(
    *,
    routes: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    q_hint: Optional[Dict[int, float]] = None,
    extra_neighbor_prob: float = 0.25,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    Ultra-light destroy for feasible-speed mode:
      1) pick one critical task (highest q if available, otherwise tail of longest route)
      2) optionally add one neighboring task on the same WS order
    This keeps neighborhood tiny and reduces repair evaluation burden.
    """
    routes = _dc_routes(routes)
    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in (seq or []))
    if not present:
        return [], set(), routes

    q_map: Dict[int, float] = {}
    if isinstance(q_hint, dict):
        for k, v in q_hint.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    q_map[kk] = vv
            except Exception:
                pass

    j_star: Optional[int] = None
    if q_map:
        cand = [(float(q_map[j]), int(j)) for j in present if int(j) in q_map]
        if cand:
            cand.sort(key=lambda x: x[0], reverse=True)
            j_star = int(cand[0][1])

    if j_star is None:
        route_rank = sorted(
            [(int(r), list(seq or [])) for r, seq in routes.items()],
            key=lambda x: (len(x[1]), int(x[0])),
            reverse=True,
        )
        for _, seq in route_rank:
            if seq:
                j_star = int(seq[-1])
                break
    if j_star is None:
        j_star = int(rng.choice(sorted(present)))

    removed_order: List[int] = [int(j_star)]
    removed: Set[int] = {int(j_star)}

    if rng.random() < float(extra_neighbor_prob):
        ws = int(getattr(evaluator, "pi", {}).get(int(j_star), -1))
        ws_seq = [int(x) for x in ((getattr(evaluator, "ws_fixed_seq", {}) or {}).get(ws, []) or [])]
        if ws_seq and (int(j_star) in ws_seq):
            p = int(ws_seq.index(int(j_star)))
            nbrs: List[int] = []
            if p > 0:
                nbrs.append(int(ws_seq[p - 1]))
            if p + 1 < len(ws_seq):
                nbrs.append(int(ws_seq[p + 1]))
            nbrs = [int(x) for x in nbrs if (int(x) in present) and (int(x) not in removed)]
            if nbrs:
                jj = int(rng.choice(nbrs))
                removed_order.append(jj)
                removed.add(jj)

    for r in list(routes.keys()):
        routes[int(r)] = [int(j) for j in routes[int(r)] if int(j) not in removed]
    return removed_order, removed, routes


def destroy_critical_batch(
    *,
    routes: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    q_hint: Optional[Dict[int, float]] = None,
    min_k: int = 3,
    max_k: int = 6,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    Medium-size critical destroy:
      - pick one bottleneck route (highest tail q / longest route fallback)
      - remove one contiguous batch around a critical task in that route
    """
    routes = _dc_routes(routes)
    present: Set[int] = set()
    for seq in routes.values():
        present.update(int(x) for x in (seq or []))
    if not present:
        return [], set(), routes

    q_map: Dict[int, float] = {}
    if isinstance(q_hint, dict):
        for k, v in q_hint.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    q_map[kk] = vv
            except Exception:
                pass

    route_scores: List[Tuple[float, int, int]] = []
    for r, seq in routes.items():
        seq2 = [int(x) for x in (seq or [])]
        if not seq2:
            continue
        q_tail = max((float(q_map.get(int(j), 0.0)) for j in seq2), default=0.0)
        route_scores.append((float(q_tail), int(len(seq2)), int(r)))
    if not route_scores:
        return [], set(), routes
    route_scores.sort(reverse=True)
    r_star = int(route_scores[0][2])
    seq_star = [int(x) for x in (routes.get(r_star, []) or [])]
    if not seq_star:
        return [], set(), routes

    k_lo = max(2, int(min_k))
    k_hi = max(k_lo, min(int(max_k), len(seq_star)))
    k = int(rng.randint(k_lo, k_hi))

    idx_star = len(seq_star) - 1
    if q_map:
        best_q = -1e100
        best_idx = idx_star
        for i, j in enumerate(seq_star):
            qv = float(q_map.get(int(j), 0.0))
            if qv > best_q:
                best_q = qv
                best_idx = int(i)
        idx_star = int(best_idx)

    if len(seq_star) <= k:
        removed_order = list(seq_star)
    else:
        start_low = max(0, int(idx_star) - (k - 1))
        start_high = min(int(idx_star), len(seq_star) - k)
        start = int(rng.randint(start_low, start_high))
        removed_order = seq_star[start : start + k]

    removed = set(int(x) for x in removed_order)
    for r in list(routes.keys()):
        routes[int(r)] = [int(j) for j in routes[int(r)] if int(j) not in removed]
    return removed_order, removed, routes


def destroy_dual_tail_critical(
    *,
    routes: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    q_hint: Optional[Dict[int, float]] = None,
    tail_window: int = 12,
    max_per_route: int = 2,
) -> Tuple[List[int], Set[int], Dict[int, List[int]]]:
    """
    Tiny critical destroy for large instances:
      - choose 1-2 bottleneck routes
      - remove route-tail critical tasks (very small neighborhood)
    """
    routes = _dc_routes(routes)
    q_map: Dict[int, float] = {}
    if isinstance(q_hint, dict):
        for k, v in q_hint.items():
            try:
                kk = int(k)
                vv = float(v)
                if math.isfinite(vv):
                    q_map[kk] = vv
            except Exception:
                pass

    route_scores: List[Tuple[float, int, int]] = []
    for r, seq in routes.items():
        rr = int(r)
        seq2 = [int(x) for x in (seq or [])]
        if not seq2:
            continue
        win = max(1, min(int(tail_window), len(seq2)))
        tail = seq2[-win:]
        tail_q = max((float(q_map.get(int(j), 0.0)) for j in tail), default=0.0)
        route_scores.append((float(tail_q), int(len(seq2)), rr))
    if not route_scores:
        return [], set(), routes

    route_scores.sort(reverse=True)
    pick_routes = [int(route_scores[0][2])]
    if len(route_scores) >= 2:
        pick_routes.append(int(route_scores[1][2]))

    removed_order: List[int] = []
    removed_set: Set[int] = set()
    per_r_cap = max(1, int(max_per_route))
    for rr in pick_routes:
        seq = [int(x) for x in (routes.get(int(rr), []) or [])]
        if not seq:
            continue
        win = max(1, min(int(tail_window), len(seq)))
        tail = seq[-win:]
        if q_map:
            tail_sorted = sorted(tail, key=lambda j: float(q_map.get(int(j), 0.0)), reverse=True)
        else:
            tail_sorted = list(reversed(tail))
        for j in tail_sorted[:per_r_cap]:
            jj = int(j)
            if jj in removed_set:
                continue
            removed_order.append(jj)
            removed_set.add(jj)
        if (len(seq) >= 2) and (len(removed_order) < 4) and (rng.random() < 0.30):
            tail_prev = int(seq[-2])
            if tail_prev not in removed_set:
                removed_order.append(tail_prev)
                removed_set.add(tail_prev)

    if not removed_set:
        return [], set(), routes
    for r in list(routes.keys()):
        routes[int(r)] = [int(j) for j in routes[int(r)] if int(j) not in removed_set]
    return removed_order, removed_set, routes


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

    # [text_corrupted]
    max_evals_per_task: Optional[int] = None,
    max_evals_total: Optional[int] = None,
    copy_inputs: bool = True,

    # [text_corrupted]cheap [text_corrupted]
    preselect_m: int = 12,
    preselect_per_pos_shelves: int = 2,

    # ordered insertion and strict-constructor switches
    removed_order: Optional[List[int]] = None,
    strict_construct: bool = False,
    strict_bruteforce_fallback: bool = False,
    require_finite_eval: bool = False,
    eval_fail_cost: float = 1e30,
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

    # [text_corrupted]shelf_seq [text_corrupted]v [text_corrupted]
    chain_of = _build_chain_of_map(shelf_seq, task_shelf_mapping)
    shelf_idx = _shelf_index(shelf_seq)

    def _ws_rank(j: int) -> int:
        ws = int(pi.get(int(j), -1))
        return int(ws_pos.get(ws, {}).get(int(j), 10**9))

    # --------------------------
    # 1) [text_corrupted] removed_list
    # --------------------------
    if removed_order is not None:
        # [text_corrupted] & [text_corrupted]removed_set [text_corrupted]
        base_set = removed_set if removed_set else set(int(x) for x in removed_order)
        seen = set()
        ordered = []
        for x in removed_order:
            xx = int(x)
            if xx in base_set and xx not in seen:
                ordered.append(xx)
                seen.add(xx)

        # [text_corrupted] removed_set [text_corrupted]ws [text_corrupted]
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

    # ===== prechecker[text_corrupted]=====
    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )

    # succ_on_chain[text_corrupted]cheap [text_corrupted]
    succ_on_chain: Dict[int, int] = {}
    for t, prev in (pre.pred_on_chain or {}).items():
        if prev is not None:
            succ_on_chain[int(prev)] = int(t)

    # remaining_unfixed[text_corrupted]routes [text_corrupted]tail [text_corrupted]
    all_tasks: Set[int] = set(int(x) for x in evaluator.J)
    present_tasks: Set[int] = set()
    for seq in routes.values():
        present_tasks.update(int(x) for x in (seq or []))
    remaining_unfixed: Set[int] = (all_tasks - present_tasks) | set(int(x) for x in removed_list)

    # [text_corrupted]
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

    def _strict_bruteforce_pick(
        *,
        j: int,
        cand_shelves_local: List[int],
    ) -> Optional[Tuple[Tuple[int, int, int], float, int, int, int, List[int]]]:
        """
        Strict fallback for constructor mode:
        bypass precheck-pruned neighborhood and evaluate full insert moves.
        """
        shelf_set = {int(s) for s in (cand_shelves_local or [])}
        if (not shelf_set) or bool(force_all_shelves):
            shelf_set = {int(s) for s in evaluator.S}
        else:
            for s_all in evaluator.S:
                shelf_set.add(int(s_all))
        shelves = sorted(shelf_set)

        best_rec: Optional[Tuple[Tuple[int, int, int], float, int, int, int, List[int]]] = None
        for r in R_ids:
            base_seq = routes[int(r)]
            pos_list = _ws_block_boundary_positions(base_seq, evaluator.pi)
            seen_route_keys: Set[Tuple[int, Tuple[int, ...]]] = set()
            for pos in pos_list:
                seq_ins = list(base_seq)
                seq_ins.insert(int(pos), int(j))
                seq_ins = _normalize_one_route_ws_and_shelf(
                    seq_ins,
                    evaluator=evaluator,
                    chain_of=chain_of,
                    shelf_idx=shelf_idx,
                )
                route_key = (int(r), tuple(int(x) for x in seq_ins))
                if route_key in seen_route_keys:
                    continue
                seen_route_keys.add(route_key)

                old_seq = routes[int(r)]
                old_place = place.get(int(j), None)
                for ss in shelves:
                    routes[int(r)] = list(seq_ins)
                    place[int(j)] = int(ss)
                    try:
                        obj_f, det = _safe_evaluate(
                            evaluator,
                            routes,
                            shelf_seq,
                            place,
                            fail_cost=eval_fail_cost,
                        )
                        if require_finite_eval:
                            try:
                                obj_ff = float(obj_f)
                                if (not math.isfinite(obj_ff)) or (obj_ff >= float(eval_fail_cost) * 0.999999):
                                    continue
                            except Exception:
                                continue
                        key = _safe_infeas_key(det, evaluator, obj=obj_f)
                        rec = (tuple(key), float(obj_f), int(r), int(pos), int(ss), list(seq_ins))
                        if (best_rec is None) or ((rec[0], rec[1]) < (best_rec[0], best_rec[1])):
                            best_rec = rec
                    finally:
                        routes[int(r)] = old_seq
                        if old_place is None:
                            place.pop(int(j), None)
                        else:
                            place[int(j)] = int(old_place)
        return best_rec

    def _cheap_insert_score(
        *,
        j: int,
        r: int,
        pos: int,
        end_s: int,
        base_seq: List[int],
    ) -> float:
        """
        cheap [text_corrupted]
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

        # [text_corrupted] j [text_corrupted] tail [text_corrupted]
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

        # ===== 1) [text_corrupted] + cheap [text_corrupted]=====
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

                # [text_corrupted]route [text_corrupted]WS[text_corrupted]+ shelf slot[text_corrupted]
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

                # [text_corrupted]end_s
                ok_s: List[int] = []
                for s in cand_shelves:
                    ss = int(s)
                    ok, _ = pre.check_insert_candidate(
                        j=j,
                        r=r,
                        pos=pos,
                        end_s=ss,
                        routes=routes,      # base routes[text_corrupted]j[text_corrupted]
                        place=place,
                        used_tail_cells=used_tail,
                        strict_move1=False, # [text_corrupted]
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
            if strict_construct:
                if strict_bruteforce_fallback:
                    brute = _strict_bruteforce_pick(j=int(j), cand_shelves_local=cand_shelves)
                    if brute is not None:
                        _, _, rr, _, ss, seq_ins = brute
                        routes[int(rr)] = list(seq_ins)
                        place[int(j)] = int(ss)
                        remaining_unfixed.discard(int(j))
                        continue
                raise RuntimeError(f"strict_construct: no legal candidate for task {j}")
            if require_finite_eval and strict_bruteforce_fallback:
                brute = _strict_bruteforce_pick(j=int(j), cand_shelves_local=cand_shelves)
                if brute is not None:
                    _, _, rr, _, ss, seq_ins = brute
                    routes[int(rr)] = list(seq_ins)
                    place[int(j)] = int(ss)
                    remaining_unfixed.discard(int(j))
                    continue
            # [text_corrupted]
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

        # [text_corrupted] cheap [text_corrupted]preselect_m [text_corrupted]
        cheap_pool.sort(key=lambda x: x[0])
        cheap_pool = cheap_pool[: max(1, int(preselect_m))]

        # ===== 2) [text_corrupted]safe evaluate[text_corrupted]====
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
                obj_f, det = _safe_evaluate(
                    evaluator,
                    routes,
                    shelf_seq,
                    place,
                    fail_cost=eval_fail_cost,
                )
                if require_finite_eval:
                    try:
                        obj_ff = float(obj_f)
                        if (not math.isfinite(obj_ff)) or (obj_ff >= float(eval_fail_cost) * 0.999999):
                            continue
                    except Exception:
                        continue
                key = _safe_infeas_key(det, evaluator, obj=obj_f)
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
            if strict_construct:
                if strict_bruteforce_fallback:
                    brute = _strict_bruteforce_pick(j=int(j), cand_shelves_local=cand_shelves)
                    if brute is not None:
                        _, _, rr, _, ss, seq_ins = brute
                        routes[int(rr)] = list(seq_ins)
                        place[int(j)] = int(ss)
                        remaining_unfixed.discard(int(j))
                        continue
                raise RuntimeError(f"strict_construct: no evaluated candidate for task {j}")
            if require_finite_eval and strict_bruteforce_fallback:
                brute = _strict_bruteforce_pick(j=int(j), cand_shelves_local=cand_shelves)
                if brute is not None:
                    _, _, rr, _, ss, seq_ins = brute
                    routes[int(rr)] = list(seq_ins)
                    place[int(j)] = int(ss)
                    remaining_unfixed.discard(int(j))
                    continue
            # Fallback to cheapest unevaluated candidate when strict mode is off.
            _, rr, pp, ss, seq_ins = cheap_pool[0]
            routes[int(rr)] = list(seq_ins)
            place[j] = int(ss)
        else:
            # [text_corrupted]infeas_key [text_corrupted]obj
            scored_cands.sort(key=lambda x: (x[0], x[1]))
            if strict_construct:
                _, _, rr, pp, ss, seq_ins = scored_cands[0]
            else:
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

    # shared evaluation budgets across ordered insertion
    max_evals_per_task: Optional[int] = None,
    max_evals_total: Optional[int] = None,
    preselect_m: int = 12,
    preselect_per_pos_shelves: int = 2,
    strict_construct: bool = False,
    strict_bruteforce_fallback: bool = False,
    require_finite_eval: bool = False,
    eval_fail_cost: float = 1e30,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Ordered repair[text_corrupted] removed_order [text_corrupted]seed-first[text_corrupted]
    [text_corrupted]repair[text_corrupted]
    """
    removed_order = [int(x) for x in (removed_order or [])]
    removed_set = set(int(x) for x in removed_order)

    return repair_greedy_insert_with_place(
        routes=routes,
        shelf_seq=shelf_seq,
        place=place,
        evaluator=evaluator,
        removed=removed_set,
        removed_order=removed_order,  # [text_corrupted]
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
        preselect_m=preselect_m,
        preselect_per_pos_shelves=preselect_per_pos_shelves,
        copy_inputs=True,
        strict_construct=strict_construct,
        strict_bruteforce_fallback=strict_bruteforce_fallback,
        require_finite_eval=require_finite_eval,
        eval_fail_cost=eval_fail_cost,
    )


def repair_greedy_insert_with_place_difficult_ordered(
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
    max_evals_per_task: Optional[int] = None,
    max_evals_total: Optional[int] = None,
    preselect_m: int = 12,
    preselect_per_pos_shelves: int = 2,
    strict_construct: bool = False,
    strict_bruteforce_fallback: bool = False,
    require_finite_eval: bool = False,
    eval_fail_cost: float = 1e30,
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Repair with a harder-first insertion order.
    Tasks with larger nominal duration are inserted earlier to reduce
    downstream dead-ends on large instances.
    """
    removed_set = set(int(x) for x in (removed or set()))
    if not removed_set:
        return _dc_routes(routes), _dc_place(place)

    d_map = getattr(evaluator, "D", {}) or {}
    pi_map = getattr(evaluator, "pi", {}) or {}
    ws_idx = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})

    def _order_key(jj: int) -> Tuple[float, int, float]:
        j = int(jj)
        d_val = float(d_map.get(j, 0.0))
        ws = int(pi_map.get(j, -1))
        rank = int(ws_idx.get(ws, {}).get(j, 10 ** 9))
        return (-d_val, rank, rng.random())

    removed_order = sorted((int(x) for x in removed_set), key=_order_key)

    return repair_greedy_insert_with_place_ordered(
        routes=routes,
        shelf_seq=shelf_seq,
        place=place,
        evaluator=evaluator,
        removed_order=removed_order,
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
        preselect_m=preselect_m,
        preselect_per_pos_shelves=preselect_per_pos_shelves,
        strict_construct=strict_construct,
        strict_bruteforce_fallback=strict_bruteforce_fallback,
        require_finite_eval=require_finite_eval,
        eval_fail_cost=eval_fail_cost,
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

    # [text_corrupted]+ [text_corrupted]
    max_evals: Optional[int] = 200,
    cap_total: int = 60,
) -> Tuple[Dict[int, int], bool]:
    """
    place [text_corrupted]
      - [text_corrupted]top_k_try
      - [text_corrupted] S[text_corrupted]cap_total
      - [text_corrupted] max_evals [text_corrupted]
    """
    improved = False
    place = _dc_place(place)

    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    eval_used = 0
    budget = None if max_evals is None else max(0, int(max_evals))

    keys = [int(j) for j in place.keys()]

    # --- [text_corrupted]q [text_corrupted]---
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

    # [text_corrupted]
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

        # [text_corrupted]
        if j in global_set:
            # [text_corrupted] cap_total[text_corrupted]S[text_corrupted]
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
    [text_corrupted]j*[text_corrupted]q [text_corrupted]ws* [text_corrupted]
    [text_corrupted]ws* [text_corrupted]ws_fixed_seq [text_corrupted]
    [text_corrupted] makespan[text_corrupted] inf -> finite[text_corrupted]

    [text_corrupted]new_shelf_seq, improved, new_obj)
    """
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    q_map = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    if not q_map:
        return shelf_seq, False, base_obj

    # [text_corrupted]makespan [text_corrupted]
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
        # [text_corrupted]
        pri = 0 if ws == ws_star else 1
        # [text_corrupted]
        ord_in_ws = ws_idx.get(ws, {}).get(j, 10**9)
        return (pri, int(ord_in_ws), orig_pos)

    # [text_corrupted] A[text_corrupted]
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

    # [text_corrupted] B[text_corrupted]
    best_seq = shelf_seq
    best_obj = base_obj

    # [text_corrupted]
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
        # [text_corrupted]
        cand_pos = [i for i, j in enumerate(seq) if i > 0 and int(evaluator.pi.get(j, -1)) == ws_star]
        if not cand_pos:
            continue
        i = int(rng.choice(cand_pos))
        j = int(seq[i])
        # [text_corrupted]0..i-1[text_corrupted]
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
    max_evals: Optional[int] = 120,
) -> Tuple[Dict[int, List[int]], bool]:
    """
    [text_corrupted]first-improvement[text_corrupted]
    [text_corrupted] routes / place[text_corrupted]shelf_seq [text_corrupted]

    SPEED FIX:
      - routes [text_corrupted]routes_norm [text_corrupted]normalize [text_corrupted]
      - cand_shelf_seq [text_corrupted]shallow copy[text_corrupted]dict(...)[text_corrupted]shelf_seq
    """
    shelf_seq = _dc_shelf_seq(shelf_seq)

    # [text_corrupted]routes [text_corrupted] normalize [text_corrupted]
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)
    eval_budget = None if (max_evals is None) else max(0, int(max_evals))
    eval_used = 0

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
                if (eval_budget is not None) and (eval_budget <= 0):
                    break
                if pos == idx:
                    continue

                cand = seq[:]
                cand.pop(idx)
                cand.insert(pos, j)

                # [text_corrupted]shallow copy [text_corrupted]
                cand_shelf_seq = dict(shelf_seq)
                cand_shelf_seq[int(c)] = cand

                obj, _ = evaluator.evaluate(routes_norm, cand_shelf_seq, place)
                eval_used += 1
                if eval_budget is not None:
                    eval_budget -= 1
                if float(obj) < base_obj - 1e-9:
                    shelf_seq[int(c)] = cand
                    return shelf_seq, True
            if (eval_budget is not None) and (eval_budget <= 0):
                break
        if (eval_budget is not None) and (eval_budget <= 0):
            break

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
    max_evals: int = 10,  # [text_corrupted] 8~12[text_corrupted]
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    Funnel cross-vehicle:
      Generate candidates (no evaluate)
        -> RulePrecheck
        -> Proxy-LB [text_corrupted]/[text_corrupted]
        -> [text_corrupted]evaluate Top-K
    """

    routes = _dc_routes(routes)
    place = _dc_place(place)

    R_ids = sorted(int(r) for r in evaluator.R)
    for r in R_ids:
        routes.setdefault(int(r), [])

    # [text_corrupted]
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
        # [text_corrupted]
        if len(cands) > int(max_s_samples):
            head = cands[: int(max_s_samples)]
            tail = cands[int(max_s_samples):]
            if tail:
                head.append(int(rng.choice(tail)))
            cands = head
        # [text_corrupted]
        out, seen = [], set()
        for s in cands:
            s = int(s)
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    # [text_corrupted](lb, cand_routes, cand_place)
    cand_pool: List[Tuple[float, Dict[int, List[int]], Dict[int, int]]] = []

    # ========== 1) relocate [text_corrupted]==========
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
                    routes=routes,   # base routes[text_corrupted]j[text_corrupted]
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

                # [text_corrupted]LB >= base_obj => [text_corrupted]
                lb = proxy.solution_lb(routes=cand_routes, place=cand_place, stop_at=base_obj if math.isfinite(base_obj) else None)
                if math.isfinite(base_obj) and lb >= base_obj - 1e-9:
                    continue

                cand_pool.append((float(lb), cand_routes, cand_place))

    # ========== 2) swap [text_corrupted]==========
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

    # [text_corrupted]
    eval_k = min(max(1, int(max_evals)), 12, len(cand_pool))

    for i in range(eval_k):
        _, cand_routes, cand_place = cand_pool[i]
        # cand_routes [text_corrupted]route-level normalize[text_corrupted] normalize
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
    """[text_corrupted]1~2[text_corrupted]"""
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

                # [text_corrupted]block [text_corrupted]
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


def cross_vehicle_ejection_chain_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    evaluator_exact: Optional[RobustEvaluator],
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]],
    rng: random.Random,
    min_seg: int = 6,
    max_seg: int = 15,
    max_trials: int = 1,
    max_pos_samples: int = 3,
    max_s_samples: int = 2,
    max_evals: int = 3,
    max_exact_trials: int = 1,
    exact_gate_rel: float = 0.003,
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    Large-segment cross-vehicle ejection-chain:
      1) move a long segment from donor route A -> receiver route B
      2) eject a shorter segment from B -> another route C
      3) exact-gate only for clearly promising candidates
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, _ = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    base_exact: Optional[float] = None
    if evaluator_exact is not None:
        try:
            base_exact_v, _ = evaluator_exact.evaluate(routes_norm, shelf_seq, place)
            base_exact = float(base_exact_v)
            if not math.isfinite(float(base_exact)):
                base_exact = None
        except Exception:
            base_exact = None

    R_ids = sorted(int(r) for r in evaluator.R)
    seg_min_eff = max(2, int(min_seg))
    seg_max_eff = max(seg_min_eff, int(max_seg))
    if len(R_ids) < 2:
        return routes_norm, place, False, base_obj
    donor_ids = [int(r) for r in R_ids if len(routes_norm.get(int(r), [])) >= seg_min_eff]
    if not donor_ids:
        return routes_norm, place, False, base_obj

    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )
    proxy = FastProxyEvaluatorLB(evaluator, pre)

    def _sample_positions(seq: List[int], k: int) -> List[int]:
        all_pos = _ws_block_boundary_positions(seq, evaluator.pi)
        all_pos = sorted(set(int(p) for p in all_pos))
        if len(all_pos) <= int(k):
            return all_pos
        must = [0, len(seq)]
        mid = [int(p) for p in all_pos if int(p) not in set(must)]
        need = max(0, int(k) - len(set(must)))
        pick = rng.sample(mid, min(need, len(mid))) if (need > 0 and mid) else []
        return sorted(set(int(x) for x in (must + pick)))

    cand_pool: List[Tuple[float, Dict[int, List[int]], Dict[int, int]]] = []
    for _ in range(max(1, int(max_trials))):
        r_from = int(rng.choice(donor_ids))
        seq_from = [int(x) for x in (routes_norm.get(int(r_from), []) or [])]
        if len(seq_from) < seg_min_eff:
            continue
        r_to = int(rng.choice([r for r in R_ids if int(r) != int(r_from)]))
        seq_to = [int(x) for x in (routes_norm.get(int(r_to), []) or [])]

        seg_hi = min(seg_max_eff, len(seq_from))
        if seg_hi < seg_min_eff:
            continue
        seg_len = int(rng.randint(seg_min_eff, seg_hi))
        i0 = int(rng.randrange(0, len(seq_from) - seg_len + 1))
        moved_block = [int(x) for x in seq_from[i0 : i0 + seg_len]]
        seq_from_rem = [int(x) for x in (seq_from[:i0] + seq_from[i0 + seg_len :])]

        r_back_choices = [int(r) for r in R_ids if int(r) != int(r_to)]
        if not r_back_choices:
            continue

        pos_to_cands = _sample_positions(seq_to, max(2, int(max_pos_samples)))
        for pos_to in pos_to_cands:
            pos_to = int(pos_to)
            seq_to_ins = list(seq_to)
            for off, jj in enumerate(moved_block):
                seq_to_ins.insert(int(pos_to) + int(off), int(jj))

            if len(seq_to_ins) <= 2:
                continue
            e_len_hi = min(max(2, seg_len // 2), len(seq_to_ins) - 1)
            e_len_lo = 2
            if e_len_hi < e_len_lo:
                continue
            e_len = int(rng.randint(e_len_lo, e_len_hi))
            ins_lo = int(pos_to)
            ins_hi = int(pos_to + seg_len - 1)
            e_starts = [
                int(st)
                for st in range(0, len(seq_to_ins) - e_len + 1)
                if ((int(st) + e_len - 1) < ins_lo) or (int(st) > ins_hi)
            ]
            if not e_starts:
                continue
            e_start = int(rng.choice(e_starts))
            ejected = [int(x) for x in seq_to_ins[e_start : e_start + e_len]]
            seq_to_fin = [int(x) for x in (seq_to_ins[:e_start] + seq_to_ins[e_start + e_len :])]

            r_back = min(r_back_choices, key=lambda rr: len(routes_norm.get(int(rr), [])))
            if rng.random() < 0.35:
                r_back = int(rng.choice(r_back_choices))
            seq_back = [int(x) for x in (routes_norm.get(int(r_back), []) or [])]
            pos_back_cands = _sample_positions(seq_back, max(2, int(max_pos_samples) - 1))
            pos_back = int(rng.choice(pos_back_cands))
            seq_back_new = list(seq_back)
            for off, jj in enumerate(ejected):
                seq_back_new.insert(int(pos_back) + int(off), int(jj))

            cand_routes = _dc_routes(routes_norm)
            cand_routes[int(r_from)] = _normalize_one_route_ws_blocks(seq_from_rem, evaluator)
            cand_routes[int(r_to)] = _normalize_one_route_ws_blocks(seq_to_fin, evaluator)
            cand_routes[int(r_back)] = _normalize_one_route_ws_blocks(seq_back_new, evaluator)
            cand_routes = _normalize_ws_blocks(cand_routes, evaluator)

            moved_tasks = set(int(x) for x in moved_block)
            moved_tasks.update(int(x) for x in ejected)
            cand_place = _dc_place(place)
            for jj in moved_tasks:
                s_cands = _cand_end_shelves_for_task(
                    int(jj),
                    place=cand_place,
                    S_near_by_j=S_near_by_j,
                    evaluator=evaluator,
                    shelf_seq=shelf_seq,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init_override,
                    must_include_prev_or_init=True,
                    max_keep=max(6, int(max_s_samples) * 2),
                )
                s_cands = _augment_shelf_candidates(
                    int(jj),
                    s_cands,
                    evaluator=evaluator,
                    rng=rng,
                    force_all_shelves=False,
                    extra_random_shelves=0,
                    cap_total=max(8, int(max_s_samples) + 2),
                )
                if not s_cands:
                    continue
                keep = [int(x) for x in s_cands[: max(1, int(max_s_samples))]]
                pick_s = int(keep[0])
                if len(keep) >= 2 and rng.random() < 0.28:
                    pick_s = int(rng.choice(keep))
                cand_place[int(jj)] = int(pick_s)

            lb = proxy.solution_lb(
                routes=cand_routes,
                place=cand_place,
                stop_at=base_obj if math.isfinite(base_obj) else None,
            )
            if math.isfinite(base_obj) and (float(lb) >= float(base_obj) - 1e-9):
                continue
            cand_pool.append((float(lb), cand_routes, cand_place))

    if not cand_pool:
        return routes_norm, place, False, base_obj
    cand_pool.sort(key=lambda x: x[0])

    eval_k = min(max(1, int(max_evals)), len(cand_pool))
    exact_limit = max(0, int(max_exact_trials))
    exact_used = 0
    gate_abs = max(1.0, abs(float(base_obj)) * max(0.0, float(exact_gate_rel)))
    for idx in range(eval_k):
        _, rr, pp = cand_pool[idx]
        obj_fast, _ = evaluator.evaluate(rr, shelf_seq, pp)
        obj_fast = float(obj_fast)
        if not (obj_fast < float(base_obj) - 1e-9):
            continue

        need_exact = bool(
            (evaluator_exact is not None)
            and (exact_used < exact_limit)
            and (obj_fast <= float(base_obj) - float(gate_abs))
        )
        if need_exact:
            exact_used += 1
            obj_exact, _ = evaluator_exact.evaluate(rr, shelf_seq, pp)
            obj_exact = float(obj_exact)
            if base_exact is None:
                if math.isfinite(obj_exact):
                    return rr, pp, True, obj_exact
            elif math.isfinite(obj_exact) and (obj_exact < float(base_exact) - 1e-9):
                return rr, pp, True, obj_exact
            continue

        return rr, pp, True, obj_fast

    return routes_norm, place, False, base_obj


def ws_micro_reorder_idle_swap_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 8,
    max_swap_span: int = 3,
    idle_window_cap: float = 240.0,
) -> Tuple[Dict[int, List[int]], bool, float]:
    """
    WS-neighborhood micro reorder:
      - swap two nearby tasks only when their q-time gap is within idle_window_cap
      - keep operation tiny (adjacent/nearby) to preserve feasibility robustness
    """
    routes = _dc_routes(routes)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)
    if not math.isfinite(base_obj):
        return routes_norm, False, base_obj

    q_map = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    pi = getattr(evaluator, "pi", {}) or {}
    route_ids = [int(r) for r in evaluator.R if len(routes_norm.get(int(r), [])) >= 2]
    if not route_ids:
        return routes_norm, False, base_obj

    span_eff = max(1, int(max_swap_span))
    window_eff = max(0.0, float(idle_window_cap))
    for _ in range(max(1, int(max_trials))):
        r = int(rng.choice(route_ids))
        seq = [int(x) for x in (routes_norm.get(int(r), []) or [])]
        if len(seq) < 2:
            continue

        i = int(rng.randrange(0, len(seq) - 1))
        j_hi = min(len(seq) - 1, int(i + span_eff))
        if j_hi <= i:
            continue
        j = int(rng.randint(i + 1, j_hi))
        a = int(seq[i])
        b = int(seq[j])
        if int(pi.get(a, -1)) == int(pi.get(b, -1)):
            continue

        qa = q_map.get(int(a), None)
        qb = q_map.get(int(b), None)
        if (qa is not None) and (qb is not None):
            try:
                if abs(float(qa) - float(qb)) > window_eff:
                    continue
            except Exception:
                pass

        cand_seq = list(seq)
        cand_seq[i], cand_seq[j] = cand_seq[j], cand_seq[i]
        cand_seq = _normalize_one_route_ws_blocks(cand_seq, evaluator)
        if cand_seq == seq:
            continue

        cand_routes = _dc_routes(routes_norm)
        cand_routes[int(r)] = cand_seq
        cand_obj, _ = evaluator.evaluate(cand_routes, shelf_seq, place)
        cand_obj = float(cand_obj)
        if cand_obj < base_obj - 1e-9:
            return cand_routes, True, cand_obj

    return routes_norm, False, base_obj


def intra_two_opt_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    rng: random.Random,
    max_trials: int = 2,
) -> Tuple[Dict[int, List[int]], bool, float]:
    """[text_corrupted] 2-opt[text_corrupted]"""
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
    """[text_corrupted] Or-opt[text_corrupted] k(1/2) [text_corrupted]"""
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
    """[text_corrupted] Cmax"""
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

                # [text_corrupted]j [text_corrupted]
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


def rebalance_bottleneck_route_once(
    *,
    routes: Dict[int, List[int]],
    place: Dict[int, int],
    shelf_seq: Dict[int, List[int]],
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init_override: Optional[Dict[int, int]],
    rng: random.Random,
    tail_k: int = 6,
    max_pos_samples: int = 6,
    max_s_samples: int = 3,
    max_evals: int = 12,
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    Large-scale escape/local-improvement:
      - find bottleneck AGV route by q-tail
      - relocate one tail task to another AGV using precheck + proxy funnel
      - exact-evaluate only top candidates
    """
    routes = _dc_routes(routes)
    place = _dc_place(place)
    routes_norm = _normalize_ws_blocks(routes, evaluator)
    base_obj, diag = evaluator.evaluate(routes_norm, shelf_seq, place)
    base_obj = float(base_obj)

    R_ids = sorted(int(r) for r in evaluator.R)
    if len(R_ids) < 2:
        return routes, place, False, base_obj

    q_map = (diag.get("q", {}) if isinstance(diag, dict) else {}) or {}
    best_r = None
    best_q = -1.0
    for r in R_ids:
        seq = [int(x) for x in (routes_norm.get(int(r), []) or [])]
        if not seq:
            continue
        q_tail = max((float(q_map.get(int(j), 0.0)) for j in seq), default=0.0)
        if q_tail > best_q:
            best_q = float(q_tail)
            best_r = int(r)
    if best_r is None:
        return routes, place, False, base_obj

    seq_star = [int(x) for x in (routes_norm.get(int(best_r), []) or [])]
    if not seq_star:
        return routes, place, False, base_obj
    tail = seq_star[-min(max(1, int(tail_k)), len(seq_star)) :]
    if not tail:
        return routes, place, False, base_obj

    pre = build_rule_prechecker(
        evaluator=evaluator,
        shelf_seq=shelf_seq,
        task_shelf_mapping=task_shelf_mapping,
        shelf_init_override=shelf_init_override,
    )
    proxy = FastProxyEvaluatorLB(evaluator, pre)

    cand_pool: List[Tuple[float, Dict[int, List[int]], Dict[int, int]]] = []
    for j in reversed(tail):
        j = int(j)
        for r_to in R_ids:
            r_to = int(r_to)
            if r_to == int(best_r):
                continue

            seq_from = [int(x) for x in routes_norm[int(best_r)] if int(x) != int(j)]
            seq_to = [int(x) for x in (routes_norm.get(int(r_to), []) or [])]
            pos_all = _ws_block_boundary_positions(seq_to, evaluator.pi)
            if len(pos_all) > int(max_pos_samples):
                must = [0, len(seq_to)]
                mid = [int(p) for p in pos_all if int(p) not in set(must)]
                need = max(0, int(max_pos_samples) - len(must))
                pick = rng.sample(mid, min(need, len(mid))) if (need > 0 and mid) else []
                pos_all = sorted(set(must + pick))

            s_cands = _cand_end_shelves_for_task(
                int(j),
                place=place,
                S_near_by_j=S_near_by_j,
                evaluator=evaluator,
                shelf_seq=shelf_seq,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init_override,
                must_include_prev_or_init=True,
                max_keep=max(3, int(max_s_samples)),
            )
            s_cands = _augment_shelf_candidates(
                int(j),
                s_cands,
                evaluator=evaluator,
                rng=rng,
                force_all_shelves=False,
                extra_random_shelves=0,
                cap_total=max(6, int(max_s_samples) + 2),
            )
            if not s_cands:
                s_cands = [int(min(int(s) for s in evaluator.S))]
            if len(s_cands) > int(max_s_samples):
                s_cands = s_cands[: int(max_s_samples)]

            used_tail = pre.build_used_tail_cells(place=place, ignore_tasks={int(j)})
            for pos in pos_all:
                pos = int(pos)
                seq_to_new = list(seq_to)
                seq_to_new.insert(pos, int(j))
                seq_to_new = _normalize_one_route_ws_blocks(seq_to_new, evaluator)
                for ss in s_cands:
                    ss = int(ss)
                    ok, _ = pre.check_insert_candidate(
                        j=int(j),
                        r=int(r_to),
                        pos=int(pos),
                        end_s=int(ss),
                        routes=routes_norm,
                        place=place,
                        used_tail_cells=used_tail,
                        strict_move1=False,
                    )
                    if not ok:
                        continue
                    cand_routes = _dc_routes(routes_norm)
                    cand_routes[int(best_r)] = list(seq_from)
                    cand_routes[int(r_to)] = list(seq_to_new)
                    cand_place = _dc_place(place)
                    cand_place[int(j)] = int(ss)
                    lb = proxy.solution_lb(
                        routes=cand_routes,
                        place=cand_place,
                        stop_at=base_obj if math.isfinite(base_obj) else None,
                    )
                    if math.isfinite(base_obj) and (lb >= base_obj - 1e-9):
                        continue
                    cand_pool.append((float(lb), cand_routes, cand_place))

    if not cand_pool:
        return routes_norm, place, False, base_obj
    cand_pool.sort(key=lambda x: x[0])

    k_eval = min(max(1, int(max_evals)), len(cand_pool))
    for idx in range(k_eval):
        _, rr, pp = cand_pool[idx]
        obj, _ = evaluator.evaluate(rr, shelf_seq, pp)
        obj = float(obj)
        if obj < base_obj - 1e-9:
            return rr, pp, True, obj
    return routes_norm, place, False, base_obj


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
    allow_non_improving: bool = False,  # [text_corrupted]shake [text_corrupted]
) -> Tuple[Dict[int, List[int]], Dict[int, int], bool, float]:
    """
    [text_corrupted]2-opt*[text_corrupted]
    - allow_non_improving=False[text_corrupted]
    - allow_non_improving=True [text_corrupted] SA [text_corrupted]
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
    ALNS [text_corrupted]
      - roulette wheel [text_corrupted]
      - segment-based [text_corrupted]w = (1-r)*w + r*(avg_score)
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
            # [text_corrupted]
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

def _flatten_route_order(routes: Dict[int, List[int]], all_tasks: List[int]) -> List[int]:
    seen: Set[int] = set()
    out: List[int] = []
    for r in sorted(routes.keys()):
        for j in (routes.get(r, []) or []):
            jj = int(j)
            if (jj in seen) or (jj not in all_tasks):
                continue
            seen.add(jj)
            out.append(jj)
    for j in all_tasks:
        if j not in seen:
            out.append(int(j))
    return out

def _flatten_ws_fixed_order(evaluator: RobustEvaluator, all_tasks: List[int]) -> List[int]:
    ws_fixed = getattr(evaluator, "ws_fixed_seq", {}) or {}
    keep = set(int(x) for x in all_tasks)
    seen: Set[int] = set()
    out: List[int] = []
    for ws in sorted(ws_fixed.keys()):
        for j in (ws_fixed.get(ws, []) or []):
            jj = int(j)
            if (jj in keep) and (jj not in seen):
                seen.add(jj)
                out.append(jj)
    for j in all_tasks:
        if j not in seen:
            out.append(int(j))
    return out

def _construct_strong_initial_solution(
    init,
    *,
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]],
    shelf_init: Optional[Dict[int, int]],
    seed: int,
    max_tries: int = 4,
    time_budget_sec: float = 6.0,
    only_when_infeasible: bool = True,
    require_feasible: bool = False,
):
    """
    Strong constructor for from-scratch feasibility.
    If require_feasible=True, use a more exhaustive construction strategy.
    """
    base = copy.deepcopy(init)
    base.routes = _dc_routes(base.routes)
    base.place = _dc_place(base.place)
    base.shelf_seq = _dc_shelf_seq(base.shelf_seq)
    base.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        base.shelf_seq,
        evaluator=evaluator,
        task_shelf_mapping=task_shelf_mapping,
    )
    base.routes = _normalize_ws_blocks(base.routes, evaluator)
    base.routes = normalize_routes_by_shelf_seq_order(base.routes, base.shelf_seq, task_shelf_mapping)

    base_obj, base_det = _safe_evaluate(evaluator, base.routes, base.shelf_seq, base.place)
    base_key = _safe_infeas_key(base_det, evaluator, obj=base_obj)
    if only_when_infeasible and tuple(base_key) == (0, 0, 0):
        return base

    all_tasks = sorted(int(j) for j in evaluator.J)
    if not all_tasks:
        return base

    rng = random.Random(int(seed) + 2027)
    t0 = time.perf_counter()
    t_budget = max(0.0, float(time_budget_sec))

    route_order = _flatten_route_order(base.routes, all_tasks)
    ws_order = _flatten_ws_fixed_order(evaluator, all_tasks)
    dur_order = sorted(all_tasks, key=lambda j: float(getattr(evaluator, "D", {}).get(int(j), 0.0)), reverse=True)
    rnd_order = list(all_tasks)
    rng.shuffle(rnd_order)

    strategy_orders = [ws_order, dur_order, route_order, rnd_order]
    local_best = base
    local_best_obj = float(base_obj)
    local_best_key = tuple(base_key)
    feas_best = copy.deepcopy(base) if tuple(base_key) == (0, 0, 0) else None
    feas_best_obj = float(base_obj) if tuple(base_key) == (0, 0, 0) else float("inf")

    R_ids = sorted(int(r) for r in evaluator.R)
    tries = max(1, int(max_tries))

    def _time_up() -> bool:
        return (t_budget > 0.0) and ((time.perf_counter() - t0) >= t_budget)

    # Stage A: strict ordered reconstruction.
    for k in range(tries):
        if _time_up():
            break
        order = list(strategy_orders[k % len(strategy_orders)])
        if k >= len(strategy_orders):
            rng.shuffle(order)

        routes0 = {int(r): [] for r in R_ids}
        place0: Dict[int, int] = {}

        if require_feasible:
            per_task_eval = 36 if k == 0 else 28
            total_eval = max(1400, 520 + 120 * len(R_ids))
            place_try = 14
            extra_shelves = 10
            force_all = True
            strict_brute = True
        else:
            per_task_eval = 22 if k == 0 else 16
            total_eval = max(320, 180 + 40 * len(R_ids))
            place_try = 10
            extra_shelves = 3
            force_all = bool(k % 2 == 1)
            strict_brute = False

        try:
            routes_c, place_c = repair_greedy_insert_with_place_ordered(
                routes=routes0,
                shelf_seq=base.shelf_seq,
                place=place0,
                evaluator=evaluator,
                removed_order=order,
                rng=rng,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init,
                max_place_try_each=place_try,
                top_k_random=1,
                extra_random_shelves=extra_shelves,
                force_all_shelves=force_all,
                max_agv_candidates=None,
                max_pos_per_route=None,
                max_evals_per_task=per_task_eval,
                max_evals_total=total_eval,
                strict_construct=True,
                strict_bruteforce_fallback=strict_brute,
            )
        except Exception:
            continue

        routes_c = _normalize_ws_blocks(routes_c, evaluator)
        routes_c = normalize_routes_by_shelf_seq_order(routes_c, base.shelf_seq, task_shelf_mapping)
        obj_c, det_c = _safe_evaluate(evaluator, routes_c, base.shelf_seq, place_c)
        key_c = _safe_infeas_key(det_c, evaluator, obj=obj_c)

        cand = copy.deepcopy(base)
        cand.routes = routes_c
        cand.place = place_c
        cand.shelf_seq = _dc_shelf_seq(base.shelf_seq)

        if tuple(key_c) == (0, 0, 0) and float(obj_c) < float(feas_best_obj) - 1e-9:
            feas_best = cand
            feas_best_obj = float(obj_c)
        if (tuple(key_c) < tuple(local_best_key)) or (tuple(key_c) == tuple(local_best_key) and float(obj_c) < float(local_best_obj) - 1e-9):
            local_best = cand
            local_best_obj = float(obj_c)
            local_best_key = tuple(key_c)
        if tuple(key_c) == (0, 0, 0):
            return cand

    # Stage B: broad fallback reconstruction.
    if tuple(local_best_key) != (0, 0, 0):
        b_tries = 3 if (not require_feasible) else max(6, tries)
        for _ in range(b_tries):
            if _time_up():
                break
            order = list(ws_order)
            rng.shuffle(order)
            routes0 = {int(r): [] for r in R_ids}
            place0: Dict[int, int] = {}
            try:
                routes_c, place_c = repair_greedy_insert_with_place_ordered(
                    routes=routes0,
                    shelf_seq=base.shelf_seq,
                    place=place0,
                    evaluator=evaluator,
                    removed_order=order,
                    rng=rng,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    max_place_try_each=12,
                    top_k_random=1,
                    extra_random_shelves=(12 if require_feasible else 6),
                    force_all_shelves=True,
                    max_agv_candidates=None,
                    max_pos_per_route=None,
                    max_evals_per_task=(40 if require_feasible else 26),
                    max_evals_total=max((2200 if require_feasible else 500), 240 + 60 * len(R_ids)),
                    strict_construct=bool(require_feasible),
                    strict_bruteforce_fallback=bool(require_feasible),
                )
            except Exception:
                continue

            routes_c = _normalize_ws_blocks(routes_c, evaluator)
            routes_c = normalize_routes_by_shelf_seq_order(routes_c, base.shelf_seq, task_shelf_mapping)
            obj_c, det_c = _safe_evaluate(evaluator, routes_c, base.shelf_seq, place_c)
            key_c = _safe_infeas_key(det_c, evaluator, obj=obj_c)

            cand = copy.deepcopy(base)
            cand.routes = routes_c
            cand.place = place_c
            cand.shelf_seq = _dc_shelf_seq(base.shelf_seq)
            if tuple(key_c) == (0, 0, 0) and float(obj_c) < float(feas_best_obj) - 1e-9:
                feas_best = cand
                feas_best_obj = float(obj_c)
            if (tuple(key_c) < tuple(local_best_key)) or (tuple(key_c) == tuple(local_best_key) and float(obj_c) < float(local_best_obj) - 1e-9):
                local_best = cand
                local_best_obj = float(obj_c)
                local_best_key = tuple(key_c)
            if tuple(key_c) == (0, 0, 0):
                return cand

    # Stage C: exhaustive restarts if feasibility is mandatory.
    if require_feasible and tuple(local_best_key) != (0, 0, 0):
        c_tries = max(8, 2 * tries)
        for k in range(c_tries):
            if _time_up():
                break
            order = list(all_tasks)
            if (k % 2) == 0:
                rng.shuffle(order)
            else:
                order.sort(key=lambda jj: (float(getattr(evaluator, "D", {}).get(int(jj), 0.0)), rng.random()), reverse=True)

            routes0 = {int(r): [] for r in R_ids}
            place0: Dict[int, int] = {}
            try:
                routes_c, place_c = repair_greedy_insert_with_place_ordered(
                    routes=routes0,
                    shelf_seq=base.shelf_seq,
                    place=place0,
                    evaluator=evaluator,
                    removed_order=order,
                    rng=rng,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    max_place_try_each=16,
                    top_k_random=1,
                    extra_random_shelves=16,
                    force_all_shelves=True,
                    max_agv_candidates=None,
                    max_pos_per_route=None,
                    max_evals_per_task=48,
                    max_evals_total=max(3200, 800 + 160 * len(R_ids)),
                    strict_construct=True,
                    strict_bruteforce_fallback=True,
                )
            except Exception:
                continue

            routes_c = _normalize_ws_blocks(routes_c, evaluator)
            routes_c = normalize_routes_by_shelf_seq_order(routes_c, base.shelf_seq, task_shelf_mapping)
            obj_c, det_c = _safe_evaluate(evaluator, routes_c, base.shelf_seq, place_c)
            key_c = _safe_infeas_key(det_c, evaluator, obj=obj_c)

            cand = copy.deepcopy(base)
            cand.routes = routes_c
            cand.place = place_c
            cand.shelf_seq = _dc_shelf_seq(base.shelf_seq)
            if tuple(key_c) == (0, 0, 0):
                return cand
            if (tuple(key_c) < tuple(local_best_key)) or (tuple(key_c) == tuple(local_best_key) and float(obj_c) < float(local_best_obj) - 1e-9):
                local_best = cand
                local_best_obj = float(obj_c)
                local_best_key = tuple(key_c)

    if feas_best is not None:
        return feas_best
    return local_best


def build_feasible_initial_solution(
    init,
    *,
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init: Optional[Dict[int, int]] = None,
    seed: int = 0,
    rounds: int = 6,
    round_time_budget_sec: float = 20.0,
):
    """
    No-history feasible initializer.
    Repeatedly runs aggressive strong-constructor rounds until a feasible solution is found.
    """
    best = copy.deepcopy(init)
    best.routes = _dc_routes(best.routes)
    best.place = _dc_place(best.place)
    best.shelf_seq = _dc_shelf_seq(best.shelf_seq)

    best_obj, best_det = _safe_evaluate(evaluator, best.routes, best.shelf_seq, best.place)
    best_key = _safe_infeas_key(best_det, evaluator, obj=best_obj)
    if tuple(best_key) == (0, 0, 0):
        return best

    cur = copy.deepcopy(best)
    tries = max(1, int(rounds))
    for k in range(tries):
        cand = _construct_strong_initial_solution(
            cur,
            evaluator=evaluator,
            S_near_by_j=S_near_by_j,
            task_shelf_mapping=task_shelf_mapping,
            shelf_init=shelf_init,
            seed=int(seed) + 104729 * int(k + 1),
            max_tries=6 + (k // 2),
            time_budget_sec=float(round_time_budget_sec),
            only_when_infeasible=False,
            require_feasible=True,
        )
        cand.routes = _normalize_ws_blocks(_dc_routes(cand.routes), evaluator)
        cand.routes = normalize_routes_by_shelf_seq_order(cand.routes, cand.shelf_seq, task_shelf_mapping)
        obj_c, det_c = _safe_evaluate(evaluator, cand.routes, cand.shelf_seq, cand.place)
        key_c = _safe_infeas_key(det_c, evaluator, obj=obj_c)

        if tuple(key_c) == (0, 0, 0):
            return cand

        # If still infeasible, run a short repair ALNS on exact evaluator.
        repair_budget = max(8.0, 0.6 * float(round_time_budget_sec))
        cand_repair = alns_minimize(
            init=cand,
            evaluator=evaluator,
            iters=max(180, 60 + 20 * (k + 1)),
            start_T=1.1,
            cool=0.997,
            S_near_by_j=S_near_by_j,
            seed=int(seed) + 8191 * int(k + 1),
            task_shelf_mapping=task_shelf_mapping,
            enable_place_tune=True,
            enable_shelf_tune=True,
            shelf_init=shelf_init,
            speed_profile="balanced",
            feasible_first=False,
            enable_strong_init=False,
            time_budget_sec=float(repair_budget),
            relabel_interval=20,
            relabel_eval_top_k=10,
        )
        cand_repair.routes = _normalize_ws_blocks(_dc_routes(cand_repair.routes), evaluator)
        cand_repair.routes = normalize_routes_by_shelf_seq_order(cand_repair.routes, cand_repair.shelf_seq, task_shelf_mapping)
        obj_r, det_r = _safe_evaluate(evaluator, cand_repair.routes, cand_repair.shelf_seq, cand_repair.place)
        key_r = _safe_infeas_key(det_r, evaluator, obj=obj_r)
        if tuple(key_r) == (0, 0, 0):
            return cand_repair

        if (tuple(key_c) < tuple(best_key)) or (tuple(key_c) == tuple(best_key) and float(obj_c) < float(best_obj) - 1e-9):
            best = cand
            best_key = tuple(key_c)
            best_obj = float(obj_c)
        if (tuple(key_r) < tuple(best_key)) or (tuple(key_r) == tuple(best_key) and float(obj_r) < float(best_obj) - 1e-9):
            best = cand_repair
            best_key = tuple(key_r)
            best_obj = float(obj_r)

        cur = cand_repair if tuple(key_r) <= tuple(key_c) else cand

    return best


def build_exact_feasible_seed(
    init,
    *,
    evaluator_partial_hard: RobustEvaluator,
    evaluator_exact: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init: Optional[Dict[int, int]] = None,
    seed: int = 0,
    attempts: int = 4,
    time_budget_sec: float = 14.0,
    per_attempt_eval_budget: int = 2800,
):
    """
    Pure from-scratch exact-feasibility driven constructor.
    Build with hard constraints while allowing incompleteness during insertion,
    then validate against exact hard evaluator.
    """
    base = copy.deepcopy(init)
    base.routes = _dc_routes(base.routes)
    base.place = _dc_place(base.place)
    base.shelf_seq = _dc_shelf_seq(base.shelf_seq)

    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    base.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        base.shelf_seq,
        evaluator=evaluator_partial_hard,
        task_shelf_mapping=clean_map,
    )
    base.routes = _normalize_ws_blocks(base.routes, evaluator_partial_hard)
    base.routes = normalize_routes_by_shelf_seq_order(base.routes, base.shelf_seq, clean_map)

    key_feas = (0, 0, 0)
    best_obj, best_det = _safe_evaluate(evaluator_exact, base.routes, base.shelf_seq, base.place)
    best_key = _safe_infeas_key(best_det, evaluator_exact, obj=best_obj)
    best_sol = copy.deepcopy(base)
    if tuple(best_key) == key_feas:
        return best_sol

    all_tasks = sorted(int(j) for j in evaluator_partial_hard.J)
    if not all_tasks:
        return best_sol

    chain_seen: Set[int] = set()
    chain_order: List[int] = []
    for c in sorted(base.shelf_seq.keys()):
        for j in (base.shelf_seq.get(int(c), []) or []):
            jj = int(j)
            if (jj in all_tasks) and (jj not in chain_seen):
                chain_seen.add(jj)
                chain_order.append(jj)
    for j in all_tasks:
        if int(j) not in chain_seen:
            chain_order.append(int(j))

    rng = random.Random(int(seed) + 7331)
    t0 = time.perf_counter()
    t_budget = max(0.0, float(time_budget_sec))
    R_ids = sorted(int(r) for r in evaluator_partial_hard.R)
    if not R_ids:
        return best_sol

    ws_order = _flatten_ws_fixed_order(evaluator_partial_hard, all_tasks)
    dur_order = sorted(all_tasks, key=lambda j: float(getattr(evaluator_partial_hard, "D", {}).get(int(j), 0.0)), reverse=True)
    route_order = _flatten_route_order(base.routes, all_tasks)
    order_bank = [chain_order, ws_order, dur_order, route_order]

    tries = max(1, int(attempts))
    for k in range(tries):
        if (t_budget > 0.0) and ((time.perf_counter() - t0) >= t_budget):
            break

        order = list(order_bank[k % len(order_bank)])
        if k >= len(order_bank):
            rng.shuffle(order)
        if (k % 3) == 2:
            order.sort(
                key=lambda j: (
                    float(getattr(evaluator_partial_hard, "D", {}).get(int(j), 0.0)),
                    rng.random(),
                ),
                reverse=True,
            )

        routes0 = {int(r): [] for r in R_ids}
        place0: Dict[int, int] = {}

        eval_budget = max(1000, int(float(per_attempt_eval_budget) * (1.0 + 0.35 * float(k))))
        eval_per_task = max(20, min(96, int(eval_budget // max(1, len(all_tasks) // 2))))
        extra_shelves = 8 + min(8, k)
        place_try = 12 + min(6, k)

        try:
            routes_c, place_c = repair_greedy_insert_with_place_ordered(
                routes=routes0,
                shelf_seq=base.shelf_seq,
                place=place0,
                evaluator=evaluator_partial_hard,
                removed_order=order,
                rng=rng,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=clean_map,
                shelf_init_override=shelf_init,
                max_place_try_each=place_try,
                top_k_random=1,
                extra_random_shelves=extra_shelves,
                force_all_shelves=True,
                max_agv_candidates=None,
                max_pos_per_route=None,
                max_evals_per_task=eval_per_task,
                max_evals_total=eval_budget,
                strict_construct=True,
                strict_bruteforce_fallback=True,
                require_finite_eval=True,
                eval_fail_cost=1e30,
            )
        except Exception:
            continue

        routes_c = _normalize_ws_blocks(routes_c, evaluator_partial_hard)
        routes_c = normalize_routes_by_shelf_seq_order(routes_c, base.shelf_seq, clean_map)
        obj_c, det_c = _safe_evaluate(evaluator_exact, routes_c, base.shelf_seq, place_c)
        key_c = _safe_infeas_key(det_c, evaluator_exact, obj=obj_c)

        cand = copy.deepcopy(base)
        cand.routes = routes_c
        cand.place = place_c
        cand.shelf_seq = _dc_shelf_seq(base.shelf_seq)

        if tuple(key_c) == key_feas:
            return cand

        if (tuple(key_c) < tuple(best_key)) or (tuple(key_c) == tuple(best_key) and float(obj_c) < float(best_obj) - 1e-9):
            best_sol = cand
            best_key = tuple(key_c)
            best_obj = float(obj_c)

    return best_sol


def build_full_coverage_seed(
    init,
    *,
    evaluator: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init: Optional[Dict[int, int]] = None,
    seed: int = 0,
):
    """
    Fast no-history constructor:
    build a full-coverage seed (all tasks scheduled exactly once) with deterministic rules.
    """
    out = copy.deepcopy(init)
    out.shelf_seq = _dc_shelf_seq(getattr(out, "shelf_seq", {}) or {})
    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    out.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        out.shelf_seq,
        evaluator=evaluator,
        task_shelf_mapping=clean_map,
    )
    chain_of = _build_chain_of_map(out.shelf_seq, clean_map)

    R_ids = sorted(int(r) for r in evaluator.R)
    routes: Dict[int, List[int]] = {int(r): [] for r in R_ids}
    place: Dict[int, int] = {}
    if not R_ids:
        out.routes = routes
        out.place = place
        return out

    all_tasks = sorted(int(j) for j in evaluator.J)
    task_order = _flatten_ws_fixed_order(evaluator, all_tasks)
    D = getattr(evaluator, "D", {}) or {}
    loads: Dict[int, float] = {int(r): 0.0 for r in R_ids}

    rng = random.Random(int(seed) + 4099)
    pi = getattr(evaluator, "pi", {}) or {}

    # predecessor on chain from shelf_seq
    pred_on_chain: Dict[int, int] = {}
    for c, seq in (out.shelf_seq or {}).items():
        arr = [int(x) for x in (seq or [])]
        for i in range(1, len(arr)):
            pred_on_chain[int(arr[i])] = int(arr[i - 1])

    all_cells_sorted = sorted(int(s) for s in evaluator.S)
    shelf_init_map = shelf_init if shelf_init is not None else (getattr(evaluator, "shelf_init", {}) or {})

    for j in task_order:
        jj = int(j)
        wsj = int(pi.get(jj, -1))

        # choose AGV by load, tie-break by route length and id.
        agv_rank = sorted(
            R_ids,
            key=lambda rr: (
                float(loads.get(int(rr), 0.0)),
                len(routes.get(int(rr), [])),
                int(rr),
            ),
        )
        r_pick = int(agv_rank[0])
        routes[r_pick].append(jj)
        loads[r_pick] = float(loads.get(r_pick, 0.0)) + float(D.get(jj, 0.0))

        cand_cells = _cand_end_shelves_for_task(
            jj,
            place=place,
            S_near_by_j=S_near_by_j,
            evaluator=evaluator,
            shelf_seq=out.shelf_seq,
            task_shelf_mapping=clean_map,
            shelf_init_override=shelf_init_map,
            must_include_prev_or_init=True,
            max_keep=10,
        )

        # deterministic priority: predecessor cell > shelf_init cell > nearest candidates > all cells.
        pri: List[int] = []
        prev_j = pred_on_chain.get(jj, None)
        if prev_j is not None and int(prev_j) in place:
            pri.append(int(place[int(prev_j)]))

        cc = int(chain_of.get(jj, -1))
        if cc in shelf_init_map:
            try:
                pri.append(int(shelf_init_map[cc]))
            except Exception:
                pass

        if jj in getattr(out, "place", {}):
            try:
                pri.append(int(out.place[jj]))
            except Exception:
                pass

        for s in cand_cells:
            pri.append(int(s))
        for s in all_cells_sorted:
            pri.append(int(s))

        seen: Set[int] = set()
        ordered_cells: List[int] = []
        for s in pri:
            if s not in seen:
                seen.add(s)
                ordered_cells.append(int(s))

        # light randomization for tie diversification while preserving priority chunks.
        if len(ordered_cells) > 4:
            head = ordered_cells[:3]
            tail = ordered_cells[3:]
            rng.shuffle(tail)
            ordered_cells = head + tail

        place[jj] = int(ordered_cells[0]) if ordered_cells else int(all_cells_sorted[0])

    routes = _normalize_ws_blocks(routes, evaluator)
    routes = normalize_routes_by_shelf_seq_order(routes, out.shelf_seq, clean_map)

    out.routes = routes
    out.place = place
    return out


def build_ws_round_robin_seed(
    init,
    *,
    evaluator: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
):
    """
    Deterministic from-scratch seed:
      1) task order follows ws_fixed_seq
      2) assign tasks round-robin to AGVs
      3) place each task back to its chain's initial shelf cell
    """
    out = copy.deepcopy(init)
    out.routes = _dc_routes(getattr(out, "routes", {}) or {})
    out.place = _dc_place(getattr(out, "place", {}) or {})
    out.shelf_seq = _dc_shelf_seq(getattr(out, "shelf_seq", {}) or {})

    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    out.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        out.shelf_seq,
        evaluator=evaluator,
        task_shelf_mapping=clean_map,
    )

    all_tasks = sorted(int(j) for j in evaluator.J)
    if not all_tasks:
        return out
    R_ids = sorted(int(r) for r in evaluator.R)
    if not R_ids:
        return out

    ws_order = _flatten_ws_fixed_order(evaluator, all_tasks)
    routes: Dict[int, List[int]] = {int(r): [] for r in R_ids}
    for idx, j in enumerate(ws_order):
        rr = int(R_ids[int(idx) % len(R_ids)])
        routes[rr].append(int(j))

    # Keep shelf_seq consistent with ws order to reduce chain/dispatch deadlocks.
    ws_pos = _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})
    pi_map = getattr(evaluator, "pi", {}) or {}
    for c in sorted(out.shelf_seq.keys()):
        seq = [int(x) for x in (out.shelf_seq.get(int(c), []) or [])]
        seq.sort(key=lambda j: (
            int(pi_map.get(int(j), 10**9)),
            int(ws_pos.get(int(pi_map.get(int(j), -1)), {}).get(int(j), 10**9)),
            int(j),
        ))
        out.shelf_seq[int(c)] = seq

    # Place each task to its chain home cell whenever available.
    chain_of = _build_chain_of_map(out.shelf_seq, clean_map)
    shelf_init_map = getattr(evaluator, "shelf_init", {}) or {}
    fallback_cell = min(int(s) for s in evaluator.S) if getattr(evaluator, "S", None) else 0
    place: Dict[int, int] = {}
    for j in all_tasks:
        jj = int(j)
        c = int(chain_of.get(jj, clean_map.get(jj, -1)))
        if c in shelf_init_map:
            place[jj] = int(shelf_init_map[int(c)])
        elif jj in out.place:
            place[jj] = int(out.place[jj])
        else:
            place[jj] = int(fallback_cell)

    routes = _normalize_ws_blocks(routes, evaluator)
    routes = normalize_routes_by_shelf_seq_order(routes, out.shelf_seq, clean_map)
    out.routes = routes
    out.place = place
    return out


def build_safe_chain_ws_seed(
    init,
    *,
    evaluator_exact: RobustEvaluator,
    S_near_by_j: Dict[int, List[int]],
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    shelf_init: Optional[Dict[int, int]] = None,
    seed: int = 0,
    tries: int = 80,
):
    """
    Deterministic-feasibility fallback:
      1) build a precedence topo order from (ws_fixed_seq + shelf_seq chain order)
      2) assign all tasks to one AGV in topo order
      3) try multiple placement assignments and keep the first exact-feasible one
    """
    out = copy.deepcopy(init)
    out.routes = _dc_routes(getattr(out, "routes", {}) or {})
    out.place = _dc_place(getattr(out, "place", {}) or {})
    out.shelf_seq = _dc_shelf_seq(getattr(out, "shelf_seq", {}) or {})
    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    out.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        out.shelf_seq,
        evaluator=evaluator_exact,
        task_shelf_mapping=clean_map,
    )
    # Reorder each shelf chain to be more consistent with WS fixed order,
    # reducing precedence-cycle risk in the fallback seed.
    ws_pos = _ws_index(getattr(evaluator_exact, "ws_fixed_seq", {}) or {})
    pi = getattr(evaluator_exact, "pi", {}) or {}
    shelf_seq_adj: Dict[int, List[int]] = _dc_shelf_seq(out.shelf_seq)
    for c in sorted(shelf_seq_adj.keys()):
        seq = [int(x) for x in (shelf_seq_adj.get(int(c), []) or [])]
        tagged = list(enumerate(seq))
        tagged.sort(
            key=lambda it: (
                int(pi.get(int(it[1]), 10**9)),
                int(ws_pos.get(int(pi.get(int(it[1]), -1)), {}).get(int(it[1]), 10**9)),
                int(it[0]),
            )
        )
        shelf_seq_adj[int(c)] = [int(j) for _, j in tagged]
    out.shelf_seq = shelf_seq_adj

    tasks = sorted(int(j) for j in evaluator_exact.J)
    if not tasks:
        return out
    task_set = set(tasks)

    succ: Dict[int, Set[int]] = defaultdict(set)
    indeg: Dict[int, int] = {int(j): 0 for j in tasks}

    def _add_edge(a: int, b: int) -> None:
        aa = int(a)
        bb = int(b)
        if (aa not in task_set) or (bb not in task_set) or (aa == bb):
            return
        if bb in succ[aa]:
            return
        succ[aa].add(bb)
        indeg[bb] = int(indeg.get(bb, 0)) + 1

    ws_fixed = getattr(evaluator_exact, "ws_fixed_seq", {}) or {}
    for ws in sorted(ws_fixed.keys()):
        seq = [int(x) for x in (ws_fixed.get(ws, []) or []) if int(x) in task_set]
        for i in range(1, len(seq)):
            _add_edge(int(seq[i - 1]), int(seq[i]))

    for c in sorted(out.shelf_seq.keys()):
        seq = [int(x) for x in (out.shelf_seq.get(c, []) or []) if int(x) in task_set]
        for i in range(1, len(seq)):
            _add_edge(int(seq[i - 1]), int(seq[i]))

    rng = random.Random(int(seed) + 19001)
    q = [int(j) for j in tasks if int(indeg.get(int(j), 0)) == 0]
    q.sort()
    topo: List[int] = []
    while q:
        j = int(q.pop(0))
        topo.append(int(j))
        nxt = sorted(int(x) for x in succ.get(int(j), set()))
        for v in nxt:
            indeg[v] = int(indeg.get(v, 0)) - 1
            if int(indeg[v]) == 0:
                q.append(int(v))
        q.sort()
    if len(topo) != len(tasks):
        # Cycle fallback: keep ws-first order to stay deterministic.
        topo = _flatten_ws_fixed_order(evaluator_exact, tasks)

    R_ids = sorted(int(r) for r in evaluator_exact.R)
    if not R_ids:
        return out
    lead_r = int(R_ids[0])
    routes_template: Dict[int, List[int]] = {int(r): [] for r in R_ids}
    routes_template[lead_r] = list(topo)
    routes_template = _normalize_ws_blocks(routes_template, evaluator_exact)
    routes_template = normalize_routes_by_shelf_seq_order(routes_template, out.shelf_seq, clean_map)

    all_cells = sorted(int(s) for s in evaluator_exact.S)
    if not all_cells:
        out.routes = routes_template
        out.place = {}
        return out

    tail_tasks: List[int] = []
    for c in sorted(out.shelf_seq.keys()):
        seq = [int(x) for x in (out.shelf_seq.get(c, []) or []) if int(x) in task_set]
        if seq:
            tail_tasks.append(int(seq[-1]))
    tail_set = set(int(x) for x in tail_tasks)

    base_place: Dict[int, int] = {}
    used_tail_cells: Set[int] = set()
    for j in tail_tasks:
        cands = _cand_end_shelves_for_task(
            int(j),
            place=base_place,
            S_near_by_j=S_near_by_j,
            evaluator=evaluator_exact,
            shelf_seq=out.shelf_seq,
            task_shelf_mapping=clean_map,
            shelf_init_override=shelf_init,
            must_include_prev_or_init=True,
            max_keep=20,
        )
        if not cands:
            cands = list(all_cells)
        pick = None
        for s in cands:
            ss = int(s)
            if ss not in used_tail_cells:
                pick = int(ss)
                break
        if pick is None:
            pick = int(cands[0]) if cands else int(all_cells[0])
        base_place[int(j)] = int(pick)
        used_tail_cells.add(int(pick))

    best_sol = copy.deepcopy(out)
    best_obj = float("inf")
    best_key = (10**9, 1, 10**12)

    n_tries = max(1, int(tries))
    for k in range(n_tries):
        place_try: Dict[int, int] = dict(base_place)
        for j in topo:
            jj = int(j)
            if jj in tail_set:
                continue
            cands = _cand_end_shelves_for_task(
                jj,
                place=place_try,
                S_near_by_j=S_near_by_j,
                evaluator=evaluator_exact,
                shelf_seq=out.shelf_seq,
                task_shelf_mapping=clean_map,
                shelf_init_override=shelf_init,
                must_include_prev_or_init=True,
                max_keep=12,
            )
            if not cands:
                cands = list(all_cells)
            if k == 0:
                pick_s = int(cands[0])
            else:
                topk = cands[: min(len(cands), 4)]
                pick_s = int(rng.choice(topk if topk else cands))
            place_try[jj] = int(pick_s)

        routes_try = _dc_routes(routes_template)
        obj_try, det_try = _safe_evaluate(evaluator_exact, routes_try, out.shelf_seq, place_try)
        key_try = _safe_infeas_key(det_try, evaluator_exact, obj=obj_try)
        if tuple(key_try) == (0, 0, 0):
            sol = copy.deepcopy(out)
            sol.routes = routes_try
            sol.place = place_try
            sol.shelf_seq = _dc_shelf_seq(out.shelf_seq)
            return sol
        if (tuple(key_try) < tuple(best_key)) or (tuple(key_try) == tuple(best_key) and float(obj_try) < float(best_obj) - 1e-9):
            best_key = tuple(key_try)
            best_obj = float(obj_try)
            best_sol = copy.deepcopy(out)
            best_sol.routes = routes_try
            best_sol.place = place_try
            best_sol.shelf_seq = _dc_shelf_seq(out.shelf_seq)

    return best_sol


def repair_shelf_seq_by_chain_violations(
    sol,
    *,
    evaluator_exact: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    max_steps: int = 80,
):
    """
    Exact-diagnostic guided shelf_seq repair:
    repeatedly read chain_order_violation_detail and reorder shelf sequence accordingly.
    """
    cur = copy.deepcopy(sol)
    cur.routes = _dc_routes(getattr(cur, "routes", {}) or {})
    cur.place = _dc_place(getattr(cur, "place", {}) or {})
    cur.shelf_seq = _dc_shelf_seq(getattr(cur, "shelf_seq", {}) or {})
    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    cur.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        cur.shelf_seq,
        evaluator=evaluator_exact,
        task_shelf_mapping=clean_map,
    )

    def _find_task_pos(routes_dict: Dict[int, List[int]], task_id: int) -> Tuple[Optional[int], Optional[int]]:
        tj = int(task_id)
        for rr, seq_rr in routes_dict.items():
            seq_i = [int(x) for x in (seq_rr or [])]
            for ii, vv in enumerate(seq_i):
                if int(vv) == tj:
                    return int(rr), int(ii)
        return None, None

    for _ in range(max(1, int(max_steps))):
        cur.routes = _normalize_ws_blocks(cur.routes, evaluator_exact)
        cur.routes = normalize_routes_by_shelf_seq_order(cur.routes, cur.shelf_seq, clean_map)
        obj, det = _safe_evaluate(evaluator_exact, cur.routes, cur.shelf_seq, cur.place)
        if math.isfinite(float(obj)):
            return cur

        info = (det.get("chain_order_violation_detail", {}) if isinstance(det, dict) else {}) or {}
        if not info:
            break

        c = int(info.get("chain", -1))
        prev_j = int(info.get("prev", -1))
        cur_j = int(info.get("cur", -1))
        if c not in cur.shelf_seq:
            break
        seq = [int(x) for x in (cur.shelf_seq.get(c, []) or [])]
        if (prev_j not in seq) or (cur_j not in seq):
            break

        # Route-level repair first: place the violating successor right after predecessor.
        r_prev, i_prev_route = _find_task_pos(cur.routes, prev_j)
        r_cur, i_cur_route = _find_task_pos(cur.routes, cur_j)
        if (r_prev is not None) and (i_prev_route is not None) and (r_cur is not None) and (i_cur_route is not None):
            seq_cur = [int(x) for x in (cur.routes.get(int(r_cur), []) or [])]
            seq_prev = [int(x) for x in (cur.routes.get(int(r_prev), []) or [])]
            if 0 <= int(i_cur_route) < len(seq_cur):
                seq_cur.pop(int(i_cur_route))
                cur.routes[int(r_cur)] = seq_cur
                # Re-locate predecessor index after deletion in same route case.
                seq_prev = [int(x) for x in (cur.routes.get(int(r_prev), []) or [])]
                try:
                    i_prev_now = int(seq_prev.index(int(prev_j)))
                except Exception:
                    i_prev_now = max(0, min(len(seq_prev), int(i_prev_route)))
                ins_pos = max(0, min(len(seq_prev), i_prev_now + 1))
                seq_prev.insert(ins_pos, int(cur_j))
                cur.routes[int(r_prev)] = seq_prev
                # Keep end shelf consistent along a violating pair to reduce hard-repair burden.
                if int(prev_j) in cur.place:
                    cur.place[int(cur_j)] = int(cur.place[int(prev_j)])

        i_prev = int(seq.index(prev_j))
        i_cur = int(seq.index(cur_j))
        if i_cur <= i_prev:
            # Violation usually means "cur is too early": move cur right after prev in this chain.
            val = int(seq.pop(i_cur))
            if i_cur < i_prev:
                i_prev -= 1
            seq.insert(i_prev + 1, val)
        else:
            # Already after prev but still too early in time: push cur later.
            if i_cur + 1 < len(seq):
                seq[i_cur], seq[i_cur + 1] = seq[i_cur + 1], seq[i_cur]
            elif i_cur > 0:
                val = int(seq.pop(i_cur))
                seq.append(val)
            else:
                break
        cur.shelf_seq[int(c)] = seq

    return cur


def repair_chain_violations_by_route_relink(
    sol,
    *,
    evaluator_exact: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]] = None,
    max_steps: int = 120,
):
    """
    Exact-diagnostic guided chain repair:
    move violating successor task onto predecessor route and delay it in both route and shelf chain.
    """
    cur = copy.deepcopy(sol)
    cur.routes = _dc_routes(getattr(cur, "routes", {}) or {})
    cur.place = _dc_place(getattr(cur, "place", {}) or {})
    cur.shelf_seq = _dc_shelf_seq(getattr(cur, "shelf_seq", {}) or {})
    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    cur.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        cur.shelf_seq,
        evaluator=evaluator_exact,
        task_shelf_mapping=clean_map,
    )

    def _find_task_pos(routes_dict: Dict[int, List[int]], task_id: int) -> Tuple[Optional[int], Optional[int]]:
        tj = int(task_id)
        for rr, seq_rr in routes_dict.items():
            seq_i = [int(x) for x in (seq_rr or [])]
            for ii, vv in enumerate(seq_i):
                if int(vv) == tj:
                    return int(rr), int(ii)
        return None, None

    best = copy.deepcopy(cur)
    best_obj, best_det = _safe_evaluate(evaluator_exact, best.routes, best.shelf_seq, best.place)
    best_key = tuple(_safe_infeas_key(best_det, evaluator_exact, obj=best_obj))

    pair_hits: Dict[Tuple[int, int], int] = defaultdict(int)
    for _ in range(max(1, int(max_steps))):
        cur.routes = _normalize_ws_blocks(cur.routes, evaluator_exact)
        cur.routes = normalize_routes_by_shelf_seq_order(cur.routes, cur.shelf_seq, clean_map)
        obj, det = _safe_evaluate(evaluator_exact, cur.routes, cur.shelf_seq, cur.place)
        key = tuple(_safe_infeas_key(det, evaluator_exact, obj=obj))
        if (key < best_key) or (key == best_key and float(obj) < float(best_obj) - 1e-9):
            best = copy.deepcopy(cur)
            best_key = key
            best_obj = float(obj)
        if key == (0, 0, 0):
            return cur

        info = (det.get("chain_order_violation_detail", {}) if isinstance(det, dict) else {}) or {}
        if not info:
            break
        c = int(info.get("chain", -1))
        prev_j = int(info.get("prev", -1))
        cur_j = int(info.get("cur", -1))
        if (prev_j <= 0) or (cur_j <= 0):
            break

        pair = (int(prev_j), int(cur_j))
        pair_hits[pair] = int(pair_hits.get(pair, 0)) + 1
        if int(pair_hits[pair]) > 8:
            # Avoid over-fixing one pair and collapsing route diversity.
            break

        # 1) Shelf-chain delay for successor.
        if c in cur.shelf_seq:
            seq_c = [int(x) for x in (cur.shelf_seq.get(int(c), []) or [])]
            if (prev_j in seq_c) and (cur_j in seq_c):
                i_prev = int(seq_c.index(prev_j))
                i_cur = int(seq_c.index(cur_j))
                val = int(seq_c.pop(i_cur))
                if i_cur < i_prev:
                    i_prev -= 1
                ins_pos = max(0, min(len(seq_c), i_prev + 1))
                seq_c.insert(ins_pos, val)
                cur.shelf_seq[int(c)] = seq_c

        # 2) Route-level delay: put successor behind predecessor on predecessor AGV.
        r_prev, i_prev_route = _find_task_pos(cur.routes, prev_j)
        r_cur, i_cur_route = _find_task_pos(cur.routes, cur_j)
        if (r_prev is None) or (i_prev_route is None) or (r_cur is None) or (i_cur_route is None):
            continue

        seq_cur = [int(x) for x in (cur.routes.get(int(r_cur), []) or [])]
        if 0 <= int(i_cur_route) < len(seq_cur):
            seq_cur.pop(int(i_cur_route))
            cur.routes[int(r_cur)] = seq_cur

        seq_prev = [int(x) for x in (cur.routes.get(int(r_prev), []) or [])]
        try:
            i_prev_now = int(seq_prev.index(int(prev_j)))
        except Exception:
            i_prev_now = max(0, min(len(seq_prev), int(i_prev_route)))

        ins_pos = max(0, min(len(seq_prev), i_prev_now + 1))
        seq_prev.insert(ins_pos, int(cur_j))
        cur.routes[int(r_prev)] = seq_prev

        if int(prev_j) in cur.place:
            cur.place[int(cur_j)] = int(cur.place[int(prev_j)])

    return best


def _build_milp_warm_hint_from_solution(
    *,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    evaluator: RobustEvaluator,
) -> Dict[str, Any]:
    j_set = set(int(x) for x in (getattr(evaluator, "J", set()) or set()))
    s_set = set(int(x) for x in (getattr(evaluator, "S", {}).keys() or []))
    j0 = _as_int_int_dict(getattr(evaluator, "J0", {}) or {})
    jd = _as_int_int_dict(getattr(evaluator, "Jd", {}) or {})
    j_i = _as_int_int_dict(getattr(evaluator, "J_I", {}) or {})

    w_map: Dict[int, int] = {}
    z_list: List[Tuple[int, int, int]] = []

    for r_raw, seq_raw in (routes or {}).items():
        try:
            r = int(r_raw)
        except Exception:
            continue
        seq = [int(x) for x in (seq_raw or []) if int(x) in j_set]
        if not seq:
            continue

        for j in seq:
            w_map[int(j)] = int(r)

        j0_r = j0.get(int(r), None)
        jd_r = jd.get(int(r), None)
        prev = int(j0_r) if j0_r is not None else None
        for j in seq:
            if prev is not None:
                z_list.append((int(prev), int(j), int(r)))
            prev = int(j)
        if prev is not None and (jd_r is not None):
            z_list.append((int(prev), int(jd_r), int(r)))

    x_map: Dict[int, int] = {}
    for j_raw, s_raw in (place or {}).items():
        try:
            j = int(j_raw)
            s = int(s_raw)
        except Exception:
            continue
        if (j in j_set) and (s in s_set):
            x_map[int(j)] = int(s)

    imm_list: List[Tuple[int, int, int]] = []
    for c_raw, seq_raw in (shelf_seq or {}).items():
        try:
            c = int(c_raw)
        except Exception:
            continue
        vt = j_i.get(int(c), None)
        if vt is None:
            continue
        seq = [int(x) for x in (seq_raw or []) if int(x) in j_set]
        if not seq:
            continue
        imm_list.append((int(vt), int(seq[0]), int(c)))
        for a, b in zip(seq[:-1], seq[1:]):
            imm_list.append((int(a), int(b), int(c)))

    return {
        "w": w_map,
        "z": z_list,
        "x": x_map,
        "immediate": imm_list,
    }


def _load_milp_bundle_solution(
    bundle_path: str,
    *,
    evaluator: RobustEvaluator,
) -> Optional[Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]]:
    if (not bundle_path) or (not os.path.exists(bundle_path)):
        return None
    try:
        with open(bundle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    j_set = set(int(x) for x in (getattr(evaluator, "J", set()) or set()))
    s_set = set(int(x) for x in (getattr(evaluator, "S", {}).keys() or []))

    routes_raw = data.get("routes", {}) if isinstance(data, dict) else {}
    shelf_raw = data.get("shelf_seq", {}) if isinstance(data, dict) else {}
    place_raw = data.get("place", {}) if isinstance(data, dict) else {}

    if not isinstance(routes_raw, dict):
        return None

    routes: Dict[int, List[int]] = {}
    for r_raw, seq_raw in routes_raw.items():
        try:
            r = int(r_raw)
        except Exception:
            continue
        routes[int(r)] = [int(x) for x in (seq_raw or []) if int(x) in j_set]

    shelf_seq: Dict[int, List[int]] = {}
    if isinstance(shelf_raw, dict):
        for c_raw, seq_raw in shelf_raw.items():
            try:
                c = int(c_raw)
            except Exception:
                continue
            shelf_seq[int(c)] = [int(x) for x in (seq_raw or []) if int(x) in j_set]

    place: Dict[int, int] = {}
    if isinstance(place_raw, dict):
        for j_raw, s_raw in place_raw.items():
            try:
                j = int(j_raw)
                s = int(s_raw)
            except Exception:
                continue
            if (j in j_set) and (s in s_set):
                place[int(j)] = int(s)

    return routes, shelf_seq, place


def _try_milp_polish_endgame(
    sol,
    *,
    evaluator: RobustEvaluator,
    task_shelf_mapping: Optional[Dict[int, int]],
    seed: int,
    time_limit_sec: float,
    lock_immediate: bool = True,
) -> Tuple[Any, float, Tuple[int, int, int], bool, str]:
    base_sol = copy.deepcopy(sol)
    base_sol.routes = _dc_routes(getattr(base_sol, "routes", {}) or {})
    base_sol.place = _dc_place(getattr(base_sol, "place", {}) or {})
    base_sol.shelf_seq = _dc_shelf_seq(getattr(base_sol, "shelf_seq", {}) or {})

    clean_map = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=False)
    base_sol.routes = _normalize_ws_blocks(base_sol.routes, evaluator)
    base_sol.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
        base_sol.shelf_seq,
        evaluator=evaluator,
        task_shelf_mapping=clean_map,
    )
    base_sol.routes = normalize_routes_by_shelf_seq_order(base_sol.routes, base_sol.shelf_seq, clean_map)

    base_obj, base_det = _safe_evaluate(evaluator, base_sol.routes, base_sol.shelf_seq, base_sol.place)
    base_obj = float(base_obj)
    base_key = tuple(_safe_infeas_key(base_det, evaluator, obj=base_obj))

    tl = float(time_limit_sec or 0.0)
    if tl <= 0.0:
        return base_sol, base_obj, base_key, False, "skip(time_limit<=0)"
    if (not math.isfinite(base_obj)) or (base_key != (0, 0, 0)):
        return base_sol, base_obj, base_key, False, "skip(base_not_feasible)"

    required_attrs = [
        "J",
        "R",
        "S",
        "pi",
        "D",
        "J0",
        "Jd",
        "J_I",
        "shelf_init",
        "agv_init",
        "d_s_pi",
        "d_pi_s",
        "d_s_s",
        "ws_fixed_seq",
    ]
    missing_attrs = [nm for nm in required_attrs if not hasattr(evaluator, nm)]
    if missing_attrs:
        miss = ",".join(str(x) for x in missing_attrs[:6])
        return base_sol, base_obj, base_key, False, f"skip(missing_eval_attrs:{miss})"

    try:
        import optimization_model
        from gurobipy import GRB
    except Exception as e:
        return base_sol, base_obj, base_key, False, f"skip(milp_import:{type(e).__name__})"

    j_set = set(int(x) for x in (getattr(evaluator, "J", set()) or set()))
    if not j_set:
        return base_sol, base_obj, base_key, False, "skip(empty_J)"

    s_map_raw = getattr(evaluator, "S", {}) or {}
    if not isinstance(s_map_raw, dict) or (not s_map_raw):
        return base_sol, base_obj, base_key, False, "skip(empty_S)"
    s_map = {int(s): v for s, v in s_map_raw.items()}

    agv_init = _as_int_int_dict(getattr(evaluator, "agv_init", {}) or {})
    r_raw = getattr(evaluator, "R", {}) or {}
    if isinstance(r_raw, dict):
        r_map = {int(r): r_raw[r] for r in r_raw.keys()}
    else:
        fallback_cell = next(iter(s_map.keys()))
        r_ids = sorted(int(x) for x in (r_raw or []))
        r_map = {int(r): int(agv_init.get(int(r), fallback_cell)) for r in r_ids}
    if not r_map:
        return base_sol, base_obj, base_key, False, "skip(empty_R)"

    pi = _as_int_int_dict(getattr(evaluator, "pi", {}) or {})
    if len(pi) < len(j_set):
        return base_sol, base_obj, base_key, False, "skip(pi_incomplete)"
    d_src = getattr(evaluator, "D", {}) or {}
    d_map: Dict[int, float] = {}
    for j in j_set:
        try:
            d_map[int(j)] = float(d_src[int(j)])
        except Exception:
            continue
    if len(d_map) != len(j_set):
        return base_sol, base_obj, base_key, False, "skip(D_incomplete)"

    j0 = _as_int_int_dict(getattr(evaluator, "J0", {}) or {})
    jd = _as_int_int_dict(getattr(evaluator, "Jd", {}) or {})
    j_i = _as_int_int_dict(getattr(evaluator, "J_I", {}) or {})
    shelf_data = _as_int_int_dict(getattr(evaluator, "shelf_init", {}) or {})
    ws_fixed_seq = _dc_shelf_seq(getattr(evaluator, "ws_fixed_seq", {}) or {})
    d_s_pi = {tuple(map(int, k)): float(v) for k, v in (getattr(evaluator, "d_s_pi", {}) or {}).items()}
    d_pi_s = {tuple(map(int, k)): float(v) for k, v in (getattr(evaluator, "d_pi_s", {}) or {}).items()}
    d_s_s = {tuple(map(int, k)): float(v) for k, v in (getattr(evaluator, "d_s_s", {}) or {}).items()}

    if not j0 or not jd or not j_i or not shelf_data or not agv_init:
        return base_sol, base_obj, base_key, False, "skip(core_maps_empty)"

    task_shelf_map = _build_chain_of_map(base_sol.shelf_seq, clean_map)
    task_shelf_map = {int(j): int(c) for j, c in task_shelf_map.items() if int(j) in j_set}
    for c_raw, seq_raw in (base_sol.shelf_seq or {}).items():
        try:
            c = int(c_raw)
        except Exception:
            continue
        for j in (seq_raw or []):
            jj = int(j)
            if jj in j_set and jj not in task_shelf_map:
                task_shelf_map[int(jj)] = int(c)
    if len(task_shelf_map) != len(j_set):
        return base_sol, base_obj, base_key, False, "skip(task_shelf_map_incomplete)"

    warm_hint = _build_milp_warm_hint_from_solution(
        routes=base_sol.routes,
        shelf_seq=base_sol.shelf_seq,
        place=base_sol.place,
        evaluator=evaluator,
    )
    warm_hint["cmax"] = float(base_obj)

    shelf_ids = set(int(x) for x in shelf_data.keys())
    used_shelves = set(int(c) for c in task_shelf_map.values())
    unused_shelves = set(int(sid) for sid in shelf_ids if sid not in used_shelves)
    shelf_virtual_tasks = {int(sid): int(j_i[int(sid)]) for sid in shelf_ids if int(sid) in j_i}
    j_i_si: Dict[int, int] = {}
    for c_raw, seq_raw in (base_sol.shelf_seq or {}).items():
        try:
            c = int(c_raw)
        except Exception:
            continue
        seq = [int(x) for x in (seq_raw or []) if int(x) in j_set]
        if seq:
            j_i_si[int(c)] = int(seq[0])

    tasks = {int(j): (int(pi[int(j)]), float(d_map[int(j)]), None, None) for j in sorted(j_set)}
    gamma_now = int(getattr(evaluator, "gamma", 0) or 0)
    stamp = int(time.time() * 1000) % 1_000_000_000
    file_prefix = f"alns_milp_polish_g{gamma_now}_s{int(seed)}_{stamp}"
    bundle_path = os.path.join(
        os.path.dirname(__file__),
        "solution_exports",
        f"{file_prefix}_bundle_gamma{gamma_now}.json",
    )

    lock_hint = {"immediate": True} if bool(lock_immediate) else None

    try:
        _, model, _, _ = optimization_model.optimize_warehouse(
            R=r_map,
            S=s_map,
            K={},
            tasks=tasks,
            AGV_positions=dict(agv_init),
            agv_positions_map={},
            bj={},
            hj={},
            map_obj=None,
            task_shelf_mapping=task_shelf_map,
            J_I=j_i,
            J_E=set(),
            J=j_set,
            J0=j0,
            Jd=jd,
            J_I_SI=j_i_si,
            shelf_data=shelf_data,
            agv_data=agv_init,
            shelf_virtual_tasks=shelf_virtual_tasks,
            unused_shelves=unused_shelves,
            file_prefix=file_prefix,
            gamma_budget=gamma_now,
            ws_fixed_seq=ws_fixed_seq,
            warm_start=None,
            warm_hint=warm_hint,
            lock_hint=lock_hint,
            d_s_pi_in=d_s_pi,
            d_pi_s_in=d_pi_s,
            d_s_s_in=d_s_s,
            time_limit=float(tl),
            no_improve_limit=min(90.0, float(tl)),
            bridge_quiet=True,
        )
    except Exception as e:
        return base_sol, base_obj, base_key, False, f"error(milp_run:{type(e).__name__})"

    sol_count = int(getattr(model, "SolCount", 0))
    status = int(getattr(model, "Status", -1))
    if sol_count <= 0:
        return base_sol, base_obj, base_key, False, f"no_incumbent(status={status})"

    loaded = _load_milp_bundle_solution(bundle_path, evaluator=evaluator)
    if loaded is None:
        return base_sol, base_obj, base_key, False, f"no_bundle(status={status})"

    routes_c, shelf_seq_c, place_c = loaded
    if not shelf_seq_c:
        shelf_seq_c = _dc_shelf_seq(base_sol.shelf_seq)
    for j in j_set:
        if int(j) not in place_c and int(j) in base_sol.place:
            place_c[int(j)] = int(base_sol.place[int(j)])

    shelf_seq_c = _ensure_shelf_seq_covers_all_tasks(
        shelf_seq_c,
        evaluator=evaluator,
        task_shelf_mapping=clean_map,
    )
    routes_c = _normalize_ws_blocks(routes_c, evaluator)
    routes_c = normalize_routes_by_shelf_seq_order(routes_c, shelf_seq_c, clean_map)

    cand_obj, cand_det = _safe_evaluate(evaluator, routes_c, shelf_seq_c, place_c)
    cand_obj = float(cand_obj)
    cand_key = tuple(_safe_infeas_key(cand_det, evaluator, obj=cand_obj))

    improved = bool((cand_key < base_key) or (cand_key == base_key and cand_obj < base_obj - 1e-9))
    if not improved:
        obj_txt = f"{cand_obj:.2f}" if math.isfinite(cand_obj) else "inf"
        base_txt = f"{base_obj:.2f}" if math.isfinite(base_obj) else "inf"
        return base_sol, base_obj, base_key, False, f"no_gain(status={status}, obj={base_txt}->{obj_txt})"

    out_sol = copy.deepcopy(base_sol)
    out_sol.routes = _dc_routes(routes_c)
    out_sol.shelf_seq = _dc_shelf_seq(shelf_seq_c)
    out_sol.place = _dc_place(place_c)

    stat_name = "OPTIMAL" if status == int(GRB.OPTIMAL) else ("TIME_LIMIT" if status == int(GRB.TIME_LIMIT) else str(status))
    return out_sol, cand_obj, cand_key, True, f"improved({stat_name}: {base_obj:.2f}->{cand_obj:.2f})"


# =========================
#         ALNS main
# =========================
def alns_minimize(
    init,
    evaluator: RobustEvaluator,
    evaluator_exact: Optional[RobustEvaluator] = None,
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

    # [text_corrupted]destroy[text_corrupted]
    enable_robust_destroy: bool = True,
    adaptive_reaction: float = 0.20,
    adaptive_segment_len: int = 50,
    adaptive_w_min: float = 0.05,
    adaptive_w_max: float = 50.0,
    reward_best: float = 33.0,
    reward_improve: float = 9.0,
    reward_accept: float = 3.0,
    reward_reject: float = 0.0,
    feasible_first: bool = True,
    enable_strong_init: bool = True,
    strong_init_tries: int = 4,
    strong_init_time_budget_sec: float = 0.0,
    speed_profile: str = "balanced",
    time_budget_sec: Optional[float] = None,
    relabel_interval: int = 20,
    relabel_max_exact_agv: int = 6,
    relabel_eval_top_k: int = 12,
    relabel_when_stagnating: bool = True,
    relabel_when_strong_shake: bool = True,
    eval_budget_total: Optional[int] = None,
    eval_budget_heavy: Optional[int] = None,
    target_feasible_obj: Optional[float] = None,
    feasible_escape_prob: float = 0.04,
    feasible_escape_relax: float = 2.00,
    eval_layering: int = 0,
    max_exact_evals_per_iter: int = 1,
    use_eval_cache: int = 0,
    eval_cache_maxsize: int = 50000,
    use_shallow_copy: int = 0,
    verbose: bool = False,
    enable_ejection_chain: int = 1,
    ejection_chain_prob: float = 0.12,
    ejection_chain_min_seg: int = 6,
    ejection_chain_max_seg: int = 15,
    enable_ws_micro_reorder: int = 1,
    ws_micro_reorder_prob: float = 0.16,
    enable_milp_polish: int = 0,
    milp_polish_time_limit_sec: float = 30.0,
    milp_polish_lock_immediate: int = 1,
    milp_polish_on_turbo: int = 0,
):
    assert S_near_by_j is not None, "[text_corrupted]S_near_by_j [text_corrupted]"
    rng = random.Random(seed)
    verbose_eff = bool(verbose)
    task_shelf_mapping = _sanitize_task_shelf_mapping(task_shelf_mapping, verbose=verbose_eff)
    profile = str(speed_profile or "balanced").strip().lower()
    turbo_mode = profile in {"turbo", "fast", "speed"}
    if (not strong_init_time_budget_sec) or (float(strong_init_time_budget_sec) <= 0.0):
        strong_init_budget = 3.0 if turbo_mode else 8.0
    else:
        strong_init_budget = float(strong_init_time_budget_sec)
    t_start = time.perf_counter()
    t_budget = None
    if time_budget_sec is not None:
        try:
            t_budget = float(time_budget_sec)
        except Exception:
            t_budget = None
    if t_budget is not None and t_budget <= 0.0:
        t_budget = None
    use_eval_cache_eff = bool(int(use_eval_cache or 0) == 1)
    use_shallow_copy_eff = bool(int(use_shallow_copy or 0) == 1)
    eval_cache_maxsize_eff = max(1000, int(eval_cache_maxsize or 0))
    enable_ejection_chain_eff = bool(int(enable_ejection_chain or 0) == 1)
    ejection_chain_prob_eff = min(1.0, max(0.0, float(ejection_chain_prob or 0.0)))
    ejection_chain_min_seg_eff = max(2, int(ejection_chain_min_seg or 2))
    ejection_chain_max_seg_eff = max(ejection_chain_min_seg_eff, int(ejection_chain_max_seg or ejection_chain_min_seg_eff))
    enable_ws_micro_reorder_eff = bool(int(enable_ws_micro_reorder or 0) == 1)
    ws_micro_reorder_prob_eff = min(1.0, max(0.0, float(ws_micro_reorder_prob or 0.0)))
    enable_milp_polish_eff = bool(int(enable_milp_polish or 0) == 1)
    try:
        milp_polish_time_limit_eff = max(0.0, float(milp_polish_time_limit_sec or 0.0))
    except Exception:
        milp_polish_time_limit_eff = 0.0
    milp_polish_lock_immediate_eff = bool(int(milp_polish_lock_immediate or 0) == 1)
    milp_polish_on_turbo_eff = bool(int(milp_polish_on_turbo or 0) == 1)

    def _log(msg: str) -> None:
        if verbose_eff:
            print(msg)

    class _EvalCounterProxy:
        def __init__(self, base, name: str):
            self._base = base
            self.name = str(name or "eval")
            self.calls = 0
            self.calls_by_tag: Dict[str, int] = defaultdict(int)
            self._tag: str = "default"

        def evaluate(self, routes, shelf_seq, place, *args, **kwargs):
            self.calls += 1
            tag = str(self._tag or "default")
            self.calls_by_tag[tag] = int(self.calls_by_tag.get(tag, 0)) + 1
            return self._base.evaluate(routes, shelf_seq, place, *args, **kwargs)

        def set_tag(self, tag: str) -> None:
            self._tag = str(tag or "default")

        def __getattr__(self, name):
            return getattr(self._base, name)

    eval_layering_eff = 1 if int(eval_layering or 0) == 1 else 0
    max_exact_evals_per_iter_eff = max(0, int(max_exact_evals_per_iter or 0))
    base_evaluator_fast = evaluator
    base_evaluator_exact = evaluator_exact if evaluator_exact is not None else evaluator

    evaluator_fast = _EvalCounterProxy(base_evaluator_fast, "fast")
    evaluator_exact_proxy = _EvalCounterProxy(base_evaluator_exact, "exact")
    layering_on = bool(eval_layering_eff == 1 and max_exact_evals_per_iter_eff > 0 and evaluator_exact is not None)
    evaluator = evaluator_fast

    def _clone_sol(sol_like):
        if not use_shallow_copy_eff:
            return copy.deepcopy(sol_like)
        out = copy.copy(sol_like)
        out.routes = _dc_routes(getattr(sol_like, "routes", {}))
        out.place = _dc_place(getattr(sol_like, "place", {}))
        out.shelf_seq = _dc_shelf_seq(getattr(sol_like, "shelf_seq", {}))
        return out

    cache_fast: OrderedDict = OrderedDict()
    cache_exact: OrderedDict = OrderedDict()
    cache_hits_fast = 0
    cache_miss_fast = 0
    cache_hits_exact = 0
    cache_miss_exact = 0

    def _serialize_solution(routes, shelf_seq, place):
        key_routes = tuple(
            (int(r), tuple(int(j) for j in (routes.get(r, []) or [])))
            for r in sorted(routes.keys())
        )
        key_shelf = tuple(
            (int(c), tuple(int(j) for j in (shelf_seq.get(c, []) or [])))
            for c in sorted(shelf_seq.keys())
        )
        key_place = tuple((int(j), int(place[j])) for j in sorted(place.keys()))
        return (key_routes, key_shelf, key_place)

    def _cache_eval(layer: str, ev_proxy, routes, shelf_seq, place):
        nonlocal cache_hits_fast, cache_miss_fast, cache_hits_exact, cache_miss_exact
        if not use_eval_cache_eff:
            return _safe_evaluate(ev_proxy, routes, shelf_seq, place)
        key = _serialize_solution(routes, shelf_seq, place)
        cache = cache_fast if str(layer) == "fast" else cache_exact
        hit = cache.get(key, None)
        if hit is not None:
            cache.move_to_end(key, last=True)
            if str(layer) == "fast":
                cache_hits_fast += 1
            else:
                cache_hits_exact += 1
            obj_h, det_h = hit
            return float(obj_h), dict(det_h) if isinstance(det_h, dict) else {}
        if str(layer) == "fast":
            cache_miss_fast += 1
        else:
            cache_miss_exact += 1
        obj_v, det_v = _safe_evaluate(ev_proxy, routes, shelf_seq, place)
        cache[key] = (float(obj_v), dict(det_v) if isinstance(det_v, dict) else {})
        cache.move_to_end(key, last=True)
        while len(cache) > int(eval_cache_maxsize_eff):
            cache.popitem(last=False)
        return float(obj_v), dict(det_v) if isinstance(det_v, dict) else {}

    # =========================
    # [text_corrupted]
    # =========================
    n_tasks = len(list(evaluator.J))
    n_s = len(list(evaluator.S))

    if eval_budget_total is None:
        eval_budget_total_eff = 10**9
    else:
        eval_budget_total_eff = max(1000, int(eval_budget_total))

    if eval_budget_heavy is None:
        eval_budget_heavy_eff = max(500, int(0.70 * float(eval_budget_total_eff)))
    else:
        eval_budget_heavy_eff = max(100, int(eval_budget_heavy))

    target_feasible_obj_eff: Optional[float] = None
    if target_feasible_obj is not None:
        try:
            tfo = float(target_feasible_obj)
            if math.isfinite(tfo):
                target_feasible_obj_eff = float(tfo)
        except Exception:
            target_feasible_obj_eff = None

    if n_tasks <= 30:
        if turbo_mode:
            max_agv_candidates = 4
            max_pos_per_route = 10
            max_place_try_each_base = 5
        else:
            max_agv_candidates = None
            max_pos_per_route = None
            max_place_try_each_base = 6
    elif n_tasks <= 120:
        if turbo_mode:
            max_agv_candidates = 3
            max_pos_per_route = 12
            max_place_try_each_base = 5
        else:
            max_agv_candidates = 3
            max_pos_per_route = 15
            max_place_try_each_base = 5
    elif n_tasks <= 260:
        if turbo_mode:
            max_agv_candidates = 2
            max_pos_per_route = 8
            max_place_try_each_base = 3
        else:
            max_agv_candidates = 3
            max_pos_per_route = 10
            max_place_try_each_base = 4
    else:
        if turbo_mode:
            max_agv_candidates = 2
            max_pos_per_route = 6
            max_place_try_each_base = 2
        else:
            max_agv_candidates = 2
            max_pos_per_route = 8
            max_place_try_each_base = 3
    if n_tasks > 30 and max_agv_candidates is None:
        max_agv_candidates = 3
    if n_s >= 300:
        max_place_try_each_base = min(max_place_try_each_base, 4)

    large_scale_mode = bool(n_tasks >= 120)
    very_large_mode = bool(n_tasks >= 260)

    init_seed = _clone_sol(init)
    if enable_strong_init:
        try:
            init_seed = _construct_strong_initial_solution(
                init_seed,
                evaluator=evaluator,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init=shelf_init,
                seed=seed,
                max_tries=int(strong_init_tries),
                time_budget_sec=float(strong_init_budget),
                only_when_infeasible=True,
            )
        except Exception as e:
            _log(f"[ALNS-init] strong constructor failed, fallback to raw init: {type(e).__name__}: {e}")

    best = _clone_sol(init_seed)
    best.routes = _dc_routes(best.routes)
    best.place = _dc_place(best.place)
    best.shelf_seq = _dc_shelf_seq(best.shelf_seq)

    best.routes = _normalize_ws_blocks(best.routes, evaluator)
    best.routes = normalize_routes_by_shelf_seq_order(best.routes, best.shelf_seq, task_shelf_mapping)
    evaluator.set_tag("init_eval")
    best_obj, best_details = _cache_eval("fast", evaluator, best.routes, best.shelf_seq, best.place)

    best_obj = float(best_obj)
    best_key = tuple(_safe_infeas_key(best_details, evaluator, obj=best_obj))
    key_feas = (0, 0, 0)

    # Fallback only when initial solution is infeasible:
    # try deterministic from-scratch seeds to avoid all-inf restarts.
    if tuple(best_key) != key_feas:
        fallback_pool: List[Tuple[str, object]] = []
        try:
            rr_seed = build_ws_round_robin_seed(
                best,
                evaluator=evaluator,
                task_shelf_mapping=task_shelf_mapping,
            )
            fallback_pool.append(("ws_round_robin", rr_seed))
        except Exception as e:
            _log(f"[ALNS-init] ws_round_robin seed failed: {type(e).__name__}: {e}")
        try:
            safe_seed = build_safe_chain_ws_seed(
                best,
                evaluator_exact=base_evaluator_exact,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init=shelf_init,
                seed=int(seed),
                tries=(24 if n_tasks > 30 else 12),
            )
            fallback_pool.append(("safe_chain_ws", safe_seed))
        except Exception as e:
            _log(f"[ALNS-init] safe_chain_ws seed failed: {type(e).__name__}: {e}")

        for seed_tag, seed_sol in fallback_pool:
            cand = _clone_sol(seed_sol)
            cand.routes = _dc_routes(cand.routes)
            cand.place = _dc_place(cand.place)
            cand.shelf_seq = _ensure_shelf_seq_covers_all_tasks(
                _dc_shelf_seq(cand.shelf_seq),
                evaluator=evaluator,
                task_shelf_mapping=task_shelf_mapping,
            )
            cand.routes = _normalize_ws_blocks(cand.routes, evaluator)
            cand.routes = normalize_routes_by_shelf_seq_order(cand.routes, cand.shelf_seq, task_shelf_mapping)

            evaluator.set_tag(f"init_recover_{seed_tag}")
            cand_obj, cand_details = _cache_eval("fast", evaluator, cand.routes, cand.shelf_seq, cand.place)
            cand_obj = float(cand_obj)
            cand_key = tuple(_safe_infeas_key(cand_details, evaluator, obj=cand_obj))

            if (tuple(cand_key) < tuple(best_key)) or (
                tuple(cand_key) == tuple(best_key) and (cand_obj < float(best_obj) - 1e-9)
            ):
                best = _clone_sol(cand)
                best_obj = float(cand_obj)
                best_details = dict(cand_details) if isinstance(cand_details, dict) else {}
                best_key = tuple(cand_key)
                _log(f"[ALNS-init] recovered better seed via {seed_tag}: key={best_key}, obj={best_obj:.2f}")
            if tuple(best_key) == key_feas:
                break

    best_feasible = _clone_sol(best) if tuple(best_key) == key_feas else None
    best_feasible_obj = float(best_obj) if tuple(best_key) == key_feas else float("inf")

    cur = _clone_sol(best)
    cur_obj = float(best_obj)
    cur_details = dict(best_details) if isinstance(best_details, dict) else {}
    cur_key = tuple(best_key)

    obj_scale = max(1.0, abs(float(best_obj)))
    if very_large_mode:
        t_scale = 0.0016
        t_cap = 18.0 if turbo_mode else 24.0
        cool_eff = max(float(cool), 0.9975 if turbo_mode else 0.9982)
    elif large_scale_mode:
        t_scale = 0.0014
        t_cap = 14.0 if turbo_mode else 18.0
        cool_eff = max(float(cool), 0.9972 if turbo_mode else 0.9978)
    else:
        t_scale = 0.0012
        t_cap = 12.0
        cool_eff = max(float(cool), 0.9960)
    T = max(float(start_T), min(float(t_cap), float(obj_scale) * float(t_scale)))

    _log(
        f"[ALNS-inner] init_obj = {best_obj:.2f} | init_infeas={best_key} "
        f"| T0={float(T):.3f} | cool_eff={float(cool_eff):.6f} "
        f"| layering={'on' if layering_on else 'off'}"
    )

    adaptive_reaction_eff = float(adaptive_reaction)
    adaptive_segment_len_eff = max(8, int(adaptive_segment_len))
    if very_large_mode:
        adaptive_reaction_eff = max(adaptive_reaction_eff, 0.26 if turbo_mode else 0.24)
        adaptive_segment_len_eff = min(adaptive_segment_len_eff, 24 if turbo_mode else 30)
    elif large_scale_mode:
        adaptive_reaction_eff = max(adaptive_reaction_eff, 0.22 if turbo_mode else 0.20)
        adaptive_segment_len_eff = min(adaptive_segment_len_eff, 36 if turbo_mode else 42)
    if int(iters) <= 320:
        adaptive_segment_len_eff = min(adaptive_segment_len_eff, max(10, int(iters) // 8))

    # =========================================================
    # ALNS Adaptive core: destroy/repair weights (the "A")
    # =========================================================
    # destroy [text_corrupted]
    destroy_names = [
        "critical_single",
        "critical_batch",
        "dual_tail_critical",
        "rand_small",
        "shaw_related",
        "chain",
        "ws_gap_bundle",
        "ws_idle_gap",
        "ws_critical",
        "robust_sensitive",
        "rand_big",
    ]
    # [text_corrupted]
    init_destroy_w = {
        "critical_single": 1.25,
        "critical_batch": 1.35 if large_scale_mode else 0.95,
        "dual_tail_critical": 1.15 if very_large_mode else 0.90,
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
        reaction=adaptive_reaction_eff,
        segment_len=adaptive_segment_len_eff,
        w_min=adaptive_w_min,
        w_max=adaptive_w_max,
    )

    # repair [text_corrupted] repair [text_corrupted]
    repair_names = ["repair_unordered", "repair_ordered", "repair_difficult"]
    init_repair_w = {"repair_unordered": 1.0, "repair_ordered": 1.0, "repair_difficult": 0.95}
    repair_pool = AdaptiveOpPool(
        repair_names,
        init_repair_w,
        reaction=adaptive_reaction_eff,
        segment_len=adaptive_segment_len_eff,
        w_min=adaptive_w_min,
        w_max=adaptive_w_max,
    )

    # [text_corrupted] destroy [text_corrupted] removed_order [text_corrupted]seed-first / gap-first[text_corrupted]
    ordered_required = {"critical_batch", "dual_tail_critical", "ws_gap_bundle", "ws_idle_gap", "ws_critical", "robust_sensitive"}

    # reward [text_corrupted]ALNS[text_corrupted]
    SCORE_BEST = float(reward_best)
    SCORE_IMPROVE = float(reward_improve)
    SCORE_ACCEPT = float(reward_accept)
    SCORE_REJECT = float(reward_reject)
    PAIR_BIAS = 0.80
    PAIR_REACTION = 0.15
    PAIR_EPS = 0.08
    PAIR_DECAY = 0.995
    pair_quality: Dict[Tuple[str, str], float] = defaultdict(float)
    long_run_mode = bool(int(iters) >= 2000)

    # destroy[text_corrupted]
    P_DESTROY_RANDOM_SMALL = 0.40
    P_DESTROY_RELATED = 0.35
    P_DESTROY_CHAIN = 0.25

    # [text_corrupted]
    if very_large_mode:
        if long_run_mode:
            # Borrow the older "260222" aggressive idea for long runs on huge instances.
            P_USE_CROSS_VEHICLE = 0.30
            P_USE_CROSS_BLOCK = 0.12
            P_USE_INTRA_2OPT = 0.20
            P_USE_INTRA_OROPT = 0.18
        else:
            P_USE_CROSS_VEHICLE = 0.16
            P_USE_CROSS_BLOCK = 0.05
            P_USE_INTRA_2OPT = 0.08
            P_USE_INTRA_OROPT = 0.08
    elif large_scale_mode:
        P_USE_CROSS_VEHICLE = 0.28
        P_USE_CROSS_BLOCK = 0.12
        P_USE_INTRA_2OPT = 0.16
        P_USE_INTRA_OROPT = 0.14
    else:
        P_USE_CROSS_VEHICLE = 0.45
        P_USE_CROSS_BLOCK = 0.25
        P_USE_INTRA_2OPT = 0.30
        P_USE_INTRA_OROPT = 0.25

    # [text_corrupted]
    if very_large_mode:
        STAG_WS = 44 if long_run_mode else 55
        STAG_LIMIT = 120
    elif large_scale_mode:
        STAG_WS = 48
        STAG_LIMIT = 150
    else:
        STAG_WS = 40
        STAG_LIMIT = 120
    P_WS_FOCUSED = 0.70
    FEAS_ESCAPE_PROB = max(0.0, min(0.25, float(feasible_escape_prob)))
    FEAS_ESCAPE_RELAX = max(0.0, float(feasible_escape_relax))

    # [text_corrupted]
    REHEAT_SOFT = max(2.50, 0.45 * float(T))
    REHEAT_STRONG = max(6.00, 0.95 * float(T))

    POST_SHAKE_ITERS = 25
    post_shake = 0
    stall = 0

    # [text_corrupted] destroy [text_corrupted] gamma>0 [text_corrupted]
    def _robust_destroy_prob(stagnating: bool) -> float:
        if (not enable_robust_destroy) or (int(getattr(evaluator, "gamma", 0)) <= 0):
            return 0.0
        return 0.20 if not stagnating else 0.30

    ws_pos_global = ws_order_idx if ws_order_idx is not None else _ws_index(getattr(evaluator, "ws_fixed_seq", {}) or {})
    pi = getattr(evaluator, "pi", {}) or {}

    elite_pool: List[Tuple[float, Any]] = []

    def _push_elite(sol_obj: float, sol_like: Any) -> None:
        if not math.isfinite(float(sol_obj)):
            return
        obj_v = float(sol_obj)
        for ex_obj, _ in elite_pool:
            if abs(float(ex_obj) - obj_v) <= 1e-9:
                return
        elite_pool.append((obj_v, _clone_sol(sol_like)))
        elite_pool.sort(key=lambda x: float(x[0]))
        if len(elite_pool) > 4:
            del elite_pool[4:]

    if best_feasible is not None and math.isfinite(float(best_feasible_obj)):
        _push_elite(float(best_feasible_obj), best_feasible)

    effective_iters = max(1, int(iters))
    iters_since_best = 0
    if very_large_mode:
        uphill_rel_cap = 0.25 if (not turbo_mode) else 0.18
        if effective_iters >= 450:
            early_stop_patience = max(200, int(0.40 * float(effective_iters)))
            early_stop_warmup = max(150, int(0.32 * float(effective_iters)))
        elif effective_iters >= 260:
            early_stop_patience = max(150, int(0.40 * float(effective_iters)))
            early_stop_warmup = max(100, int(0.30 * float(effective_iters)))
        else:
            early_stop_patience = max(100, int(0.30 * float(effective_iters)))
            early_stop_warmup = max(70, int(0.26 * float(effective_iters)))
    elif large_scale_mode:
        uphill_rel_cap = 0.35 if (not turbo_mode) else 0.25
        early_stop_patience = max(120, int(0.30 * float(effective_iters)))
        early_stop_warmup = max(80, int(0.28 * float(effective_iters)))
    else:
        uphill_rel_cap = 0.45 if (not turbo_mode) else 0.35
        early_stop_patience = max(90, int(0.26 * float(effective_iters)))
        early_stop_warmup = max(70, int(0.28 * float(effective_iters)))

    effective_relabel_interval = int(relabel_interval)
    if turbo_mode:
        effective_relabel_interval = 0
    elif very_large_mode:
        effective_relabel_interval = max(0, int(effective_relabel_interval))
        if effective_relabel_interval > 0:
            effective_relabel_interval = max(60, effective_relabel_interval)
    elif large_scale_mode and effective_relabel_interval > 0:
        effective_relabel_interval = max(35, effective_relabel_interval)

    effective_relabel_top_k = int(relabel_eval_top_k)
    if turbo_mode:
        effective_relabel_top_k = min(effective_relabel_top_k, 4)
    elif very_large_mode:
        effective_relabel_top_k = min(effective_relabel_top_k, 3)
    elif n_tasks >= 30:
        effective_relabel_top_k = min(effective_relabel_top_k, 8)

    if (target_feasible_obj_eff is not None) and (best_feasible is not None) and (float(best_feasible_obj) <= float(target_feasible_obj_eff) + 1e-9):
        _log(
            f"[ALNS-inner] init already reaches target feasible obj <= {float(target_feasible_obj_eff):.2f}; skip iterations."
        )
        effective_iters = 0

    no_improve_early_stop_on = True
    if very_large_mode and int(getattr(evaluator, "gamma", 0)) == 0 and int(effective_iters) <= 1500:
        # Gamma=0 short/medium runs are often warm-started near-feasible; avoid stopping too early.
        no_improve_early_stop_on = False

    calls_last_report = int(getattr(evaluator, "calls", 0))
    exact_calls_last_report = int(getattr(evaluator_exact_proxy, "calls", 0))
    for it in range(effective_iters):
        if (target_feasible_obj_eff is not None) and (best_feasible is not None) and (float(best_feasible_obj) <= float(target_feasible_obj_eff) + 1e-9):
            _log(
                f"[ALNS-inner] stop by target feasible obj: {float(best_feasible_obj):.2f} <= {float(target_feasible_obj_eff):.2f} at iter {it}/{effective_iters}"
            )
            break
        if int(getattr(evaluator, "calls", 0)) >= int(eval_budget_total_eff):
            _log(
                f"[ALNS-inner] stop by eval budget: {int(eval_budget_total_eff)} calls "
                f"at iter {it}/{effective_iters}"
            )
            break
        if (t_budget is not None) and ((time.perf_counter() - t_start) >= t_budget):
            _log(f"[ALNS-inner] stop by time budget: {t_budget:.2f}s at iter {it}/{effective_iters}")
            break
        improved_best_this_iter = False
        exact_evals_this_iter = 0

        # [text_corrupted]stall [text_corrupted]post_shake [text_corrupted]
        stagnating = (stall >= STAG_WS)
        post_mode = (post_shake > 0)
        strong_shake = (stall >= STAG_LIMIT)

        # [text_corrupted] post_mode [text_corrupted]
        if stagnating or post_mode:
            T = max(T, REHEAT_SOFT)
        if strong_shake:
            T = max(T, REHEAT_STRONG)

        forced_diversify = bool(
            very_large_mode and (stall >= max(28, STAG_WS - 8)) and ((it % 24) == 0)
        )

        if strong_shake:
            if feasible_first and (best_feasible is not None):
                cur = _clone_sol(best_feasible)
                cur_obj = float(best_feasible_obj)
                cur_details = {}
                cur_key = key_feas
            else:
                cur = _clone_sol(best)
                cur_obj = float(best_obj)
                cur_details = dict(best_details) if isinstance(best_details, dict) else {}
                cur_key = tuple(best_key)

        if (
            large_scale_mode
            and (not strong_shake)
            and (stall >= STAG_WS)
            and (len(elite_pool) >= 2)
            and ((it % (36 if very_large_mode else 24)) == 0)
        ):
            pick_hi = min(len(elite_pool) - 1, 2)
            pick_idx = 1 if pick_hi >= 1 else 0
            if pick_hi > 1:
                pick_idx = int(rng.randint(1, pick_hi))
            elite_obj, elite_sol = elite_pool[pick_idx]
            cur = _clone_sol(elite_sol)
            cur_obj = float(elite_obj)
            cur_details = {}
            cur_key = key_feas
            T = max(float(T), REHEAT_STRONG)
            post_shake = max(int(post_shake), 12 if very_large_mode else 8)

        cur_feasible = (tuple(cur_key) == key_feas)
        fast_feasible_mode = bool(cur_feasible and (not strong_shake) and (not stagnating) and (not post_mode))

        # topk/extra[text_corrupted]post_mode [text_corrupted] stagnating
        if strong_shake:
            topk_insert = 3
            extra_shelves = 2
        elif stagnating:
            topk_insert = 3
            extra_shelves = 2
        elif post_mode:
            topk_insert = 2
            extra_shelves = 1
        elif fast_feasible_mode:
            topk_insert = 1
            extra_shelves = 0
        else:
            topk_insert = 1
            extra_shelves = 0

        force_all_shelves = bool(strong_shake)

        # ===== [text_corrupted] stall [text_corrupted]post_mode [text_corrupted]=====
        if strong_shake:
            if very_large_mode:
                repair_evals_per_task = 10 if turbo_mode else 8
                repair_evals_total = 60 if turbo_mode else 48
            elif large_scale_mode:
                repair_evals_per_task = 18 if turbo_mode else 14
                repair_evals_total = 110 if turbo_mode else 90
            else:
                repair_evals_per_task = 40 if turbo_mode else 35
                repair_evals_total = 220 if turbo_mode else 190
        elif stagnating:
            if very_large_mode:
                repair_evals_per_task = 8 if turbo_mode else 6
                repair_evals_total = 42 if turbo_mode else 32
            elif large_scale_mode:
                repair_evals_per_task = 14 if turbo_mode else 11
                repair_evals_total = 90 if turbo_mode else 72
            else:
                repair_evals_per_task = 30 if turbo_mode else 24
                repair_evals_total = 150 if turbo_mode else 130
        elif post_mode:
            if very_large_mode:
                repair_evals_per_task = 6 if turbo_mode else 5
                repair_evals_total = 26 if turbo_mode else 22
            elif large_scale_mode:
                repair_evals_per_task = 10 if turbo_mode else 8
                repair_evals_total = 62 if turbo_mode else 52
            else:
                repair_evals_per_task = 20 if turbo_mode else 16
                repair_evals_total = 100 if turbo_mode else 90
        elif fast_feasible_mode:
            if very_large_mode:
                repair_evals_per_task = 2 if turbo_mode else 1
                repair_evals_total = 8 if turbo_mode else 6
            elif large_scale_mode:
                repair_evals_per_task = 3 if turbo_mode else 2
                repair_evals_total = 14 if turbo_mode else 10
            else:
                repair_evals_per_task = 5 if turbo_mode else 4
                repair_evals_total = 24 if turbo_mode else 20
        else:
            if very_large_mode:
                repair_evals_per_task = 4 if turbo_mode else 3
                repair_evals_total = 20 if turbo_mode else 16
            elif large_scale_mode:
                repair_evals_per_task = 8 if turbo_mode else 6
                repair_evals_total = 50 if turbo_mode else 36
            else:
                repair_evals_per_task = 14 if turbo_mode else 10
                repair_evals_total = 80 if turbo_mode else 55

        if very_large_mode and long_run_mode:
            # In long runs, keep repair search a bit deeper to avoid early-quality plateau.
            if fast_feasible_mode:
                repair_evals_per_task = max(repair_evals_per_task, 3 if turbo_mode else 2)
                repair_evals_total = max(repair_evals_total, 14 if turbo_mode else 10)
            elif stagnating:
                repair_evals_per_task = max(repair_evals_per_task, 9 if turbo_mode else 7)
                repair_evals_total = max(repair_evals_total, 54 if turbo_mode else 40)

        # =====================================================
        # -------------- 1) destroy + repair --------------
        # =====================================================
        # ========== [text_corrupted]=========
        chosen_destroy = None
        chosen_repair = None

        # ========== [text_corrupted] removed_set [text_corrupted]==========
        def _ws_rank_local(jj: int) -> int:
            ws = int(pi.get(int(jj), -1))
            return int(ws_pos_global.get(ws, {}).get(int(jj), 10 ** 9))

        def _order_from_set(rem_set: Set[int]) -> List[int]:
            lst = [int(x) for x in rem_set]
            rng.shuffle(lst)
            lst.sort(key=lambda x: (_ws_rank_local(x), rng.random()))
            return lst

        # ========== [text_corrupted] destroy+repair [text_corrupted] ==========
        def _adaptive_destroy_repair() -> Tuple[Dict[int, List[int]], Dict[int, int], str, str]:
            nonlocal stagnating, strong_shake, post_mode
            near_stall = bool(very_large_mode and (stall >= max(14, STAG_WS // 3)))

            def _destroy_frac(kind: str) -> Tuple[float, float]:
                k = str(kind or "small")
                if very_large_mode:
                    if k == "big":
                        if strong_shake:
                            return (0.10, 0.22)
                        if stagnating:
                            return (0.07, 0.16)
                        if near_stall:
                            return (0.05, 0.11)
                        return (0.03, 0.08)
                    if strong_shake:
                        return (0.03, 0.07)
                    if stagnating:
                        return (0.024, 0.055)
                    if near_stall:
                        return (0.018, 0.040)
                    return (0.012, 0.030) if fast_feasible_mode else (0.02, 0.05)
                if large_scale_mode:
                    if k == "big":
                        return (0.09, 0.20) if (stagnating or strong_shake) else (0.07, 0.16)
                    return (0.03, 0.10) if fast_feasible_mode else (0.05, 0.13)
                if k == "big":
                    return (0.35, 0.65)
                return (0.15, 0.35)

            # --------- 1) [text_corrupted]destroy [text_corrupted]---------
            cand_destroy: List[str] = []
            if forced_diversify:
                cand_destroy = ["critical_batch", "rand_big", "ws_gap_bundle", "chain"]
                if very_large_mode:
                    cand_destroy.append("dual_tail_critical")
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            elif strong_shake:
                if very_large_mode:
                    cand_destroy = ["critical_batch", "critical_single", "dual_tail_critical", "ws_critical", "rand_small", "rand_big", "chain"]
                elif large_scale_mode:
                    cand_destroy = ["critical_batch", "critical_single", "ws_critical", "ws_gap_bundle", "rand_small", "chain"]
                else:
                    cand_destroy = ["ws_gap_bundle", "ws_idle_gap", "rand_big"]
            elif fast_feasible_mode:
                if very_large_mode:
                    cand_destroy = ["critical_batch", "critical_single", "dual_tail_critical", "ws_critical", "rand_small"]
                    if near_stall or ((it % 14) == 0):
                        cand_destroy.append("ws_gap_bundle")
                    if near_stall or ((it % 16) == 0):
                        cand_destroy.append("rand_big")
                    if near_stall and ((it % 10) == 0):
                        cand_destroy.append("chain")
                else:
                    cand_destroy = ["critical_single", "ws_critical"]
                if (not very_large_mode) and ((it % 18) == 0):
                    cand_destroy.append("ws_gap_bundle")
                if (not very_large_mode) and ((it % 12) == 0):
                    cand_destroy.append("rand_small")
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            elif stagnating:
                if very_large_mode:
                    cand_destroy = ["critical_batch", "critical_single", "dual_tail_critical", "ws_critical", "rand_small", "rand_big", "chain"]
                elif large_scale_mode:
                    cand_destroy = ["critical_batch", "critical_single", "ws_gap_bundle", "ws_critical", "rand_small", "chain", "shaw_related"]
                else:
                    cand_destroy = ["critical_single", "ws_gap_bundle", "ws_idle_gap", "ws_critical", "rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            elif post_mode:
                if very_large_mode:
                    cand_destroy = ["critical_batch", "critical_single", "dual_tail_critical", "ws_critical", "rand_small", "rand_big", "chain"]
                elif large_scale_mode:
                    cand_destroy = ["critical_batch", "critical_single", "ws_gap_bundle", "ws_critical", "rand_small", "shaw_related", "chain"]
                else:
                    cand_destroy = ["critical_single", "ws_gap_bundle", "ws_critical", "rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")
            else:
                cand_destroy = ["critical_single", "rand_small", "shaw_related", "chain"]
                if enable_robust_destroy and int(getattr(evaluator, "gamma", 0)) > 0:
                    cand_destroy.append("robust_sensitive")

            if strong_shake:
                eps_destroy = 0.22 if very_large_mode else 0.16
            elif stagnating:
                eps_destroy = 0.14 if very_large_mode else 0.10
            elif near_stall:
                eps_destroy = 0.08
            elif fast_feasible_mode and very_large_mode:
                eps_destroy = 0.05
            else:
                eps_destroy = 0.03
            if cand_destroy and (rng.random() < float(eps_destroy)):
                dname = str(rng.choice(cand_destroy))
            else:
                dname = destroy_pool.pick(rng, cand_destroy)

            # --------- 2) [text_corrupted] destroy ---------
            if dname == "rand_small":
                lo, hi = _destroy_frac("small")
                remove_frac = rng.uniform(lo, hi)
                removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed)

            elif dname == "critical_single":
                q_hint = cur_details.get("q", {}) if isinstance(cur_details, dict) else {}
                removed_order, removed, routes_half = destroy_critical_single(
                    routes=cur.routes,
                    evaluator=evaluator,
                    rng=rng,
                    q_hint=q_hint if isinstance(q_hint, dict) else None,
                    extra_neighbor_prob=(0.15 if turbo_mode else 0.25),
                )

            elif dname == "critical_batch":
                q_hint = cur_details.get("q", {}) if isinstance(cur_details, dict) else {}
                if very_large_mode:
                    k_min, k_max = 2, 4
                elif large_scale_mode:
                    k_min, k_max = 3, 5
                else:
                    k_min, k_max = 3, 6
                removed_order, removed, routes_half = destroy_critical_batch(
                    routes=cur.routes,
                    evaluator=evaluator,
                    rng=rng,
                    q_hint=q_hint if isinstance(q_hint, dict) else None,
                    min_k=int(k_min),
                    max_k=int(k_max),
                )

            elif dname == "dual_tail_critical":
                q_hint = cur_details.get("q", {}) if isinstance(cur_details, dict) else {}
                removed_order, removed, routes_half = destroy_dual_tail_critical(
                    routes=cur.routes,
                    evaluator=evaluator,
                    rng=rng,
                    q_hint=q_hint if isinstance(q_hint, dict) else None,
                    tail_window=(10 if very_large_mode else 12),
                    max_per_route=(2 if very_large_mode else 1),
                )
                if not removed_order:
                    lo, hi = _destroy_frac("small")
                    remove_frac = rng.uniform(lo, hi)
                    removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                    removed_order = _order_from_set(removed)

            elif dname == "rand_big":
                lo, hi = _destroy_frac("big")
                remove_frac = rng.uniform(lo, hi)
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
                    # [text_corrupted]
                    lo, hi = _destroy_frac("small")
                    remove_frac = rng.uniform(lo, hi)
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
                # [text_corrupted]name
                lo, hi = _destroy_frac("small")
                remove_frac = rng.uniform(lo, hi)
                removed, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed)

            removed_set = set(int(x) for x in (removed_order or [])) if removed_order else set()
            # [text_corrupted] repair
            missing = _missing_tasks_in_routes(cur.routes, evaluator)
            if missing:
                miss_list = sorted((int(x) for x in missing), key=_ws_rank_local)
                # ordered destroy [text_corrupted]seed-first [text_corrupted]
                if removed_order is None:
                    removed_order = []
                for x in miss_list:
                    if int(x) not in removed_set:
                        removed_order.append(int(x))
                        removed_set.add(int(x))
            if not removed_set:
                # [text_corrupted]
                lo, hi = _destroy_frac("small")
                remove_frac = rng.uniform(lo, hi)
                removed_set, routes_half = destroy_random(cur.routes, remove_frac, rng)
                removed_order = _order_from_set(removed_set)
                removed_set = set(int(x) for x in removed_order)

            # --------- 3) [text_corrupted]repair[text_corrupted]destroy [text_corrupted] ordered[text_corrupted]--------
            if dname in ordered_required:
                cand_repair = ["repair_ordered", "repair_difficult"]
            else:
                cand_repair = ["repair_unordered", "repair_ordered", "repair_difficult"]

            repair_eval_pt = int(repair_evals_per_task)
            repair_eval_total = int(repair_evals_total)
            place_try_each_eff = int(max_place_try_each_base)
            extra_shelves_eff = int(extra_shelves)
            if very_large_mode:
                preselect_m_eff = 4
                preselect_per_pos_eff = 1
            elif large_scale_mode:
                preselect_m_eff = 6
                preselect_per_pos_eff = 1
            elif fast_feasible_mode:
                preselect_m_eff = 8
                preselect_per_pos_eff = 1
            else:
                preselect_m_eff = 12
                preselect_per_pos_eff = 2
            repair_require_finite = bool(feasible_first and cur_feasible and (not very_large_mode))
            repair_strict = bool(repair_require_finite and (not large_scale_mode) and (not stagnating))
            repair_strict_bruteforce = bool(repair_require_finite and (not large_scale_mode))
            if dname == "critical_batch":
                repair_eval_pt = max(repair_eval_pt, 3 if very_large_mode else 4)
                repair_eval_total = max(repair_eval_total, 18 if very_large_mode else 28)
                preselect_m_eff = max(preselect_m_eff, 6 if very_large_mode else 8)
                preselect_per_pos_eff = max(preselect_per_pos_eff, 1)
            if dname == "dual_tail_critical":
                repair_eval_pt = min(repair_eval_pt, 2 if very_large_mode else repair_eval_pt)
                repair_eval_total = min(repair_eval_total, 12 if very_large_mode else repair_eval_total)
                place_try_each_eff = min(place_try_each_eff, 3)
                extra_shelves_eff = min(extra_shelves_eff, 1)
            if fast_feasible_mode and (dname == "critical_single"):
                repair_eval_pt = min(repair_eval_pt, 3 if turbo_mode else 2)
                repair_eval_total = min(repair_eval_total, 10 if turbo_mode else 8)
                place_try_each_eff = min(place_try_each_eff, 3)
                extra_shelves_eff = 0
                preselect_m_eff = min(preselect_m_eff, 4)
            if forced_diversify:
                repair_eval_pt = max(repair_eval_pt, 4 if very_large_mode else 5)
                repair_eval_total = max(repair_eval_total, 22 if very_large_mode else 36)
            if very_large_mode and near_stall and fast_feasible_mode:
                repair_eval_pt = max(repair_eval_pt, 3 if turbo_mode else 2)
                repair_eval_total = max(repair_eval_total, 14 if turbo_mode else 10)
                preselect_m_eff = max(preselect_m_eff, 5)

            # lightweight pair credit: bias repair roulette by (destroy,repair) historical quality
            cand_r = [c for c in cand_repair if c in repair_pool.w]
            use_pair_credit = bool(stagnating or post_mode or strong_shake)
            if strong_shake:
                eps_repair = 0.16 if very_large_mode else 0.10
            elif stagnating:
                eps_repair = 0.10 if very_large_mode else 0.07
            elif near_stall:
                eps_repair = 0.05
            else:
                eps_repair = 0.02
            if not cand_r:
                rname = repair_pool.pick(rng, cand_repair)
            elif rng.random() < float(eps_repair):
                rname = str(rng.choice(cand_r))
            elif not use_pair_credit:
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

            # --------- 4) [text_corrupted] repair ---------
            evaluator.set_tag("repair")
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
                    max_place_try_each=place_try_each_eff,
                    top_k_random=topk_insert,
                    extra_random_shelves=extra_shelves_eff,
                    force_all_shelves=force_all_shelves,
                    max_agv_candidates=max_agv_candidates,
                    max_pos_per_route=max_pos_per_route,
                    max_evals_per_task=repair_eval_pt,
                    max_evals_total=repair_eval_total,
                    preselect_m=preselect_m_eff,
                    preselect_per_pos_shelves=preselect_per_pos_eff,
                    strict_construct=repair_strict,
                    strict_bruteforce_fallback=repair_strict_bruteforce,
                    require_finite_eval=repair_require_finite,
                    eval_fail_cost=1e30,
                )
            elif rname == "repair_difficult":
                routes_new_, place_new_ = repair_greedy_insert_with_place_difficult_ordered(
                    routes=routes_half,
                    shelf_seq=cur.shelf_seq,
                    place=cur.place,
                    evaluator=evaluator,
                    removed=set(int(x) for x in removed_set),
                    rng=rng,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    max_place_try_each=place_try_each_eff,
                    top_k_random=topk_insert,
                    extra_random_shelves=extra_shelves_eff,
                    force_all_shelves=force_all_shelves,
                    max_agv_candidates=max_agv_candidates,
                    max_pos_per_route=max_pos_per_route,
                    max_evals_per_task=repair_eval_pt,
                    max_evals_total=repair_eval_total,
                    preselect_m=preselect_m_eff,
                    preselect_per_pos_shelves=preselect_per_pos_eff,
                    strict_construct=repair_strict,
                    strict_bruteforce_fallback=repair_strict_bruteforce,
                    require_finite_eval=repair_require_finite,
                    eval_fail_cost=1e30,
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
                    max_place_try_each=place_try_each_eff,
                    top_k_random=topk_insert,
                    extra_random_shelves=extra_shelves_eff,
                    force_all_shelves=force_all_shelves,
                    max_agv_candidates=max_agv_candidates,
                    max_pos_per_route=max_pos_per_route,
                    max_evals_per_task=repair_eval_pt,
                    max_evals_total=repair_eval_total,
                    preselect_m=preselect_m_eff,
                    preselect_per_pos_shelves=preselect_per_pos_eff,
                    strict_construct=repair_strict,
                    strict_bruteforce_fallback=repair_strict_bruteforce,
                    require_finite_eval=repair_require_finite,
                    eval_fail_cost=1e30,
                )

            return routes_new_, place_new_, dname, rname

        # ========== [text_corrupted] destroy+repair ==========
        evaluator.set_tag("destroy_repair")
        routes_new, place_new, chosen_destroy, chosen_repair = _adaptive_destroy_repair()

        # strong_shake [text_corrupted]+ [text_corrupted]
        if strong_shake:
            stall = 0
            post_shake = int(POST_SHAKE_ITERS)


        # =====================================================
        # =====================================================
        #  Gate[text_corrupted]heavy local search
        # =====================================================
        # =====================================================
        #  Gate[text_corrupted]heavy local search
        # =====================================================
        # =====================================================
        #  Gate[text_corrupted] details[text_corrupted]heavy local search
        # =====================================================
        routes_new = _normalize_ws_blocks(routes_new, evaluator)

        # [text_corrupted]heavy local[text_corrupted] destroy+repair [text_corrupted]SA
        shelf_seq_new = cur.shelf_seq

        # [text_corrupted]v([text_corrupted]routes[text_corrupted]) [text_corrupted]shelf_seq [text_corrupted] routes[text_corrupted]ws_fixed_seq
        routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)

        evaluator.set_tag("gate_eval")
        cand_obj_fast, cand_details_fast = _cache_eval("fast", evaluator, routes_new, shelf_seq_new, place_new)

        cand_obj_fast = float(cand_obj_fast)
        cand_details_fast = dict(cand_details_fast) if isinstance(cand_details_fast, dict) else {}

        delta0 = cand_obj_fast - float(cur_obj)

        if very_large_mode and long_run_mode:
            p_min = 0.10 if turbo_mode else 0.08
        else:
            p_min = 0.25 if turbo_mode else 0.22
        gate_delta = -math.log(p_min) * float(T)  # [text_corrupted]3*T

        # if current state is infeasible, force heavy local to prioritize repairs
        infeasible_cur = (cur_key != (0, 0, 0))

        allow_heavy_local = int(getattr(evaluator, "calls", 0)) < int(eval_budget_heavy_eff)
        do_balance_move = bool(
            large_scale_mode
            and cur_feasible
            and allow_heavy_local
            and (stall >= max(18, STAG_WS // 2))
            and ((it % (20 if very_large_mode else 14)) == 0)
        )
        if do_balance_move:
            evaluator.set_tag("bottleneck_move")
            if very_large_mode:
                b_tail_k, b_pos, b_s, b_eval = 3, 3, 2, 4
            else:
                b_tail_k, b_pos, b_s, b_eval = 5, 4, 2, 6
            rb_routes, rb_place, rb_improved, rb_obj = rebalance_bottleneck_route_once(
                routes=routes_new,
                place=place_new,
                shelf_seq=shelf_seq_new,
                evaluator=evaluator,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init,
                rng=rng,
                tail_k=b_tail_k,
                max_pos_samples=b_pos,
                max_s_samples=b_s,
                max_evals=b_eval,
            )
            if rb_improved and (float(rb_obj) < float(cand_obj_fast) - 1e-9):
                routes_new, place_new = rb_routes, rb_place
                evaluator.set_tag("gate_eval")
                cand_obj_fast, cand_details_fast = _cache_eval("fast", evaluator, routes_new, shelf_seq_new, place_new)
                cand_obj_fast = float(cand_obj_fast)
                cand_details_fast = dict(cand_details_fast) if isinstance(cand_details_fast, dict) else {}
                delta0 = cand_obj_fast - float(cur_obj)
        infeas_force_heavy = bool(
            infeasible_cur and (
                strong_shake
                or stagnating
                or (it < (30 if turbo_mode else 45))
                or (cand_obj_fast < float(cur_obj) - 1e-9)
                or ((it % (3 if turbo_mode else 4)) == 0)
            )
        )

        if fast_feasible_mode:
            if very_large_mode:
                if str(chosen_destroy) == "critical_single":
                    light_stride = 32 if turbo_mode else 36
                    improve_prob = 0.03
                else:
                    light_stride = 22 if turbo_mode else 26
                    improve_prob = 0.05
            elif large_scale_mode:
                if str(chosen_destroy) == "critical_single":
                    light_stride = 22 if turbo_mode else 25
                    improve_prob = 0.05
                else:
                    light_stride = 16 if turbo_mode else 18
                    improve_prob = 0.09
            elif str(chosen_destroy) == "critical_single":
                light_stride = 15 if turbo_mode else 18
                improve_prob = 0.08
            else:
                light_stride = 9 if turbo_mode else 11
                improve_prob = 0.20
            do_heavy_local = bool(
                allow_heavy_local and (
                    ((it % light_stride) == 0)
                    or ((delta0 <= 0.0) and (rng.random() < improve_prob))
                )
            )
        else:
            do_heavy_local = bool(
                strong_shake
                or (stagnating and ((it % (2 if turbo_mode else 2)) == 0))
                or (post_mode and ((it % (2 if turbo_mode else 3)) == 0))
                or infeas_force_heavy
                or (delta0 <= gate_delta)
            ) and allow_heavy_local
            if very_large_mode and (not strong_shake):
                do_heavy_local = bool(do_heavy_local and ((it % (4 if turbo_mode else 3)) == 0))
            elif large_scale_mode and (not strong_shake):
                do_heavy_local = bool(do_heavy_local and ((it % 2) == 0))

        if very_large_mode:
            gamma_now = int(getattr(evaluator, "gamma", 0))
            if long_run_mode:
                if strong_shake:
                    stride_h = 10
                elif stagnating:
                    stride_h = 24 if gamma_now > 0 else 20
                else:
                    stride_h = 45 if gamma_now > 0 else 36
            else:
                if strong_shake:
                    stride_h = 14
                elif stagnating:
                    stride_h = 48 if gamma_now > 0 else 40
                else:
                    stride_h = 90 if gamma_now > 0 else 70
            do_heavy_local = bool(do_heavy_local and ((it % stride_h) == 0))
            if forced_diversify:
                do_heavy_local = False

        # cand_obj / cand_details [text_corrupted] fast [text_corrupted] heavy local[text_corrupted]
        cand_obj = float(cand_obj_fast)
        cand_details = dict(cand_details_fast)


        # =====================================================
        # -------------- 2) [text_corrupted] --------------
        # =====================================================
        if do_heavy_local:
            evaluator.set_tag("heavy_local")
            if very_large_mode:
                cv_trials = 1
                cv_pos_samples = 4
                cv_s_samples = 3
                cv_eval_budget = 4
                block_trials = 1
                intra_trials = 1
                star_trials = 5 if stagnating else 2
            elif large_scale_mode:
                cv_trials = 2
                cv_pos_samples = 6
                cv_s_samples = 4
                cv_eval_budget = 6
                block_trials = 1
                intra_trials = 1
                star_trials = 8 if stagnating else 4
            else:
                cv_trials = 3
                cv_pos_samples = 8
                cv_s_samples = 5
                cv_eval_budget = 8
                block_trials = 2
                intra_trials = 2
                star_trials = 24 if stagnating else 8
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
                    try_swap=bool(stagnating or strong_shake),  # [text_corrupted]/[text_corrupted] swap
                    max_trials=cv_trials,
                    max_pos_samples=cv_pos_samples,
                    max_s_samples=cv_s_samples,
                    max_evals=cv_eval_budget,
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
                    max_trials=block_trials,
                )
                if improved_blk:
                    routes_new, place_new = routes_blk, place_blk

            p_ejection = float(ejection_chain_prob_eff)
            if stagnating or strong_shake:
                p_ejection = min(0.95, max(p_ejection, p_ejection * 1.65))
            elif very_large_mode and (not long_run_mode):
                p_ejection = min(p_ejection, 0.08)
            if enable_ejection_chain_eff and (rng.random() < p_ejection):
                exact_for_ejection = evaluator_exact_proxy if layering_on else None
                routes_ej, place_ej, improved_ej, _ = cross_vehicle_ejection_chain_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    evaluator_exact=exact_for_ejection,
                    S_near_by_j=S_near_by_j,
                    task_shelf_mapping=task_shelf_mapping,
                    shelf_init_override=shelf_init,
                    rng=rng,
                    min_seg=ejection_chain_min_seg_eff,
                    max_seg=ejection_chain_max_seg_eff,
                    max_trials=(1 if very_large_mode else 2),
                    max_pos_samples=(2 if very_large_mode else 3),
                    max_s_samples=(2 if very_large_mode else 3),
                    max_evals=(2 if very_large_mode else 3),
                    max_exact_trials=1,
                    exact_gate_rel=0.003,
                )
                if improved_ej:
                    routes_new, place_new = routes_ej, place_ej

            p_star = 0.15 if not stagnating else 0.55
            trials_star = star_trials
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
                    max_trials=intra_trials,
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
                    max_trials=intra_trials,
                    max_block=2,
                )
                if improved_or:
                    routes_new = routes_new3

            p_ws_micro = float(ws_micro_reorder_prob_eff)
            if stagnating:
                p_ws_micro = min(0.90, max(p_ws_micro, p_ws_micro * 1.45))
            elif very_large_mode:
                p_ws_micro = min(p_ws_micro, 0.12)
            if cur_feasible and enable_ws_micro_reorder_eff and (rng.random() < p_ws_micro):
                routes_ws, improved_ws, _ = ws_micro_reorder_idle_swap_once(
                    routes=routes_new,
                    place=place_new,
                    shelf_seq=cur.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    max_trials=(6 if very_large_mode else (8 if large_scale_mode else 10)),
                    max_swap_span=(2 if very_large_mode else 3),
                    idle_window_cap=(180.0 if very_large_mode else 240.0),
                )
                if improved_ws:
                    routes_new = routes_ws

            if stagnating and ((it % 6) == 0):
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
            place_tune_on_iter = bool(enable_place_tune)
            shelf_tune_on_iter = bool(enable_shelf_tune and task_shelf_mapping)
            if very_large_mode and (not strong_shake):
                place_tune_on_iter = bool(stagnating and ((it % 120) == 0))
                shelf_tune_on_iter = bool(stagnating and ((it % 160) == 0))

            if place_tune_on_iter:
                if strong_shake:
                    do_tune = ((it % (6 if very_large_mode else (4 if large_scale_mode else 2))) == 0)
                    global_try = 1 if large_scale_mode else 3
                elif stagnating:
                    do_tune = ((it % (10 if very_large_mode else (6 if large_scale_mode else 3))) == 0)
                    global_try = 1 if large_scale_mode else 2
                else:
                    do_tune = ((it % (42 if very_large_mode else (24 if large_scale_mode else 10))) == 0)
                    global_try = 0

                if do_tune:
                    if strong_shake:
                        tune_eval_budget = 24 if very_large_mode else (48 if large_scale_mode else 120)
                        tune_cap = 14 if very_large_mode else (24 if large_scale_mode else 50)
                    elif stagnating:
                        tune_eval_budget = 16 if very_large_mode else (32 if large_scale_mode else 80)
                        tune_cap = 12 if very_large_mode else (20 if large_scale_mode else 45)
                    else:
                        tune_eval_budget = (8 if turbo_mode else 10) if very_large_mode else ((16 if turbo_mode else 20) if large_scale_mode else (36 if turbo_mode else 45))
                        tune_cap = 10 if very_large_mode else (16 if large_scale_mode else 36)
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
                        max_evals=tune_eval_budget,
                        cap_total=tune_cap,
                    )

            # [text_corrupted] shelf_seq [text_corrupted]shelf_tune [text_corrupted]
            shelf_seq_new = cur.shelf_seq

            if shelf_tune_on_iter:
                if very_large_mode:
                    do_reloc = strong_shake or (stagnating and ((it % 10) == 0)) or ((it % 40) == 0)
                elif large_scale_mode:
                    do_reloc = strong_shake or (stagnating and ((it % 6) == 0)) or ((it % 24) == 0)
                else:
                    do_reloc = strong_shake or (stagnating and ((it % 3) == 0)) or ((it % 12) == 0)
                if do_reloc:
                    if strong_shake:
                        reloc_eval_budget = 18 if very_large_mode else (35 if large_scale_mode else 70)
                    elif stagnating:
                        reloc_eval_budget = 12 if very_large_mode else (24 if large_scale_mode else 45)
                    else:
                        reloc_eval_budget = (8 if turbo_mode else 10) if very_large_mode else ((14 if turbo_mode else 18) if large_scale_mode else (24 if turbo_mode else 30))
                    shelf_seq_new, _ = local_shelf_seq_relocate_once(
                        routes=routes_new,
                        shelf_seq=cur.shelf_seq,
                        place=place_new,
                        evaluator=evaluator,
                        task_shelf_mapping=task_shelf_mapping,
                        rng=rng,
                        max_evals=reloc_eval_budget,
                    )

                if very_large_mode:
                    do_promote = strong_shake or (stagnating and ((it % 12) == 0)) or ((it % 50) == 0)
                elif large_scale_mode:
                    do_promote = strong_shake or (stagnating and ((it % 6) == 0)) or ((it % 28) == 0)
                else:
                    do_promote = strong_shake or (stagnating and ((it % 2) == 0)) or ((it % 18) == 0)
                if do_promote:
                    shelf_seq_new, _, _ = intensify_shelf_seq_promote_critical_ws_once(
                        routes=routes_new,
                        shelf_seq=shelf_seq_new,
                        place=place_new,
                        evaluator=evaluator,
                        rng=rng,
                        max_trials=(6 if very_large_mode else (10 if large_scale_mode else 14)) if stagnating else (4 if very_large_mode else (6 if large_scale_mode else 8)),
                    )

            if (it % (30 if turbo_mode else 35)) == 0:
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

            if (stall >= STAG_WS) and ((it % 20) == 0):
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

        # [text_corrupted]do_heavy_local=False[text_corrupted]
        # [text_corrupted]routes_new/ place_new [text_corrupted] destroy+repair[text_corrupted]shelf_seq_new=cur.shelf_seq[text_corrupted]
        # cand_obj [text_corrupted]cand_obj_fast[text_corrupted]SA accept[text_corrupted]

        # =====================================================
        # -------------- 4) evaluate + relabel --------------
        # =====================================================
        # =====================================================
        # -------------- 4) evaluate + relabel --------------
        # =====================================================
        if do_heavy_local:
            # heavy local [text_corrupted] routes_new / shelf_seq_new / place_new[text_corrupted]
            routes_new = _normalize_ws_blocks(routes_new, evaluator)
            routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)
            evaluator.set_tag("post_heavy_eval")
            cand_obj, cand_details = _cache_eval("fast", evaluator, routes_new, shelf_seq_new, place_new)

            cand_obj = float(cand_obj)
            cand_details = dict(cand_details) if isinstance(cand_details, dict) else {}
        else:
            # do_heavy_local=False[text_corrupted]Gate [text_corrupted]evaluate [text_corrupted] cand_obj/cand_details
            cand_obj = float(cand_obj)
            cand_details = dict(cand_details) if isinstance(cand_details, dict) else {}

        periodic_relabel = (effective_relabel_interval > 0 and ((it % effective_relabel_interval) == 0))
        stagnating_relabel = bool(
            relabel_when_stagnating
            and stagnating
            and ((it % (26 if very_large_mode else (14 if large_scale_mode else (10 if turbo_mode else 8)))) == 0)
        )
        strong_relabel = bool(
            relabel_when_strong_shake
            and strong_shake
            and ((it % (14 if very_large_mode else (9 if large_scale_mode else (5 if turbo_mode else 4)))) == 0)
        )
        do_relabel = bool(periodic_relabel or stagnating_relabel or strong_relabel)
        if do_relabel:
            evaluator.set_tag("relabel")
            routes_rl, obj_rl = _relabel_best_routes(
                routes_new,
                evaluator=evaluator,
                shelf_seq=shelf_seq_new,
                place=place_new,
                max_exact_agv=int(relabel_max_exact_agv),
                eval_top_k=int(effective_relabel_top_k),
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init,
            )
            if obj_rl < float(cand_obj) - 1e-9:
                routes_new = routes_rl

                # [text_corrupted]relabel [text_corrupted] v [text_corrupted]shelf_seq [text_corrupted]
                routes_new = normalize_routes_by_shelf_seq_order(routes_new, shelf_seq_new, task_shelf_mapping)

                # [text_corrupted]relabel [text_corrupted] routes[text_corrupted]details[text_corrupted]SA [text_corrupted] cand_key[text_corrupted]
                evaluator.set_tag("relabel_eval")
                cand_obj, cand_details = _cache_eval("fast", evaluator, routes_new, shelf_seq_new, place_new)
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
        # [text_corrupted] cur/best[text_corrupted]reward [text_corrupted]accept [text_corrupted]cur [text_corrupted]
        prev_cur_obj = float(cur_obj)
        prev_cur_key = tuple(cur_key)
        prev_best_obj = float(best_obj)
        prev_best_key = tuple(best_key)

        cand_key = tuple(_safe_infeas_key(cand_details, evaluator, obj=cand_obj))
        if layering_on and (exact_evals_this_iter < int(max_exact_evals_per_iter_eff)):
            cand_feas_fast = (tuple(cand_key) == key_feas)
            exact_gate = False
            if cand_feas_fast:
                # Exact eval only for promising feasible candidates.
                if tuple(prev_cur_key) != key_feas:
                    exact_gate = True
                elif float(cand_obj) <= float(prev_cur_obj) + max(1.0, 0.0015 * abs(float(prev_cur_obj))):
                    exact_gate = True
                elif (best_feasible is not None) and (
                    float(cand_obj) <= float(best_feasible_obj) + max(1.0, 0.0015 * abs(float(best_feasible_obj)))
                ):
                    exact_gate = True
            if exact_gate:
                evaluator_exact_proxy.set_tag("layer_exact")
                cand_obj_exact, cand_det_exact = _cache_eval("exact", evaluator_exact_proxy, routes_new, shelf_seq_new, place_new)
                exact_evals_this_iter += 1
                cand_obj = float(cand_obj_exact)
                cand_details = dict(cand_det_exact) if isinstance(cand_det_exact, dict) else {}
                cand_key = tuple(_safe_infeas_key(cand_details, evaluator_exact_proxy, obj=cand_obj))

        cand_feas = (tuple(cand_key) == key_feas)
        prev_cur_feas = (tuple(prev_cur_key) == key_feas)
        too_uphill_feas = False
        if prev_cur_feas and cand_feas and math.isfinite(float(prev_cur_obj)) and math.isfinite(float(cand_obj)):
            rel_up = (float(cand_obj) - float(prev_cur_obj)) / max(1.0, abs(float(prev_cur_obj)))
            too_uphill_feas = bool(rel_up > float(uphill_rel_cap))

        accept = False
        target_hit = False

        if feasible_first:
            # Feasibility-first acceptance: never move from feasible back to worse infeasible states.
            if cand_key < prev_cur_key:
                accept = True
            elif cand_key == prev_cur_key:
                if cand_obj < prev_cur_obj - 1e-9:
                    accept = True
                elif too_uphill_feas:
                    accept = False
                else:
                    delta = float(cand_obj) - float(prev_cur_obj)
                    x_obj = -delta / max(1e-9, float(T))
                    if x_obj >= 700.0:
                        prob = 1.0
                    elif x_obj <= -700.0:
                        prob = 0.0
                    else:
                        prob = math.exp(x_obj)
                    if rng.random() < prob:
                        accept = True
            else:
                # cand infeas [text_corrupted]
                if prev_cur_feas:
                    accept = False
                    # Controlled escape from feasible local minima when search stagnates.
                    if (stall >= STAG_WS) and (FEAS_ESCAPE_PROB > 0.0):
                        cur_inf = _safe_infeas_scalar(cur_details, evaluator, obj=cur_obj)
                        cand_inf = _safe_infeas_scalar(cand_details, evaluator, obj=cand_obj)
                        allowed_inf = max(1.0, cur_inf) * (1.0 + FEAS_ESCAPE_RELAX)
                        if cand_inf <= allowed_inf:
                            esc_prob = min(
                                FEAS_ESCAPE_PROB,
                                FEAS_ESCAPE_PROB * (1.0 + 0.02 * max(0, stall - STAG_WS)),
                            )
                            if rng.random() < esc_prob:
                                accept = True
                else:
                    accept = False
        else:
            # [text_corrupted]infeas-SA [text_corrupted]infeas
            if cand_key < prev_cur_key:
                accept = True
            elif cand_key == prev_cur_key:
                if cand_obj < prev_cur_obj - 1e-9:
                    accept = True
                elif too_uphill_feas:
                    accept = False
                else:
                    delta = float(cand_obj) - float(prev_cur_obj)
                    x_obj = -delta / max(1e-9, float(T))
                    if x_obj >= 700.0:
                        prob = 1.0
                    elif x_obj <= -700.0:
                        prob = 0.0
                    else:
                        prob = math.exp(x_obj)
                    if rng.random() < prob:
                        accept = True
            else:
                delta_inf = _safe_infeas_scalar(cand_details, evaluator, obj=cand_obj) - _safe_infeas_scalar(cur_details, evaluator, obj=cur_obj)
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

            # best [text_corrupted]prev_best[text_corrupted]
            if (cand_key < prev_best_key) or (cand_key == prev_best_key and cand_obj < prev_best_obj - 1e-9):
                best = _clone_sol(cur)
                best_obj = float(cand_obj)
                best_key = tuple(cand_key)
                best_details = dict(cur_details) if isinstance(cur_details, dict) else {}
                improved_best_this_iter = True

            if cand_feas and (cand_obj < best_feasible_obj - 1e-9):
                best_feasible = _clone_sol(cur)
                best_feasible_obj = float(cand_obj)
                _push_elite(float(best_feasible_obj), best_feasible)

                if (target_feasible_obj_eff is not None) and (float(best_feasible_obj) <= float(target_feasible_obj_eff) + 1e-9):
                    target_hit = True
            elif cand_feas:
                _push_elite(float(cand_obj), cur)

        # =====================================================
        # -------------- 5.5) Adaptive weight update ----------
        # =====================================================
        # [text_corrupted] ALNS[text_corrupted]reward [text_corrupted]cand [text_corrupted] prev_cur / prev_best
        if accept:
            is_new_best = (cand_key < prev_best_key) or (cand_key == prev_best_key and cand_obj < prev_best_obj - 1e-9)
            is_improve_cur = (cand_key < prev_cur_key) or (cand_key == prev_cur_key and cand_obj < prev_cur_obj - 1e-9)

            if is_new_best:
                reward = SCORE_BEST
            elif is_improve_cur:
                reward = SCORE_IMPROVE
            else:
                reward = max(0.05, 0.20 * SCORE_ACCEPT)
        else:
            reward = SCORE_REJECT

        reward_for_pool = max(0.0, min(float(adaptive_w_max), float(reward) / max(1.0, float(SCORE_IMPROVE))))
        if chosen_destroy is not None:
            destroy_pool.record(chosen_destroy, reward_for_pool)
        if chosen_repair is not None:
            repair_pool.record(chosen_repair, reward_for_pool)
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
        if ((it + 1) % max(1, int(adaptive_segment_len_eff)) == 0) and pair_quality:
            for pk in list(pair_quality.keys()):
                qv = float(pair_quality.get(pk, 0.0)) * PAIR_DECAY
                if qv < 1e-6:
                    pair_quality.pop(pk, None)
                else:
                    pair_quality[pk] = qv

        destroy_pool.maybe_update(it)
        repair_pool.maybe_update(it)

        if target_hit:
            _log(
                f"[ALNS-inner] hit target feasible obj: {float(best_feasible_obj):.2f} <= {float(target_feasible_obj_eff):.2f} at iter {it + 1}"
            )
            break

        # =====================================================
        # -------------- 6) cool & stall --------------
        # =====================================================
        T *= float(cool_eff)
        iters_since_best = 0 if improved_best_this_iter else (iters_since_best + 1)
        stall = 0 if improved_best_this_iter else (stall + 1)

        if stagnating or post_mode:
            T = max(float(T), REHEAT_SOFT)

        if post_shake > 0:
            post_shake -= 1

        if (
            no_improve_early_stop_on
            and
            (effective_iters >= 120)
            and (not long_run_mode)
            and ((it + 1) >= int(early_stop_warmup))
            and (iters_since_best >= int(early_stop_patience))
            and (best_feasible is not None)
        ):
            _log(
                f"[ALNS-inner] early-stop by no-best-improve: "
                f"idle={iters_since_best}, iter={it + 1}/{effective_iters}"
            )
            break

        if (it + 1) % 50 == 0:
            calls_now = int(getattr(evaluator, "calls", 0))
            calls_delta = max(0, calls_now - calls_last_report)
            calls_last_report = calls_now
            exact_calls_now = int(getattr(evaluator_exact_proxy, "calls", 0))
            exact_calls_delta = max(0, exact_calls_now - exact_calls_last_report)
            exact_calls_last_report = exact_calls_now
            _log(
                f"[ALNS-inner] iter {it + 1}/{iters} | cur={cur_obj:.2f} | "
                f"best={best_obj:.2f} | stall={stall} | T={T:.4f} | post={post_shake} "
                f"| dCalls50={calls_delta} | dExact50={exact_calls_delta}"

            )
            _log(f"[ALNS-adapt] destroy top = {destroy_pool.topk(4)}")
            _log(f"[ALNS-adapt] repair  top = {repair_pool.topk(2)}")


    _log(f"[ALNS-inner] best_obj after SA = {best_obj:.2f}")

    final_sol = best_feasible if (feasible_first and best_feasible is not None) else best
    final_obj = best_feasible_obj if (feasible_first and best_feasible is not None) else best_obj

    final_sol.routes = _normalize_ws_blocks(final_sol.routes, evaluator)
    do_final_relabel = bool((not turbo_mode) and (effective_relabel_interval > 0) and (not very_large_mode))
    if do_final_relabel:
        evaluator.set_tag("final_relabel")
        best_rl, best_rl_obj = _relabel_best_routes(
            final_sol.routes,
            evaluator=evaluator,
            shelf_seq=final_sol.shelf_seq,
            place=final_sol.place,
            max_exact_agv=int(relabel_max_exact_agv),
            eval_top_k=int(effective_relabel_top_k),
            task_shelf_mapping=task_shelf_mapping,
            shelf_init_override=shelf_init,
        )
        if best_rl_obj < float(final_obj) - 1e-9:
            final_sol.routes = best_rl
            final_obj = float(best_rl_obj)

    final_tail_rounds = 0 if (turbo_mode or large_scale_mode) else 1
    for _ in range(final_tail_rounds):
        routes_int, place_int, improved_int, obj_int = intensify_critical_tail_once(
            routes=final_sol.routes,
            place=final_sol.place,
            shelf_seq=final_sol.shelf_seq,
            evaluator=evaluator,
            S_near_by_j=S_near_by_j,
            task_shelf_mapping=task_shelf_mapping,
            shelf_init_override=shelf_init,
            tail_k=2,
        )
        if improved_int and float(obj_int) < float(final_obj) - 1e-9:
            final_sol.routes, final_sol.place = routes_int, place_int
            final_obj = float(obj_int)
        else:
            break

    # Quality-oriented endgame on large instances: a few exact, high-value moves.
    if large_scale_mode and math.isfinite(float(final_obj)):
        no_gain = 0
        end_rounds = 6 if very_large_mode else 8
        for rr in range(int(end_rounds)):
            if no_gain >= 3:
                break
            improved_any = False

            evaluator.set_tag("final_balance")
            rb_routes, rb_place, rb_improved, rb_obj = rebalance_bottleneck_route_once(
                routes=final_sol.routes,
                place=final_sol.place,
                shelf_seq=final_sol.shelf_seq,
                evaluator=evaluator,
                S_near_by_j=S_near_by_j,
                task_shelf_mapping=task_shelf_mapping,
                shelf_init_override=shelf_init,
                rng=rng,
                tail_k=4 if very_large_mode else 6,
                max_pos_samples=4 if very_large_mode else 6,
                max_s_samples=2 if very_large_mode else 3,
                max_evals=6 if very_large_mode else 10,
            )
            if rb_improved and (float(rb_obj) < float(final_obj) - 1e-9):
                final_sol.routes, final_sol.place = rb_routes, rb_place
                final_obj = float(rb_obj)
                improved_any = True

            if (rr % 2) == 0 or (not improved_any):
                evaluator.set_tag("final_2opt")
                st_routes, st_place, st_improved, st_obj = cross_vehicle_2opt_star_once(
                    routes=final_sol.routes,
                    place=final_sol.place,
                    shelf_seq=final_sol.shelf_seq,
                    evaluator=evaluator,
                    rng=rng,
                    max_trials=6 if very_large_mode else 10,
                    allow_non_improving=False,
                )
                if st_improved and (float(st_obj) < float(final_obj) - 1e-9):
                    final_sol.routes, final_sol.place = st_routes, st_place
                    final_obj = float(st_obj)
                    improved_any = True

            if improved_any:
                no_gain = 0
            else:
                no_gain += 1

    allow_milp_polish = bool(enable_milp_polish_eff and (milp_polish_on_turbo_eff or (not turbo_mode)))
    if allow_milp_polish and math.isfinite(float(final_obj)):
        milp_tl = float(milp_polish_time_limit_eff)
        if t_budget is not None:
            elapsed_sec = time.perf_counter() - t_start
            rem_sec = max(0.0, float(t_budget) - float(elapsed_sec))
            milp_tl = min(float(milp_tl), max(0.0, float(rem_sec) - 0.2))
        if milp_tl >= 1.0:
            evaluator_for_milp = evaluator_exact_proxy if layering_on else evaluator
            milp_sol, milp_obj, milp_key, milp_improved, milp_msg = _try_milp_polish_endgame(
                final_sol,
                evaluator=evaluator_for_milp,
                task_shelf_mapping=task_shelf_mapping,
                seed=int(seed),
                time_limit_sec=float(milp_tl),
                lock_immediate=milp_polish_lock_immediate_eff,
            )
            _log(f"[ALNS-milp] {milp_msg}")
            if milp_improved:
                final_sol = milp_sol
                final_obj = float(milp_obj)
                _log(f"[ALNS-milp] accepted key={milp_key} obj={final_obj:.2f}")

    if layering_on:
        evaluator_exact_proxy.set_tag("final_exact_recheck")
        final_obj_exact, final_det_exact = _cache_eval(
            "exact",
            evaluator_exact_proxy,
            final_sol.routes,
            final_sol.shelf_seq,
            final_sol.place,
        )
        final_key_exact = tuple(_safe_infeas_key(final_det_exact, evaluator_exact_proxy, obj=final_obj_exact))
        if final_key_exact == key_feas and math.isfinite(float(final_obj_exact)):
            final_obj = float(final_obj_exact)

    _log(f"[ALNS-inner] final_best_obj = {final_obj:.2f}")
    _log(f"[ALNS-inner] eval_calls_fast = {int(getattr(evaluator, 'calls', 0))}")
    if layering_on:
        _log(f"[ALNS-inner] eval_calls_exact = {int(getattr(evaluator_exact_proxy, 'calls', 0))}")
    if hasattr(evaluator, "calls_by_tag"):
        try:
            top_tags = sorted(
                [(str(k), int(v)) for k, v in dict(getattr(evaluator, "calls_by_tag", {})).items()],
                key=lambda kv: kv[1],
                reverse=True,
            )
            _log(f"[ALNS-inner] eval_calls_by_tag_fast_top = {top_tags[:8]}")
        except Exception:
            pass
    if layering_on and hasattr(evaluator_exact_proxy, "calls_by_tag"):
        try:
            top_tags_exact = sorted(
                [(str(k), int(v)) for k, v in dict(getattr(evaluator_exact_proxy, "calls_by_tag", {})).items()],
                key=lambda kv: kv[1],
                reverse=True,
            )
            _log(f"[ALNS-inner] eval_calls_by_tag_exact_top = {top_tags_exact[:8]}")
        except Exception:
            pass
    if use_eval_cache_eff:
        _log(
            f"[ALNS-cache] fast_hit={cache_hits_fast} fast_miss={cache_miss_fast} "
            f"exact_hit={cache_hits_exact} exact_miss={cache_miss_exact} "
            f"fast_size={len(cache_fast)} exact_size={len(cache_exact)}"
        )
    # [text_corrupted] routes [text_corrupted]shelf_seq [text_corrupted] ws_fixed_seq[text_corrupted]    final_sol.routes = _normalize_ws_blocks(final_sol.routes, evaluator)
    final_sol.routes = normalize_routes_by_shelf_seq_order(final_sol.routes, final_sol.shelf_seq, task_shelf_mapping)
    if bool(verbose_eff):
        _ = basic_feasibility_check_level0(
            routes=final_sol.routes,
            shelf_seq=final_sol.shelf_seq,
            place=final_sol.place,
            evaluator=(evaluator_exact_proxy if layering_on else evaluator),
            task_shelf_mapping=task_shelf_mapping,
            verbose=True,
        )
    return final_sol

