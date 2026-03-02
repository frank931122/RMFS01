from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from evaluator import RobustEvaluator


def _build_toy_instance() -> Tuple[Dict, Dict[int, List[int]], Dict[int, List[int]], Dict[int, int]]:
    J = {1, 2}
    R = {1: (0, 0)}
    S = {1: (0, 0), 2: (1, 0), 3: (2, 0)}
    pi = {1: 1, 2: 1}
    D = {1: 2.0, 2: 3.0}

    J0 = {1: -1}
    Jd = {1: -2}
    J_I = {1: -1001, 2: -1002}

    shelf_data = {1: 1, 2: 2}
    agv_data = {1: 1}

    d_s_s = {}
    for a in S:
        for b in S:
            d_s_s[(int(a), int(b))] = float(abs(int(a) - int(b)))
    d_s_pi = {}
    d_pi_s = {}
    for s in S:
        for j in J:
            travel = float(abs(int(s) - (1 + (int(j) % 2))) + 1.0)
            d_s_pi[(int(s), int(j))] = travel
            d_pi_s[(int(j), int(s))] = travel

    ws_fixed_seq = {1: [1, 2]}

    evaluator_kwargs = dict(
        J=J,
        R=R,
        S=S,
        pi=pi,
        D=D,
        J0=J0,
        Jd=Jd,
        J_I=J_I,
        shelf_data=shelf_data,
        agv_data=agv_data,
        d_s_pi=d_s_pi,
        d_pi_s=d_pi_s,
        d_s_s=d_s_s,
        gamma=0,
        ws_fixed_seq=ws_fixed_seq,
        ws_setup_rule="flat",
        lock_place=True,
        detach_on_mismatch=False,
        envelope_shared_resources=True,
        enable_chain_repair=True,
        chain_repair_max_iters=8,
        enable_cell_repair=True,
        cell_repair_max_iters=8,
        timeline_mode="off",
        record_cell_repair_log=False,
        collect_v_arcs=False,
        cell_conflict_mode="hard",
        allow_incomplete=False,
    )
    routes = {1: [1, 2]}
    shelf_seq = {1: [1], 2: [2]}
    place = {1: 1, 2: 2}
    return evaluator_kwargs, routes, shelf_seq, place


def _almost_eq(a: float, b: float, tol: float = 1e-9) -> bool:
    if math.isfinite(float(a)) and math.isfinite(float(b)):
        return abs(float(a) - float(b)) <= float(tol)
    return (not math.isfinite(float(a))) and (not math.isfinite(float(b)))


def main() -> int:
    evaluator_kwargs, routes, shelf_seq, place = _build_toy_instance()

    # Exact mode must match legacy default behavior.
    e_legacy = RobustEvaluator(**evaluator_kwargs)
    e_exact = RobustEvaluator(mode="exact", **evaluator_kwargs)
    obj_legacy, det_legacy = e_legacy.evaluate(routes, shelf_seq, place)
    obj_exact, det_exact = e_exact.evaluate(routes, shelf_seq, place)

    if not _almost_eq(float(obj_legacy), float(obj_exact)):
        raise AssertionError(f"exact mode mismatch: legacy={obj_legacy}, exact={obj_exact}")
    if bool(det_legacy.get("feasible", False)) != bool(det_exact.get("feasible", False)):
        raise AssertionError("exact mode feasible flag mismatch")

    # Fast mode mapping must enforce required switches.
    fast_kwargs = dict(evaluator_kwargs)
    fast_kwargs.update(
        {
            "timeline_mode": "full",
            "record_cell_repair_log": True,
            "collect_v_arcs": True,
            "enable_cell_repair": True,
            "cell_repair_max_iters": 99,
            "enable_chain_repair": True,
            "chain_repair_max_iters": 99,
            "cell_conflict_mode": "hard",
            "allow_incomplete": False,
        }
    )
    e_fast = RobustEvaluator(mode="fast", **fast_kwargs)
    assert e_fast.mode == "fast"
    assert e_fast.enable_cell_repair is False
    assert int(e_fast.cell_repair_max_iters) == 0
    assert e_fast.enable_chain_repair is False
    assert int(e_fast.chain_repair_max_iters) == 0
    assert e_fast.timeline_mode == "off"
    assert e_fast.record_cell_repair_log is False
    assert e_fast.collect_v_arcs is False
    assert e_fast.cell_conflict_mode == "penalty"
    assert e_fast.allow_incomplete is True

    print("[verify_accuracy] PASS: exact mode consistency + fast mode switch mapping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
