# milp_solver.py
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from gurobipy import GRB

import optimization_model  # 你现在的 MILP 在这里：optimize_warehouse


def _build_task_shelf_mapping_from_shelf_seq(
    shelf_seq: Optional[Dict[int, List[int]]],
    J: Set[int],
) -> Dict[int, Optional[int]]:
    """
    用 shelf_seq 反推 task_shelf_mapping：task -> shelf_id
    """
    mp: Dict[int, Optional[int]] = {}
    if not isinstance(shelf_seq, dict):
        return mp
    Jset = set(int(x) for x in J)
    for c, seq in shelf_seq.items():
        try:
            cc = int(c)
        except Exception:
            continue
        for j in (seq or []):
            try:
                jj = int(j)
            except Exception:
                continue
            if jj in Jset:
                mp[jj] = cc
    return mp


def _build_tasks_from_pi_D(
    J: Set[int],
    pi: Dict[int, int],
    D: Dict[int, float],
) -> Dict[int, Tuple[Optional[int], float, Any, Any]]:
    """
    optimize_warehouse 只会用 tasks[j] 的 (ws, dur, _, _) 来生成 pi/D（针对真实任务 j in J）
    """
    out: Dict[int, Tuple[Optional[int], float, Any, Any]] = {}
    for j in J:
        jj = int(j)
        out[jj] = (int(pi[jj]), float(D[jj]), None, None)
    return out


def solve_milp_bridge(
    *,
    prefix: str,
    gamma: int,
    warm_hint: Optional[dict] = None,
    fix_structure: bool = False,
    time_limit: float = 300.0,
    verbose: bool = False,

    # ---- data from main_align_bridge.py ----
    J: Set[int],
    R: dict,
    S: dict,
    pi: Dict[int, int],
    D: Dict[int, float],
    J0: Dict[int, int],
    Jd: Dict[int, int],
    J_I: Dict[int, int],
    shelf_data: Dict[int, int],
    agv_data: Dict[int, int],
    d_s_pi: dict,
    d_pi_s: dict,
    d_s_s: dict,
    Delta_s_pi: Optional[dict] = None,
    Delta_pi_s: Optional[dict] = None,
    Delta_s_s: Optional[dict] = None,
    ws_fixed_seq: Optional[Dict[int, List[int]]] = None,

    # structure (from ALNS bundle)
    fixed_routes: Optional[Dict[int, List[int]]] = None,
    fixed_shelf_seq: Optional[Dict[int, List[int]]] = None,
    fixed_place: Optional[Dict[int, int]] = None,
) -> dict:
    """
    返回给 main_align_bridge.py：
      {
        "status": "...",
        "feasible": bool,
        "obj": float|None,
        "runtime": float|None,
        "mipgap": float|None
      }
    """

    # 关键：如果 evaluator 那边已经 infeasible，warm_hint 会是 None
    # 这时继续跑 MILP 没意义（还会巨慢），直接跳过。
    if warm_hint is None:
        return {
            "status": "SKIP_NO_WARMHINT",
            "feasible": False,
            "obj": None,
            "runtime": 0.0,
            "mipgap": None,
        }

    # ========== 准备 optimize_warehouse 所需的最小输入 ==========
    Jset = set(int(x) for x in J)

    # tasks：由 pi/D 生成（真实任务即可）
    tasks = _build_tasks_from_pi_D(Jset, pi, D)

    # shelf_seq：用 ALNS 的结构（你就是要验证“这套结构”）
    shelf_seq = fixed_shelf_seq if isinstance(fixed_shelf_seq, dict) else {}

    # task_shelf_mapping：从 shelf_seq 反推
    task_shelf_mapping = _build_task_shelf_mapping_from_shelf_seq(shelf_seq, Jset)

    # unused_shelves：没有任何真实任务的货架
    shelf_ids = set(int(k) for k in shelf_data.keys())
    used_shelves = set(int(v) for v in task_shelf_mapping.values() if v is not None)
    unused_shelves = set(int(sid) for sid in shelf_ids if sid not in used_shelves)

    # shelf_virtual_tasks：你模型里 vt = J_I[shelf]
    shelf_virtual_tasks = {int(sid): int(J_I[int(sid)]) for sid in shelf_ids if int(sid) in J_I}

    # J_E / J_I_SI：你当前 optimize_warehouse 里基本不用它们，给空即可
    J_E: Set[int] = set()
    J_I_SI: Dict[int, int] = {}
    for c, seq in shelf_seq.items():
        seq_clean = [int(j) for j in (seq or []) if int(j) in Jset]
        if seq_clean:
            J_I_SI[int(c)] = int(seq_clean[0])

    # lock_hint：fix_structure=True 时锁 w/x/z/immediate（v 永远不锁）
    lock_hint = None
    if fix_structure:
        lock_hint = {"w": True, "x": True, "z": True, "immediate": True, "v": False}

    # 其它 optimize_warehouse 形式参数（基本不用）
    AGV_positions = dict(agv_data)          # 占位给一个像样的 dict
    agv_positions_map = {}
    bj = {}
    hj = {}
    map_obj = None
    K_dummy = {}  # 我们会把 d_s_pi/d_pi_s/d_s_s 直接喂给 optimize_warehouse，让它不再依赖 K

    # ===== 运行 MILP（由 optimization_model.optimize_warehouse 真正求解）=====
    file_prefix = f"{prefix}_bridge_milpG{int(gamma)}"

    try:
        _, model, _, _ = optimization_model.optimize_warehouse(
            R=R,
            S=S,
            K=K_dummy,
            tasks=tasks,
            AGV_positions=AGV_positions,
            agv_positions_map=agv_positions_map,
            bj=bj,
            hj=hj,
            map_obj=map_obj,
            task_shelf_mapping=task_shelf_mapping,
            J_I=J_I,
            J_E=J_E,
            J=Jset,
            J0=J0,
            Jd=Jd,
            J_I_SI=J_I_SI,
            shelf_data=shelf_data,
            agv_data=agv_data,
            shelf_virtual_tasks=shelf_virtual_tasks,
            unused_shelves=unused_shelves,
            file_prefix=file_prefix,
            gamma_budget=int(gamma),
            ws_fixed_seq=ws_fixed_seq,
            warm_start=None,
            warm_hint=warm_hint,
            lock_hint=lock_hint,

            # ===== 下面三个参数需要你在 optimization_model.py 里按我第 2 部分加上 =====
            d_s_pi_in=d_s_pi,
            d_pi_s_in=d_pi_s,
            d_s_s_in=d_s_s,
            time_limit=float(time_limit),
            no_improve_limit=min(120.0, float(time_limit)),   # 你可以自己调
            bridge_quiet=(not bool(verbose)),
        )
    except TypeError as e:
        # 说明 optimization_model.optimize_warehouse 还没按第2部分加参数
        return {
            "status": f"TYPE_ERROR({e})",
            "feasible": False,
            "obj": None,
            "runtime": 0.0,
            "mipgap": None,
        }
    except Exception as e:
        return {
            "status": f"ERROR({e})",
            "feasible": False,
            "obj": None,
            "runtime": 0.0,
            "mipgap": None,
        }

    status = int(model.Status)
    runtime = float(getattr(model, "Runtime", float("nan")))
    solcnt = int(getattr(model, "SolCount", 0))

    # “feasible” 以是否有 incumbent 为准
    feasible = solcnt > 0 and status not in (GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED)

    if status == GRB.OPTIMAL:
        return {"status": "OPTIMAL", "feasible": True, "obj": float(model.ObjVal), "runtime": runtime, "mipgap": 0.0}

    if status == GRB.TIME_LIMIT:
        obj = float(model.ObjVal) if solcnt > 0 else None
        gap = float(getattr(model, "MIPGap", float("nan"))) if solcnt > 0 else None
        return {"status": "TIME_LIMIT", "feasible": feasible, "obj": obj, "runtime": runtime, "mipgap": gap}

    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return {"status": "INFEASIBLE", "feasible": False, "obj": None, "runtime": runtime, "mipgap": None}

    return {"status": f"OTHER({status})", "feasible": feasible, "obj": None, "runtime": runtime, "mipgap": None}