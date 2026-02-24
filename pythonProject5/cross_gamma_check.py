# cross_gamma_check.py
from __future__ import annotations

import csv
import inspect
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

Number = float


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _try_int(x: Any) -> Any:
    try:
        # 注意：json 里 key 常是字符串数字
        if isinstance(x, str) and x.strip().isdigit():
            return int(x)
        if isinstance(x, (int, float)) and int(x) == x:
            return int(x)
        return x
    except Exception:
        return x


def _normalize_routes(routes_by_agv: Any) -> Dict[int, List[int]]:
    """
    允许 routes_by_agv:
      - {1: [..], 2: [..]}
      - {"1": ["3","4"], ...}
    """
    if routes_by_agv is None:
        return {}
    if not isinstance(routes_by_agv, dict):
        raise TypeError(f"routes_by_agv must be dict, got {type(routes_by_agv)}")

    out: Dict[int, List[int]] = {}
    for k, v in routes_by_agv.items():
        agv = _try_int(k)
        if not isinstance(agv, int):
            raise ValueError(f"AGV id must be int-like, got {k}")
        if v is None:
            out[agv] = []
            continue
        if not isinstance(v, list):
            raise TypeError(f"routes_by_agv[{k}] must be list, got {type(v)}")
        out[agv] = [int(_try_int(t)) for t in v]
    return out


def _normalize_shelf_seq(shelf_seq: Any) -> Dict[int, List[int]]:
    """
    允许 shelf_seq:
      - {1: [..], 2: [..]}
      - {"1": ["3","4"], ...}
    """
    if shelf_seq is None:
        return {}
    if not isinstance(shelf_seq, dict):
        raise TypeError(f"shelf_seq must be dict, got {type(shelf_seq)}")

    out: Dict[int, List[int]] = {}
    for k, v in shelf_seq.items():
        sid = _try_int(k)
        if not isinstance(sid, int):
            raise ValueError(f"Shelf id must be int-like, got {k}")
        if v is None:
            out[sid] = []
            continue
        if not isinstance(v, list):
            raise TypeError(f"shelf_seq[{k}] must be list, got {type(v)}")
        out[sid] = [int(_try_int(t)) for t in v]
    return out


def _normalize_place(place: Any) -> Dict[int, int]:
    """
    允许 place:
      - dict: {task: shelfpos}
      - list: [(task, shelfpos), ...] / [[task, shelfpos], ...]
    """
    if place is None:
        return {}
    if isinstance(place, dict):
        out: Dict[int, int] = {}
        for k, v in place.items():
            tk = int(_try_int(k))
            sv = int(_try_int(v))
            out[tk] = sv
        return out

    if isinstance(place, list):
        out2: Dict[int, int] = {}
        for item in place:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"place list item must be (task,s), got {item}")
            tk = int(_try_int(item[0]))
            sv = int(_try_int(item[1]))
            out2[tk] = sv
        return out2

    raise TypeError(f"place must be dict or list, got {type(place)}")


def normalize_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    把 bundle 里常见字段统一成可评估的格式。
    """
    b = dict(bundle)

    # 常见 key 名：你的导出里基本就是这几个
    routes = b.get("routes_by_agv") or b.get("routes") or b.get("routesByAgv")
    shelf_seq = b.get("shelf_seq") or b.get("shelfSeq")
    place = b.get("place") or b.get("placement")

    b["routes_by_agv"] = _normalize_routes(routes)
    b["shelf_seq"] = _normalize_shelf_seq(shelf_seq)
    b["place"] = _normalize_place(place)
    return b


def _parse_eval_result(res: Any) -> Tuple[Number, Dict[str, Any]]:
    """
    兼容 evaluator 返回：
      - float/int: makespan
      - (makespan, details_dict)
      - details_dict（里面含 makespan）
    """
    details: Dict[str, Any] = {}

    if isinstance(res, (int, float)):
        return float(res), details

    if isinstance(res, dict):
        details = dict(res)
        # 常见字段名尝试
        for k in ("makespan", "C_max_AGV", "cmax", "C_task_max", "objective", "obj"):
            if k in details and isinstance(details[k], (int, float)):
                return float(details[k]), details
        raise ValueError(f"dict result has no makespan-like key: keys={list(details.keys())}")

    if isinstance(res, tuple) and len(res) >= 1:
        ms = res[0]
        if not isinstance(ms, (int, float)):
            raise ValueError(f"tuple[0] not numeric makespan: {ms}")
        if len(res) >= 2 and isinstance(res[1], dict):
            details = dict(res[1])
        return float(ms), details

    raise TypeError(f"Unknown evaluator result type: {type(res)}")


def evaluate_bundle_one_gamma(
    *,
    evaluator: Any,
    routes_by_agv: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    gamma_eval: int,
) -> Tuple[Number, Dict[str, Any]]:
    """
    尝试用不同调用方式兼容你当前 evaluator 的接口。
    """
    # 1) 如果 evaluator 有 set_gamma / set_Gamma，优先用
    if hasattr(evaluator, "set_gamma") and callable(getattr(evaluator, "set_gamma")):
        evaluator.set_gamma(gamma_eval)
    elif hasattr(evaluator, "set_Gamma") and callable(getattr(evaluator, "set_Gamma")):
        evaluator.set_Gamma(gamma_eval)
    else:
        # 2) 否则尝试直接写属性
        if hasattr(evaluator, "gamma"):
            try:
                setattr(evaluator, "gamma", gamma_eval)
            except Exception:
                pass

    # 3) 尝试调用 evaluate（用 kwargs 和 positional 两种）
    #    你现在的项目里 evaluate 基本能跑通，这里只做“最大兼容”
    eval_fn = getattr(evaluator, "evaluate", None)
    if eval_fn is None or not callable(eval_fn):
        raise AttributeError("evaluator has no callable .evaluate(...)")

    # 如果 evaluate 支持 gamma 参数，尽量传进去（避免只改属性不生效）
    sig = None
    try:
        sig = inspect.signature(eval_fn)
    except Exception:
        sig = None

    can_pass_gamma = False
    if sig is not None:
        can_pass_gamma = ("gamma" in sig.parameters) or ("Gamma" in sig.parameters) or ("gamma_eval" in sig.parameters)

    # 调用顺序：kwargs 优先（更不容易弄错位置）
    last_err: Optional[Exception] = None
    call_patterns = []

    if can_pass_gamma:
        call_patterns.extend([
            lambda: eval_fn(routes_by_agv=routes_by_agv, shelf_seq=shelf_seq, place=place, gamma=gamma_eval),
            lambda: eval_fn(routes_by_agv=routes_by_agv, shelf_seq=shelf_seq, place=place, Gamma=gamma_eval),
            lambda: eval_fn(routes_by_agv, shelf_seq, place, gamma=gamma_eval),
            lambda: eval_fn(routes_by_agv, shelf_seq, place, Gamma=gamma_eval),
        ])

    call_patterns.extend([
        lambda: eval_fn(routes_by_agv=routes_by_agv, shelf_seq=shelf_seq, place=place),
        lambda: eval_fn(routes_by_agv, shelf_seq, place),
    ])

    for fn in call_patterns:
        try:
            res = fn()
            return _parse_eval_result(res)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"All evaluator call patterns failed. Last error: {last_err}")


def cross_gamma_suite(
    *,
    prefix: str,
    bundles_by_gamma: Dict[int, Dict[str, Any]],
    eval_gammas: List[int],
    evaluator_factory: Callable[[int], Any],
    milp_opt_by_gamma: Optional[Dict[int, float]] = None,
    out_dir: str = "solution_exports",
) -> List[Dict[str, Any]]:
    """
    对每个 source_gamma 的 bundle，评估到所有 eval_gammas：
      输出长表：prefix_crossGamma_suite.csv
      输出矩阵：prefix_crossGamma_matrix.csv
    """
    _ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []

    for src_g, bundle in bundles_by_gamma.items():
        nb = normalize_bundle(bundle)
        routes_by_agv = nb["routes_by_agv"]
        shelf_seq = nb["shelf_seq"]
        place = nb["place"]

        for eg in eval_gammas:
            evaluator = evaluator_factory(eg)
            ms, details = evaluate_bundle_one_gamma(
                evaluator=evaluator,
                routes_by_agv=routes_by_agv,
                shelf_seq=shelf_seq,
                place=place,
                gamma_eval=eg,
            )

            opt = None
            regret = None
            ratio = None
            if milp_opt_by_gamma is not None and eg in milp_opt_by_gamma and milp_opt_by_gamma[eg] is not None:
                opt = float(milp_opt_by_gamma[eg])
                regret = ms - opt
                ratio = (ms / opt) if opt > 0 else None

            # 尽量从 details 里抠一些你常用的诊断字段
            bottleneck_task = details.get("bottleneck_task") or details.get("bottleneckTask")
            penalties = details.get("penalties")

            rows.append({
                "prefix": prefix,
                "source_gamma": int(src_g),
                "eval_gamma": int(eg),
                "makespan": float(ms),
                "milp_opt_at_eval_gamma": opt,
                "regret_vs_opt": regret,
                "ratio_vs_opt": ratio,
                "bottleneck_task": bottleneck_task,
                "penalties": json.dumps(penalties, ensure_ascii=False) if penalties is not None else None,
            })

    # 1) 写长表
    suite_csv = os.path.join(out_dir, f"{prefix}_crossGamma_suite.csv")
    with open(suite_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 2) 写矩阵（source_gamma 行，eval_gamma 列）
    matrix: Dict[int, Dict[int, float]] = {}
    for r in rows:
        sg = int(r["source_gamma"])
        eg = int(r["eval_gamma"])
        matrix.setdefault(sg, {})[eg] = float(r["makespan"])

    matrix_csv = os.path.join(out_dir, f"{prefix}_crossGamma_matrix.csv")
    eval_cols = [int(g) for g in eval_gammas]
    with open(matrix_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["source_gamma"] + [f"eval_gamma_{g}" for g in eval_cols]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for sg in sorted(matrix.keys()):
            row = {"source_gamma": sg}
            for g in eval_cols:
                row[f"eval_gamma_{g}"] = matrix[sg].get(g, "")
            w.writerow(row)

    # 3) 控制台打印一个小表（方便你肉眼看）
    print("\n[CrossGamma] makespan matrix (source_gamma -> eval_gamma):")
    header = "source\\eval | " + " | ".join([f"{g:>6d}" for g in eval_cols])
    print(header)
    print("-" * len(header))
    for sg in sorted(matrix.keys()):
        line = f"{sg:>10d} | " + " | ".join([f"{matrix[sg].get(g, float('nan')):6.2f}" for g in eval_cols])
        print(line)

    print(f"\n[CrossGamma] exported:\n  - {suite_csv}\n  - {matrix_csv}\n")
    return rows
