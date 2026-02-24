# verify_crossgamma.py
from __future__ import annotations
import inspect
from typing import Any
import os
import json
import math
import argparse
import inspect
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

from evaluator import RobustEvaluator


# -----------------------------
# Helpers: load bundle
# -----------------------------
def _intify_routes(routes_raw: Any) -> Dict[int, List[int]]:
    routes_raw = routes_raw or {}
    out: Dict[int, List[int]] = {}
    for r, seq in routes_raw.items():
        rr = int(r)
        out[rr] = [int(x) for x in (seq or [])]
    return out


def _intify_shelf_seq(shelf_raw: Any) -> Dict[int, List[int]]:
    shelf_raw = shelf_raw or {}
    out: Dict[int, List[int]] = {}
    for c, seq in shelf_raw.items():
        cc = int(c)
        out[cc] = [int(x) for x in (seq or [])]
    return out


def _intify_place(place_raw: Any) -> Dict[int, int]:
    place_raw = place_raw or {}
    out: Dict[int, int] = {}
    for j, s in place_raw.items():
        out[int(j)] = int(s)
    return out


def load_bundle(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"bundle not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)
    b["routes"] = _intify_routes(b.get("routes") or b.get("routes_by_agv"))
    b["shelf_seq"] = _intify_shelf_seq(b.get("shelf_seq"))
    b["place"] = _intify_place(b.get("place"))
    return b


# -----------------------------
# Helpers: structure checks
# -----------------------------
def _dup_list(lst: List[int]) -> List[int]:
    seen = set()
    dup = set()
    for x in lst:
        if x in seen:
            dup.add(x)
        seen.add(x)
    return sorted(dup)


def structure_check(
    *,
    tasks_df: pd.DataFrame,
    bundle: dict,
    max_show: int = 20,
) -> None:
    J_real = set(int(x) for x in tasks_df["Task"].tolist())
    task2shelf = {int(r.Task): int(r.Shelf) for r in tasks_df.itertuples(index=False)}

    routes = bundle["routes"]
    shelf_seq = bundle["shelf_seq"]
    place = bundle["place"]

    route_tasks: List[int] = []
    for _, seq in routes.items():
        for j in seq:
            if int(j) in J_real:
                route_tasks.append(int(j))

    shelf_tasks: List[int] = []
    for c, seq in shelf_seq.items():
        for j in seq:
            if int(j) in J_real:
                shelf_tasks.append(int(j))

    miss_route = sorted(J_real - set(route_tasks))
    miss_shelf = sorted(J_real - set(shelf_tasks))
    dup_route = _dup_list(route_tasks)
    dup_shelf = _dup_list(shelf_tasks)

    miss_place = sorted([j for j in J_real if j not in place])

    wrong_chain = []
    for c, seq in shelf_seq.items():
        cc = int(c)
        for j in seq:
            jj = int(j)
            if jj not in J_real:
                continue
            true_c = task2shelf.get(jj, None)
            if true_c is None:
                continue
            if int(true_c) != cc:
                wrong_chain.append((jj, cc, int(true_c)))

    print("=" * 80)
    print("[STRUCTURE CHECK]")
    print(f"  |J_real|={len(J_real)}")
    print(f"  routes:  cover={len(set(route_tasks))} dup={len(dup_route)} miss={len(miss_route)}")
    print(f"  shelf_seq: cover={len(set(shelf_tasks))} dup={len(dup_shelf)} miss={len(miss_shelf)}")
    print(f"  place: cover={len(place)} miss_real_tasks={len(miss_place)}")
    print(f"  wrong_chain(task in wrong shelf key) = {len(wrong_chain)}")

    if miss_route:
        print("  missing_in_routes:", miss_route[:max_show])
    if miss_shelf:
        print("  missing_in_shelf_seq:", miss_shelf[:max_show])
    if dup_route:
        print("  duplicate_in_routes:", dup_route[:max_show])
    if dup_shelf:
        print("  duplicate_in_shelf_seq:", dup_shelf[:max_show])
    if miss_place:
        print("  missing_place:", miss_place[:max_show])
    if wrong_chain:
        print("  wrong_chain samples (task, in_shelf, true_shelf):", wrong_chain[:max_show])
    print("=" * 80)


# -----------------------------
# Helpers: evaluator creation (try to reuse your scenario builder)
# -----------------------------
def _deep_collect_named_values(
    root: Any,
    target_names: set[str],
    *,
    max_depth: int = 6,
    max_nodes: int = 6000,
) -> dict[str, Any]:
    """
    在对象图里递归搜：dict key / attribute name 命中 target_names 的值。
    兼容 __dict__ / __slots__ / property（通过 getattr 探测少量候选名）。
    """
    found: dict[str, Any] = {}
    seen = set()
    stack = [(root, 0)]
    nodes = 0

    # 常见“中间容器”属性名（帮助穿透 Scenario->ctx/map 等）
    common_children = {
        "ctx", "data", "map", "map_obj", "scenario", "env",
        "movement_manager", "time_manager", "mm", "tm",
        "problem", "instance", "inputs",
    }

    # 为了支持 slots/property：我们只对 “target_names ∪ common_children” 做 getattr 探测
    probe_names = set(target_names) | common_children

    while stack:
        cur, d = stack.pop()
        if cur is None:
            continue

        oid = id(cur)
        if oid in seen:
            continue
        seen.add(oid)

        nodes += 1
        if nodes > max_nodes:
            break

        if d > max_depth:
            continue

        # dict
        if isinstance(cur, dict):
            for k, v in cur.items():
                ks = str(k)
                if ks in target_names and ks not in found:
                    found[ks] = v
                stack.append((v, d + 1))
            continue

        # list/tuple/set
        if isinstance(cur, (list, tuple, set)):
            for v in cur:
                stack.append((v, d + 1))
            continue

        # try vars(__dict__)
        dd = None
        try:
            dd = vars(cur)
        except Exception:
            dd = None

        if isinstance(dd, dict):
            for name, v in dd.items():
                if name in target_names and name not in found:
                    found[name] = v
                stack.append((v, d + 1))

        # probe slots/property/common fields
        for name in probe_names:
            if name in found:
                continue
            try:
                if hasattr(cur, name):
                    v = getattr(cur, name)
                    if name in target_names and name not in found:
                        found[name] = v
                    stack.append((v, d + 1))
            except Exception:
                continue

    return found


def _coerce_int_int_dict(x: Any) -> Any:
    """若 x 是 dict，尽量转成 int->int；否则原样返回。"""
    if not isinstance(x, dict):
        return x
    out = {}
    for k, v in x.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            # 保留原值，避免硬崩
            out[k] = v
    return out


def build_evaluator_from_prefix(prefix: str, gamma: int, debug: bool = False):
    """
    用 build_scenario_from_prefix 读入场景（拿到 J/R/S/pi/D/距离等），
    然后按 main.py 的口径补齐 RobustEvaluator 必需的：
      - shelf_data  <- scenario.shelf_init
      - agv_data    <- scenario.agv_init
      - J0/Jd       <- 由 agv_id 生成 (1000+r)/(2000+r)
    """
    import inspect
    import math
    from scenario import build_scenario_from_prefix
    from utils import distance
    from evaluator import RobustEvaluator

    # 1) build Scenario（注意：你这里的 build_scenario_from_prefix 明确需要 distance）
    sc = build_scenario_from_prefix(
        prefix=prefix,
        distance=distance,
        D_setup=2.0,
        gamma_budget=0,
        K_near=12,
    )

    # 2) 从 Scenario 补齐 main.py 里 evaluator 需要的字段
    #    Scenario 里叫 shelf_init / agv_init，本质就是 “id -> cell_index”
    shelf_data = {int(k): int(v) for k, v in (getattr(sc, "shelf_init", {}) or {}).items()}
    agv_data   = {int(k): int(v) for k, v in (getattr(sc, "agv_init", {}) or {}).items()}

    # 如果 sc.R 不是 dict，而 agv_data 才是“真实 AGV 列表”，以 agv_data 为准
    agv_ids = sorted(set(int(r) for r in (agv_data.keys() if agv_data else getattr(sc, "R", []))))

    J0 = {int(r): 1000 + int(r) for r in agv_ids}
    Jd = {int(r): 2000 + int(r) for r in agv_ids}

    # 3) 距离与 Delta：Scenario 已有 d_*，但 Delta_* 可能没有，我们用拷贝兜底
    d_s_pi = getattr(sc, "d_s_pi", {}) or {}
    d_pi_s = getattr(sc, "d_pi_s", {}) or {}
    d_s_s  = getattr(sc, "d_s_s", {}) or {}

    Delta_s_pi = getattr(sc, "Delta_s_pi", None) or dict(d_s_pi)
    Delta_pi_s = getattr(sc, "Delta_pi_s", None) or dict(d_pi_s)
    Delta_s_s  = getattr(sc, "Delta_s_s",  None) or dict(d_s_s)

    # 4) 组装 RobustEvaluator 参数（按 main.py cross-gamma 工厂的口径）
    base_kwargs = dict(
        J=getattr(sc, "J", None),
        R=getattr(sc, "R", None),
        S=getattr(sc, "S", None),
        pi=getattr(sc, "pi", None),
        D=getattr(sc, "D", None),
        D_setup=getattr(sc, "D_setup", 2.0),

        J0=J0,
        Jd=Jd,
        J_I=getattr(sc, "J_I", None),

        shelf_data=shelf_data,
        agv_data=agv_data,

        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,

        gamma=int(gamma),
        ws_fixed_seq=getattr(sc, "ws_fixed_seq", None),
        ws_setup_rule="flat",
        lock_place=True,
        detach_on_mismatch=False,
        envelope_shared_resources=True,

        enable_chain_repair=True,
        chain_repair_max_iters=8,
        enable_cell_repair=True,
        cell_repair_max_iters=15,

        cell_conflict_mode="hard",

        timeline_mode="full" if debug else "off",
        collect_v_arcs=True if debug else False,
        record_cell_repair_log=False,

        enable_init_cell_lock=True,
        init_lock_max_iters=2,
        init_lock_tol=1e-6,
        init_lock_verbose=False,
    )

    # 5) 用签名过滤（关键：忽略 *args/**kwargs，不要把 **_ignored_kwargs 当必需）
    sig = inspect.signature(RobustEvaluator.__init__)
    filtered = {}
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name in base_kwargs and base_kwargs[name] is not None:
            filtered[name] = base_kwargs[name]

    # 6) 构造 evaluator
    try:
        ev = RobustEvaluator(**filtered)
    except Exception as e:
        raise RuntimeError(
            f"Failed to construct RobustEvaluator. "
            f"Type(sc)={type(sc)} | Error={type(e).__name__}: {e}\n"
            f"Provided keys={sorted(list(filtered.keys()))}"
        )

    return ev, sc



def extract_task_shelf_mapping(runtime_obj: Any, ev: RobustEvaluator) -> Optional[Dict[Any, Any]]:
    """
    Try to locate the mapping that triggers your WARN.
    """
    # 1) evaluator attribute
    if hasattr(ev, "task_shelf_mapping"):
        mp = getattr(ev, "task_shelf_mapping")
        if isinstance(mp, dict):
            return mp

    # 2) return object dict
    if isinstance(runtime_obj, dict):
        mp = runtime_obj.get("task_shelf_mapping", None)
        if isinstance(mp, dict):
            return mp

    # 3) tuple/list elements
    if isinstance(runtime_obj, (tuple, list)):
        for it in runtime_obj:
            if isinstance(it, dict) and "task_shelf_mapping" in it and isinstance(it["task_shelf_mapping"], dict):
                return it["task_shelf_mapping"]

    return None


def summarize_penalties(diag: dict) -> Dict[str, Any]:
    penalties = diag.get("penalties", {}) if isinstance(diag, dict) else {}
    if not isinstance(penalties, dict):
        return {"penalties": penalties, "penalty_keys": None}

    # keep only numeric-ish for quick view
    num_p = {}
    for k, v in penalties.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            num_p[str(k)] = fv
    return {"penalties": penalties, "penalty_keys": sorted(num_p.keys()), "penalties_numeric": num_p}


def unscheduled_list(diag: dict) -> List[int]:
    # try common keys
    for k in ("unscheduled_tasks", "missing_tasks", "unserved_tasks"):
        v = diag.get(k, None) if isinstance(diag, dict) else None
        if isinstance(v, (list, tuple, set)):
            return [int(x) for x in v]
    return []


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default="solution_exports")
    ap.add_argument("--tag", default="alns")

    ap.add_argument("--source-gammas", default="0,5,10,15")
    ap.add_argument("--eval-gammas", default="0,5,10,15")

    ap.add_argument("--export-diag", action="store_true", help="dump diag json per (srcG,evalG)")
    ap.add_argument("--export-matrix", action="store_true", help="write matrix csv")

    args = ap.parse_args()

    prefix = args.prefix
    outdir = args.outdir
    tag = args.tag

    src_gammas = [int(x) for x in args.source_gammas.split(",") if str(x).strip() != ""]
    eval_gammas = [int(x) for x in args.eval_gammas.split(",") if str(x).strip() != ""]

    # ---- load tasks.csv
    tasks_path = os.path.join("scenario", prefix, "tasks.csv")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"tasks.csv not found: {tasks_path}")

    df = pd.read_csv(tasks_path)

    need_cols = {"Task", "Shelf", "Workstation", "Duration"}
    miss_cols = [c for c in need_cols if c not in df.columns]
    if miss_cols:
        raise RuntimeError(f"tasks.csv missing columns: {miss_cols}")

    if "WSOrder" not in df.columns:
        # fallback
        df["WSOrder"] = df.groupby("Workstation").cumcount() + 1

    print("=" * 80)
    print("[TASKS.CSV CHECK]")
    print(f"  rows={len(df)}")
    print(f"  Shelf NaN={int(df['Shelf'].isna().sum())} | Workstation NaN={int(df['Workstation'].isna().sum())} | WSOrder NaN={int(df['WSOrder'].isna().sum())}")
    print(f"  dtypes:\n{df.dtypes}")
    print("=" * 80)

    # ---- build evaluators once per eval_gamma (consistent with main)
    evaluators: Dict[int, RobustEvaluator] = {}
    runtime_objs: Dict[int, Any] = {}

    for g in eval_gammas:
        ev, ret = build_evaluator_from_prefix(prefix, g, debug=bool(args.export_diag))

        evaluators[int(g)] = ev
        runtime_objs[int(g)] = ret

    # ---- print mapping None issue (if we can extract it)
    print("\n" + "=" * 80)
    print("[RUNTIME task_shelf_mapping CHECK] (why you see: 6 None/NaN/illegal)")
    print("=" * 80)
    for g in sorted(evaluators.keys()):
        ev = evaluators[g]
        mp = extract_task_shelf_mapping(runtime_objs[g], ev)
        if not isinstance(mp, dict):
            print(f"  eval_gamma={g}: cannot locate runtime task_shelf_mapping on evaluator/scenario return.")
            continue

        bad = []
        for k, v in mp.items():
            if v is None:
                bad.append((k, v))
            else:
                try:
                    if isinstance(v, float) and math.isnan(v):
                        bad.append((k, v))
                except Exception:
                    pass

        print(f"  eval_gamma={g}: mapping_size={len(mp)} | bad(None/NaN)={len(bad)}")
        if bad:
            print("    bad samples (key,value):", bad[:20])

            # heuristic: if keys are not in real Task ids, they are likely virtual tasks (e.g., J_I)
            real_tasks = set(int(x) for x in df["Task"].tolist())
            bad_keys = [bk for bk, _ in bad]
            outside = [bk for bk in bad_keys if int(bk) not in real_tasks]
            if outside:
                print("    note: many bad keys are NOT real tasks from tasks.csv -> very likely virtual tasks (e.g., J_I).")
    print("=" * 80 + "\n")

    # ---- evaluate all pairs
    matrix: Dict[int, Dict[int, float]] = {sg: {} for sg in src_gammas}

    for sg in src_gammas:
        bundle_path = os.path.join(outdir, f"{prefix}_{tag}_bundle_gamma{sg}.json")
        b = load_bundle(bundle_path)

        print("\n" + "#" * 90)
        print(f"[BUNDLE] source_gamma={sg} | path={bundle_path} | cmax_in_bundle={b.get('cmax')}")
        print("#" * 90)
        structure_check(tasks_df=df, bundle=b, max_show=20)

        routes = b["routes"]
        shelf_seq = b["shelf_seq"]
        place = b["place"]

        for eg in eval_gammas:
            ev = evaluators[int(eg)]
            obj, diag = ev.evaluate(routes, shelf_seq, place)
            try:
                obj_f = float(obj)
            except Exception:
                obj_f = float("inf")

            if not isinstance(diag, dict):
                diag = {}

            uns = unscheduled_list(diag)
            pen_info = summarize_penalties(diag)

            matrix[sg][eg] = obj_f

            print(f"[EVAL] srcG={sg} -> evalG={eg} | obj={obj_f}")
            if uns:
                print(f"       unscheduled_cnt={len(uns)} | first={uns[:20]}")
            pn = pen_info.get("penalties_numeric", {})
            if pn:
                # show top few numeric penalties
                top = sorted(pn.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:10]
                print(f"       penalties_numeric_top10={top}")
            else:
                # still print raw penalties if exists
                if diag.get("penalties", None) is not None:
                    print(f"       penalties(raw)={diag.get('penalties')}")
            # show feasibility flag if exists
            if "feasible" in diag:
                print(f"       feasible={diag.get('feasible')}")

            if args.export_diag:
                out_path = os.path.join(outdir, f"{prefix}_verify_srcG{sg}_evalG{eg}_diag.json")
                payload = {
                    "prefix": prefix,
                    "source_gamma": sg,
                    "eval_gamma": eg,
                    "obj": obj_f,
                    "bundle_cmax": b.get("cmax"),
                    "routes": routes,
                    "shelf_seq": shelf_seq,
                    "place": place,
                    "diag": diag,
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                print(f"       [EXPORT] wrote {out_path}")

    # ---- export / print matrix
    print("\n" + "=" * 80)
    print("[MATRIX] obj(srcG -> evalG)")
    print("=" * 80)
    header = ["source_gamma"] + [f"evalG{g}" for g in eval_gammas]
    print(",".join(header))
    for sg in src_gammas:
        row = [str(sg)] + [str(matrix[sg].get(eg, "")) for eg in eval_gammas]
        print(",".join(row))
    print("=" * 80)

    if args.export_matrix:
        mat_path = os.path.join(outdir, f"{prefix}_verify_matrix.csv")
        rows = []
        for sg in src_gammas:
            r = {"source_gamma": sg}
            for eg in eval_gammas:
                r[f"eval_gamma_{eg}"] = matrix[sg].get(eg, float("nan"))
            rows.append(r)
        pd.DataFrame(rows).to_csv(mat_path, index=False)
        print(f"[EXPORT] matrix csv -> {mat_path}")


if __name__ == "__main__":
    main()
