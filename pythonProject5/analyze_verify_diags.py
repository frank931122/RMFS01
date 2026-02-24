#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import math
import glob
import argparse
from typing import Dict, Any, Tuple, Optional

import pandas as pd


def _safe_float(x, default=float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def _int_key_float_map(m: Any) -> Dict[int, float]:
    if not isinstance(m, dict):
        return {}
    out = {}
    for k, v in m.items():
        kk = _safe_int(k, None)
        if kk is None:
            continue
        out[kk] = _safe_float(v)
    return out


def _int_key_int_map(m: Any) -> Dict[int, int]:
    if not isinstance(m, dict):
        return {}
    out = {}
    for k, v in m.items():
        kk = _safe_int(k, None)
        vv = _safe_int(v, None)
        if kk is None or vv is None:
            continue
        out[kk] = vv
    return out


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_src_eval_from_filename(fname: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_srcG(\d+)_evalG(\d+)_diag\.json$", fname)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _get_tasks_meta(prefix: str) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    tasks_csv = os.path.join("scenario", prefix, "tasks.csv")
    df = pd.read_csv(tasks_csv)
    ws_of = {int(r["Task"]): int(r["Workstation"]) for _, r in df.iterrows()}
    shelf_of = {int(r["Task"]): int(r["Shelf"]) for _, r in df.iterrows()}
    wsorder_of = {int(r["Task"]): int(r["WSOrder"]) for _, r in df.iterrows()} if "WSOrder" in df.columns else {}
    return ws_of, shelf_of, wsorder_of


def _unwrap_meta_and_diag(diag_path: str, obj: Any) -> Tuple[dict, dict]:
    """
    支持三种导出风格：
      A) 文件本身就是 diag（含 p/q/penalties/timeline）
      B) 文件是 wrapper，真正 diag 在 obj["diag"] 或 obj["details"] 等字段
      C) 文件是 wrapper，只保存了 "diag_json": "xxx.json" 指向另一个文件
    返回：(meta, diag)
    """
    meta = obj if isinstance(obj, dict) else {}
    diag = {}

    if isinstance(obj, dict):
        # B) wrapper 内嵌 diag
        for k in ("diag", "details", "evaluator_diag", "eval_diag"):
            if isinstance(obj.get(k), dict):
                diag = obj[k]
                return meta, diag

        # C) 指向另一个 diag_json 文件
        diag_json = obj.get("diag_json", None)
        if isinstance(diag_json, str) and diag_json.strip():
            p = diag_json.strip()
            # 允许相对路径：相对当前 diag 文件目录
            if not os.path.isabs(p):
                p = os.path.join(os.path.dirname(diag_path), p)
            if os.path.exists(p):
                o2 = _load_json(p)
                if isinstance(o2, dict):
                    meta2 = dict(meta)
                    meta2["_diag_json_resolved"] = p
                    return meta2, o2

    # A) 直接当作 diag
    if isinstance(obj, dict):
        diag = obj
    return meta, diag


def _bundle_place(prefix: str, tag: str, src_g: int, outdir: str) -> Dict[int, int]:
    path = os.path.join(outdir, f"{prefix}_{tag}_bundle_gamma{int(src_g)}.json")
    if not os.path.exists(path):
        return {}
    b = _load_json(path)
    place = b.get("place", {}) or {}
    return _int_key_int_map(place)


def _summarize_one(
    *,
    diag_path: str,
    prefix: str,
    tag: str,
    outdir: str,
    big_thr: float,
    ws_of: Dict[int, int],
    shelf_of: Dict[int, int],
    wsorder_of: Dict[int, int],
) -> dict:
    base = os.path.basename(diag_path)
    parsed = _parse_src_eval_from_filename(base)
    if parsed is None:
        return {}

    src_g, eval_g = parsed

    root = _load_json(diag_path)
    meta, diag = _unwrap_meta_and_diag(diag_path, root)

    # 这三个优先从 diag 取；取不到再从 meta 取
    feasible = diag.get("feasible", None)
    if feasible is None:
        feasible = meta.get("feasible", None)

    penalties = diag.get("penalties", None)
    if penalties is None:
        penalties = meta.get("penalties", None)

    # p/q
    p = _int_key_float_map(diag.get("p", {}) or {})
    q = _int_key_float_map(diag.get("q", {}) or {})

    # makespan：优先 C_task_max，再看 meta 里有没有 obj/makespan/cmax，最后 max(q)
    c_task_max = diag.get("C_task_max", None)
    if c_task_max is None:
        for k in ("makespan", "obj", "cmax", "eval_cmax"):
            if meta.get(k, None) is not None:
                c_task_max = meta.get(k)
                break
    if c_task_max is None:
        c_task_max = max(q.values()) if q else float("nan")
    c_task_max = _safe_float(c_task_max)

    # end_shelf_final：用于看 place 是否被 evaluator 改写
    end_final = _int_key_int_map(diag.get("end_shelf_final", {}) or {})
    place_in = _bundle_place(prefix, tag, src_g, outdir)
    place_changed_cnt = None
    if place_in and end_final:
        place_changed_cnt = sum(
            1 for j, s in place_in.items()
            if j in end_final and int(end_final[j]) != int(s)
        )

    print("\n" + "=" * 92)
    print(f"[DIAG] {base}")
    print(f"  srcG={src_g} evalG={eval_g} | makespan≈{c_task_max}")
    print(f"  feasible={feasible} | place_changed_cnt={place_changed_cnt}")
    print(f"  penalties={penalties}")

    # 如果还是没有 q，说明你压根没把原始 diag 保存下来
    if not q:
        keys = list(meta.keys())[:60] if isinstance(meta, dict) else []
        print("  !!! WARNING: cannot find diag['q'] in this file.")
        print("  meta keys sample:", keys)
        if isinstance(meta, dict) and meta.get("_diag_json_resolved"):
            print("  diag_json resolved to:", meta.get("_diag_json_resolved"))
        return {"src_g": src_g, "eval_g": eval_g, "cmax": c_task_max, "missing_q": True}

    # 找 BIG-M-like
    big_tasks = [(j, qj, p.get(j, float("nan"))) for j, qj in q.items()
                 if (not math.isfinite(qj)) or (qj >= big_thr)]
    big_tasks.sort(key=lambda x: (-x[1], x[0]))

    if big_tasks:
        print(f"\n  !!! BIG tasks (q >= {big_thr} or non-finite): count={len(big_tasks)}")
        print("  task |  WS | shelf | WSOrder |       p |       q")
        for (j, qj, pj) in big_tasks[:40]:
            ws = ws_of.get(j, None)
            sh = shelf_of.get(j, None)
            wo = wsorder_of.get(j, None)
            print(f"  {j:>4} | {str(ws):>3} | {str(sh):>5} | {str(wo):>7} | {pj:>7.2f} | {qj:>7.2f}")
    else:
        # 打印 top-q，帮助你看到 q 的量级是否正常
        top = sorted(q.items(), key=lambda kv: kv[1], reverse=True)[:15]
        print("\n  top-q tasks:")
        print("  task |  WS | shelf | WSOrder |       q")
        for j, qj in top:
            ws = ws_of.get(j, None)
            sh = shelf_of.get(j, None)
            wo = wsorder_of.get(j, None)
            print(f"  {j:>4} | {str(ws):>3} | {str(sh):>5} | {str(wo):>7} | {qj:>7.2f}")

    return {"src_g": src_g, "eval_g": eval_g, "cmax": c_task_max, "missing_q": False}


def _compare_q(prefix: str, outdir: str, src_g: int, eval_a: int, eval_b: int,
               ws_of: Dict[int, int], shelf_of: Dict[int, int], wsorder_of: Dict[int, int], topk: int = 30):
    pa = os.path.join(outdir, f"{prefix}_verify_srcG{src_g}_evalG{eval_a}_diag.json")
    pb = os.path.join(outdir, f"{prefix}_verify_srcG{src_g}_evalG{eval_b}_diag.json")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return

    ra = _load_json(pa); rb = _load_json(pb)
    ma, da = _unwrap_meta_and_diag(pa, ra)
    mb, db = _unwrap_meta_and_diag(pb, rb)

    qa = _int_key_float_map(da.get("q", {}) or {})
    qb = _int_key_float_map(db.get("q", {}) or {})
    if not qa or not qb:
        return

    rows = []
    for j in sorted(set(qa.keys()) | set(qb.keys())):
        a = qa.get(j, float("nan"))
        b = qb.get(j, float("nan"))
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        rows.append((j, a, b, a - b))
    rows.sort(key=lambda x: abs(x[3]), reverse=True)

    print("\n" + "-" * 92)
    print(f"[DIFF-Q] srcG={src_g} compare evalG{eval_a} vs evalG{eval_b}")
    print(" task |  WS | shelf | WSOrder |     qA |     qB |   (qA-qB)")
    for (j, a, b, d) in rows[:topk]:
        ws = ws_of.get(j, None)
        sh = shelf_of.get(j, None)
        wo = wsorder_of.get(j, None)
        print(f" {j:>4} | {str(ws):>3} | {str(sh):>5} | {str(wo):>7} | {a:>6.2f} | {b:>6.2f} | {d:>+9.2f}")
    print("-" * 92 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--outdir", default="solution_exports")
    ap.add_argument("--tag", default="alns")
    ap.add_argument("--big-thr", type=float, default=9000.0)
    ap.add_argument("--compare", default="0,5", help="compare two eval gammas, e.g. 0,5")
    args = ap.parse_args()

    prefix = args.prefix
    outdir = args.outdir

    ws_of, shelf_of, wsorder_of = _get_tasks_meta(prefix)

    pattern = os.path.join(outdir, f"{prefix}_verify_srcG*_evalG*_diag.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No diag files match: {pattern}")

    # 逐文件总结
    src_set = set()
    for path in files:
        r = _summarize_one(
            diag_path=path,
            prefix=prefix,
            tag=args.tag,
            outdir=outdir,
            big_thr=float(args.big_thr),
            ws_of=ws_of,
            shelf_of=shelf_of,
            wsorder_of=wsorder_of,
        )
        if r and "src_g" in r:
            src_set.add(int(r["src_g"]))

    # 对比 q：默认 eval0 vs eval5
    try:
        a, b = args.compare.split(",", 1)
        eval_a = int(a.strip()); eval_b = int(b.strip())
    except Exception:
        eval_a, eval_b = 0, 5

    for src_g in sorted(src_set):
        _compare_q(prefix, outdir, src_g, eval_a, eval_b, ws_of, shelf_of, wsorder_of, topk=30)


if __name__ == "__main__":
    main()
