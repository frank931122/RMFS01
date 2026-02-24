#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd

from map_generator import load_map_csv, create_map_from_components
from utils import distance
from evaluator import RobustEvaluator
from export_eval_diag import export_evaluator_diagnostics


def idx2rc(idx: int, W: int) -> Tuple[int, int]:
    return divmod(int(idx) - 1, int(W))


def load_bundle(prefix: str, tag: str, source_gamma: int, outdir: str) -> dict:
    path = os.path.join(outdir, f"{prefix}_{tag}_bundle_gamma{int(source_gamma)}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bundle not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        b = json.load(f)

    routes_raw = b.get("routes", {}) or {}
    shelf_seq_raw = b.get("shelf_seq", {}) or {}
    place_raw = b.get("place", {}) or {}

    routes = {int(r): [int(x) for x in (seq or [])] for r, seq in routes_raw.items()}
    shelf_seq = {int(c): [int(x) for x in (seq or [])] for c, seq in shelf_seq_raw.items()}
    place = {int(j): int(s) for j, s in place_raw.items()}

    b2 = dict(b)
    b2["routes"] = routes
    b2["shelf_seq"] = shelf_seq
    b2["place"] = place
    b2["_path"] = path
    return b2


def build_ctx(prefix: str) -> dict:
    scen_dir = os.path.join("scenario", prefix)
    tasks_csv = os.path.join(scen_dir, "tasks.csv")
    if not os.path.exists(tasks_csv):
        raise FileNotFoundError(f"tasks.csv not found: {tasks_csv}")

    tasks_df = pd.read_csv(tasks_csv)
    if "WSOrder" not in tasks_df.columns:
        tasks_df["WSOrder"] = tasks_df.groupby("Workstation").cumcount() + 1

    shelf_data, agv_data, ws_indices, sp_indices, W, H = load_map_csv(prefix)

    ws_fixed_seq = (
        tasks_df.sort_values(["Workstation", "WSOrder", "Task"])
        .groupby("Workstation")["Task"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )

    J = set(int(x) for x in tasks_df["Task"].tolist())
    pi = {int(r["Task"]): int(r["Workstation"]) for _, r in tasks_df.iterrows()}
    D = {int(r["Task"]): float(r["Duration"]) for _, r in tasks_df.iterrows()}
    task_shelf = {int(r["Task"]): int(r["Shelf"]) for _, r in tasks_df.iterrows()}

    map_obj = create_map_from_components(
        width=W, height=H,
        sp_indices=sp_indices,
        ws_indices=ws_indices,
        shelf_data=shelf_data,
        agv_data=agv_data
    )
    R = map_obj.extract_AGVs()

    # storage points S / workstations K
    S = {int(sp): idx2rc(int(sp), W) for sp in sp_indices}
    K = {i + 1: idx2rc(ws_indices[i], W) for i in range(len(ws_indices))}

    # dists
    d_s_pi, d_pi_s, d_s_s = {}, {}, {}
    for s in S:
        for j in J:
            d_s_pi[(int(s), int(j))] = distance(int(s), int(pi[int(j)]), S, K)
    for j in J:
        for s in S:
            d_pi_s[(int(j), int(s))] = distance(int(pi[int(j)]), int(s), K, S)
    for s in S:
        for sp in S:
            d_s_s[(int(s), int(sp))] = distance(int(s), int(sp), S, S)

    Delta_s_pi = dict(d_s_pi)
    Delta_pi_s = dict(d_pi_s)
    Delta_s_s = dict(d_s_s)

    # dummies
    shelf_ids = sorted(int(x) for x in shelf_data.keys())
    agv_ids = sorted(int(x) for x in agv_data.keys())
    J0 = {aid: 1000 + int(aid) for aid in agv_ids}
    Jd = {aid: 2000 + int(aid) for aid in agv_ids}
    J_I = {sid: 3000 + int(sid) for sid in shelf_ids}

    return dict(
        tasks_df=tasks_df,
        ws_fixed_seq=ws_fixed_seq,
        J=J, R=R, S=S, K=K,
        pi=pi, D=D,
        task_shelf=task_shelf,
        shelf_data=shelf_data,
        agv_data=agv_data,
        J0=J0, Jd=Jd, J_I=J_I,
        d_s_pi=d_s_pi, d_pi_s=d_pi_s, d_s_s=d_s_s,
        Delta_s_pi=Delta_s_pi, Delta_pi_s=Delta_pi_s, Delta_s_s=Delta_s_s,
        W=W, H=H
    )


def make_evaluator(ctx: dict, eval_gamma: int, *, overrides: Optional[dict] = None, timeline_full: bool = True) -> RobustEvaluator:
    # baseline = 你 main.py cross-gamma evaluator_factory_for_cross 的风格
    kw = dict(
        J=ctx["J"], R=ctx["R"], S=ctx["S"],
        pi=ctx["pi"],
        D=ctx["D"],
        J0=ctx["J0"], Jd=ctx["Jd"], J_I=ctx["J_I"],
        shelf_data=ctx["shelf_data"], agv_data=ctx["agv_data"],
        d_s_pi=ctx["d_s_pi"], d_pi_s=ctx["d_pi_s"], d_s_s=ctx["d_s_s"],
        Delta_s_pi=ctx["Delta_s_pi"], Delta_pi_s=ctx["Delta_pi_s"], Delta_s_s=ctx["Delta_s_s"],
        gamma=int(eval_gamma),
        ws_fixed_seq=ctx["ws_fixed_seq"],
        ws_setup_rule="flat",
        lock_place=True,
        detach_on_mismatch=False,
        envelope_shared_resources=True,

        enable_chain_repair=True,
        chain_repair_max_iters=8,
        enable_cell_repair=True,
        cell_repair_max_iters=15,

        cell_conflict_mode="hard",

        timeline_mode="full" if timeline_full else "off",
        collect_v_arcs=True if timeline_full else False,
        record_cell_repair_log=True if timeline_full else False,

        enable_init_cell_lock=True,
        init_lock_max_iters=2,
        init_lock_tol=1e-6,
        init_lock_verbose=False,
    )

    if overrides:
        kw.update(overrides)

    return RobustEvaluator(**kw)


def find_big_tasks(diag: dict, big_thr: float) -> List[Tuple[int, float, float]]:
    p = diag.get("p", {}) or {}
    q = diag.get("q", {}) or {}
    p2 = {int(k): float(v) for k, v in p.items()}
    q2 = {int(k): float(v) for k, v in q.items()}

    out = []
    for j, qj in q2.items():
        pj = p2.get(j, float("nan"))
        if (not math.isfinite(qj)) or (qj >= big_thr) or (math.isfinite(pj) and pj >= big_thr):
            out.append((int(j), float(pj), float(qj)))

    out.sort(key=lambda x: (x[1], x[0]))  # 按 p 从小到大：找到“第一个跳到 BIG_M 的任务”
    return out


def build_pred_on_chain(shelf_seq: Dict[int, List[int]]) -> Dict[int, Optional[int]]:
    pred = {}
    for c, seq in shelf_seq.items():
        seq2 = [int(x) for x in (seq or [])]
        for i, j in enumerate(seq2):
            pred[int(j)] = int(seq2[i - 1]) if i > 0 else None
    return pred


def print_first_trigger(
    *,
    ctx: dict,
    shelf_seq: Dict[int, List[int]],
    place_in: Dict[int, int],
    diag: dict,
    big_thr: float
) -> None:
    big = find_big_tasks(diag, big_thr)
    if not big:
        print(f"[DBG] No BIG tasks under thr={big_thr}.")
        return

    pred = build_pred_on_chain(shelf_seq)

    ws_of = {int(r["Task"]): int(r["Workstation"]) for _, r in ctx["tasks_df"].iterrows()}
    sh_of = {int(r["Task"]): int(r["Shelf"]) for _, r in ctx["tasks_df"].iterrows()}
    wo_of = {int(r["Task"]): int(r["WSOrder"]) for _, r in ctx["tasks_df"].iterrows()} if "WSOrder" in ctx["tasks_df"].columns else {}

    j0, p0, q0 = big[0]
    print("\n" + "=" * 90)
    print(f"[FIRST-BIG] thr={big_thr} | first_task={j0} | p={p0:.2f} q={q0:.2f}")
    print(f"  meta: WS={ws_of.get(j0)} | Shelf={sh_of.get(j0)} | WSOrder={wo_of.get(j0)} | place={place_in.get(j0)}")
    pj = pred.get(j0, None)
    if pj is not None:
        print(f"  chain predecessor on same shelf_seq: prev_task={pj} | place(prev)={place_in.get(int(pj))}")
    else:
        print("  chain predecessor: None (chain head)")
    print("=" * 90)

    # timeline 里找记录（如果有）
    tl = diag.get("timeline", []) or []
    if not tl:
        print("[FIRST-BIG] timeline is empty -> 请确保 evaluator.timeline_mode='full'")
        return

    rec_by = {}
    for rec in tl:
        try:
            tj = int(rec.get("Task"))
        except Exception:
            continue
        rec_by[tj] = rec

    def show_rec(title: str, j: int):
        rec = rec_by.get(int(j), None)
        if not isinstance(rec, dict):
            print(f"[TIMELINE] {title}: Task {j} not found in timeline.")
            return
        # 打印关键字段（字段不存在就显示 None）
        keys = [
            "AGV", "Task", "Chain", "WS",
            "home_before", "end_s",
            "pick_start", "arrival_ws", "ws_start", "ws_end",
            "arrive_cell_nom", "arrive_cell_act",
            "dt1", "dt2_nom", "dt2_eff", "dt3_nom", "dt3_eff",
            "place_lb", "place_wait_due_to_lb"
        ]
        print(f"\n[TIMELINE] {title} (Task {j})")
        for k in keys:
            if k in rec:
                print(f"  {k}: {rec.get(k)}")
        # 额外：把 record 自己有哪些 key 列出来（便于你继续深挖）
        print("  (record keys sample):", list(rec.keys())[:40])

    show_rec("FIRST-BIG", j0)
    if pj is not None:
        show_rec("PREV-ON-CHAIN", int(pj))


def run_one(
    *,
    prefix: str,
    tag: str,
    source_gamma: int,
    eval_gamma: int,
    outdir: str,
    export: bool,
    export_tag: str,
    big_thr: float,
    verbose_eval: bool
) -> Tuple[float, dict]:
    ctx = build_ctx(prefix)
    bundle = load_bundle(prefix, tag, source_gamma, outdir)
    routes = bundle["routes"]
    shelf_seq = bundle["shelf_seq"]
    place = bundle["place"]

    ev = make_evaluator(ctx, eval_gamma, timeline_full=True)
    obj, diag = ev.evaluate(routes, shelf_seq, place, verbose=bool(verbose_eval))
    obj_f = float(obj) if math.isfinite(float(obj)) else float("inf")

    penalties = diag.get("penalties", {}) if isinstance(diag, dict) else {}
    print("\n" + "-" * 90)
    print(f"[EVAL] srcG={source_gamma} -> evalG={eval_gamma} | obj={obj_f}")
    print(f"  penalties={penalties}")
    print("-" * 90)

    print_first_trigger(ctx=ctx, shelf_seq=shelf_seq, place_in=place, diag=diag, big_thr=big_thr)

    if export and isinstance(diag, dict):
        try:
            export_evaluator_diagnostics(
                prefix=prefix,
                export_tag=export_tag,
                source_gamma=int(source_gamma),
                eval_gamma=int(eval_gamma),
                routes=routes,
                shelf_seq=shelf_seq,
                place=place,
                milp_cmax=float(bundle.get("cmax", float("nan"))),
                eval_cmax=float(obj_f),
                diag=diag,
                outdir=outdir,
            )
            print(f"[EXPORT] diagnostics exported -> tag={export_tag} (outdir={outdir})")
        except Exception as e:
            print(f"[EXPORT] failed: {type(e).__name__}: {e}")

    return obj_f, diag


def ablation(
    *,
    prefix: str,
    tag: str,
    source_gamma: int,
    eval_gamma: int,
    outdir: str,
    big_thr: float
) -> None:
    ctx = build_ctx(prefix)
    bundle = load_bundle(prefix, tag, source_gamma, outdir)
    routes = bundle["routes"]
    shelf_seq = bundle["shelf_seq"]
    place = bundle["place"]

    cases = [
        ("BASE(cross-gamma style)", {}),
        ("init_lock_OFF", {"enable_init_cell_lock": False, "init_lock_max_iters": 0}),
        ("init_lock_ITERS10", {"enable_init_cell_lock": True, "init_lock_max_iters": 10}),
        ("cell_repair_OFF", {"enable_cell_repair": False, "cell_repair_max_iters": 0}),
        ("chain_repair_OFF", {"enable_chain_repair": False, "chain_repair_max_iters": 0}),
        ("envelope_shared_OFF", {"envelope_shared_resources": False}),
        ("lock_place_FALSE", {"lock_place": False}),
        ("detach_on_mismatch_TRUE", {"detach_on_mismatch": True}),
        ("cell_conflict_penalty", {"cell_conflict_mode": "penalty"}),
    ]

    rows = []
    for name, ov in cases:
        ev = make_evaluator(ctx, eval_gamma, overrides=ov, timeline_full=False)  # ablation 不需要 full timeline
        obj, diag = ev.evaluate(routes, shelf_seq, place, verbose=False)
        obj_f = float(obj) if math.isfinite(float(obj)) else float("inf")

        big_cnt = 0
        if isinstance(diag, dict) and ("q" in diag):
            try:
                q2 = {int(k): float(v) for k, v in (diag.get("q", {}) or {}).items()}
                big_cnt = sum(1 for _, qv in q2.items() if (not math.isfinite(qv)) or (qv >= big_thr))
            except Exception:
                big_cnt = -1

        rows.append((name, obj_f, big_cnt))

    print("\n" + "=" * 90)
    print(f"[ABLATION] srcG={source_gamma} -> evalG={eval_gamma} | big_thr={big_thr}")
    print(" case | makespan | big_task_cnt(q>=thr)")
    print("-" * 90)
    for name, obj_f, big_cnt in rows:
        print(f" {name:<24} | {obj_f:>8.2f} | {big_cnt:>6}")
    print("=" * 90 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tag", default="alns")
    ap.add_argument("--source-gamma", type=int, required=True)
    ap.add_argument("--eval-gamma", type=int, required=True)
    ap.add_argument("--outdir", default="solution_exports")
    ap.add_argument("--big-thr", type=float, default=9000.0)

    ap.add_argument("--export", action="store_true")
    ap.add_argument("--export-tag", default="")
    ap.add_argument("--verbose-eval", action="store_true")
    ap.add_argument("--ablation", action="store_true")

    args = ap.parse_args()

    export_tag = args.export_tag.strip()
    if not export_tag:
        export_tag = f"debugBig_srcG{args.source_gamma}_evalG{args.eval_gamma}"

    run_one(
        prefix=args.prefix,
        tag=args.tag,
        source_gamma=int(args.source_gamma),
        eval_gamma=int(args.eval_gamma),
        outdir=args.outdir,
        export=bool(args.export),
        export_tag=export_tag,
        big_thr=float(args.big_thr),
        verbose_eval=bool(args.verbose_eval),
    )

    if args.ablation:
        ablation(
            prefix=args.prefix,
            tag=args.tag,
            source_gamma=int(args.source_gamma),
            eval_gamma=int(args.eval_gamma),
            outdir=args.outdir,
            big_thr=float(args.big_thr),
        )


if __name__ == "__main__":
    main()
