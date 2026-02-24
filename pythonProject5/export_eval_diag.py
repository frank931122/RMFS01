# export_eval_diag.py
import os
import json
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


def _to_builtin(obj: Any) -> Any:
    """Convert common non-JSON types (numpy/pandas) into Python builtins."""
    # numpy scalar
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    if isinstance(obj, dict):
        # JSON requires string keys
        return {str(k): _to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_builtin(x) for x in obj]

    if isinstance(obj, (int, float, str, bool)) or obj is None:
        # normalize NaN/inf
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    # fallback
    return str(obj)


def _write_df(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_robust_segments_from_timeline(
    timeline_df: pd.DataFrame,
    eps: float = 1e-9
) -> pd.DataFrame:
    """
    从 evaluator timeline 里提取“发生鲁棒膨胀”的 travel 段：
      - dt2: home_before -> WS
      - dt3: WS -> end_s

    规则：dt*_eff - dt*_nom > eps 即认为该段被“膨胀/变化”。
    """
    if timeline_df is None or timeline_df.empty:
        return pd.DataFrame(columns=[
            "Task", "AGV", "Chain", "WS",
            "seg_type", "from_node", "to_node",
            "nom", "eff", "delta",
            "home_before", "end_s",
            "ws_start", "ws_end",
            "arrive_cell_nom", "arrive_cell_act",
            "arrive_cell_gap",
            "cell_wait"
        ])

    df = timeline_df.copy()

    # Try numeric conversion where possible
    for col in [
        "Task", "AGV", "Chain", "WS", "home_before", "end_s",
        "dt2_nom", "dt2_eff", "dt3_nom", "dt3_eff",
        "ws_start", "ws_end", "arrive_cell_nom", "arrive_cell_act"
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows: List[Dict[str, Any]] = []

    def safe_int(x) -> Optional[int]:
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            return int(x)
        except Exception:
            return None

    def safe_float(x) -> Optional[float]:
        try:
            if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
                return None
            return float(x)
        except Exception:
            return None

    for _, r in df.iterrows():
        j = safe_int(r.get("Task"))
        agv = safe_int(r.get("AGV"))
        chain = safe_int(r.get("Chain"))
        ws = safe_int(r.get("WS"))
        hb = safe_int(r.get("home_before"))
        end_s = safe_int(r.get("end_s"))

        ws_start = safe_float(r.get("ws_start"))
        ws_end = safe_float(r.get("ws_end"))
        ac_nom = safe_float(r.get("arrive_cell_nom"))
        ac_act = safe_float(r.get("arrive_cell_act"))

        arrive_cell_gap = None
        if ac_nom is not None and ac_act is not None:
            arrive_cell_gap = ac_act - ac_nom

        cell_wait = None
        dt3_eff = safe_float(r.get("dt3_eff"))
        if ac_act is not None and ws_end is not None and dt3_eff is not None:
            # 如果 evaluator 里 dt3_eff 只表示“行走”，则这里会体现额外等待
            cell_wait = ac_act - (ws_end + dt3_eff)

        # ---- dt2 segment: home_before -> WS ----
        dt2_nom = safe_float(r.get("dt2_nom"))
        dt2_eff = safe_float(r.get("dt2_eff"))
        if dt2_nom is not None and dt2_eff is not None:
            dlt = dt2_eff - dt2_nom
            if dlt > eps:
                rows.append({
                    "Task": j, "AGV": agv, "Chain": chain, "WS": ws,
                    "seg_type": "dt2(home->WS)",
                    "from_node": hb,
                    "to_node": f"WS{ws}" if ws is not None else None,
                    "nom": dt2_nom, "eff": dt2_eff, "delta": dlt,
                    "home_before": hb, "end_s": end_s,
                    "ws_start": ws_start, "ws_end": ws_end,
                    "arrive_cell_nom": ac_nom, "arrive_cell_act": ac_act,
                    "arrive_cell_gap": arrive_cell_gap,
                    "cell_wait": cell_wait,
                })

        # ---- dt3 segment: WS -> end_s ----
        dt3_nom = safe_float(r.get("dt3_nom"))
        dt3_eff = safe_float(r.get("dt3_eff"))
        if dt3_nom is not None and dt3_eff is not None:
            dlt = dt3_eff - dt3_nom
            if dlt > eps:
                rows.append({
                    "Task": j, "AGV": agv, "Chain": chain, "WS": ws,
                    "seg_type": "dt3(WS->end)",
                    "from_node": f"WS{ws}" if ws is not None else None,
                    "to_node": end_s,
                    "nom": dt3_nom, "eff": dt3_eff, "delta": dlt,
                    "home_before": hb, "end_s": end_s,
                    "ws_start": ws_start, "ws_end": ws_end,
                    "arrive_cell_nom": ac_nom, "arrive_cell_act": ac_act,
                    "arrive_cell_gap": arrive_cell_gap,
                    "cell_wait": cell_wait,
                })

    out = pd.DataFrame(rows)
    if not out.empty:
        # 让你一眼看出“谁贡献最大”
        out = out.sort_values(["delta", "Task"], ascending=[False, True]).reset_index(drop=True)
    return out


def export_evaluator_diagnostics(
    *,
    prefix: str,
    export_tag: str,
    source_gamma: int,
    eval_gamma: int,
    routes: Dict[int, List[int]],
    shelf_seq: Dict[int, List[int]],
    place: Dict[int, int],
    milp_cmax: float,
    eval_cmax: float,
    diag: Dict[str, Any],
    outdir: str = "solution_exports",
) -> Dict[str, str]:
    """
    把 evaluator 在“给定结构（通常是 MILP bundle）”上的所有诊断信息导出：
      - *_diag.json              （全量 diag + routes/shelf_seq/place + meta）
      - *_timeline.csv           （timeline_mode=full 才有）
      - *_pq.csv                 （p/q）
      - *_robust_segments.csv    （哪些 dt2/dt3 被膨胀了）
      - *_v_arcs.csv             （collect_v_arcs=True 才有）
      - *_end_shelf_final.csv    （若 evaluator 做了 end_shelf_final）
    """
    os.makedirs(outdir, exist_ok=True)

    base = f"{prefix}_{export_tag}_srcG{int(source_gamma)}_evalG{int(eval_gamma)}"
    paths: Dict[str, str] = {}

    # ---- JSON: full diag ----
    json_path = os.path.join(outdir, f"{base}_diag.json")
    payload = {
        "meta": {
            "prefix": prefix,
            "export_tag": export_tag,
            "source_gamma": int(source_gamma),
            "eval_gamma": int(eval_gamma),
            "milp_cmax": float(milp_cmax),
            "eval_cmax": float(eval_cmax),
            "export_time": datetime.now().isoformat(timespec="seconds"),
        },
        "routes": {str(int(r)): [int(t) for t in (seq or [])] for r, seq in routes.items()},
        "shelf_seq": {str(int(c)): [int(t) for t in (seq or [])] for c, seq in shelf_seq.items()},
        "place": {str(int(j)): int(s) for j, s in place.items()},
        "diag": _to_builtin(diag),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    paths["diag_json"] = json_path

    # ---- p/q ----
    p = diag.get("p", {}) or {}
    q = diag.get("q", {}) or {}
    try:
        p2 = {int(k): float(v) for k, v in p.items()}
        q2 = {int(k): float(v) for k, v in q.items()}
    except Exception:
        p2, q2 = {}, {}
    pq_rows = []
    for j in sorted(set(p2.keys()) | set(q2.keys())):
        pq_rows.append({"Task": int(j), "p": p2.get(j), "q": q2.get(j), "end_s": place.get(int(j))})
    pq_df = pd.DataFrame(pq_rows)
    pq_path = os.path.join(outdir, f"{base}_pq.csv")
    _write_df(pq_df, pq_path)
    paths["pq_csv"] = pq_path

    # ---- timeline ----
    timeline = diag.get("timeline", []) or []
    if isinstance(timeline, list) and len(timeline) > 0 and isinstance(timeline[0], dict):
        tl_df = pd.DataFrame(timeline)
        tl_path = os.path.join(outdir, f"{base}_timeline.csv")
        _write_df(tl_df, tl_path)
        paths["timeline_csv"] = tl_path

        # ---- robust segments ----
        seg_df = build_robust_segments_from_timeline(tl_df)
        seg_path = os.path.join(outdir, f"{base}_robust_segments.csv")
        _write_df(seg_df, seg_path)
        paths["robust_segments_csv"] = seg_path
    else:
        # timeline_mode 不是 full 时会没有 timeline
        paths["timeline_csv"] = ""
        paths["robust_segments_csv"] = ""

    # ---- v_arcs ----
    V = diag.get("V_arcs", []) or []
    if isinstance(V, list) and len(V) > 0:
        v_df = pd.DataFrame(V, columns=["i", "j", "s", "s_prime"])
        v_path = os.path.join(outdir, f"{base}_v_arcs.csv")
        _write_df(v_df, v_path)
        paths["v_arcs_csv"] = v_path
    else:
        paths["v_arcs_csv"] = ""

    # ---- end_shelf_final ----
    end_final = diag.get("end_shelf_final", {}) or {}
    if isinstance(end_final, dict) and len(end_final) > 0:
        ef_rows = [{"Task": int(k), "end_shelf_final": int(v), "place_input": place.get(int(k))} for k, v in end_final.items()]
        ef_df = pd.DataFrame(ef_rows).sort_values("Task")
        ef_path = os.path.join(outdir, f"{base}_end_shelf_final.csv")
        _write_df(ef_df, ef_path)
        paths["end_shelf_final_csv"] = ef_path
    else:
        paths["end_shelf_final_csv"] = ""

    print("\n[EXPORT-EVAL] evaluator diagnostics exported:")
    for k, pth in paths.items():
        if pth:
            print(f"  - {k}: {pth}")
    print("")
    return paths
