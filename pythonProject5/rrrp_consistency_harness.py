
"""
rrr_integration.py

Utilities to parse MILP run logs (with Chinese section headers as in the user's output)
and export the selected/needed data into an Excel workbook for downstream use in `rrr`.

Usage (CLI):
    python rrr_integration.py --log /path/to/milp_log.txt --out /path/to/rrr.xlsx

Usage (API):
    from rrr_integration import parse_milp_log, export_to_excel
    with open("milp_log.txt","r",encoding="utf-8") as f:
        text = f.read()
    result = parse_milp_log(text)
    export_to_excel(result, "rrr.xlsx")
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import argparse
from collections import defaultdict

# ---------- Data containers ----------

@dataclass
class TaskSchedule:
    task: int
    p: float
    q: float
    agv: int
    end_shelf: int

@dataclass
class GammaResult:
    gamma: int
    C_max_AGV: Optional[float] = None
    w: List[Tuple[int,int]] = field(default_factory=list)  # (j, r)
    x: List[Tuple[int,int]] = field(default_factory=list)  # (j, s)
    z: List[Tuple[int,int,int]] = field(default_factory=list)  # (j, j2, r)
    v: List[Tuple[int,int,int,int]] = field(default_factory=list)  # (i, j, s, s2)
    agv_assignments: Dict[int, List[int]] = field(default_factory=dict)  # AGV -> [tasks]
    shelf_seq: Dict[int, List[int]] = field(default_factory=dict)  # Shelf -> [tasks]
    task_info: List[Tuple[int,int,float]] = field(default_factory=list)  # (task, workstation, duration)
    schedule: List[TaskSchedule] = field(default_factory=list)

@dataclass
class ScenarioMeta:
    map_info: Optional[str] = None
    task_rows: Optional[int] = None
    ws_set: Optional[List[int]] = None
    ws_order: Optional[Dict[int, List[int]]] = None
    scenario_line: Optional[str] = None  # raw line like "[Scenario] |J|=8 ... Γ=0"
    J: Optional[int] = None
    R: Optional[int] = None
    S: Optional[int] = None
    K: Optional[int] = None

@dataclass
class ParseResult:
    scenario: ScenarioMeta
    gammas: Dict[int, GammaResult]


# ---------- Parser implementation ----------

_re_run_gamma = re.compile(r'\[RUN\]\s*开始优化：γ\s*=\s*(\d+)')
_re_cmax = re.compile(r'C_max_AGV\s*=\s*([0-9]+(?:\.[0-9]+)?)')
_re_w = re.compile(r'w\[(\d+),\s*(\d+)\]\s*=\s*1')
_re_x = re.compile(r'x\[(\d+),\s*(\d+)\]\s*=\s*1')
# Accept both straight and curly apostrophe in "j’,r"
_re_z = re.compile(r'z\[(\d+),\s*(\d+),\s*(\d+)\]\s*=\s*1')
_re_v = re.compile(r'v\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\s*=\s*1')
_re_agv_assign = re.compile(r'AGV\s*(\d+):\s*\[([0-9,\s]+)\]')
_re_task_line = re.compile(
    r'\[Task\s+(\d+)\]\s*p=([0-9]+(?:\.[0-9]+)?),\s*q=([0-9]+(?:\.[0-9]+)?),\s*AGV=(\d+),\s*EndShelf=(\d+)'
)
_re_taskinfo = re.compile(r'Task\s+(\d+):\s*Workstation=(\d+),\s*Duration=([0-9]+(?:\.[0-9]+)?)')
_re_shelf_line = re.compile(r'Shelf\s+(\d+):\s*\[([0-9,\s]+)\]')

_re_map = re.compile(r'\[MAP\]\s*(.*)')
_re_task_meta = re.compile(r'\[TASK\]\s*读取\s*tasks\.csv\s*行数=(\d+),\s*WS集合=\[([0-9,\s]+)\]')
_re_wsorder = re.compile(r'\[WSOrder\]\s*固定顺序：\s*(\{.*\})')
_re_scenario_counts = re.compile(r'\[Scenario\]\s*\|J\|\s*=\s*(\d+)\s*\|R\|\s*=\s*(\d+)\s*\|S\|\s*=\s*(\d+)\s*\|K\|\s*=\s*(\d+)')


def _append_unique(lst, item):
    if item not in lst:
        lst.append(item)


def parse_milp_log(text: str) -> ParseResult:
    lines = text.splitlines()
    scenario = ScenarioMeta()
    gammas: Dict[int, GammaResult] = {}

    current_gamma: Optional[int] = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()

        # --- Global metadata
        m = _re_map.search(line)
        if m and scenario.map_info is None:
            scenario.map_info = m.group(1).strip()

        m = _re_task_meta.search(line)
        if m and scenario.task_rows is None:
            scenario.task_rows = int(m.group(1))
            ws_list = [int(x.strip()) for x in m.group(2).split(",") if x.strip()]
            scenario.ws_set = ws_list

        m = _re_wsorder.search(line)
        if m and scenario.ws_order is None:
            try:
                # dict string uses Python-like syntax; use json-friendly transform
                # Replace single quotes with double quotes safely (keys are ints)
                wsorder_str = m.group(1)
                safe = wsorder_str.replace("'", '"')
                scenario.ws_order = json.loads(safe)
            except Exception:
                scenario.ws_order = None

        if '[Scenario]' in line and scenario.scenario_line is None:
            scenario.scenario_line = line
            mc = _re_scenario_counts.search(line)
            if mc:
                scenario.J = int(mc.group(1))
                scenario.R = int(mc.group(2))
                scenario.S = int(mc.group(3))
                scenario.K = int(mc.group(4))

        # --- Gamma section start
        mg = _re_run_gamma.search(line)
        if mg:
            current_gamma = int(mg.group(1))
            if current_gamma not in gammas:
                gammas[current_gamma] = GammaResult(gamma=current_gamma)
            i += 1
            continue

        # If we haven't started a gamma section, skip gamma-specific parsers
        if current_gamma is None:
            i += 1
            continue

        gobj = gammas[current_gamma]

        # Cmax
        mc = _re_cmax.search(line)
        if mc:
            gobj.C_max_AGV = float(mc.group(1))

        # Section parsers with blocks
        # w
        if line.strip().startswith("=== w[j,r] = 1"):
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mw = _re_w.search(s)
                if mw:
                    _append_unique(gobj.w, (int(mw.group(1)), int(mw.group(2))))
                j += 1
            i = j
            continue

        # x
        if line.strip().startswith("=== x[j,s] = 1"):
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mx = _re_x.search(s)
                if mx:
                    _append_unique(gobj.x, (int(mx.group(1)), int(mx.group(2))))
                j += 1
            i = j
            continue

        # z
        if "=== z[" in line and "= 1" in line:
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mz = _re_z.search(s)
                if mz:
                    _append_unique(gobj.z, (int(mz.group(1)), int(mz.group(2)), int(mz.group(3))))
                j += 1
            i = j
            continue

        # v
        if "=== v[" in line and "搬架弧" in line:
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mv = _re_v.search(s)
                if mv:
                    _append_unique(gobj.v, (int(mv.group(1)), int(mv.group(2)), int(mv.group(3)), int(mv.group(4))))
                j += 1
            i = j
            continue

        # AGV assignments
        if line.strip().startswith("==== 优化完成, 任务分配结果"):
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                ma = _re_agv_assign.search(s)
                if ma:
                    agv = int(ma.group(1))
                    tasks = [int(x.strip()) for x in ma.group(2).split(",") if x.strip()]
                    gobj.agv_assignments[agv] = tasks
                j += 1
            i = j
            continue

        # Schedule line: [Task N] p=..., q=..., AGV=..., EndShelf=...
        if line.strip().startswith("[Task "):
            ms = _re_task_line.search(line.replace("，", ","))
            if ms:
                gobj.schedule.append(
                    TaskSchedule(
                        task=int(ms.group(1)),
                        p=float(ms.group(2)),
                        q=float(ms.group(3)),
                        agv=int(ms.group(4)),
                        end_shelf=int(ms.group(5))
                    )
                )

        # Task info
        if line.strip().startswith("=== 任务信息"):
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mt = _re_taskinfo.search(s)
                if mt:
                    gobj.task_info.append((int(mt.group(1)), int(mt.group(2)), float(mt.group(3))))
                j += 1
            i = j
            continue

        # Shelf sequences
        if line.strip().startswith("=== 货架上的任务序列"):
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("==="):
                    break
                mshelf = _re_shelf_line.search(s)
                if mshelf:
                    shelf = int(mshelf.group(1))
                    tasks = [int(x.strip()) for x in mshelf.group(2).split(",") if x.strip()]
                    gobj.shelf_seq[shelf] = tasks
                j += 1
            i = j
            continue

        i += 1

    return ParseResult(scenario=scenario, gammas=gammas)


# ---------- Export helpers ----------

def _df_schedule(sched: List[TaskSchedule]) -> pd.DataFrame:
    df = pd.DataFrame([{
        "task": t.task, "p": t.p, "q": t.q, "agv": t.agv, "end_shelf": t.end_shelf
    } for t in sched])
    if not df.empty:
        df = df.sort_values(["p", "task"]).reset_index(drop=True)
    return df

def _df_pairs(name1, name2, lst: List[Tuple[int,int]]) -> pd.DataFrame:
    return pd.DataFrame(lst, columns=[name1, name2]).sort_values([name1, name2]).reset_index(drop=True)

def _df_triples(c1, c2, c3, lst: List[Tuple[int,int,int]]) -> pd.DataFrame:
    return pd.DataFrame(lst, columns=[c1, c2, c3]).sort_values([c1, c2, c3]).reset_index(drop=True)

def _df_quads(c1, c2, c3, c4, lst: List[Tuple[int,int,int,int]]) -> pd.DataFrame:
    return pd.DataFrame(lst, columns=[c1, c2, c3, c4]).sort_values([c1, c2, c3, c4]).reset_index(drop=True)

def _df_agv_assign(d: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for agv, tasks in sorted(d.items()):
        rows.append({"AGV": agv, "tasks": tasks, "num_tasks": len(tasks)})
    return pd.DataFrame(rows)

def _df_shelf_seq(d: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for shelf, tasks in sorted(d.items()):
        rows.append({"shelf": shelf, "tasks": tasks, "num_tasks": len(tasks)})
    return pd.DataFrame(rows)

def _df_task_info(lst: List[Tuple[int,int,float]]) -> pd.DataFrame:
    return pd.DataFrame(lst, columns=["task", "workstation", "duration"]).sort_values("task").reset_index(drop=True)


def export_to_excel(parsed: ParseResult, out_path: str):
    # Summary sheet
    summary_rows = []
    for g, gres in sorted(parsed.gammas.items()):
        summary_rows.append({
            "gamma": g,
            "C_max_AGV": gres.C_max_AGV,
            "num_tasks": len({t.task for t in gres.schedule}),
            "num_w": len(gres.w),
            "num_x": len(gres.x),
            "num_z": len(gres.z),
            "num_v": len(gres.v),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("gamma").reset_index(drop=True)

    # Scenario sheet
    scen = parsed.scenario
    df_scen = pd.DataFrame([{
        "map_info": scen.map_info,
        "task_rows": scen.task_rows,
        "ws_set": scen.ws_set,
        "ws_order": scen.ws_order,
        "scenario_line": scen.scenario_line,
        "J": scen.J, "R": scen.R, "S": scen.S, "K": scen.K
    }])

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_scen.to_excel(writer, sheet_name="Scenario", index=False)

        for g, gres in sorted(parsed.gammas.items()):
            # One sheet per data category
            _df_schedule(gres.schedule).to_excel(writer, sheet_name=f"G{g}_Schedule", index=False)
            _df_pairs("j","r", gres.w).to_excel(writer, sheet_name=f"G{g}_w", index=False)
            _df_pairs("j","s", gres.x).to_excel(writer, sheet_name=f"G{g}_x", index=False)
            _df_triples("j","j2","r", gres.z).to_excel(writer, sheet_name=f"G{g}_z", index=False)
            _df_quads("i","j","s","s2", gres.v).to_excel(writer, sheet_name=f"G{g}_v", index=False)
            _df_agv_assign(gres.agv_assignments).to_excel(writer, sheet_name=f"G{g}_AGV", index=False)
            _df_shelf_seq(gres.shelf_seq).to_excel(writer, sheet_name=f"G{g}_ShelfSeq", index=False)
            _df_task_info(gres.task_info).to_excel(writer, sheet_name=f"G{g}_TaskInfo", index=False)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to MILP log text file")
    ap.add_argument("--out", required=True, help="Path to output Excel file")
    args = ap.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()
    parsed = parse_milp_log(text)
    export_to_excel(parsed, args.out)
    print(f"[OK] Exported parsed results to: {args.out}")

if __name__ == "__main__":
    main()
