from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _build_task_to_cell_map_from_place(place_obj: Any) -> Dict[int, Any]:
    """
    尝试从 diag["place"] 里提取 Task -> cell 的映射。
    兼容 place 是 dict / list 的多种结构。
    """
    mapping: Dict[int, Any] = {}

    if isinstance(place_obj, dict):
        # 常见情况1：place[task] = {...} 或 place[str(task)] = {...}
        for k, v in place_obj.items():
            task = _safe_int(k)
            if task is None:
                # 也可能是 place["records"] 这种结构，留到后面搜索
                continue

            # v 可能是 cell 本身，也可能是一个 dict 里包含 cell 字段
            if isinstance(v, dict):
                cell = None
                for ck in ("cell", "cell_id", "Cell", "to_cell", "place_cell", "node", "to_node"):
                    if ck in v:
                        cell = v.get(ck)
                        break
                if cell is not None:
                    mapping[task] = cell
            else:
                mapping[task] = v

        # 常见情况2：place 不是按 task 做 key，而是内部有一堆 record
        # 这里做一次兜底扫描
        for _, v in place_obj.items():
            if isinstance(v, dict):
                task = _safe_int(v.get("Task", v.get("task")))
                if task is None:
                    continue
                cell = None
                for ck in ("cell", "cell_id", "Cell", "to_cell", "place_cell", "node", "to_node"):
                    if ck in v:
                        cell = v.get(ck)
                        break
                if cell is not None:
                    mapping[task] = cell

    elif isinstance(place_obj, list):
        # 常见情况：place 是记录列表，每条包含 Task/cell
        for item in place_obj:
            if not isinstance(item, dict):
                continue
            task = _safe_int(item.get("Task", item.get("task")))
            if task is None:
                continue
            cell = None
            for ck in ("cell", "cell_id", "Cell", "to_cell", "place_cell", "node", "to_node"):
                if ck in item:
                    cell = item.get(ck)
                    break
            if cell is not None:
                mapping[task] = cell

    return mapping


def _ensure_cell_column(seg: pd.DataFrame, diag_obj: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    保证 seg 里存在 'cell' 列。
    返回： (新seg, cell来源说明)
    """
    seg = seg.copy()

    if "cell" in seg.columns:
        return seg, "segments.csv 已自带 cell 列"

    # 1) 优先从 diag["place"] 补
    place_obj = diag_obj.get("place")
    task_to_cell = _build_task_to_cell_map_from_place(place_obj)
    if task_to_cell and "Task" in seg.columns:
        seg["cell"] = seg["Task"].map(task_to_cell)
        if seg["cell"].notna().any():
            return seg, "cell 由 diag.json 的 place 映射补齐（Task -> cell）"

    # 2) 兜底：用 to_node 当 cell（很多数据里去 cell 的目的节点就是 cell）
    if "to_node" in seg.columns:
        seg["cell"] = seg["to_node"]
        return seg, "segments.csv 无 cell；兜底使用 to_node 作为 cell（仅表示目的节点）"

    # 3) 再兜底：from_node
    if "from_node" in seg.columns:
        seg["cell"] = seg["from_node"]
        return seg, "segments.csv 无 cell/to_node；兜底使用 from_node 作为 cell（仅表示起点节点）"

    raise KeyError("segments.csv 中既没有 cell，也没有 to_node/from_node，无法构造 cell 列")


def _print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(f"\n[{title}]")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(df.head(max_rows).to_string(index=False))


def _find_place_record_for_task(place_obj: Any, task: int) -> Any:
    """
    在 diag["place"] 里尽量找到与 task 对应的那条记录（用于打印）。
    """
    if place_obj is None:
        return None

    # dict: 直接用 key 取
    if isinstance(place_obj, dict):
        for k in (task, str(task)):
            if k in place_obj:
                return place_obj[k]
        # 扫描 dict 的 value 看有没有 Task 字段
        for _, v in place_obj.items():
            if isinstance(v, dict):
                t = _safe_int(v.get("Task", v.get("task")))
                if t == task:
                    return v
        return None

    # list: 扫描
    if isinstance(place_obj, list):
        for v in place_obj:
            if isinstance(v, dict):
                t = _safe_int(v.get("Task", v.get("task")))
                if t == task:
                    return v
        return None

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        type=str,
        default="demo01_milp_bundle_srcG1_evalG1",
        help="文件名前缀，不带后缀。例：demo01_milp_bundle_srcG1_evalG1",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="solution_exports",
        help="导出文件所在目录（相对或绝对路径）",
    )
    parser.add_argument(
        "--task",
        type=int,
        default=4,
        help="想重点查看的 Task 编号",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="打印 TopK 异常任务/记录",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir)
    prefix = args.case

    diag_path = base_dir / f"{prefix}_diag.json"
    seg_path = base_dir / f"{prefix}_robust_segments.csv"
    tl_path = base_dir / f"{prefix}_timeline.csv"

    print(f"diag: {diag_path} exists= {diag_path.exists()}")
    print(f"seg : {seg_path} exists= {seg_path.exists()}")
    print(f"tl  : {tl_path} exists= {tl_path.exists()}")

    if not diag_path.exists() or not seg_path.exists() or not tl_path.exists():
        print("\n!! 有文件不存在，请检查 case 前缀或目录 --dir")
        return

    diag_obj = _read_json(diag_path)
    seg = pd.read_csv(seg_path)
    tl = pd.read_csv(tl_path)

    print("\n[diag keys]")
    print(list(diag_obj.keys()))

    print("\n[segments columns]")
    print(list(seg.columns))

    seg2, cell_src = _ensure_cell_column(seg, diag_obj)
    print(f"\n[cell column] OK: 已保证 seg 里有 'cell' 列；来源：{cell_src}")

    print("\n[timeline columns]")
    print(list(tl.columns))

    # 1) segments：按 arrive_cell_gap / cell_wait 找异常（如果列存在）
    if "arrive_cell_gap" in seg2.columns:
        worst_gap = (
            seg2.assign(arrive_cell_gap=seg2["arrive_cell_gap"].fillna(0))
            .sort_values("arrive_cell_gap", ascending=False)
        )
        _print_df(f"segments Top{args.topk} by arrive_cell_gap", worst_gap, max_rows=args.topk)

    if "cell_wait" in seg2.columns:
        worst_wait = (
            seg2.assign(cell_wait=seg2["cell_wait"].fillna(0))
            .sort_values("cell_wait", ascending=False)
        )
        _print_df(f"segments Top{args.topk} by cell_wait", worst_wait, max_rows=args.topk)

    # 2) timeline：重点看 place_wait_due_to_lb（如果存在）
    if "place_wait_due_to_lb" in tl.columns:
        worst_place_wait = (
            tl.assign(place_wait_due_to_lb=tl["place_wait_due_to_lb"].fillna(0))
            .sort_values("place_wait_due_to_lb", ascending=False)
        )
        _print_df(f"timeline Top{args.topk} by place_wait_due_to_lb", worst_place_wait, max_rows=args.topk)

    # 3) 你的 Task 聚焦输出
    if "Task" in tl.columns:
        tl_task = tl[tl["Task"] == args.task]
        _print_df(f"timeline where Task=={args.task}", tl_task, max_rows=50)
    else:
        print("\n!! timeline.csv 中没有 Task 列，无法按 Task 过滤")

    if "Task" in seg2.columns:
        seg_task = seg2[seg2["Task"] == args.task]
        _print_df(f"segments where Task=={args.task}", seg_task, max_rows=50)
    else:
        print("\n!! segments.csv 中没有 Task 列，无法按 Task 过滤")

    # 4) 把 diag["place"] 里与 Task 对应的记录尽量打印出来
    place_rec = _find_place_record_for_task(diag_obj.get("place"), args.task)
    print(f"\n[diag.place record for Task=={args.task}]")
    if place_rec is None:
        print("(not found)")
    else:
        # 直接打印原始结构，便于你对照字段名
        print(place_rec)


if __name__ == "__main__":
    main()
