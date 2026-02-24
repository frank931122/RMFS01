#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
task_generator.py
- 以 seed 控制的可复现随机，生成 (Shelf, Workstation) 任务；
- 默认：允许重复（同一货架可有多个任务，且可在同一工位重复出现）；
- 若加 --unique-pairs 则恢复“互异 (s, w)”的旧逻辑；
- 写出到 scenario/<prefix>/tasks.csv；
- 同步镜像导出到 solution_exports/<prefix>_taskInfo.csv / _taskShelf.csv（便于旧管线/动画使用）
- 新增：为每个工作站生成固定顺序 WSOrder（从 1 开始）
"""

import os
import argparse
import pandas as pd
import numpy as np


def write_tasks_csv(prefix: str,
                    outdir: str = "scenario",
                    seed: int | None = None,
                    num_tasks: int = 50,
                    shelf_pool=None,
                    ws_pool=None,
                    unique_pairs: bool = False,
                    dur_low: int = 5,
                    dur_high: int = 11):
    """
    生成 tasks.csv 并镜像导出到 solution_exports。
    参数:
      - prefix: 场景名（目录：scenario/<prefix>/tasks.csv）
      - seed: 随机种子（None 则每次不同）
      - num_tasks: 任务数量
      - shelf_pool: 可选货架ID列表（默认 [1..6]）
      - ws_pool: 可选工位ID列表（默认 [1..2]）
      - unique_pairs: True=互异 (s,w)；False=允许重复（默认）
      - dur_low, dur_high: 工位加工时长的随机区间 [low, high)
    """
    scen_dir = os.path.join(outdir, prefix)
    os.makedirs(scen_dir, exist_ok=True)
    path = os.path.join(scen_dir, "tasks.csv")

    # 池默认
    if shelf_pool is None:
        shelf_pool = [1, 2, 3, 4, 5, 6]
    if ws_pool is None:
        ws_pool = [1, 2]

    # 可复现随机
    rng = np.random.default_rng(seed)

    # 生成 (Shelf, Workstation)
    rows = []
    if unique_pairs:
        # —— 兼容旧逻辑：互异 (s, w) —— #
        all_pairs = [(s, w) for s in shelf_pool for w in ws_pool]
        if num_tasks > len(all_pairs):
            raise ValueError(f"num_tasks={num_tasks} 超过可用互异对数 {len(all_pairs)}")
        chosen_idx = rng.choice(len(all_pairs), size=num_tasks, replace=False)
        pairs = [all_pairs[i] for i in chosen_idx]
        shelves = [p[0] for p in pairs]
        wss     = [p[1] for p in pairs]
    else:
        # —— 新逻辑（默认）：允许重复，独立抽样 —— #
        shelves = rng.choice(shelf_pool, size=num_tasks, replace=True).tolist()
        wss     = rng.choice(ws_pool,   size=num_tasks, replace=True).tolist()

    # 随机工位加工时长
    durations = rng.integers(dur_low, dur_high, size=num_tasks)

    for tid, (sid, ws, dur) in enumerate(zip(shelves, wss, durations), start=1):
        rows.append((int(tid), int(sid), int(ws), float(dur)))

    df = pd.DataFrame(rows, columns=["Task", "Shelf", "Workstation", "Duration"])

    # 新增：为每个工作站生成固定顺序（从 1 开始）
    df["WSOrder"] = df.groupby("Workstation").cumcount() + 1

    # 写 scenario
    df.to_csv(path, index=False)
    mode_str = "互异(s,w)" if unique_pairs else "允许重复"
    print(f"[TASK] 写出 {path} 行数={len(df)} (seed={seed}, num_tasks={num_tasks}, 模式={mode_str})")

    # —— 同步镜像：导出到 solution_exports（兼容旧仿真/动画） —— #
    outdir2 = "solution_exports"
    os.makedirs(outdir2, exist_ok=True)
    info_csv  = os.path.join(outdir2, f"{prefix}_taskInfo.csv")
    shelf_csv = os.path.join(outdir2, f"{prefix}_taskShelf.csv")

    # 这里把 WSOrder 一并导出，便于核对
    df[["Task", "Workstation", "Duration", "WSOrder"]].to_csv(info_csv, index=False)
    df[["Task", "Shelf"]].to_csv(shelf_csv, index=False)

    print(f"[TASK] 已同步导出到 {info_csv} / {shelf_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="demo01", help="场景前缀（scenario/<prefix>/tasks.csv）")
    ap.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    ap.add_argument("--num_tasks", type=int, default=40, help="任务数")
    ap.add_argument("--unique-pairs", action="store_true",
                    help="使用互异 (Shelf, Workstation) 生成（默认不互异，允许重复）")
    ap.add_argument("--dur_low", type=int, default=5, help="加工时长下界（含）")
    ap.add_argument("--dur_high", type=int, default=11, help="加工时长上界（不含）")
    args = ap.parse_args()

    write_tasks_csv(prefix=args.prefix,
                    seed=args.seed,
                    num_tasks=args.num_tasks,
                    unique_pairs=args.unique_pairs,
                    dur_low=args.dur_low,
                    dur_high=args.dur_high)
