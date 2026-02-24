# export_solution.py
import os
import pandas as pd

def export_solution_csv(result,
                        p_vars, q_vars,
                        task_shelf_mapping,
                        tasks_dict,
                        gamma_show: int = 0,
                        file_prefix: str = "case01"):
    """
    将 Gurobi 求解结果导出为 3 个 CSV，放到 solution_exports/ 目录。

    Parameters
    ----------
    result : dict[int, list[int]]
        {AGV_ID: [task_id 按执行顺序排好]}
    p_vars, q_vars : gurobipy tupledict
        优化模型中的 p、q 变量字典。只导出 gamma=gamma_show 那一层。
    task_shelf_mapping : dict[int, int | None]
        task → shelf_id  映射。
    gamma_show : int
        选择导出哪一个 γ 段（默认 0）。
    file_prefix : str
        写文件名前缀，生成 3 个文件：
        <prefix>_taskSeq.csv / <prefix>_taskTiming.csv / <prefix>_taskShelf.csv
    """
    os.makedirs("solution_exports", exist_ok=True)

    # --- 兼容垫片：若 p/q 的键不是 (task, γ) 而是单 task，则包装成 (task, 0) ---
    def _wrap_gamma0(td):
        try:
            keys = list(td.keys())
        except Exception:
            keys = list(td)
        if not keys:
            return td
        # 已经是 (task, γ) 形状则直接返回
        if isinstance(keys[0], tuple) and len(keys[0]) == 2:
            return td
        # 否则包装成 (task, 0)
        return {(k, 0): td[k] for k in keys}

    p_vars = _wrap_gamma0(p_vars)
    q_vars = _wrap_gamma0(q_vars)

    # 1) 任务序列
    seq_df = pd.DataFrame(
        [{"AGV_ID": agv, "seq": str(task_list)}
         for agv, task_list in result.items()])
    seq_df.to_csv(f"solution_exports/{file_prefix}_taskSeq.csv", index=False)

    # 2) 任务时间
    rows = []
    for (t, γ), p_var in p_vars.items():
        if γ == gamma_show:
            rows.append({"Task": t,
                         "p": round(p_var.X, 4),
                         "q": round(q_vars[t, γ].X, 4)})
    pd.DataFrame(rows).to_csv(
        f"solution_exports/{file_prefix}_taskTiming.csv", index=False)

    # 3) 货架映射
    rows = [{"Task": t,
             "Shelf": task_shelf_mapping.get(t)}
            for agv in result.values() for t in agv]
    pd.DataFrame(rows).to_csv(
        f"solution_exports/{file_prefix}_taskShelf.csv", index=False)

    print(f"[export] 已写 3 份 CSV 到 solution_exports/ 目录\n")
    # 4️⃣ 任务‑工作站‑时长 信息（仿真需要知道 ws 互斥、服务时长）
    info_rows = []
    for t, (ws, dur, *_rest) in tasks_dict.items():
        info_rows.append({"Task": t,
                          "Workstation": ws,
                          "Duration": dur})
    pd.DataFrame(info_rows).to_csv(
        f"solution_exports/{file_prefix}_taskInfo.csv", index=False)

    print("[export] 任务‑工作站信息亦已导出。\n")