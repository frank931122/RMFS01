import argparse
import csv
import os
import re
from typing import Dict, Tuple, Optional, List


def _find_col(cols: List[str], patterns: List[str]) -> Optional[str]:
    """在列名里用正则找第一个匹配的列。"""
    for pat in patterns:
        rgx = re.compile(pat, re.IGNORECASE)
        for c in cols:
            if rgx.search(c):
                return c
    return None


def _candidate_paths(prefix: str, gamma: int) -> List[str]:
    fname = f"{prefix}_p_q_times_allGamma_gamma{gamma}.csv"

    cwd = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    cands = [
        os.path.join(cwd, fname),
        os.path.join(cwd, "solution_exports", fname),
        os.path.join(script_dir, fname),
        os.path.join(root_dir, "solution_exports", fname),
        os.path.join(root_dir, fname),
    ]

    # 去重 + 规范化
    seen = set()
    res = []
    for p in cands:
        p2 = os.path.normpath(p)
        if p2 not in seen:
            seen.add(p2)
            res.append(p2)
    return res


def _walk_find(start_dir: str, target_name: str, max_depth: int = 3) -> Optional[str]:
    """从 start_dir 往下最多搜 max_depth 层，找文件名完全匹配的文件。"""
    start_dir = os.path.abspath(start_dir)
    if not os.path.isdir(start_dir):
        return None

    base_depth = start_dir.rstrip("\\/").count(os.sep)
    for root, dirs, files in os.walk(start_dir):
        depth = root.rstrip("\\/").count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if target_name in files:
            return os.path.join(root, target_name)
    return None


def find_pq_file(prefix: str, gamma: int) -> Optional[str]:
    fname = f"{prefix}_p_q_times_allGamma_gamma{gamma}.csv"

    # 1) 先按候选路径试
    for p in _candidate_paths(prefix, gamma):
        if os.path.exists(p):
            return p

    # 2) 兜底：有限深度搜索
    cwd = os.path.abspath(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    for start in [cwd, script_dir, root_dir]:
        found = _walk_find(start, fname, max_depth=4)
        if found:
            return found

    return None


def load_pq_wide_or_long(path: str) -> Tuple[Dict[int, float], Dict[int, float], str]:
    """
    支持两种格式：
    - wide: 每行 Task，包含 q0/q1（或 q[0]/q[1]）
    - long: 每行 Task + gamma + q（列名可能是 g/Gamma/Layer 等）
    返回：q0, q1, mode
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []

        task_col = _find_col(cols, [r"^task$", r"\btask\b", r"^Task$"])
        if task_col is None:
            raise ValueError(f"无法识别 Task 列。列名={cols}")

        # 尝试 wide
        q0_col = _find_col(cols, [r"^q0$", r"q[^a-zA-Z0-9]*0\b", r"q\[\s*0\s*\]"])
        q1_col = _find_col(cols, [r"^q1$", r"q[^a-zA-Z0-9]*1\b", r"q\[\s*1\s*\]"])

        if q0_col and q1_col:
            q0: Dict[int, float] = {}
            q1: Dict[int, float] = {}
            for row in reader:
                try:
                    j = int(float(row[task_col]))
                    q0[j] = float(row[q0_col])
                    q1[j] = float(row[q1_col])
                except Exception:
                    continue
            return q0, q1, "wide"

        # 尝试 long
        gamma_col = _find_col(cols, [r"^g$", r"\bgamma\b", r"\blayer\b", r"^Γ$"])
        q_col = _find_col(cols, [r"^q$", r"\bq\b", r"q_time", r"q\("])

        if gamma_col and q_col:
            q0: Dict[int, float] = {}
            q1: Dict[int, float] = {}
            for row in reader:
                try:
                    j = int(float(row[task_col]))
                    gg = int(float(row[gamma_col]))
                    qv = float(row[q_col])
                except Exception:
                    continue
                if gg == 0:
                    q0[j] = qv
                elif gg == 1:
                    q1[j] = qv
            return q0, q1, "long"

        raise ValueError(
            "无法识别 CSV 格式：既没找到 (q0,q1) 也没找到 (gamma,q) 列。\n"
            f"列名={cols}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", type=str, required=True)
    ap.add_argument("--gamma", type=int, required=True)
    args = ap.parse_args()

    path = find_pq_file(args.prefix, args.gamma)
    if not path:
        fname = f"{args.prefix}_p_q_times_allGamma_gamma{args.gamma}.csv"
        print("[ERROR] 没找到目标文件：", fname)
        print("你可以先在项目根目录执行：")
        print(rf'  python -c "import glob; print(' + "'\\n'.join(glob.glob('**/*p_q_times_allGamma*gamma1*.csv', recursive=True)))" + r')"')
        print("或者检查 main 是否在本次 run 中执行了 export_solution_csv。")
        return

    print(f"[OK] 找到文件：{path}")
    q0, q1, mode = load_pq_wide_or_long(path)
    print(f"[OK] 解析模式：{mode} | q0条数={len(q0)} q1条数={len(q1)}")

    common = sorted(set(q0.keys()) & set(q1.keys()))
    if not common:
        print("[WARN] 没有同时存在 q0/q1 的任务。请打开 CSV 看看格式或列名。")
        return

    diffs = []
    for j in common:
        diffs.append((j, q0[j], q1[j], q1[j] - q0[j]))
    diffs.sort(key=lambda x: x[3], reverse=True)

    all_d = [d for _, _, _, d in diffs]
    print("\nTop-10 最大 (q1-q0)：")
    for j, a, b, d in diffs[:10]:
        print(f"  Task {j:>2}: q0={a:>8.2f}  q1={b:>8.2f}  diff={d:>8.2f}")

    print(f"\nDiff 统计：min={min(all_d):.2f}, max={max(all_d):.2f}, avg={sum(all_d)/len(all_d):.2f}")


if __name__ == "__main__":
    main()
