# -*- coding: utf-8 -*-
# python rebuild_gamma0.py --out solution_exports --prefix demo01
import argparse, csv
from pathlib import Path

X = {1:23, 2:14, 3:25, 4:50, 5:41, 6:14, 7:41, 8:52, 9:42, 10:25, 11:43}
SEQ = [
    (3,7.00,15.00,1,25), (1,9.00,15.00,2,23), (9,18.00,24.00,3,42),
    (10,26.00,35.00,2,25), (5,30.00,38.00,1,41), (7,37.00,47.00,3,41),
    (2,46.00,54.00,1,14), (6,49.00,57.00,2,14), (8,58.00,66.00,3,52),
    (11,66.00,75.00,1,43), (4,68.00,75.00,2,50),
]
WS = {
  1:(2,6.0), 2:(2,8.0), 3:(1,8.0), 4:(2,7.0), 5:(2,8.0),
  6:(1,8.0), 7:(1,10.0), 8:(2,8.0), 9:(2,6.0), 10:(1,9.0), 11:(1,9.0),
}

def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[OK] write {path} ({len(rows)} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="solution_exports")
    ap.add_argument("--prefix", default="demo01")
    args = ap.parse_args()

    out = Path(args.out)
    pre = args.prefix

    # taskEndShelf_gamma0.csv
    write_csv(
        out / f"{pre}_taskEndShelf_gamma0.csv",
        ["task_id","end_shelf"],
        [(t, X[t]) for t in sorted(X)]
    )

    # wsPlan_gamma0.csv
    write_csv(
        out / f"{pre}_wsPlan_gamma0.csv",
        ["task_id","workstation","duration"],
        [(t, WS[t][0], WS[t][1]) for t in sorted(WS)]
    )

    # taskSeq_gamma0.csv
    write_csv(
        out / f"{pre}_taskSeq_gamma0.csv",
        ["task_id","p","q","agv","end_shelf"],
        SEQ
    )

if __name__ == "__main__":
    main()
