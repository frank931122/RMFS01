# -*- coding: utf-8 -*-
from __future__ import annotations
import os, re, json, argparse, time
from typing import Dict, List, Tuple, Optional
import pandas as pd

# ----------------- 小工具 -----------------
def _safe_path(path: str) -> str:
    """若 path 已存在则自动追加 _v1/_v2/... 防止覆盖。"""
    base, ext = os.path.splitext(path)
    k = 1
    out = path
    while os.path.exists(out):
        out = f"{base}_v{k}{ext}"
        k += 1
    return out

def _try_read_csv(path: Optional[str]) -> pd.DataFrame:
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            print(f"[WARN] 读取 CSV 失败：{path} | {e}")
    return pd.DataFrame()

def _short(sheet_name: str) -> str:
    # Excel 工作表名最长31字符
    name = sheet_name.replace("/", "_").replace("\\", "_")
    return name[:31]

def _find_files(prefix: str, gamma: int, outdir: str) -> Dict[str, Dict[str, Tuple[str, float]]]:
    """扫描 solution_exports/ 下与 prefix/gamma 相关的 CSV，按 '后缀' 分组。
    返回: groups[suffix][key] = (path, mtime)
      - key 取值: "occ_agvs", "occ_cells", "occ_workstations", "occ_shelves",
                  "audit_agvs", "audit_cells", "audit_workstations", "audit_shelves",
                  "audit_summary"
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}_(occ|audit)_([A-Za-z0-9]+)_gamma{gamma}(?P<suffix>.*)\.csv$"
    )
    groups: Dict[str, Dict[str, Tuple[str, float]]] = {}
    if not os.path.isdir(outdir):
        return groups
    for fname in os.listdir(outdir):
        if not fname.endswith(".csv"):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        typ = m.group(1)      # 'occ' or 'audit'
        kind = m.group(2)     # e.g. 'agvs' / 'cells' / 'workstations' / 'shelves' / 'summary'
        suffix = m.group("suffix") or ""
        key = f"{typ}_{kind}"
        path = os.path.join(outdir, fname)
        groups.setdefault(suffix, {})[key] = (path, os.path.getmtime(path))
    return groups

def _choose_suffix(groups: Dict[str, Dict[str, Tuple[str,float]]],
                   suffix: Optional[str], pick_latest: bool) -> str:
    if not groups:
        raise FileNotFoundError("在 solution_exports/ 下未发现任何 occ/audit CSV。")

    if suffix is not None:
        cand = [s for s in groups if suffix in s]
        if len(cand) == 1:
            return cand[0]
        if len(cand) == 0:
            raise FileNotFoundError(f"没有找到包含后缀 '{suffix}' 的这一批导出文件。")
        # 多个匹配：选文件数最多那组
        cand.sort(key=lambda s: (len(groups[s]), max(t for _, t in groups[s].values())), reverse=True)
        return cand[0]

    if pick_latest:
        # 以每组的“最新修改时间”作为选择依据
        def group_latest_mtime(sfx: str) -> float:
            return max(t for _, t in groups[sfx].values())
        return max(groups.keys(), key=group_latest_mtime)

    # 默认：若只有一组就选它；否则选文件数最多且时间最近的一组
    if len(groups) == 1:
        return list(groups.keys())[0]
    return sorted(groups.keys(),
                  key=lambda sfx: (len(groups[sfx]), max(t for _, t in groups[sfx].values())),
                  reverse=True)[0]

def _read_bundle_and_meta(prefix: str, gamma: int, outdir: str):
    bundle_path = os.path.join(outdir, f"{prefix}_bundle_gamma{gamma}.json")
    meta_path   = os.path.join(outdir, f"{prefix}_result_meta_gamma{gamma}.json")

    bundle = None; meta = None
    if os.path.exists(bundle_path):
        try:
            with open(bundle_path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 bundle 失败：{bundle_path} | {e}")

    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] 读取 meta 失败：{meta_path} | {e}")

    return bundle_path if bundle else None, bundle, meta_path if meta else None, meta

def _to_df_bundle(bundle: dict) -> Dict[str, pd.DataFrame]:
    """把 bundle 中的 routes/place/shelf_seq/p/q 等转成若干 DataFrame。"""
    dfs = {}
    if not bundle:
        return dfs

    # routes
    routes = bundle.get("routes", {})
    if routes:
        rows = [{"AGV_ID": int(r), "seq": json.dumps([int(x) for x in seq or []])}
                for r, seq in routes.items()]
        dfs["bundle_routes"] = pd.DataFrame(rows).sort_values("AGV_ID")

    # place
    place = bundle.get("place", {})
    if place:
        rows = [{"Task": int(j), "EndShelf": int(s)} for j, s in place.items()]
        dfs["bundle_place"] = pd.DataFrame(rows).sort_values("Task")

    # shelf_seq
    shelf_seq = bundle.get("shelf_seq", {})
    if shelf_seq:
        rows = [{"shelf_id": int(c), "seq": json.dumps([int(x) for x in seq or []])}
                for c, seq in shelf_seq.items()]
        dfs["bundle_shelf_seq"] = pd.DataFrame(rows).sort_values("shelf_id")

    # p/q & cmax
    p_map = bundle.get("p", {})
    q_map = bundle.get("q", {})
    if p_map or q_map:
        all_tasks = sorted({int(j) for j in list(p_map.keys()) + list(q_map.keys())})
        rows = [{"Task": j,
                 "p": float(p_map.get(str(j), p_map.get(j, float("nan")))),
                 "q": float(q_map.get(str(j), q_map.get(j, float("nan"))))}
                for j in all_tasks]
        df = pd.DataFrame(rows).sort_values("Task")
        df["cmax_bundle"] = float(bundle.get("cmax", float("nan")))
        dfs["bundle_times"] = df

    # v/z/w（如有）
    if bundle.get("w_assignments"):
        dfs["bundle_w_assignments"] = pd.DataFrame(
            [{"Task": int(t), "AGV_ID": int(r)} for t, r in bundle["w_assignments"]]
        ).sort_values(["AGV_ID", "Task"])
    if bundle.get("z_edges"):
        dfs["bundle_z_edges"] = pd.DataFrame(
            [{"i": int(i), "j": int(j), "AGV_ID": int(r)} for i, j, r in bundle["z_edges"]]
        ).sort_values(["AGV_ID", "i", "j"])
    if bundle.get("v_arcs"):
        dfs["bundle_v_arcs"] = pd.DataFrame(
            [{"i": int(i), "j": int(j), "s_from": int(s), "s_to": int(sp)} for i, j, s, sp in bundle["v_arcs"]]
        )

    return dfs

def _to_df_meta(meta: dict) -> Dict[str, pd.DataFrame]:
    dfs = {}
    if not meta:
        return dfs
    # meta 主体
    flat = []
    for k in ("status", "obj_val", "cmax", "runtime", "mip_gap", "gamma_budget"):
        if k in meta:
            flat.append({"key": k, "value": meta[k]})
    if flat:
        dfs["meta"] = pd.DataFrame(flat)

    # sizes
    if isinstance(meta.get("sizes"), dict):
        dfs["meta_sizes"] = pd.DataFrame(
            [{"name": k, "size": v} for k, v in meta["sizes"].items()]
        )

    # sets
    if isinstance(meta.get("sets"), dict):
        sets_rows = []
        for sk, sv in meta["sets"].items():
            sets_rows.append({"set": sk, "value": json.dumps(sv, ensure_ascii=False)})
        dfs["meta_sets"] = pd.DataFrame(sets_rows)
    return dfs

# ----------------- 主逻辑：合并到 Excel -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="场景前缀，如 demo01")
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--suffix", default=None, help="指定要打包的那一批后缀（支持子串匹配），如 _sim2_20251027-180932")
    ap.add_argument("--pick-latest", action="store_true", help="不指定 suffix 时，自动选择最新的一批")
    ap.add_argument("--outfile", default=None, help="自定义输出 xlsx 路径；默认写到 solution_exports/")
    args = ap.parse_args()

    prefix = args.prefix
    gamma  = int(args.gamma)
    outdir = "solution_exports"
    os.makedirs(outdir, exist_ok=True)

    groups = _find_files(prefix, gamma, outdir)
    chosen_suffix = _choose_suffix(groups, args.suffix, args.pick_latest)
    files = groups[chosen_suffix]

    # 读取四类 occ 与四类 audit + summary（存在才写，缺失就跳过）
    occ_agvs  = _try_read_csv(dict(files).get("occ_agvs"))
    occ_cells = _try_read_csv(dict(files).get("occ_cells"))
    occ_ws    = _try_read_csv(dict(files).get("occ_workstations"))
    occ_shelf = _try_read_csv(dict(files).get("occ_shelves"))

    audit_agv  = _try_read_csv(dict(files).get("audit_agvs"))
    audit_cell = _try_read_csv(dict(files).get("audit_cells"))
    audit_ws   = _try_read_csv(dict(files).get("audit_workstations"))
    audit_shelf= _try_read_csv(dict(files).get("audit_shelves"))
    audit_sum  = _try_read_csv(dict(files).get("audit_summary"))

    # 尝试读取 bundle / meta（与后缀无关）
    bundle_path, bundle, meta_path, meta = _read_bundle_and_meta(prefix, gamma, outdir)
    bundle_dfs = _to_df_bundle(bundle) if bundle else {}
    meta_dfs   = _to_df_meta(meta) if meta else {}

    # 目标 xlsx 路径
    base_name = f"{prefix}_packed_gamma{gamma}{chosen_suffix}.xlsx"
    out_path = args.outfile or os.path.join(outdir, base_name)
    out_path = _safe_path(out_path)

    # 写 Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # 记录当前打包信息
        info_rows = []
        for k, (p, m) in sorted(files.items()):
            info_rows.append({"category": k, "file": os.path.basename(p), "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(m))})
        if bundle_path: info_rows.append({"category": "bundle_json", "file": os.path.basename(bundle_path), "mtime": ""})
        if meta_path:   info_rows.append({"category": "meta_json",   "file": os.path.basename(meta_path),   "mtime": ""})
        pd.DataFrame(info_rows).to_excel(writer, sheet_name=_short("pack_info"), index=False)

        # occ
        if not occ_agvs.empty:  occ_agvs.to_excel(writer, sheet_name=_short("occ_agvs"), index=False)
        if not occ_cells.empty: occ_cells.to_excel(writer, sheet_name=_short("occ_cells"), index=False)
        if not occ_ws.empty:    occ_ws.to_excel(writer, sheet_name=_short("occ_workstations"), index=False)
        if not occ_shelf.empty: occ_shelf.to_excel(writer, sheet_name=_short("occ_shelves"), index=False)

        # audits
        if not audit_agv.empty:   audit_agv.to_excel(writer, sheet_name=_short("audit_agvs"), index=False)
        if not audit_cell.empty:  audit_cell.to_excel(writer, sheet_name=_short("audit_cells"), index=False)
        if not audit_ws.empty:    audit_ws.to_excel(writer, sheet_name=_short("audit_workstations"), index=False)
        if not audit_shelf.empty: audit_shelf.to_excel(writer, sheet_name=_short("audit_shelves"), index=False)
        if not audit_sum.empty:   audit_sum.to_excel(writer, sheet_name=_short("audit_summary"), index=False)

        # bundle/meta
        for name, df in bundle_dfs.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=_short(name), index=False)
        for name, df in meta_dfs.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=_short(name), index=False)

    print(f"[OK] Packed Excel -> {out_path}")

if __name__ == "__main__":
    main()
