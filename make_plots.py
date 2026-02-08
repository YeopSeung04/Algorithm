# make_plots.py
from __future__ import annotations
import argparse
import os
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_fig(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _try_float(x: str) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    try:
        return float(s)
    except Exception:
        return None


def _get_sorted_unique(rows: List[Dict[str, str]], key: str) -> List:
    vals = []
    for r in rows:
        v = r.get(key, "")
        fv = _try_float(v)
        vals.append(fv if fv is not None else v)
    # remove empties
    vals2 = [v for v in vals if v != "" and v is not None]
    # numeric sort if possible
    if vals2 and all(isinstance(v, (int, float, np.floating)) for v in vals2):
        return sorted(set(float(v) for v in vals2))
    return sorted(set(str(v) for v in vals2))


def _group_key(r: Dict[str, str], keys: List[str]) -> Tuple:
    out = []
    for k in keys:
        v = r.get(k, "")
        fv = _try_float(v)
        out.append(fv if fv is not None else v)
    return tuple(out)


def _median_cpu(rows: List[Dict[str, str]], algo: str) -> float:
    xs = [_try_float(r["cpu_sec"]) for r in rows if r.get("algo") == algo]
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan")
    return float(np.median(xs))


def _agg_rank_matrix(
    ranking_rows: List[Dict[str, str]],
    row_key: str,
    col_key: str,
    algos: List[str],
    fixed: Dict[str, str] | None = None,
) -> Tuple[np.ndarray, List, List]:
    """
    Returns matrix [len(rows), len(cols)] for a single algo? -> NO.
    We build a 3D notion: but to keep it simple for paper:
      - For each cell (row_key, col_key), we compute avg rank per algo.
      - We will output one heatmap per algo OR we can output best algo only.
    Here we output one heatmap per algo (avg rank).
    """
    fixed = fixed or {}
    row_vals = _get_sorted_unique(ranking_rows, row_key)
    col_vals = _get_sorted_unique(ranking_rows, col_key)

    mats = {}
    for algo in algos:
        mat = np.full((len(row_vals), len(col_vals)), np.nan, dtype=np.float64)
        for i, rv in enumerate(row_vals):
            for j, cv in enumerate(col_vals):
                xs = []
                for r in ranking_rows:
                    if r.get("algo") != algo:
                        continue
                    if fixed:
                        ok = True
                        for fk, fv in fixed.items():
                            if str(r.get(fk, "")) != str(fv):
                                ok = False
                                break
                        if not ok:
                            continue
                    rrv = _try_float(r.get(row_key, "")); rrv = rrv if rrv is not None else r.get(row_key, "")
                    rcv = _try_float(r.get(col_key, "")); rcv = rcv if rcv is not None else r.get(col_key, "")
                    if str(rrv) == str(rv) and str(rcv) == str(cv):
                        xs.append(_try_float(r.get("rank", "")))
                xs = [x for x in xs if x is not None]
                if xs:
                    mat[i, j] = float(np.mean(xs))
        mats[algo] = mat

    return mats, row_vals, col_vals


def _agg_speedup_matrix(
    results_rows: List[Dict[str, str]],
    row_key: str,
    col_key: str,
    algos: List[str],
    base_algo: str = "DIJKSTRA",
    fixed: Dict[str, str] | None = None,
) -> Tuple[Dict[str, np.ndarray], List, List]:
    """
    speedup(cell, algo) = median_cpu(base_algo) / median_cpu(algo)
    aggregated within each cell
    """
    fixed = fixed or {}
    row_vals = _get_sorted_unique(results_rows, row_key)
    col_vals = _get_sorted_unique(results_rows, col_key)

    mats = {}
    for algo in algos:
        mat = np.full((len(row_vals), len(col_vals)), np.nan, dtype=np.float64)
        for i, rv in enumerate(row_vals):
            for j, cv in enumerate(col_vals):
                cell = []
                for r in results_rows:
                    if fixed:
                        ok = True
                        for fk, fv in fixed.items():
                            if str(r.get(fk, "")) != str(fv):
                                ok = False
                                break
                        if not ok:
                            continue
                    rrv = _try_float(r.get(row_key, "")); rrv = rrv if rrv is not None else r.get(row_key, "")
                    rcv = _try_float(r.get(col_key, "")); rcv = rcv if rcv is not None else r.get(col_key, "")
                    if str(rrv) == str(rv) and str(rcv) == str(cv):
                        cell.append(r)

                if not cell:
                    continue

                base = _median_cpu(cell, base_algo)
                m = _median_cpu(cell, algo)
                if np.isfinite(base) and np.isfinite(m) and m > 0:
                    mat[i, j] = base / m
        mats[algo] = mat

    return mats, row_vals, col_vals


def _plot_heatmap(mat: np.ndarray, row_labels: List, col_labels: List,
                  title: str, cbar: str, out_path: str, rotate_x: int = 30) -> None:
    plt.figure(figsize=(1.2 * max(4, len(col_labels)), 0.6 * max(4, len(row_labels))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label=cbar)
    plt.yticks(range(len(row_labels)), [str(x) for x in row_labels])
    plt.xticks(range(len(col_labels)), [str(x) for x in col_labels], rotation=rotate_x, ha="right")
    plt.title(title)
    save_fig(out_path)


def rank_heatmap_family_overall(ranking_rows: List[Dict[str, str]], outdir: str) -> None:
    families = sorted(set(r["family"] for r in ranking_rows))
    algos = sorted(set(r["algo"] for r in ranking_rows))

    mat = np.full((len(families), len(algos)), np.nan)
    for i, fam in enumerate(families):
        for j, algo in enumerate(algos):
            xs = [_try_float(r["rank"]) for r in ranking_rows if r["family"] == fam and r["algo"] == algo]
            xs = [x for x in xs if x is not None]
            if xs:
                mat[i, j] = float(np.mean(xs))

    _plot_heatmap(
        mat, families, algos,
        title="Rank Heatmap (avg rank per family)",
        cbar="Average Rank (lower is better)",
        out_path=os.path.join(outdir, "rank_heatmap_family_overall.png"),
    )


def speedup_heatmap_family_overall(results_rows: List[Dict[str, str]], outdir: str) -> None:
    families = sorted(set(r["family"] for r in results_rows))
    algos = sorted(set(r["algo"] for r in results_rows))
    if "DIJKSTRA" not in algos:
        raise ValueError("DIJKSTRA missing in results.")

    mat = np.full((len(families), len(algos)), np.nan)

    for i, fam in enumerate(families):
        base = [_try_float(r["cpu_sec"]) for r in results_rows if r["family"] == fam and r["algo"] == "DIJKSTRA"]
        base = [x for x in base if x is not None]
        base_med = float(np.median(base)) if base else np.nan
        for j, algo in enumerate(algos):
            xs = [_try_float(r["cpu_sec"]) for r in results_rows if r["family"] == fam and r["algo"] == algo]
            xs = [x for x in xs if x is not None]
            med = float(np.median(xs)) if xs else np.nan
            if np.isfinite(base_med) and np.isfinite(med) and med > 0:
                mat[i, j] = base_med / med

    _plot_heatmap(
        mat, families, algos,
        title="Speedup Heatmap (median CPU speedup per family)",
        cbar="Speedup vs Dijkstra (median)",
        out_path=os.path.join(outdir, "speedup_heatmap_family_overall.png"),
    )


def scatter_expansion_cpu(results_rows: List[Dict[str, str]], outdir: str) -> None:
    algos = sorted(set(r["algo"] for r in results_rows))
    plt.figure(figsize=(9, 6))
    for algo in algos:
        xs = [_try_float(r["expansions"]) for r in results_rows if r["algo"] == algo]
        ys = [_try_float(r["cpu_sec"]) for r in results_rows if r["algo"] == algo]
        xs = [x for x in xs if x is not None]
        ys = [y for y in ys if y is not None]
        if xs and ys:
            plt.scatter(xs, ys, s=8, alpha=0.35, label=algo)

    plt.xlabel("Expansions")
    plt.ylabel("CPU seconds")
    plt.title("Expansions vs CPU (all queries)")
    plt.legend(markerscale=2, fontsize=9)
    save_fig(os.path.join(outdir, "expansion_cpu_scatter.png"))


def heatmaps_family_x_p(ranking_rows: List[Dict[str, str]], results_rows: List[Dict[str, str]], outdir: str) -> None:
    # rows=family, cols=p
    algos_rank = sorted(set(r["algo"] for r in ranking_rows))
    rank_mats, fams, ps = _agg_rank_matrix(ranking_rows, "family", "p", algos_rank)

    for algo, mat in rank_mats.items():
        _plot_heatmap(
            mat, fams, ps,
            title=f"Avg Rank Heatmap (family × p) | algo={algo}",
            cbar="Average Rank (lower is better)",
            out_path=os.path.join(outdir, f"rank_family_x_p__{algo}.png"),
        )

    algos_speed = sorted(set(r["algo"] for r in results_rows))
    speed_mats, fams2, ps2 = _agg_speedup_matrix(results_rows, "family", "p", algos_speed, base_algo="DIJKSTRA")
    for algo, mat in speed_mats.items():
        _plot_heatmap(
            mat, fams2, ps2,
            title=f"Speedup Heatmap (family × p) | algo={algo}",
            cbar="Speedup vs Dijkstra (median)",
            out_path=os.path.join(outdir, f"speedup_family_x_p__{algo}.png"),
        )


def heatmaps_family_x_n(ranking_rows: List[Dict[str, str]], results_rows: List[Dict[str, str]], outdir: str) -> None:
    # rows=family, cols=n
    algos_rank = sorted(set(r["algo"] for r in ranking_rows))
    rank_mats, fams, ns = _agg_rank_matrix(ranking_rows, "family", "n", algos_rank)

    for algo, mat in rank_mats.items():
        _plot_heatmap(
            mat, fams, ns,
            title=f"Avg Rank Heatmap (family × n) | algo={algo}",
            cbar="Average Rank (lower is better)",
            out_path=os.path.join(outdir, f"rank_family_x_n__{algo}.png"),
        )

    algos_speed = sorted(set(r["algo"] for r in results_rows))
    speed_mats, fams2, ns2 = _agg_speedup_matrix(results_rows, "family", "n", algos_speed, base_algo="DIJKSTRA")
    for algo, mat in speed_mats.items():
        _plot_heatmap(
            mat, fams2, ns2,
            title=f"Speedup Heatmap (family × n) | algo={algo}",
            cbar="Speedup vs Dijkstra (median)",
            out_path=os.path.join(outdir, f"speedup_family_x_n__{algo}.png"),
        )


def heatmaps_ws_beta(ranking_rows: List[Dict[str, str]], results_rows: List[Dict[str, str]], outdir: str) -> None:
    # WS only: rows=p, cols=ws_beta
    rr = [r for r in ranking_rows if r.get("family") == "WS" and str(r.get("ws_beta", "")).strip() != ""]
    rs = [r for r in results_rows if r.get("family") == "WS" and str(r.get("ws_beta", "")).strip() != ""]
    if not rr or not rs:
        return

    algos_rank = sorted(set(r["algo"] for r in rr))
    rank_mats, ps, betas = _agg_rank_matrix(rr, "p", "ws_beta", algos_rank)
    for algo, mat in rank_mats.items():
        _plot_heatmap(
            mat, ps, betas,
            title=f"Avg Rank Heatmap (WS: p × beta) | algo={algo}",
            cbar="Average Rank (lower is better)",
            out_path=os.path.join(outdir, f"rank_WS_p_x_beta__{algo}.png"),
        )

    algos_speed = sorted(set(r["algo"] for r in rs))
    speed_mats, ps2, betas2 = _agg_speedup_matrix(rs, "p", "ws_beta", algos_speed, base_algo="DIJKSTRA")
    for algo, mat in speed_mats.items():
        _plot_heatmap(
            mat, ps2, betas2,
            title=f"Speedup Heatmap (WS: p × beta) | algo={algo}",
            cbar="Speedup vs Dijkstra (median)",
            out_path=os.path.join(outdir, f"speedup_WS_p_x_beta__{algo}.png"),
        )


def heatmaps_comm_ratio(ranking_rows: List[Dict[str, str]], results_rows: List[Dict[str, str]], outdir: str) -> None:
    # COMM only: rows=p, cols=comm_ratio
    rr = [r for r in ranking_rows if r.get("family") == "COMM" and str(r.get("comm_ratio", "")).strip() != ""]
    rs = [r for r in results_rows if r.get("family") == "COMM" and str(r.get("comm_ratio", "")).strip() != ""]
    if not rr or not rs:
        return

    algos_rank = sorted(set(r["algo"] for r in rr))
    rank_mats, ps, ratios = _agg_rank_matrix(rr, "p", "comm_ratio", algos_rank)
    for algo, mat in rank_mats.items():
        _plot_heatmap(
            mat, ps, ratios,
            title=f"Avg Rank Heatmap (COMM: p × ratio) | algo={algo}",
            cbar="Average Rank (lower is better)",
            out_path=os.path.join(outdir, f"rank_COMM_p_x_ratio__{algo}.png"),
        )

    algos_speed = sorted(set(r["algo"] for r in rs))
    speed_mats, ps2, ratios2 = _agg_speedup_matrix(rs, "p", "comm_ratio", algos_speed, base_algo="DIJKSTRA")
    for algo, mat in speed_mats.items():
        _plot_heatmap(
            mat, ps2, ratios2,
            title=f"Speedup Heatmap (COMM: p × ratio) | algo={algo}",
            cbar="Speedup vs Dijkstra (median)",
            out_path=os.path.join(outdir, f"speedup_COMM_p_x_ratio__{algo}.png"),
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=str)
    ap.add_argument("--ranking", required=True, type=str)
    ap.add_argument("--outdir", default="plots", type=str)
    args = ap.parse_args()

    results_rows = read_csv(args.results)
    ranking_rows = read_csv(args.ranking)

    os.makedirs(args.outdir, exist_ok=True)

    # 기존 3개
    scatter_expansion_cpu(results_rows, args.outdir)
    rank_heatmap_family_overall(ranking_rows, args.outdir)
    speedup_heatmap_family_overall(results_rows, args.outdir)

    # 확장 heatmaps
    heatmaps_family_x_p(ranking_rows, results_rows, args.outdir)
    heatmaps_family_x_n(ranking_rows, results_rows, args.outdir)
    heatmaps_ws_beta(ranking_rows, results_rows, args.outdir)
    heatmaps_comm_ratio(ranking_rows, results_rows, args.outdir)

    print("[OK] plots saved ->", args.outdir)


if __name__ == "__main__":
    main()
