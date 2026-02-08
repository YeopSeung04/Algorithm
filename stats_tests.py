# stats_tests.py
from __future__ import annotations
import argparse
import os
import math
import pandas as pd
import numpy as np

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta: P(x>y) - P(x<y)
    O(n^2)지만 family별 graph 수가 크지 않으면 충분.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    gt = 0
    lt = 0
    for a in x:
        gt += np.sum(a > y)
        lt += np.sum(a < y)
    n = len(x) * len(y)
    if n == 0:
        return float("nan")
    return (gt - lt) / n

def try_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """
    Wilcoxon signed-rank p-value (two-sided).
    scipy 있으면 사용, 없으면 sign test 근사로 대체.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[~np.isnan(d)]
    d = d[d != 0.0]
    if len(d) < 5:
        return float("nan")

    # scipy 있으면 Wilcoxon
    try:
        from scipy.stats import wilcoxon
        _, p = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
        return float(p)
    except Exception:
        # fallback: sign test (binomial) 근사
        pos = int(np.sum(d > 0))
        neg = int(np.sum(d < 0))
        n = pos + neg
        if n == 0:
            return float("nan")
        k = min(pos, neg)

        # two-sided binomial p-value: 2 * sum_{i=0..k} C(n,i)/2^n
        # compute in log space for stability
        def logC(n, r):
            return math.lgamma(n+1) - math.lgamma(r+1) - math.lgamma(n-r+1)

        logs = [logC(n, i) - n*math.log(2.0) for i in range(k+1)]
        m = max(logs)
        s = sum(math.exp(li - m) for li in logs)
        p = 2.0 * math.exp(m) * s
        return float(min(1.0, p))

def make_pairs(df: pd.DataFrame, algo_a: str, algo_b: str) -> pd.DataFrame:
    """
    (family,n,p,weight,seed,graph_idx) 단위로 algo별 median cpu를 만들어 paired 비교 데이터 생성
    """
    keys = ["family", "n", "p", "weight", "seed", "graph_idx"]
    g = df.groupby(keys + ["algo"], dropna=False)["cpu_sec"].median().reset_index()

    a = g[g["algo"] == algo_a].rename(columns={"cpu_sec": "cpu_a"}).drop(columns=["algo"])
    b = g[g["algo"] == algo_b].rename(columns={"cpu_sec": "cpu_b"}).drop(columns=["algo"])

    m = a.merge(b, on=keys, how="inner")
    return m

def write_latex_table(path: str, rows: list[dict], caption: str, label: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # simple latex table writer
    header = r"""\begin{table}[H]
\centering
\caption{%s}
\label{%s}
\begin{tabular}{l r r c}
\toprule
Graph Family & p-value & Cliff's $\delta$ & Significant ($\alpha=0.05$) \\
\midrule
""" % (caption, label)

    lines = [header]
    for r in rows:
        fam = r["family"]
        p = r["p_value"]
        d = r["cliffs_delta"]
        sig = "Yes" if (isinstance(p, float) and not math.isnan(p) and p < 0.05) else "No"
        p_str = (f"{p:.4g}" if (isinstance(p, float) and not math.isnan(p)) else "NA")
        d_str = (f"{d:.3f}" if (isinstance(d, float) and not math.isnan(d)) else "NA")
        lines.append(f"{fam} & {p_str} & {d_str} & {sig} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outtex", default="tables_paper/latex_wilcoxon_basic.tex")
    ap.add_argument("--algo_a", default="A*")
    ap.add_argument("--algo_b", default="DIJKSTRA")
    args = ap.parse_args()

    df = pd.read_csv(args.results)
    for c in ["cpu_sec", "n", "p", "graph_idx", "seed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    algos = set(df["algo"].astype(str).unique())
    if args.algo_a not in algos:
        # 혹시 CSV가 "A"로 바뀌었으면 자동 대체
        if "A" in algos:
            args.algo_a = "A"
    if args.algo_b not in algos and "DIJKSTRA" not in algos:
        raise ValueError(f"Algo not found. available={sorted(list(algos))}")

    paired = make_pairs(df, args.algo_a, args.algo_b)

    out_rows = []
    for fam, sub in paired.groupby("family"):
        x = sub["cpu_a"].to_numpy(dtype=float)
        y = sub["cpu_b"].to_numpy(dtype=float)
        p = try_wilcoxon(x, y)
        d = cliffs_delta(x, y)
        out_rows.append({"family": fam, "p_value": p, "cliffs_delta": d})

    # family 이름 고정 정렬
    fam_order = ["ER", "BA", "WS", "GRID", "COMM"]
    out_rows.sort(key=lambda r: fam_order.index(r["family"]) if r["family"] in fam_order else 999)

    write_latex_table(
        args.outtex,
        out_rows,
        caption=f"Wilcoxon signed-rank test 및 Cliff's delta ({args.algo_a} vs {args.algo_b})",
        label="tab:wilcoxon_effect",
    )
    print("[OK] wrote ->", args.outtex)

if __name__ == "__main__":
    main()


# basic 결과에서 A* vs Dijkstra
# python stats_tests.py --results out_paper/results_basic_20260208_085258.csv \
#   --outtex tables_paper/latex_wilcoxon_basic.tex
#
# # extended에서도 A* vs Dijkstra (원하면)
# python stats_tests.py --results out_paper/results_extended_bidij_alt_20260208_085647.csv \
#   --outtex tables_paper2/latex_wilcoxon_extended.tex