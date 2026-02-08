# make_tables.py
from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # 안전하게 숫자 변환
    for c in ["cpu_sec", "expansions", "relaxations", "path_cost", "optimal_cost", "ok_optimal"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["n", "p", "ws_beta", "comm_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def median_summary(df: pd.DataFrame) -> pd.DataFrame:
    # family×algo 단위로 중앙값 요약
    g = df.groupby(["family", "algo"], dropna=False)
    out = g.agg(
        median_cpu_sec=("cpu_sec", "median"),
        median_exp=("expansions", "median"),
        median_rel=("relaxations", "median"),
        ok_rate=("ok_optimal", "mean"),
        n_samples=("cpu_sec", "count"),
    ).reset_index()

    # ms로도 추가
    out["median_cpu_ms"] = out["median_cpu_sec"] * 1000.0
    return out

def speedup_vs_dijkstra(summary: pd.DataFrame) -> pd.DataFrame:
    # family별 Dijkstra 기준 speedup
    pivot = summary.pivot(index="family", columns="algo", values="median_cpu_sec")
    if "DIJKSTRA" not in pivot.columns:
        raise ValueError("DIJKSTRA not found in algo column.")
    base = pivot["DIJKSTRA"]
    speed = (base.values[:, None] / pivot).copy()
    speed.columns = [f"speedup_{c}" for c in speed.columns]
    speed = speed.reset_index()
    return speed

def winrate_graph_level(df: pd.DataFrame, algo_a: str, algo_b: str) -> pd.DataFrame:
    # 그래프 인스턴스 단위: (family,n,p,weight,seed,graph_idx)에서
    # algo별 median cpu를 비교하여 승률 계산
    keys = ["family", "n", "p", "weight", "seed", "graph_idx"]
    g = df.groupby(keys + ["algo"])["cpu_sec"].median().reset_index()
    pa = g[g["algo"] == algo_a].rename(columns={"cpu_sec": "cpu_a"}).drop(columns=["algo"])
    pb = g[g["algo"] == algo_b].rename(columns={"cpu_sec": "cpu_b"}).drop(columns=["algo"])
    m = pa.merge(pb, on=keys, how="inner")
    m["a_wins"] = (m["cpu_a"] < m["cpu_b"]).astype(int)
    out = m.groupby(["family"])["a_wins"].agg(win_rate="mean", n_graphs="count").reset_index()
    out["algo_a"] = algo_a
    out["algo_b"] = algo_b
    return out

def to_latex_table(df: pd.DataFrame, path: str, caption: str, label: str, floatfmt: str = "0.3f") -> None:
    # LaTeX table 파일로 저장(본문에 \input{}로 넣기 쉬움)
    latex = df.to_latex(index=False, float_format=lambda x: format(x, floatfmt))
    block = []
    block.append("\\begin{table}[H]")
    block.append("\\centering")
    block.append(latex)
    block.append(f"\\caption{{{caption}}}")
    block.append(f"\\label{{{label}}}")
    block.append("\\end{table}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(block))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", default="tables_out")
    ap.add_argument("--mode", default="basic")  # basic / extended
    args = ap.parse_args()

    df = load_csv(args.results)
    os.makedirs(args.outdir, exist_ok=True)

    summ = median_summary(df)
    summ = summ.sort_values(["family", "algo"]).reset_index(drop=True)
    summ.to_csv(os.path.join(args.outdir, f"table_summary_{args.mode}.csv"), index=False)

    sp = speedup_vs_dijkstra(summ)
    sp.to_csv(os.path.join(args.outdir, f"table_speedup_{args.mode}.csv"), index=False)

    # 기본은 A vs Dijkstra, 확장은 A/ALT/BI 다 보고 싶으면 추가로 만들자
    wr_ad = winrate_graph_level(df, "A*", "DIJKSTRA") if "A*" in df["algo"].unique() else winrate_graph_level(df, "A", "DIJKSTRA")
    wr_ad.to_csv(os.path.join(args.outdir, f"table_winrate_A_vs_D_{args.mode}.csv"), index=False)

    # LaTeX 테이블도 같이 생성
    to_latex_table(
        summ[["family","algo","median_cpu_ms","median_exp","ok_rate","n_samples"]],
        os.path.join(args.outdir, f"latex_summary_{args.mode}.tex"),
        caption=f"Family별 알고리즘 성능 요약({args.mode})",
        label=f"tab:summary_{args.mode}",
        floatfmt="0.3f",
    )
    to_latex_table(
        sp,
        os.path.join(args.outdir, f"latex_speedup_{args.mode}.tex"),
        caption=f"Dijkstra 대비 speedup({args.mode})",
        label=f"tab:speedup_{args.mode}",
        floatfmt="0.3f",
    )
    to_latex_table(
        wr_ad,
        os.path.join(args.outdir, f"latex_winrate_{args.mode}.tex"),
        caption=f"그래프 인스턴스 단위 승률({args.mode})",
        label=f"tab:winrate_{args.mode}",
        floatfmt="0.3f",
    )

    print("[OK] tables ->", args.outdir)

if __name__ == "__main__":
    main()
