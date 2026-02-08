# plot_feature_importance.py
from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outpng", default="figures/plots_paper2/feature_importance.png")
    ap.add_argument("--mode", default="basic")
    args = ap.parse_args()

    # 1. 로딩 시 Warning 방지
    df = pd.read_csv(args.results, low_memory=False)

    numeric_cols = ["cpu_sec", "n", "p", "avg_degree", "degree_cv", "clustering", "diam_est", "aspl_est", "w_mean",
                    "w_std",
                    "meta_side", "meta_m", "meta_c", "meta_k", "meta_beta", "seed", "graph_idx"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2. Graph-level dataset (A* vs Dijkstra)
    keys = ["family", "n", "p", "weight", "seed", "graph_idx"]
    existing_keys = [k for k in keys if k in df.columns]

    g = df.groupby(existing_keys + ["algo"])["cpu_sec"].median().reset_index()
    piv = g.pivot(index=existing_keys, columns="algo", values="cpu_sec").reset_index()

    if args.mode == "basic":
        algo_a = "A*" if "A*" in piv.columns else ("A" if "A" in piv.columns else None)
        algo_b = "DIJKSTRA" if "DIJKSTRA" in piv.columns else None
        if algo_a is None or algo_b is None:
            print(f"[Error] Available algos: {piv.columns.tolist()}")
            raise ValueError("Need A*(or A) and DIJKSTRA in CSV.")
        piv["y"] = (piv[algo_a] < piv[algo_b]).astype(int)
    else:
        raise ValueError("Use mode=basic for interpretable coefficient importance.")

    # 3. Feature 선정 (결측치 위험이 큰 meta_ 변수는 제외하고 공통 구조 지표 위주로 설정)
    feature_candidates = [
        "n", "avg_degree", "degree_cv", "clustering", "diam_est", "aspl_est", "w_mean", "w_std"
    ]
    actual_features = [c for c in feature_candidates if c in df.columns]

    # [에러 해결 지점] 중복 컬럼 방지를 위해 set() 사용
    all_needed_cols = list(set(existing_keys + actual_features))
    feat_df = df[all_needed_cols].drop_duplicates(subset=existing_keys)

    data = piv.merge(feat_df, on=existing_keys, how="left")

    # 4. 결측치 제거 및 최종 feature 확정
    data = data.dropna(subset=actual_features + ["y"])

    if len(data) == 0:
        print("[!] 데이터가 0개입니다. 결측치가 많은 피쳐를 feature_candidates에서 제외하세요.")
        return

    X = data[actual_features].to_numpy()
    y = data["y"].to_numpy()

    # 5. 로지스틱 회귀 및 계수(Importance) 추출
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs")),
    ])
    pipe.fit(X, y)

    clf = pipe.named_steps["clf"]
    coef = clf.coef_[0]
    imp = np.abs(coef)  # 절대값이 클수록 성능 결정에 중요한 피쳐

    # 정렬 및 시각화
    idx = np.argsort(-imp)
    feat_sorted = [actual_features[i] for i in idx]
    imp_sorted = imp[idx]

    os.makedirs(os.path.dirname(args.outpng), exist_ok=True)
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(feat_sorted)))
    plt.bar(range(len(feat_sorted)), imp_sorted, color=colors)
    plt.xticks(range(len(feat_sorted)), feat_sorted, rotation=45, ha="right")
    plt.ylabel("Importance (|Standardized Coefficients|)")
    plt.title(f"Graph Feature Importance: Which feature decides A* vs Dijkstra?\n(Accuracy: {pipe.score(X, y):.2f})")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(args.outpng, dpi=300)
    plt.close()

    print(f"[OK] 분석 완료! 결과 이미지 저장됨 -> {args.outpng}")
    print(f"[*] 분석에 사용된 데이터 수: {len(data)}")


if __name__ == "__main__":
    main()



# python plot_feature_importance.py \
#   --results out_paper/results_basic_20260208_085258.csv \
#   --outpng figures/plots_paper2/feature_importance.png \
#   --mode basic
