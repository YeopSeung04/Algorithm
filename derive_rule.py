# derive_rule.py
from __future__ import annotations
import argparse, os
import pandas as pd
import numpy as np


def build_graph_level_dataset(df: pd.DataFrame, algos: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    그래프 인스턴스 단위로 algo별 median cpu를 만들고, best algo 라벨을 붙인다.
    """
    keys = ["family", "n", "p", "weight", "seed", "graph_idx"]

    # 1. 존재하는 컬럼만 사용하여 그룹화
    existing_keys = [k for k in keys if k in df.columns]
    g = df.groupby(existing_keys + ["algo"], dropna=False)["cpu_sec"].median().reset_index()

    # 2. Pivot: columns=algo
    piv = g.pivot(index=existing_keys, columns="algo", values="cpu_sec").reset_index()

    # 3. Best algo 결정
    use_algos = [a for a in algos if a in piv.columns]
    if len(use_algos) < 2:
        print(f"[Warning] 사용할 수 있는 알고리즘이 부족합니다: {list(piv.columns)}")
        return pd.DataFrame(), []

    cpu_mat = piv[use_algos].to_numpy()
    # 모든 알고리즘이 NaN인 행 제외
    valid_rows = ~np.isnan(cpu_mat).all(axis=1)
    piv = piv[valid_rows].copy()
    cpu_mat = piv[use_algos].to_numpy()

    best_idx = np.nanargmin(cpu_mat, axis=1)
    piv["best_algo"] = [use_algos[i] for i in best_idx]

    return piv, use_algos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outtxt", default="tables_paper/derived_rule.txt")
    ap.add_argument("--mode", default="basic")
    ap.add_argument("--tree_depth", type=int, default=3)
    args = ap.parse_args()

    # CSV 로드 (DtypeWarning 방지)
    df = pd.read_csv(args.results, low_memory=False)

    # 숫자형 변환
    numeric_cols = ["cpu_sec", "n", "p", "seed", "graph_idx", "avg_degree", "degree_cv", "clustering",
                    "diam_est", "aspl_est", "w_mean", "w_std", "meta_side", "meta_m", "meta_c", "meta_k", "meta_beta"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Feature 컬럼 설정
    # feature_candidates = [
    #     "n", "p", "avg_degree", "degree_cv", "clustering", "diam_est", "aspl_est", "w_mean", "w_std",
    #     "meta_side", "meta_m", "meta_c", "meta_k", "meta_beta",
    # ]
    # features = [c for c in feature_candidates if c in df.columns]
    # 모든 그래프 패밀리가 공통으로 가지는 물리적 지표들만 선택
    feature_candidates = [
        "n", "avg_degree", "degree_cv", "clustering",
        "diam_est", "aspl_est", "w_mean", "w_std"
    ]
    # meta_ 로 시작하는 파라미터들은 패밀리마다 다르므로 일단 제외하거나,
    # 꼭 쓰고 싶다면 0으로 채워야(fillna) 합니다.
    features = [c for c in feature_candidates if c in df.columns]
    # 알고리즘 선택
    all_algos = sorted(df["algo"].astype(str).unique())
    if args.mode == "basic":
        target_algos = ["A*", "DIJKSTRA"]
        if "A*" not in all_algos and "A" in all_algos:
            target_algos = ["A", "DIJKSTRA"]
    else:
        target_algos = all_algos

    # 그래프 레벨 데이터셋 빌드
    piv, use_algos = build_graph_level_dataset(df, target_algos)
    if piv.empty:
        print("데이터셋 생성 실패.")
        return

    # 그래프 레벨 피쳐 병합 (ValueError: label 'n' is not unique 해결 지점)
    keys = ["family", "n", "p", "weight", "seed", "graph_idx"]
    existing_keys = [k for k in keys if k in df.columns]

    # 중복 컬럼 방지를 위해 set 사용
    keep_cols = list(set(existing_keys + features))
    feat_df = df[keep_cols].drop_duplicates(subset=existing_keys)

    data = piv.merge(feat_df, on=existing_keys, how="left")

    # [중요] 결측치 확인 및 제거
    print(f"[*] 초기 데이터 수: {len(data)}")
    missing = data[features].isnull().sum()
    if missing.any():
        print("[*] 컬럼별 결측치 발생 현황:\n", missing[missing > 0])

    data_clean = data.dropna(subset=features + ["best_algo"])
    print(f"[*] 결측치 제거 후 남은 데이터 수: {len(data_clean)}")

    if len(data_clean) == 0:
        print("[!] 에러: 학습할 데이터가 0개입니다. features 리스트를 조정하거나 CSV를 확인하세요.")
        return

    X = data_clean[features].to_numpy()
    y = data_clean["best_algo"].astype(str).to_numpy()

    # Scikit-learn 모델링
    try:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        print("scikit-learn이 설치되어 있지 않습니다.")
        return

    clf = DecisionTreeClassifier(max_depth=args.tree_depth, random_state=7)

    # 클래스가 1개인 경우 처리 (분류 불가)
    if len(np.unique(y)) < 2:
        print(f"[!] 경고: 모든 데이터의 최적 알고리즘이 '{np.unique(y)[0]}'로 동일합니다. 교차 검증을 건너뜁니다.")
        acc = 1.0
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
        acc = cross_val_score(clf, X, y, cv=skf).mean()

    clf.fit(X, y)
    rule = export_text(clf, feature_names=features)

    # 결과 저장
    os.makedirs(os.path.dirname(args.outtxt), exist_ok=True)
    with open(args.outtxt, "w", encoding="utf-8") as f:
        f.write(f"[MODE] {args.mode}\n")
        f.write(f"[ALGOS] {use_algos}\n")
        f.write(f"[CV] stratified-5fold accuracy={acc:.3f}\n\n")
        f.write(rule)

    print(f"[OK] 결과 저장 완료 -> {args.outtxt}")
    print(f"[CV] accuracy={acc:.3f}")


if __name__ == "__main__":
    main()

# # 기본: A vs Dijkstra 룰
# python derive_rule.py --results out_paper/results_basic_20260208_085258.csv \
#   --outtxt tables_paper/derived_rule_basic.txt --mode basic --tree_depth 3
#
# # 확장: A/ALT/BI/DIJKSTRA 중 best 선택 룰
# python derive_rule.py --results out_paper/results_extended_bidij_alt_20260208_085647.csv \
#   --outtxt tables_paper2/derived_rule_extended.txt --mode extended --tree_depth 3
