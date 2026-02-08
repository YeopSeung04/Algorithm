# selection_rule.py
from __future__ import annotations
import argparse
import csv
from typing import Dict, List, Tuple
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def group_graph_level(rows: List[Dict[str, str]]) -> List[Dict]:
    """
    graph 단위로 aggregation.
    key: (family,n,p,weight,seed,graph_idx,ws_beta,comm_ratio)
    label: A* median cpu < Dijkstra median cpu ? 1 : 0
    features: graph_features (동일 값이 row마다 복제되어 있으므로 1개만 가져옴)
    """
    key_fields = ["family", "n", "p", "weight", "seed", "graph_idx", "ws_beta", "comm_ratio"]
    feats_prefix = "meta_"  # meta feature까지 포함하려면 여기서 처리

    buckets: Dict[Tuple, List[Dict[str, str]]] = {}
    for r in rows:
        k = tuple(r[f] for f in key_fields)
        buckets.setdefault(k, []).append(r)

    out = []
    for k, rs in buckets.items():
        # cpu lists
        dij = [float(x["cpu_sec"]) for x in rs if x["algo"] == "DIJKSTRA"]
        ast = [float(x["cpu_sec"]) for x in rs if x["algo"] == "A*"]
        if not dij or not ast:
            continue
        y = 1 if np.median(ast) < np.median(dij) else 0

        # pick one row to extract features
        r0 = rs[0]
        feats = {}
        for kk, vv in r0.items():
            if kk.startswith("n_") or kk in ["density", "avg_degree", "degree_cv", "clustering", "w_mean", "w_std", "diam_est", "aspl_est"] or kk.startswith("meta_"):
                # attempt to parse as float/int
                try:
                    feats[kk] = float(vv) if vv != "" else 0.0
                except Exception:
                    feats[kk] = 0.0

        out.append({
            "key": k,
            "label_astar_wins": y,
            **feats
        })

    return out


def rule_based_predict(sample: Dict) -> int:
    """
    단순 룰 baseline (논문에서 '해석 가능한 모델'로 넣기 좋음)
    - Grid/좌표형(여기서는 family meta로 간접 판단 가능) -> A* 우세
    - clustering 높고(>0.25) degree_cv 낮으면 A* 우세 (탐색이 heuristic에 잘 줄어드는 편)
    - community 강하면(추정: clustering 높고 aspl_est 높거나, meta params 활용) -> Dijkstra 우세
    """
    fam = ""
    # meta_family가 있으면 사용, 없으면 family를 key에서 파싱해야 함(여기선 meta_family 기대)
    if "meta_family" in sample:
        fam = str(sample["meta_family"])
    # fallback: key[0] = family
    if not fam and "key" in sample:
        fam = str(sample["key"][0])

    clustering = float(sample.get("clustering", 0.0))
    degree_cv = float(sample.get("degree_cv", 0.0))
    aspl = float(sample.get("aspl_est", 0.0))

    if fam == "GRID":
        return 1
    if fam == "COMM":
        return 0

    # heuristic-friendly
    if clustering > 0.25 and degree_cv < 0.60:
        return 1

    # community-ish / long-range
    if clustering > 0.35 and aspl > 10.0:
        return 0

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=str)
    ap.add_argument("--ml", action="store_true", help="use sklearn logistic regression if available")
    args = ap.parse_args()

    rows = read_csv(args.results)
    graphs = group_graph_level(rows)
    if not graphs:
        print("No graph-level data.")
        return

    # rule baseline
    y_true = np.array([g["label_astar_wins"] for g in graphs], dtype=np.int32)
    y_pred = np.array([rule_based_predict(g) for g in graphs], dtype=np.int32)
    acc = float((y_true == y_pred).mean())
    print(f"[RULE] accuracy={acc:.3f} (n_graphs={len(graphs)})")

    if args.ml:
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            # features
            feat_keys = [k for k in graphs[0].keys()
                         if k not in ["key", "label_astar_wins"] and isinstance(graphs[0][k], (int, float))]
            X = np.array([[float(g.get(k, 0.0)) for k in feat_keys] for g in graphs], dtype=np.float64)
            y = y_true

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=5000, solver="lbfgs")
            )
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
            print(f"[ML] accuracy={accuracy_score(yte, yp):.3f} (features={len(feat_keys)})")

            # feature importance (absolute coef)
            coefs = np.abs(clf.coef_[0])
            idx = np.argsort(coefs)[::-1][:10]
            print("[ML] top features:")
            for i in idx:
                print(f"  {feat_keys[i]}  |coef|={coefs[i]:.4f}")

        except Exception as e:
            print("[ML] sklearn not available or failed:", e)


if __name__ == "__main__":
    main()
