# eval_selection_groupkfold.py
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

KEYS = ["family","n","p","weight","seed","graph_idx"]

def build_features(df: pd.DataFrame):
    # 그래프 레벨 feature + family one-hot
    feat_cols = [
        "avg_degree","degree_cv","clustering","diam_est","aspl_est",
        "w_mean","w_std","density","n_edges",
        "meta_side","meta_m","meta_c","meta_k","meta_beta",
        "comm_ratio","ws_beta",
    ]
    feat_cols = [c for c in feat_cols if c in df.columns]

    base = df[KEYS + feat_cols].drop_duplicates(subset=KEYS).copy()

    # numeric
    for c in feat_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)

    base["feat_n"] = pd.to_numeric(base["n"], errors="coerce").fillna(0.0)
    base["feat_p"] = pd.to_numeric(base["p"], errors="coerce").fillna(0.0)

    fam = pd.get_dummies(base["family"], prefix="fam")
    base = pd.concat([base, fam], axis=1)

    used_cols = ["feat_n","feat_p"] + feat_cols + list(fam.columns)
    X = base[used_cols].to_numpy()
    groups = base[KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    return base, X, groups, used_cols

def median_cpu_by_graph(df: pd.DataFrame):
    g = df.groupby(KEYS+["algo"], dropna=False)["cpu_sec"].median().reset_index()
    piv = g.pivot(index=KEYS, columns="algo", values="cpu_sec").reset_index()
    return piv

def eval_binary(df: pd.DataFrame):
    piv = median_cpu_by_graph(df)
    piv["y"] = (piv["A*"] < piv["DIJKSTRA"]).astype(int)

    base, X, groups, used_cols = build_features(df)
    data = base.merge(piv[KEYS+["y"]], on=KEYS, how="left")
    y = data["y"].astype(int).to_numpy()

    gkf = GroupKFold(n_splits=5)
    y_true=[]; y_pred=[]; y_prob=[]
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        prob = pipe.predict_proba(X[te])[:,1]
        y_true.append(y[te]); y_pred.append(pred); y_prob.append(prob)

    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred); y_prob=np.concatenate(y_prob)
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred)
    auc=roc_auc_score(y_true,y_prob)
    cm=confusion_matrix(y_true,y_pred)
    baseline=max(y.mean(), 1-y.mean())
    return acc,f1,auc,baseline,cm

def eval_multiclass(df: pd.DataFrame):
    piv = median_cpu_by_graph(df)
    algos = [c for c in ["A*","DIJKSTRA","BI_DIJKSTRA","ALT"] if c in piv.columns]
    piv["best"] = piv[algos].idxmin(axis=1)

    base, X, groups, used_cols = build_features(df)
    data = base.merge(piv[KEYS+["best"]], on=KEYS, how="left")
    y = data["best"].astype(str).to_numpy()

    # baseline majority
    vals, cnt = np.unique(y, return_counts=True)
    baseline = cnt.max()/cnt.sum()

    gkf = GroupKFold(n_splits=5)
    y_true=[]; y_pred=[]
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, multi_class="multinomial"))
        ])
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        y_true.append(y[te]); y_pred.append(pred)

    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    acc=accuracy_score(y_true,y_pred)
    macro_f1=f1_score(y_true,y_pred,average="macro")
    weighted_f1=f1_score(y_true,y_pred,average="weighted")
    report = classification_report(y_true,y_pred,zero_division=0)
    return acc, macro_f1, weighted_f1, baseline, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--mode", choices=["basic","extended"], required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.results, low_memory=False)

    if args.mode=="basic":
        acc,f1,auc,base,cm = eval_binary(df)
        print(f"[BASIC] GroupKFold5  acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}  baseline={base:.3f}")
        print("[BASIC] Confusion matrix [[TN,FP],[FN,TP]] =", cm.tolist())
    else:
        acc, mf1, wf1, base, rep = eval_multiclass(df)
        print(f"[EXT] GroupKFold5  acc={acc:.3f}  macro_f1={mf1:.3f}  weighted_f1={wf1:.3f}  baseline={base:.3f}")
        print(rep)

if __name__ == "__main__":
    main()
