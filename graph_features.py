# graph_features.py
from __future__ import annotations
import random
import math
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import networkx as nx


def _edge_weight_stats(G: nx.Graph) -> Tuple[float, float]:
    ws = [float(G[u][v].get("w", 1.0)) for u, v in G.edges()]
    if not ws:
        return 0.0, 0.0
    w = np.array(ws, dtype=np.float64)
    return float(w.mean()), float(w.std(ddof=1) if len(w) > 1 else 0.0)


def _degree_stats(G: nx.Graph) -> Tuple[float, float]:
    deg = np.array([d for _, d in G.degree()], dtype=np.float64)
    if len(deg) == 0:
        return 0.0, 0.0
    mu = float(deg.mean())
    sd = float(deg.std(ddof=1) if len(deg) > 1 else 0.0)
    cv = sd / mu if mu > 1e-9 else 0.0
    return mu, cv


def _approx_diameter_and_aspl(
    G: nx.Graph,
    seed: int,
    samples: int = 8,
) -> Tuple[float, float]:
    """
    가중 그래프에서 diameter/avg shortest path length를 근사.
    - samples개의 랜덤 source에서 single_source_dijkstra_path_length 수행
    - diameter ~= max dist
    - aspl ~= 평균 dist의 평균
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return 0.0, 0.0

    k = min(samples, len(nodes))
    srcs = rng.sample(nodes, k)

    maxd = 0.0
    aspl_acc = 0.0
    aspl_cnt = 0

    for s in srcs:
        dist = nx.single_source_dijkstra_path_length(G, s, weight="w")
        if len(dist) <= 1:
            continue
        vals = list(dist.values())
        local_max = max(vals)
        if local_max > maxd:
            maxd = local_max

        # exclude self=0
        vals2 = [v for v in vals if v > 0]
        if vals2:
            aspl_acc += sum(vals2) / len(vals2)
            aspl_cnt += 1

    aspl = aspl_acc / aspl_cnt if aspl_cnt > 0 else 0.0
    return float(maxd), float(aspl)


def compute_features(
    G: nx.Graph,
    meta: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    그래프 하나에 대한 feature dict 반환.
    meta(패밀리/파라미터)는 그대로 feature에 포함시켜도 됨(회귀/분석에 도움).
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    avg_deg, deg_cv = _degree_stats(G)
    clustering = float(nx.average_clustering(G)) if n >= 3 else 0.0
    w_mean, w_std = _edge_weight_stats(G)

    diam_est, aspl_est = _approx_diameter_and_aspl(G, seed=seed, samples=8)

    feats = {
        "n_nodes": int(n),
        "n_edges": int(m),
        "density": float(nx.density(G)) if n > 1 else 0.0,
        "avg_degree": float(avg_deg),
        "degree_cv": float(deg_cv),
        "clustering": float(clustering),
        "w_mean": float(w_mean),
        "w_std": float(w_std),
        "diam_est": float(diam_est),
        "aspl_est": float(aspl_est),
    }

    # meta 파라미터도 같이 넣어주면 분석/룰 만들기 쉬움
    # (예: WS beta, COMM ratio 등)
    for k, v in meta.items():
        feats[f"meta_{k}"] = v

    return feats
