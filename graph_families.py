# graph_families.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import networkx as nx

Vec2 = Tuple[float, float]


@dataclass
class GraphInstance:
    G: nx.Graph
    pos: Optional[Dict[int, Vec2]]  # 좌표 기반 heuristic용. 없으면 ALT로 보완 가능.
    meta: dict


def _ensure_connected(G: nx.Graph, rng: random.Random) -> None:
    if nx.is_connected(G):
        return
    comps = list(nx.connected_components(G))
    for i in range(len(comps) - 1):
        a = rng.choice(list(comps[i]))
        b = rng.choice(list(comps[i + 1]))
        G.add_edge(a, b)


def _assign_weights(G: nx.Graph, weight: str, rng: random.Random) -> None:
    # nonnegative weights only (shortest path assumption)
    if weight == "uniform_1_9":
        for u, v in G.edges():
            G[u][v]["w"] = rng.randint(1, 9)
    elif weight == "uniform_1_1":
        for u, v in G.edges():
            G[u][v]["w"] = 1
    elif weight == "lognormal":
        # heavier tail but clipped
        for u, v in G.edges():
            w = float(np.random.lognormal(mean=0.0, sigma=0.6))
            G[u][v]["w"] = int(max(1, min(20, round(w * 3))))
    else:
        raise ValueError(f"Unknown weight distribution: {weight}")


def make_er(n: int, p: float, seed: int, weight: str) -> GraphInstance:
    rng = random.Random(seed)
    G = nx.gnp_random_graph(n=n, p=p, seed=seed, directed=False)
    _ensure_connected(G, rng)
    _assign_weights(G, weight, rng)
    pos = nx.spring_layout(G, seed=seed, k=1.2 / math.sqrt(max(2, n)))
    pos2 = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return GraphInstance(G=G, pos=pos2, meta={"family": "ER", "n": n, "p": p, "seed": seed, "weight": weight})


def make_ba(n: int, m: int, seed: int, weight: str) -> GraphInstance:
    # Barabasi-Albert (scale-free). m must be < n
    rng = random.Random(seed)
    m = max(1, min(m, n - 1))
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    _ensure_connected(G, rng)
    _assign_weights(G, weight, rng)
    pos = nx.spring_layout(G, seed=seed, k=1.0 / math.sqrt(max(2, n)))
    pos2 = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return GraphInstance(G=G, pos=pos2, meta={"family": "BA", "n": n, "m": m, "seed": seed, "weight": weight})


def make_ws(n: int, k: int, beta: float, seed: int, weight: str) -> GraphInstance:
    # Watts-Strogatz (small-world). k must be even.
    rng = random.Random(seed)
    k = max(2, min(k, n - 1))
    if k % 2 == 1:
        k += 1
    G = nx.watts_strogatz_graph(n=n, k=k, p=beta, seed=seed)
    _ensure_connected(G, rng)
    _assign_weights(G, weight, rng)
    pos = nx.spring_layout(G, seed=seed, k=1.0 / math.sqrt(max(2, n)))
    pos2 = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return GraphInstance(G=G, pos=pos2, meta={"family": "WS", "n": n, "k": k, "beta": beta, "seed": seed, "weight": weight})


def make_grid(side: int, seed: int, weight: str, diag: bool = False) -> GraphInstance:
    # side x side grid => n=side^2
    rng = random.Random(seed)
    G2 = nx.grid_2d_graph(side, side)
    if diag:
        for x in range(side):
            for y in range(side):
                if x + 1 < side and y + 1 < side:
                    G2.add_edge((x, y), (x + 1, y + 1))
                if x + 1 < side and y - 1 >= 0:
                    G2.add_edge((x, y), (x + 1, y - 1))

    # relabel to int nodes
    mapping = {node: i for i, node in enumerate(G2.nodes())}
    G = nx.relabel_nodes(G2, mapping, copy=True)

    _assign_weights(G, weight, rng)

    # perfect grid positions (good for Euclidean heuristic)
    inv = {i: node for node, i in mapping.items()}
    pos = {}
    for i, (x, y) in inv.items():
        pos[int(i)] = (float(x), float(y))

    return GraphInstance(G=G, pos=pos, meta={"family": "GRID", "side": side, "n": side * side, "seed": seed, "weight": weight, "diag": diag})


def make_community(
    n: int, communities: int, p_in: float, p_out: float, seed: int, weight: str
) -> GraphInstance:
    # simple stochastic block model: equal sizes
    rng = random.Random(seed)
    communities = max(2, min(communities, n))
    sizes = [n // communities] * communities
    for i in range(n % communities):
        sizes[i] += 1

    probs = [[p_out for _ in range(communities)] for _ in range(communities)]
    for i in range(communities):
        probs[i][i] = p_in

    G = nx.stochastic_block_model(sizes, probs, seed=seed, directed=False, selfloops=False)
    _ensure_connected(G, rng)
    _assign_weights(G, weight, rng)

    pos = nx.spring_layout(G, seed=seed, k=1.1 / math.sqrt(max(2, n)))
    pos2 = {int(k): (float(v[0]), float(v[1])) for k, v in pos.items()}
    return GraphInstance(G=G, pos=pos2, meta={"family": "COMM", "n": n, "c": communities, "p_in": p_in, "p_out": p_out, "seed": seed, "weight": weight})
