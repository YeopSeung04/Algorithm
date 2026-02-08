# shortest_path_algs.py
from __future__ import annotations
import math
import time
import heapq
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Iterable

import networkx as nx

Vec2 = Tuple[float, float]


@dataclass
class RunStats:
    cpu_sec: float
    expansions: int      # settled/popped-finalized nodes
    relaxations: int     # successful relax operations
    path_cost: float
    found: bool


def _w(G: nx.Graph, u: int, v: int) -> float:
    return float(G[u][v].get("w", 1.0))


def dijkstra(G: nx.Graph, s: int, t: int) -> RunStats:
    t0 = time.perf_counter()

    dist: Dict[int, float] = {s: 0.0}
    parent: Dict[int, int] = {}
    pq = [(0.0, s)]
    settled = set()
    expansions = 0
    relax = 0

    while pq:
        d, u = heapq.heappop(pq)
        if u in settled:
            continue
        settled.add(u)
        expansions += 1

        if u == t:
            break

        for v in G.neighbors(u):
            nd = d + _w(G, u, v)
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
                relax += 1

    cpu = time.perf_counter() - t0
    if t in dist:
        return RunStats(cpu, expansions, relax, dist[t], True)
    return RunStats(cpu, expansions, relax, float("inf"), False)


def astar(G: nx.Graph, s: int, t: int, pos: Optional[Dict[int, Vec2]] = None, h_scale: float = 1.0) -> RunStats:
    """
    A* with admissible heuristic if h_scale<=1 and pos is consistent Euclidean-like.
    If pos is None, heuristic is 0 (=> Dijkstra behavior + overhead).
    """
    t0 = time.perf_counter()

    def h(u: int) -> float:
        if not pos:
            return 0.0
        (x1, y1) = pos[u]
        (x2, y2) = pos[t]
        return h_scale * math.hypot(x1 - x2, y1 - y2)

    g: Dict[int, float] = {s: 0.0}
    pq = [(h(s), s)]
    open_best_f: Dict[int, float] = {s: h(s)}
    closed = set()
    expansions = 0
    relax = 0

    while pq:
        f_u, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1

        if u == t:
            break

        gu = g[u]
        for v in G.neighbors(u):
            nd = gu + _w(G, u, v)
            if nd < g.get(v, float("inf")):
                g[v] = nd
                fv = nd + h(v)
                open_best_f[v] = fv
                heapq.heappush(pq, (fv, v))
                relax += 1

    cpu = time.perf_counter() - t0
    if t in g:
        return RunStats(cpu, expansions, relax, g[t], True)
    return RunStats(cpu, expansions, relax, float("inf"), False)


def bidirectional_dijkstra(G: nx.Graph, s: int, t: int) -> RunStats:
    """
    Bidirectional Dijkstra (nonnegative weights).
    Stops when min frontier sums exceed best found.
    """
    t0 = time.perf_counter()

    dist_f: Dict[int, float] = {s: 0.0}
    dist_b: Dict[int, float] = {t: 0.0}
    pq_f = [(0.0, s)]
    pq_b = [(0.0, t)]
    settled_f = set()
    settled_b = set()

    best = float("inf")
    meet = None

    expansions = 0
    relax = 0

    while pq_f and pq_b:
        # termination check (valid for nonnegative weights)
        if pq_f[0][0] + pq_b[0][0] >= best:
            break

        # expand forward
        df, u = heapq.heappop(pq_f)
        if u not in settled_f:
            settled_f.add(u)
            expansions += 1
            if u in dist_b:
                cand = df + dist_b[u]
                if cand < best:
                    best = cand
                    meet = u
            for v in G.neighbors(u):
                nd = df + _w(G, u, v)
                if nd < dist_f.get(v, float("inf")):
                    dist_f[v] = nd
                    heapq.heappush(pq_f, (nd, v))
                    relax += 1

        # expand backward
        db, u2 = heapq.heappop(pq_b)
        if u2 not in settled_b:
            settled_b.add(u2)
            expansions += 1
            if u2 in dist_f:
                cand = db + dist_f[u2]
                if cand < best:
                    best = cand
                    meet = u2
            for v in G.neighbors(u2):
                nd = db + _w(G, u2, v)
                if nd < dist_b.get(v, float("inf")):
                    dist_b[v] = nd
                    heapq.heappush(pq_b, (nd, v))
                    relax += 1

    cpu = time.perf_counter() - t0
    if best < float("inf"):
        return RunStats(cpu, expansions, relax, best, True)
    return RunStats(cpu, expansions, relax, float("inf"), False)


# -------------------------
# ALT (A* + Landmarks)
# -------------------------
@dataclass
class AltPreproc:
    landmarks: List[int]
    dist_from: List[Dict[int, float]]  # dist_from[i][v] = d(L_i, v)


def _pick_landmarks(G: nx.Graph, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    if not nodes:
        return []
    # farthest-point-ish: start random, then iteratively pick farthest by dijkstra distance
    L = [rng.choice(nodes)]
    for _ in range(1, k):
        # compute distances from last landmark
        d = nx.single_source_dijkstra_path_length(G, L[-1], weight="w")
        # pick node with max distance among reachable
        far = max(d.items(), key=lambda it: it[1])[0]
        if far in L:
            far = rng.choice(nodes)
        L.append(int(far))
    return L


def alt_preprocess(G: nx.Graph, k_landmarks: int, seed: int) -> AltPreproc:
    landmarks = _pick_landmarks(G, k_landmarks, seed)
    dist_from: List[Dict[int, float]] = []
    for L in landmarks:
        dist = nx.single_source_dijkstra_path_length(G, L, weight="w")
        dist_from.append({int(v): float(d) for v, d in dist.items()})
    return AltPreproc(landmarks=landmarks, dist_from=dist_from)


def alt_astar(G: nx.Graph, s: int, t: int, pre: AltPreproc) -> RunStats:
    """
    ALT heuristic: h(u,t) = max_L |d(L,t) - d(L,u)|   (admissible for undirected, nonnegative)
    """
    t0 = time.perf_counter()

    def h(u: int) -> float:
        best = 0.0
        for distL in pre.dist_from:
            du = distL.get(u, float("inf"))
            dt = distL.get(t, float("inf"))
            if du < float("inf") and dt < float("inf"):
                val = abs(dt - du)
                if val > best:
                    best = val
        return best

    g: Dict[int, float] = {s: 0.0}
    pq = [(h(s), s)]
    closed = set()
    expansions = 0
    relax = 0

    while pq:
        f_u, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expansions += 1

        if u == t:
            break

        gu = g[u]
        for v in G.neighbors(u):
            nd = gu + _w(G, u, v)
            if nd < g.get(v, float("inf")):
                g[v] = nd
                heapq.heappush(pq, (nd + h(v), v))
                relax += 1

    cpu = time.perf_counter() - t0
    if t in g:
        return RunStats(cpu, expansions, relax, g[t], True)
    return RunStats(cpu, expansions, relax, float("inf"), False)
