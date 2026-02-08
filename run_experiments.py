# run_experiments.py
from __future__ import annotations
import argparse
import os
import csv
import time
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx

from graph_families import (
    GraphInstance, make_er, make_ba, make_ws, make_grid, make_community
)
from shortest_path_algs import (
    dijkstra, astar, bidirectional_dijkstra,
    alt_preprocess, alt_astar, AltPreproc
)
from graph_features import compute_features


def sample_queries(G: nx.Graph, q: int, seed: int, far_frac: float = 0.3) -> List[Tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    queries: List[Tuple[int, int]] = []
    if len(nodes) < 2:
        return queries

    def rand_pair():
        a, b = rng.sample(nodes, 2)
        return int(a), int(b)

    far_k = int(round(q * far_frac))

    for _ in range(far_k):
        s, _ = rand_pair()
        dist = nx.single_source_dijkstra_path_length(G, s, weight="w")
        if len(dist) < 2:
            continue
        items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        top = items[: max(1, len(items) // 10)]
        t = int(rng.choice(top)[0])
        if s != t:
            queries.append((s, t))

    while len(queries) < q:
        s, t = rand_pair()
        if s != t:
            queries.append((s, t))

    return queries[:q]

def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # ✅ union of all keys (prevents "dict contains fields not in fieldnames")
    field_set = set()
    for r in rows:
        field_set.update(r.keys())

    # 안정적 순서: 자주 보는 컬럼 먼저, 나머지는 알파벳
    preferred = [
        "family", "n", "p", "weight", "seed", "graph_idx", "query_id", "s", "t",
        "algo", "cpu_sec", "expansions", "relaxations",
        "path_cost", "optimal_cost", "ok_optimal",
        "ws_beta", "comm_ratio",
        "n_nodes", "n_edges", "density", "avg_degree", "degree_cv", "clustering",
        "w_mean", "w_std", "diam_est", "aspl_est", "ws_beta", "comm_ratio", "grid_side",
    ]
    fields = []
    for k in preferred:
        if k in field_set:
            fields.append(k)
            field_set.remove(k)
    fields.extend(sorted(field_set))

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def aggregate_ranking(rows: List[Dict], group_keys: List[str]) -> List[Dict]:
    bucket: Dict[Tuple, Dict[str, List[float]]] = {}
    for r in rows:
        g = tuple(r[k] for k in group_keys)
        algo = r["algo"]
        cpu = float(r["cpu_sec"])
        bucket.setdefault(g, {}).setdefault(algo, []).append(cpu)

    out: List[Dict] = []
    for g, amap in bucket.items():
        med = []
        for algo, xs in amap.items():
            m = float(np.median(xs))
            med.append((algo, m))
        med.sort(key=lambda x: x[1])

        for rank, (algo, m) in enumerate(med, start=1):
            row = {k: v for k, v in zip(group_keys, g)}
            row.update({
                "rank": rank,
                "algo": algo,
                "median_cpu_sec": m,
                "n_samples": len(amap[algo]),
            })
            out.append(row)
    return out


def build_graph(family: str, n: int, p: float, seed: int, weight: str,
                ws_beta: float, comm_ratio: float, grid_side: int) -> GraphInstance:
    """
    family별 파라미터 매핑:
    - ER: n,p
    - BA: m ~ p*n/2
    - WS: k ~ p*n, beta = ws_beta (sweep)
    - GRID: side ~ round(sqrt(n))
    - COMM: p_in/p_out ratio = comm_ratio (sweep), base는 p로 scale
    """
    if family == "ER":
        return make_er(n=n, p=p, seed=seed, weight=weight)
    if family == "BA":
        m = max(1, int(round(p * n / 2)))
        return make_ba(n=n, m=m, seed=seed, weight=weight)
    if family == "WS":
        k = max(2, int(round(p * n)))
        beta = ws_beta
        return make_ws(n=n, k=k, beta=beta, seed=seed, weight=weight)
    if family == "GRID":
        side = max(4, int(grid_side))
        return make_grid(side=side, seed=seed, weight=weight, diag=False)
    if family == "COMM":
        c = 4
        # base density p로 p_in을 설정하고 ratio로 p_out을 정함
        p_in = min(0.7, max(0.10, p * 2.0))
        p_out = max(0.005, p_in / max(1.0, comm_ratio))
        return make_community(n=n, communities=c, p_in=p_in, p_out=p_out, seed=seed, weight=weight)

    raise ValueError(f"Unknown family: {family}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--families", type=str, default="ER,BA,WS,GRID,COMM")
    ap.add_argument("--n_list", type=str, default="30,60,120,240")
    ap.add_argument("--p_list", type=str, default="0.06,0.10,0.14")
    ap.add_argument("--weights", type=str, default="uniform_1_9")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=7)

    ap.add_argument("--bidij", action="store_true")
    ap.add_argument("--alt", action="store_true")
    ap.add_argument("--k_landmarks", type=int, default=8)

    # WS beta sweep, COMM ratio sweep
    ap.add_argument("--ws_beta_list", type=str, default="0.01,0.05,0.10,0.30")
    ap.add_argument("--grid_side_list", type=str, default="6,8,12,16")
    ap.add_argument("--comm_ratio_list", type=str, default="3,6,12")

    args = ap.parse_args()

    families = [x.strip() for x in args.families.split(",") if x.strip()]
    n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
    p_list = [float(x.strip()) for x in args.p_list.split(",") if x.strip()]
    weights = [x.strip() for x in args.weights.split(",") if x.strip()]

    ws_betas = [float(x.strip()) for x in args.ws_beta_list.split(",") if x.strip()]
    comm_ratios = [float(x.strip()) for x in args.comm_ratio_list.split(",") if x.strip()]
    grid_sides = [int(x.strip()) for x in args.grid_side_list.split(",") if x.strip()]

    ts = time.strftime("%Y%m%d_%H%M%S")
    # --- tag output names (basic vs extended) ---
    # mode_tag = "extended" if (args.bidij or args.alt) else "basic"
    # 더 구체적으로:
    mode_tag = "extended_bidij_alt" if (args.bidij and args.alt) else ("extended_bidij" if args.bidij else ("extended_alt" if args.alt else "basic"))

    results_path = os.path.join(args.outdir, f"results_{mode_tag}_{ts}.csv")
    ranking_path = os.path.join(args.outdir, f"ranking_{mode_tag}_{ts}.csv")

    rows: List[Dict] = []
    run_id = 0

    for family in families:
        for weight in weights:

            # GRID는 n_list 대신 grid_side_list로 스윕
            if family == "GRID":
                n_iter = [None]  # dummy
                side_iter = grid_sides
            else:
                n_iter = n_list
                side_iter = [0]  # dummy

            for n in n_iter:
                for p in p_list:

                    beta_iter = ws_betas if family == "WS" else [0.10]
                    ratio_iter = comm_ratios if family == "COMM" else [6.0]

                    for ws_beta in beta_iter:
                        for comm_ratio in ratio_iter:
                            for grid_side in side_iter:
                                for r in range(args.runs):
                                    seed = args.seed0 + run_id
                                    run_id += 1

                                    # GRID가 아니면 grid_side는 의미 없음
                                    grid_side_used = grid_side if family == "GRID" else 0

                                    # GRID면 n = side^2, 아니면 n 그대로
                                    n_used = (grid_side_used * grid_side_used) if family == "GRID" else int(n)

                                    inst = build_graph(
                                        family=family, n=n_used, p=p, seed=seed, weight=weight,
                                        ws_beta=ws_beta, comm_ratio=comm_ratio, grid_side=grid_side_used
                                    )

                                    G = inst.G
                                    pos = inst.pos

                                    # compute features once per graph
                                    feats = compute_features(G, inst.meta, seed=seed)

                                    qlist = sample_queries(G, q=args.queries, seed=seed)

                                    alt_pre: Optional[AltPreproc] = None
                                    if args.alt:
                                        alt_pre = alt_preprocess(G, k_landmarks=args.k_landmarks, seed=seed)

                                    for qi, (s, t) in enumerate(qlist):
                                        base = dijkstra(G, s, t)
                                        opt_cost = base.path_cost

                                        def row_common(algo: str, stats, ok_opt: int) -> Dict:
                                            rc = {
                                                "family": family,
                                                # GRID는 n_used로 기록되게 하는 게 깔끔함
                                                "n": inst.meta.get("n", n_used),
                                                "p": round(p, 4),
                                                "weight": weight,
                                                "seed": seed,
                                                "graph_idx": r,
                                                "query_id": qi,
                                                "s": s,
                                                "t": t,
                                                "algo": algo,
                                                "cpu_sec": stats.cpu_sec,
                                                "expansions": stats.expansions,
                                                "relaxations": stats.relaxations,
                                                "path_cost": stats.path_cost,
                                                "optimal_cost": opt_cost,
                                                "ok_optimal": ok_opt,
                                                # sweep params
                                                "ws_beta": ws_beta if family == "WS" else "",
                                                "comm_ratio": comm_ratio if family == "COMM" else "",
                                                # GRID side도 결과에 박아두면 논문 표/필터링이 쉬움
                                                "grid_side": grid_side_used if family == "GRID" else "",
                                            }
                                            rc.update(feats)
                                            return rc

                                        rows.append(row_common("DIJKSTRA", base, 1))

                                        a = astar(G, s, t, pos=pos, h_scale=1.0)
                                        ok_a = int(abs(a.path_cost - opt_cost) < 1e-6) if a.found and base.found else 0
                                        rows.append(row_common("A*", a, ok_a))

                                        if args.bidij:
                                            b = bidirectional_dijkstra(G, s, t)
                                            ok_b = int(
                                                abs(b.path_cost - opt_cost) < 1e-6) if b.found and base.found else 0
                                            rows.append(row_common("BI_DIJKSTRA", b, ok_b))

                                        if args.alt and alt_pre is not None:
                                            aa = alt_astar(G, s, t, alt_pre)
                                            ok_alt = int(
                                                abs(aa.path_cost - opt_cost) < 1e-6) if aa.found and base.found else 0
                                            rows.append(row_common("ALT", aa, ok_alt))

                                    print(
                                        f"[{family}] n={inst.meta.get('n', n_used)} p={p:.3f} w={weight} "
                                        f"beta={ws_beta} ratio={comm_ratio} side={grid_side_used} seed={seed} "
                                        f"done ({len(qlist)} queries)"
                                    )

    write_csv(results_path, rows)
    print(f"[OK] results -> {results_path}")

    # ranking group keys
    group_keys = ["family", "n", "p", "weight", "ws_beta", "comm_ratio", "grid_side"]
    ranking = aggregate_ranking(rows, group_keys=group_keys)
    write_csv(ranking_path, ranking)
    print(f"[OK] ranking -> {ranking_path}")

    by_algo = {}
    for r in rows:
        by_algo.setdefault(r["algo"], []).append(int(r["ok_optimal"]))
    for a, xs in by_algo.items():
        rate = sum(xs) / max(1, len(xs))
        print(f"[sanity] {a}: ok_optimal rate={rate:.3f}")


if __name__ == "__main__":
    main()
