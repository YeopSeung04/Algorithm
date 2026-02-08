import os
import math
import time
import random
import csv
from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, Iterable, List

import numpy as np
import pygame
import networkx as nx
import imageio.v2 as imageio

Vec2 = Tuple[float, float]
Edge = Tuple[int, int]


# -----------------------------
# Graph
# -----------------------------
def make_graph(n: int = 40, p: float = 0.10, seed: int = 7) -> Tuple[nx.Graph, Dict[int, Vec2]]:
    random.seed(seed)
    G = nx.gnp_random_graph(n=n, p=p, seed=seed, directed=False)

    # ensure connected-ish
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        for i in range(len(comps) - 1):
            a = random.choice(list(comps[i]))
            b = random.choice(list(comps[i + 1]))
            G.add_edge(a, b)

    # weights
    for u, v in G.edges():
        G[u][v]["w"] = random.randint(1, 9)

    pos = nx.spring_layout(G, seed=seed, k=1.2 / math.sqrt(max(2, n)))
    return G, {k: (float(v[0]), float(v[1])) for k, v in pos.items()}


def norm_to_rect(pos: Dict[int, Vec2], rect: pygame.Rect, pad: int = 22) -> Dict[int, Tuple[int, int]]:
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    def tx(x: float) -> int:
        if maxx - minx < 1e-9:
            return rect.centerx
        return int(rect.left + pad + (x - minx) * (rect.width - 2 * pad) / (maxx - minx))

    def ty(y: float) -> int:
        if maxy - miny < 1e-9:
            return rect.centery
        return int(rect.top + pad + (y - miny) * (rect.height - 2 * pad) / (maxy - miny))

    return {k: (tx(x), ty(y)) for k, (x, y) in pos.items()}


# -----------------------------
# Common state
# -----------------------------
@dataclass
class StepState:
    title: str
    visited: Set[int]
    frontier: Set[int]
    current: Optional[int]
    path_edges: Set[Edge]
    highlight_edges: Set[Edge]
    dist: Dict[int, float]
    extra: str


def _edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def reconstruct_path(parent: Dict[int, int], goal: int) -> Set[Edge]:
    edges: Set[Edge] = set()
    cur = goal
    while cur in parent:
        p = parent[cur]
        edges.add(_edge(cur, p))
        cur = p
    return edges


# -----------------------------
# Algorithms: BFS / DFS / Dijkstra / A* / Prim
# -----------------------------
def step_bfs(G: nx.Graph, start: int, goal: int) -> Iterable[StepState]:
    from collections import deque
    q = deque([start])
    parent: Dict[int, int] = {}
    visited: Set[int] = {start}
    frontier: Set[int] = {start}

    while q:
        cur = q.popleft()
        frontier.discard(cur)

        yield StepState(
            title="BFS",
            visited=set(visited),
            frontier=set(frontier),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in visited else set(),
            highlight_edges=set(),
            dist={},
            extra="done" if cur == goal else f"pop={cur}"
        )

        if cur == goal:
            break

        for nb in G.neighbors(cur):
            if nb not in visited:
                visited.add(nb)
                parent[nb] = cur
                q.append(nb)
                frontier.add(nb)

    yield StepState(
        title="BFS",
        visited=set(visited),
        frontier=set(),
        current=goal if goal in visited else None,
        path_edges=reconstruct_path(parent, goal) if goal in visited else set(),
        highlight_edges=set(),
        dist={},
        extra="done"
    )


def step_dfs(G: nx.Graph, start: int, goal: int) -> Iterable[StepState]:
    stack = [start]
    parent: Dict[int, int] = {}
    visited: Set[int] = set()
    frontier: Set[int] = {start}

    while stack:
        cur = stack.pop()
        frontier.discard(cur)
        if cur in visited:
            continue
        visited.add(cur)

        yield StepState(
            title="DFS",
            visited=set(visited),
            frontier=set(frontier),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in visited else set(),
            highlight_edges=set(),
            dist={},
            extra="done" if cur == goal else f"pop={cur}"
        )

        if cur == goal:
            break

        nbs = list(G.neighbors(cur))
        random.shuffle(nbs)
        for nb in nbs:
            if nb not in visited:
                parent.setdefault(nb, cur)
                stack.append(nb)
                frontier.add(nb)

    yield StepState(
        title="DFS",
        visited=set(visited),
        frontier=set(),
        current=goal if goal in visited else None,
        path_edges=reconstruct_path(parent, goal) if goal in visited else set(),
        highlight_edges=set(),
        dist={},
        extra="done"
    )


def step_dijkstra(G: nx.Graph, start: int, goal: int) -> Iterable[StepState]:
    import heapq
    dist = {start: 0.0}
    parent: Dict[int, int] = {}
    pq = [(0.0, start)]
    settled: Set[int] = set()
    frontier: Set[int] = {start}

    while pq:
        d, cur = heapq.heappop(pq)
        if cur in settled:
            continue
        settled.add(cur)
        frontier.discard(cur)

        yield StepState(
            title="DIJKSTRA",
            visited=set(settled),
            frontier=set(frontier),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in settled else set(),
            highlight_edges=set(),
            dist=dict(dist),
            extra="done" if cur == goal else f"settle={cur} d={d:.1f}"
        )

        if cur == goal:
            break

        highlight: Set[Edge] = set()
        for nb in G.neighbors(cur):
            w = float(G[cur][nb]["w"])
            nd = d + w
            if nb not in dist or nd < dist[nb]:
                dist[nb] = nd
                parent[nb] = cur
                heapq.heappush(pq, (nd, nb))
                frontier.add(nb)
                highlight.add(_edge(cur, nb))

        yield StepState(
            title="DIJKSTRA",
            visited=set(settled),
            frontier=set(frontier),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in dist else set(),
            highlight_edges=highlight,
            dist=dict(dist),
            extra="relax"
        )

    yield StepState(
        title="DIJKSTRA",
        visited=set(settled),
        frontier=set(),
        current=goal if goal in settled else None,
        path_edges=reconstruct_path(parent, goal) if goal in settled else set(),
        highlight_edges=set(),
        dist=dict(dist),
        extra="done"
    )


def step_astar(G: nx.Graph, pos: Dict[int, Vec2], start: int, goal: int) -> Iterable[StepState]:
    import heapq

    def h(a: int, b: int) -> float:
        ax, ay = pos[a]
        bx, by = pos[b]
        return math.hypot(ax - bx, ay - by)

    g = {start: 0.0}
    f = {start: h(start, goal)}
    parent: Dict[int, int] = {}
    open_pq = [(f[start], start)]
    open_set: Set[int] = {start}
    closed: Set[int] = set()

    while open_pq:
        _, cur = heapq.heappop(open_pq)
        if cur in closed:
            continue
        open_set.discard(cur)
        closed.add(cur)

        yield StepState(
            title="A*",
            visited=set(closed),
            frontier=set(open_set),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in closed else reconstruct_path(parent, cur),
            highlight_edges=set(),
            dist=dict(g),
            extra="done" if cur == goal else f"pick={cur} g={g[cur]:.1f}"
        )

        if cur == goal:
            break

        highlight: Set[Edge] = set()
        for nb in G.neighbors(cur):
            w = float(G[cur][nb]["w"])
            tentative = g[cur] + w
            if nb in closed and tentative >= g.get(nb, float("inf")):
                continue
            if tentative < g.get(nb, float("inf")):
                parent[nb] = cur
                g[nb] = tentative
                f[nb] = tentative + h(nb, goal)
                heapq.heappush(open_pq, (f[nb], nb))
                open_set.add(nb)
                highlight.add(_edge(cur, nb))

        yield StepState(
            title="A*",
            visited=set(closed),
            frontier=set(open_set),
            current=cur,
            path_edges=reconstruct_path(parent, goal) if goal in g else reconstruct_path(parent, cur),
            highlight_edges=highlight,
            dist=dict(g),
            extra="expand"
        )

    yield StepState(
        title="A*",
        visited=set(closed),
        frontier=set(),
        current=goal if goal in closed else None,
        path_edges=reconstruct_path(parent, goal) if goal in closed else set(),
        highlight_edges=set(),
        dist=dict(g),
        extra="done"
    )


def step_prim(G: nx.Graph, start: int) -> Iterable[StepState]:
    import heapq
    in_mst: Set[int] = {start}
    mst_edges: Set[Edge] = set()
    pq: List[Tuple[float, int, int]] = []

    for nb in G.neighbors(start):
        heapq.heappush(pq, (float(G[start][nb]["w"]), start, nb))

    while pq and len(in_mst) < G.number_of_nodes():
        w, u, v = heapq.heappop(pq)
        if v in in_mst:
            continue

        in_mst.add(v)
        e = _edge(u, v)
        mst_edges.add(e)

        yield StepState(
            title="PRIM",
            visited=set(in_mst),
            frontier=set(),
            current=v,
            path_edges=set(mst_edges),
            highlight_edges={e},
            dist={},
            extra=f"add {u}-{v} (w={w:.0f})"
        )

        for nb in G.neighbors(v):
            if nb not in in_mst:
                heapq.heappush(pq, (float(G[v][nb]["w"]), v, nb))

    yield StepState(
        title="PRIM",
        visited=set(in_mst),
        frontier=set(),
        current=None,
        path_edges=set(mst_edges),
        highlight_edges=set(),
        dist={},
        extra="done"
    )


# -----------------------------
# Genetic Algorithm: TSP-ish on node positions
# -----------------------------
def ga_distance_matrix(pos: Dict[int, Vec2], nodes: List[int]) -> np.ndarray:
    n = len(nodes)
    coords = np.array([pos[i] for i in nodes], dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        D[i] = np.sqrt(dx * dx + dy * dy)
    return D


def ga_tour_length(tour: np.ndarray, D: np.ndarray) -> float:
    # closed tour
    return float(np.sum(D[tour, np.roll(tour, -1)]))


def ox_crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(a)
    i, j = sorted(random.sample(range(n), 2))
    child = np.full(n, -1, dtype=np.int32)
    child[i:j] = a[i:j]
    used = set(child[i:j].tolist())

    idx = j % n
    for k in range(n):
        cand = b[(j + k) % n]
        if int(cand) not in used:
            child[idx] = cand
            used.add(int(cand))
            idx = (idx + 1) % n

    # safety
    if (child == -1).any():
        missing = [x for x in a.tolist() if x not in used]
        for t in range(n):
            if child[t] == -1:
                child[t] = missing.pop(0)
    return child


def mutate_swap(tour: np.ndarray, rate: float = 0.15) -> None:
    if random.random() < rate:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]


def step_genetic_tsp(
    pos: Dict[int, Vec2],
    max_gen: int = 2000,
    pop_size: int = 80,
    elite: int = 6,
    mut_rate: float = 0.20,
    seed: int = 7,
) -> Iterable[StepState]:
    random.seed(seed)
    np.random.seed(seed)

    nodes = sorted(pos.keys())
    n = len(nodes)
    if n < 4:
        # trivial
        yield StepState("GA", set(), set(), None, set(), set(), {}, "done")
        return

    D = ga_distance_matrix(pos, nodes)

    def rand_tour():
        t = np.arange(n, dtype=np.int32)
        np.random.shuffle(t)
        return t

    pop = [rand_tour() for _ in range(pop_size)]
    best_len = float("inf")
    best_tour = pop[0].copy()

    for gen in range(max_gen):
        lens = np.array([ga_tour_length(t, D) for t in pop], dtype=np.float64)
        order = np.argsort(lens)
        pop = [pop[i] for i in order.tolist()]
        lens = lens[order]

        if float(lens[0]) < best_len:
            best_len = float(lens[0])
            best_tour = pop[0].copy()

        # build edges for best tour
        tour_nodes = [nodes[int(i)] for i in best_tour.tolist()]
        path_edges: Set[Edge] = set()
        for i in range(n):
            u = tour_nodes[i]
            v = tour_nodes[(i + 1) % n]
            path_edges.add(_edge(u, v))

        yield StepState(
            title="GA",
            visited=set(),            # not meaningful
            frontier=set(),           # not meaningful
            current=None,
            path_edges=path_edges,
            highlight_edges=set(),
            dist={},
            extra=f"gen={gen} best={best_len:.3f}"
        )

        # selection: keep elite
        new_pop = [pop[i].copy() for i in range(min(elite, pop_size))]

        # tournament selection
        def pick_parent() -> np.ndarray:
            k = 4
            cand = random.sample(range(pop_size), k)
            best = min(cand, key=lambda idx: lens[idx])
            return pop[best]

        while len(new_pop) < pop_size:
            p1 = pick_parent()
            p2 = pick_parent()
            child = ox_crossover(p1, p2)
            mutate_swap(child, rate=mut_rate)
            new_pop.append(child)

        pop = new_pop

    # final
    yield StepState(
        title="GA",
        visited=set(),
        frontier=set(),
        current=None,
        path_edges=set(),  # keep last drawn state already shows best
        highlight_edges=set(),
        dist={},
        extra="done"
    )


# -----------------------------
# UI Components
# -----------------------------
class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label
        self.enabled = True

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, is_hover: bool = False):
        bg = (20, 24, 34)
        border = (46, 52, 72)
        if not self.enabled:
            bg = (14, 16, 22)
            border = (30, 34, 46)
        elif is_hover:
            bg = (26, 30, 44)
        pygame.draw.rect(screen, bg, self.rect, border_radius=12)
        pygame.draw.rect(screen, border, self.rect, width=1, border_radius=12)

        txt = font.render(self.label, True, (230, 235, 245) if self.enabled else (120, 125, 135))
        screen.blit(txt, (self.rect.left + 12, self.rect.centery - txt.get_height() // 2))

    def hit(self, pos) -> bool:
        return self.enabled and self.rect.collidepoint(pos)


class Slider:
    def __init__(self, rect: pygame.Rect, vmin: int, vmax: int, value: int, step: int = 1):
        self.rect = rect
        self.vmin = vmin
        self.vmax = vmax
        self.value = value
        self.step = step
        self.dragging = False

    def _value_from_x(self, x: int) -> int:
        x = max(self.rect.left, min(x, self.rect.right))
        t = (x - self.rect.left) / max(1, self.rect.width)
        raw = self.vmin + t * (self.vmax - self.vmin)
        snapped = int(round(raw / self.step) * self.step)
        return max(self.vmin, min(self.vmax, snapped))

    def handle_event(self, event) -> bool:
        changed = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                nv = self._value_from_x(event.pos[0])
                if nv != self.value:
                    self.value = nv
                    changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                nv = self._value_from_x(event.pos[0])
                if nv != self.value:
                    self.value = nv
                    changed = True
        return changed

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, label: str):
        pygame.draw.rect(screen, (18, 20, 28), self.rect, border_radius=10)
        pygame.draw.rect(screen, (46, 52, 72), self.rect, width=1, border_radius=10)

        t = (self.value - self.vmin) / max(1, (self.vmax - self.vmin))
        kx = int(self.rect.left + t * self.rect.width)
        ky = self.rect.centery
        pygame.draw.circle(screen, (150, 220, 255), (kx, ky), 8)
        pygame.draw.circle(screen, (60, 68, 80), (kx, ky), 8, 1)

        txt = font.render(f"{label}: {self.value}", True, (230, 235, 245))
        screen.blit(txt, (self.rect.left, self.rect.top - 22))


# -----------------------------
# Render
# -----------------------------
def draw_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    G: nx.Graph,
    spos: Dict[int, Tuple[int, int]],
    state: StepState,
    start: int,
    goal: int,
    font: pygame.font.Font,
    elapsed_cpu_sec: float,
    steps: int,
    done: bool,
):
    pygame.draw.rect(screen, (10, 12, 18), rect, border_radius=14)
    pygame.draw.rect(screen, (28, 32, 44), rect, width=1, border_radius=14)

    # Background edges: GA는 완전그래프가 아니라 "노드만" 배경이 깔끔함
    if state.title != "GA":
        for u, v in G.edges():
            pygame.draw.line(screen, (22, 26, 34), spos[u], spos[v], 1)

    # highlight edges
    for (u, v) in state.highlight_edges:
        pygame.draw.line(screen, (180, 140, 255), spos[u], spos[v], 3)

    # path edges (final path / mst / GA best tour)
    for (u, v) in state.path_edges:
        if u in spos and v in spos:
            pygame.draw.line(screen, (150, 220, 255), spos[u], spos[v], 3 if state.title == "GA" else 4)

    # nodes
    for n in G.nodes():
        x, y = spos[n]
        r = 6
        if n == start:
            color = (240, 240, 240); r = 8
        elif n == goal and state.title != "GA":  # GA는 goal 개념 없음
            color = (255, 120, 120); r = 8
        elif state.current == n:
            color = (255, 215, 120); r = 9
        elif n in state.frontier:
            color = (120, 255, 170)
        elif n in state.visited:
            color = (90, 160, 255)
        else:
            color = (60, 68, 80)
        pygame.draw.circle(screen, color, (x, y), r)

    # left header
    title = f"{state.title}   |   {state.extra}"
    t = font.render(title, True, (230, 235, 245))
    screen.blit(t, (rect.left + 14, rect.top + 10))

    # right header (steps + cpu time)
    time_str = f"{elapsed_cpu_sec:7.4f}s"
    step_str = f"steps={steps:4d}"
    right = f"{step_str}  {time_str}"
    col = (120, 255, 170) if done else (180, 180, 200)
    rt = font.render(right, True, col)
    screen.blit(rt, (rect.right - rt.get_width() - 14, rect.top + 10))


def surface_to_frame(screen: pygame.Surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(screen)
    return np.transpose(arr, (1, 0, 2))


# -----------------------------
# Benchmark + CSV
# -----------------------------
def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / max(1, (len(xs) - 1))
    return m, math.sqrt(v)


def run_benchmark(
    runs: int,
    base_seed: int,
    node_n: int,
    edge_p: float,
    ga_gens: int,
    ga_pop: int,
    ga_elite: int,
    ga_mut: float,
) -> List[Dict[str, object]]:
    """
    렌더링 없이 알고리즘만 돌려서 'pure algorithm time' 측정.
    CSV row 리스트로 반환.
    """
    results: List[Dict[str, object]] = []

    algo_names = ["DIJKSTRA", "BFS", "A*", "PRIM", "DFS", "GA"]

    for i in range(runs):
        seed = base_seed + i
        G, pos = make_graph(n=node_n, p=edge_p, seed=seed)
        nodes = list(G.nodes())
        start, goal = 0, max(nodes)

        steppers = {
            "DIJKSTRA": iter(step_dijkstra(G, start, goal)),
            "BFS": iter(step_bfs(G, start, goal)),
            "A*": iter(step_astar(G, pos, start, goal)),
            "PRIM": iter(step_prim(G, start)),
            "DFS": iter(step_dfs(G, start, goal)),
            "GA": iter(step_genetic_tsp(pos, max_gen=ga_gens, pop_size=ga_pop, elite=ga_elite, mut_rate=ga_mut, seed=seed)),
        }

        for algo in algo_names:
            steps = 0
            cpu = 0.0
            done = False
            it = steppers[algo]

            while True:
                t0 = time.perf_counter()
                try:
                    s = next(it)
                    cpu += (time.perf_counter() - t0)
                    steps += 1
                    if s.extra == "done":
                        done = True
                        break
                except StopIteration:
                    cpu += (time.perf_counter() - t0)
                    done = True
                    break

            results.append({
                "seed": seed,
                "nodes": node_n,
                "edge_p": edge_p,
                "algo": algo,
                "cpu_sec": cpu,
                "steps": steps,
                "done": int(done),
                "ga_gens": ga_gens,
                "ga_pop": ga_pop,
                "ga_elite": ga_elite,
                "ga_mut": ga_mut,
            })

    return results


def save_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Main App
# -----------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # window
    W, H = 1500, 1750
    sidebar_w = 320
    margin = 26
    gap = 18

    pygame.init()
    pygame.display.set_caption("Algorithm Visual Suite (6 Panels + Sidebar + Recorder + Benchmark)")
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo", 18)
    font_sm = pygame.font.SysFont("Menlo", 16)

    # mac IME 안정
    pygame.key.start_text_input()

    # parameters
    seed = 7
    node_n = 40
    edge_p = 0.10

    # GA parameters (UI 확장하고 싶으면 슬라이더로 만들면 됨)
    ga_gens = 800
    ga_pop = 80
    ga_elite = 6
    ga_mut = 0.20

    # layout: 6 panels stacked
    def build_layout_rects() -> List[pygame.Rect]:
        left_w = W - sidebar_w - 3 * margin
        panel_h = (H - 2 * margin - 5 * gap) // 6
        rects_: List[pygame.Rect] = []
        y = margin
        x = margin
        for _ in range(6):
            rects_.append(pygame.Rect(x, y, left_w, panel_h))
            y += panel_h + gap
        return rects_

    rects = build_layout_rects()

    # globals
    G: nx.Graph
    pos: Dict[int, Vec2]
    spos_list: List[Dict[int, Tuple[int, int]]]
    start: int
    goal: int

    order = ["DIJKSTRA", "BFS", "A*", "PRIM", "DFS", "GA"]

    # algorithm runtime stats (pure algorithm cpu time)
    algo_cpu = {k: 0.0 for k in order}
    algo_steps = {k: 0 for k in order}
    algo_done = {k: False for k in order}

    # benchmark cache (for Save CSV)
    bench_rows: List[Dict[str, object]] = []

    def init_graph():
        nonlocal G, pos, spos_list, start, goal
        G, pos = make_graph(n=node_n, p=edge_p, seed=seed)
        nodes = list(G.nodes())
        start, goal = 0, max(nodes)
        # map positions into each rect
        spos_list = [norm_to_rect(pos, r, pad=34) for r in rects]

    def build_steppers():
        return {
            "DIJKSTRA": iter(step_dijkstra(G, start, goal)),
            "BFS": iter(step_bfs(G, start, goal)),
            "A*": iter(step_astar(G, pos, start, goal)),
            "PRIM": iter(step_prim(G, start)),
            "DFS": iter(step_dfs(G, start, goal)),
            "GA": iter(step_genetic_tsp(pos, max_gen=ga_gens, pop_size=ga_pop, elite=ga_elite, mut_rate=ga_mut, seed=seed)),
        }

    def reset_stats():
        for k in order:
            algo_cpu[k] = 0.0
            algo_steps[k] = 0
            algo_done[k] = False

    def reset_all():
        nonlocal steppers, states
        steppers = build_steppers()
        states = {k: next(it) for k, it in steppers.items()}
        reset_stats()

    def new_graph(randomize_all: bool = True):
        nonlocal seed, edge_p
        seed += 1
        if randomize_all:
            edge_p = max(0.06, min(0.18, edge_p + random.uniform(-0.03, 0.03)))
        init_graph()
        reset_all()

    # initialize
    steppers: Dict[str, Iterable[StepState]]
    states: Dict[str, StepState]
    init_graph()
    reset_all()

    auto = False
    auto_fps = 8
    accum = 0.0

    # recorder
    recording = False
    writer = None
    out_path = None
    record_fps = 60

    def rec_start():
        nonlocal recording, writer, out_path
        if recording:
            return
        out_path = os.path.join(script_dir, f"record_suite_seed{seed}_n{node_n}.mp4")
        writer = imageio.get_writer(out_path, fps=record_fps, codec="libx264", quality=8)
        recording = True
        print(f"[REC ON] -> {out_path}")

    def rec_stop_and_save():
        nonlocal recording, writer, out_path
        if not recording:
            return
        if writer is not None:
            writer.close()
            writer = None
        recording = False
        print(f"[REC OFF] saved -> {out_path}")

    def save_png():
        png_path = os.path.join(script_dir, f"snapshot_suite_seed{seed}_n{node_n}.png")
        pygame.image.save(screen, png_path)
        print(f"[PNG] saved -> {png_path}")

    def step_all():
        """
        한 스텝씩 진행.
        각 알고리즘의 '순수 계산 시간'만 누적 (AUTO 속도 영향 제거).
        """
        for k in order:
            if algo_done[k]:
                continue

            t0 = time.perf_counter()
            try:
                states[k] = next(steppers[k])
                algo_cpu[k] += (time.perf_counter() - t0)
                algo_steps[k] += 1
                if states[k].extra == "done":
                    algo_done[k] = True
            except StopIteration:
                algo_cpu[k] += (time.perf_counter() - t0)
                algo_done[k] = True

    def do_benchmark():
        """
        여러 seed 자동 반복.
        평균/표준편차 출력 + bench_rows 채움.
        """
        nonlocal bench_rows
        runs = 20
        base = seed  # 현재 seed부터
        print(f"[BENCH] runs={runs} base_seed={base} nodes={node_n} p={edge_p:.3f} GA(gens={ga_gens},pop={ga_pop})")

        bench_rows = run_benchmark(
            runs=runs,
            base_seed=base,
            node_n=node_n,
            edge_p=edge_p,
            ga_gens=ga_gens,
            ga_pop=ga_pop,
            ga_elite=ga_elite,
            ga_mut=ga_mut,
        )

        # summary
        for algo in order:
            xs = [float(r["cpu_sec"]) for r in bench_rows if r["algo"] == algo]
            ss = [int(r["steps"]) for r in bench_rows if r["algo"] == algo]
            m, s = mean_std(xs)
            sm, ssig = mean_std([float(x) for x in ss])
            print(f"  {algo:8s}  cpu_mean={m:.6f}s  cpu_std={s:.6f}s   steps_mean={sm:.2f}  steps_std={ssig:.2f}")

        print("[BENCH] done. (Use 'Save CSV' to write file)")

    def save_bench_csv():
        if not bench_rows:
            print("[CSV] No benchmark data. Run Benchmark first.")
            return
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(script_dir, f"bench_{ts}_n{node_n}_p{edge_p:.3f}.csv")
        save_csv(path, bench_rows)
        print(f"[CSV] saved -> {path}")

    # Sidebar UI layout
    sidebar_rect = pygame.Rect(W - sidebar_w - margin, margin, sidebar_w, H - 2 * margin)

    slider_nodes = Slider(
        rect=pygame.Rect(sidebar_rect.left + 18, sidebar_rect.top + 70, sidebar_rect.width - 36, 16),
        vmin=10, vmax=140, value=node_n, step=1
    )

    # buttons
    btn_w = sidebar_rect.width - 36
    bx = sidebar_rect.left + 18
    by = sidebar_rect.top + 120
    bh = 44
    b_gap = 12

    btn_step = Button(pygame.Rect(bx, by + 0 * (bh + b_gap), btn_w, bh), "STEP (Space)")
    btn_auto = Button(pygame.Rect(bx, by + 1 * (bh + b_gap), btn_w, bh), "AUTO Toggle (Enter)")
    btn_new = Button(pygame.Rect(bx, by + 2 * (bh + b_gap), btn_w, bh), "Random Graph (N)")
    btn_rec = Button(pygame.Rect(bx, by + 3 * (bh + b_gap), btn_w, bh), "REC Start (R)")
    btn_save = Button(pygame.Rect(bx, by + 4 * (bh + b_gap), btn_w, bh), "REC Stop + Save")
    btn_png = Button(pygame.Rect(bx, by + 5 * (bh + b_gap), btn_w, bh), "Save PNG Snapshot")
    btn_bench = Button(pygame.Rect(bx, by + 6 * (bh + b_gap), btn_w, bh), "Benchmark (20 runs)")
    btn_csv = Button(pygame.Rect(bx, by + 7 * (bh + b_gap), btn_w, bh), "Save CSV (bench)")

    status_msg = ""
    status_t = 0.0

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        mouse = pygame.mouse.get_pos()
        clicked = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # slider
            if slider_nodes.handle_event(event):
                node_n = slider_nodes.value
                new_graph(randomize_all=True)
                status_msg = f"Nodes: {node_n}"
                status_t = 2.0

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                clicked = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    step_all()
                elif event.key == pygame.K_RETURN:
                    auto = not auto

            # mac IME 대응: TEXTINPUT로도 받기
            if event.type == pygame.TEXTINPUT:
                ch = event.text.lower()
                if ch == "n":
                    new_graph(randomize_all=True)
                elif ch == "r":
                    if recording:
                        rec_stop_and_save()
                    else:
                        rec_start()

        # sidebar button click
        if clicked:
            if btn_step.hit(mouse):
                step_all()
            elif btn_auto.hit(mouse):
                auto = not auto
            elif btn_new.hit(mouse):
                new_graph(randomize_all=True)
            elif btn_rec.hit(mouse):
                rec_start()
            elif btn_save.hit(mouse):
                rec_stop_and_save()
            elif btn_png.hit(mouse):
                save_png()
            elif btn_bench.hit(mouse):
                do_benchmark()
                status_msg = "Benchmark complete (see console)"
                status_t = 2.0
            elif btn_csv.hit(mouse):
                save_bench_csv()
                status_msg = "CSV saved (see console)"
                status_t = 2.0

        # auto stepping
        if auto:
            accum += dt
            if accum >= 1.0 / auto_fps:
                accum = 0.0
                step_all()

        # draw
        screen.fill((6, 8, 12))

        # panels
        for i, k in enumerate(order):
            draw_panel(
                screen=screen,
                rect=rects[i],
                G=G,
                spos=spos_list[i],
                state=states[k],
                start=start,
                goal=goal,
                font=font,
                elapsed_cpu_sec=algo_cpu[k],
                steps=algo_steps[k],
                done=algo_done[k],
            )

        # sidebar
        pygame.draw.rect(screen, (10, 12, 18), sidebar_rect, border_radius=16)
        pygame.draw.rect(screen, (28, 32, 44), sidebar_rect, width=1, border_radius=16)

        title = font.render("Controls", True, (230, 235, 245))
        screen.blit(title, (sidebar_rect.left + 18, sidebar_rect.top + 16))

        slider_nodes.draw(screen, font_sm, "Nodes")

        st1 = f"AUTO: {'ON' if auto else 'OFF'}"
        st2 = f"REC: {'ON' if recording else 'OFF'}"
        s1 = font_sm.render(st1, True, (150, 155, 165))
        s2 = font_sm.render(st2, True, (255, 120, 120) if recording else (150, 155, 165))
        screen.blit(s1, (sidebar_rect.left + 18, sidebar_rect.top + 90))
        screen.blit(s2, (sidebar_rect.left + 18 + 150, sidebar_rect.top + 90))

        def draw_btn(b: Button):
            b.draw(screen, font, is_hover=b.rect.collidepoint(mouse))

        draw_btn(btn_step)
        draw_btn(btn_auto)
        draw_btn(btn_new)
        draw_btn(btn_rec)
        draw_btn(btn_save)
        draw_btn(btn_png)
        draw_btn(btn_bench)
        draw_btn(btn_csv)

        hint = "SPACE(step)  ENTER(auto)  N(new)  R(rec toggle)  ESC(quit)"
        hsurf = font_sm.render(hint, True, (150, 155, 165))
        screen.blit(hsurf, (margin, H - margin - 18))

        if status_t > 0:
            status_t -= dt
            msg = font_sm.render(status_msg, True, (230, 235, 245))
            screen.blit(msg, (sidebar_rect.left + 18, sidebar_rect.bottom - 32))

        pygame.display.flip()

        # record frame
        if recording and writer is not None:
            writer.append_data(surface_to_frame(screen))

    if writer is not None:
        writer.close()
        print(f"[REC END] saved -> {out_path}")

    pygame.quit()


if __name__ == "__main__":
    main()
