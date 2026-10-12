"""Backward dependency closures: which nodes an owned set depends on.

A halo is exact when it holds every node whose layer-0 state can reach an owned
output. Each operator below inverts one family's forward message-passing
schedule, one stage at a time. All work on boolean membership vectors over the
mesh nodes of a `Hierarchy` (or over the finest vertices, for the merged mesh).

`back(A, x)` is the primitive: for edges u -> v stored as A[u, v], it adds to x
every source u of some destination v already in x. Nodes are never removed:
every update in these models reads the destination's own previous state too.

Families
--------
sequential (sweep, as this project's Hi-LAM-style model runs it): one layer is
    same-level message passing at every level, then a full up-cascade
    (finest -> coarsest), then a full down-cascade back, each stage consuming
    what the previous stage wrote. Backward, one layer is
        lateral( up_back^k( down_back^k( S ) ) ),   k = n_levels - 1
    -- reverse the down-cascade first (expands toward coarser levels), then
    the up-cascade (back toward finer ones), then one same-level hop.
    Validated against autograd to set equality (scripts/validate_autograd.py).
parallel: same-level, up and down edges all read the previous layer's values,
    so each vertical level costs a layer.
merged (GraphCast-style multi-mesh): every level's same-level edges mapped
    onto the shared finest vertices; the closure is the L-hop ball.
vcycle (U-Net / BSMS-style pooling): the hierarchy is traversed once per model
    on a schedule of ("mp", level), ("up", fine, coarse), ("down", coarse, fine)
    stages, inverted in reverse order.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sp

from .graphs import Hierarchy, merged_vertex_map


def adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    """Store edges u -> v as A[u, v]."""
    data = np.ones(edges.shape[1], dtype=np.float32)
    return sp.csr_matrix((data, (edges[0], edges[1])), shape=(n, n))


def back(a: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    return x | (a @ x.astype(np.float32) > 0)


class Operators:
    """Sparse operators for one hierarchy, built once and reused."""

    def __init__(self, h: Hierarchy):
        self.h = h
        n = h.n
        self.k = len(h.levels) - 1
        self.lat_by_level = {lv: adjacency(e, n) for lv, e in h.same.items()}
        self.lateral = sum(self.lat_by_level.values()).tocsr()
        self.up_by_pair = {p: adjacency(e, n) for p, e in h.up.items()}
        self.down_by_pair = {p: adjacency(e, n) for p, e in h.down.items()}
        self.up = sum(self.up_by_pair.values()).tocsr()
        self.down = sum(self.down_by_pair.values()).tocsr()
        self._merged = None

    # -- per-layer backward steps -------------------------------------------

    def sequential_layer(self, x: np.ndarray) -> np.ndarray:
        for _ in range(self.k):
            x = back(self.down, x)
        for _ in range(self.k):
            x = back(self.up, x)
        return back(self.lateral, x)

    def parallel_layer(self, x: np.ndarray) -> np.ndarray:
        xf = x.astype(np.float32)
        return x | (self.lateral @ xf > 0) | (self.down @ xf > 0) | (self.up @ xf > 0)

    # -- merged multi-mesh --------------------------------------------------

    @property
    def merged(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """(adjacency over the finest vertices, hierarchy-node -> vertex map)."""
        if self._merged is None:
            vmap = merged_vertex_map(self.h)
            n_fine = self.h.count[self.h.levels[0]]
            edges = np.concatenate([vmap[e] for e in self.h.same.values()], axis=1)
            self._merged = (adjacency(edges, n_fine), vmap)
        return self._merged

    def merged_core(self, owned: np.ndarray) -> np.ndarray:
        """Owned hierarchy nodes, every level, mapped onto finest vertices."""
        adj, vmap = self.merged
        core = np.zeros(adj.shape[0], dtype=bool)
        core[vmap[owned]] = True
        return core

    def merged_layer(self, x: np.ndarray) -> np.ndarray:
        return back(self.merged[0], x)

    # -- V-cycle ------------------------------------------------------------

    def vcycle(self, x: np.ndarray, schedule: list[tuple]) -> np.ndarray:
        for stage in reversed(schedule):
            kind = stage[0]
            if kind == "mp":
                x = back(self.lat_by_level[stage[1]], x)
            elif kind == "up":
                x = back(self.up_by_pair[(stage[1], stage[2])], x)
            elif kind == "down":
                x = back(self.down_by_pair[(stage[1], stage[2])], x)
            else:
                raise ValueError(f"unknown stage {stage!r}")
        return x

    # -- helpers ------------------------------------------------------------

    def owned_mask(self, assignment: np.ndarray, pid: int) -> np.ndarray:
        return assignment == pid


class Rollout:
    """Grid nodes needed one autoregressive step back, sequential cascade.

    output[g] <- grid_h[g] and mesh[f] for f in m2g_back(g); mesh[f] <- the
    mesh closure of `layers` processor layers seeded at f; only the finest
    level's layer-0 states come from the grid (encode-once g2m), the coarser
    levels' are position-only. So one step back is
        need | g2m_back( closure( m2g_back(need) ) restricted to the finest level )
    """

    def __init__(self, ops: Operators, g2m: np.ndarray, m2g: np.ndarray, n_grid: int):
        h = ops.h
        self.ops = ops
        ones = np.ones(g2m.shape[1], dtype=np.float32)
        self.g2m_back = sp.csr_matrix((ones, (g2m[0], g2m[1])), shape=(n_grid, h.n))
        ones = np.ones(m2g.shape[1], dtype=np.float32)
        self.m2g_back = sp.csr_matrix((ones, (m2g[0], m2g[1])), shape=(h.n, n_grid))
        self.finest = np.zeros(h.n, dtype=bool)
        self.finest[h.level_slice(h.levels[0])] = True

    def step(self, need: np.ndarray, layers: int) -> np.ndarray:
        mesh = self.m2g_back @ need.astype(np.float32) > 0
        mesh = iterate(mesh, self.ops.sequential_layer, layers) & self.finest
        return need | (self.g2m_back @ mesh.astype(np.float32) > 0)


def one_pass_per_level(levels: list[int]) -> list[tuple]:
    """BSMS-GNN's V-cycle: one message pass per level on the way down and
    again on the way up, pooling and unpooling between levels."""
    down_leg = []
    for fine, coarse in zip(levels, levels[1:]):
        down_leg += [("mp", fine), ("up", fine, coarse)]
    up_leg = []
    for fine, coarse in reversed(list(zip(levels, levels[1:]))):
        up_leg += [("down", coarse, fine), ("mp", fine)]
    return down_leg + [("mp", levels[-1])] + up_leg


def block_schedule(fine: int, coarse: int, blocks: list[tuple[str, int]]) -> list[tuple]:
    """Two-level schedule in MS-MGN notation, e.g. [("H", 1), ("L", 11), ("H", 1)]."""
    out, at = [], "H"
    for tier, count in blocks:
        if tier != at:
            out.append(("up", fine, coarse) if tier == "L" else ("down", coarse, fine))
            at = tier
        out += [("mp", fine if tier == "H" else coarse)] * count
    return out


def iterate(x0: np.ndarray, layer: Callable[[np.ndarray], np.ndarray],
            layers: int) -> np.ndarray:
    x = x0.copy()
    for _ in range(layers):
        x = layer(x)
    return x


def saturate(x0: np.ndarray, layer: Callable[[np.ndarray], np.ndarray],
             max_layers: int = 1000) -> tuple[int, list[int]]:
    """L_sat (the first depth at which the closure is every node) and the
    closure size at every depth from 0 to L_sat."""
    x = x0.copy()
    sizes = [int(x.sum())]
    while sizes[-1] < x.size:
        if len(sizes) > max_layers:
            raise RuntimeError("closure did not saturate")
        x = layer(x)
        sizes.append(int(x.sum()))
    return len(sizes) - 1, sizes
