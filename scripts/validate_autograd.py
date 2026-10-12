"""Check every closure operator against automatic differentiation.

Exactness does not depend on the weights, so a network with random weights is
a valid probe: run it, backpropagate from the owned outputs, and read off which
inputs carry a nonzero gradient. That is the true dependency set, and each
operator must reproduce it exactly -- set equality, not containment. The union
over random initialisations guards against an exact cancellation hiding a live
path. Everything is float64: in float32 the larger cones report spurious gaps,
which are autograd underflow at the cone edge, not operator errors.

The probe networks run each family's schedule on the real edge sets, with a
generic message-passing update: a destination's new state is a function of its
own state and of its in-edge sources, which is all the dependency structure
there is. The sweep schedule is this project's model,
weather_gnn.model.hilam.Processor._generic_step: same-level passes at every
level, then the up-cascade, then the down-cascade.

Seeds are node 500 of a level (and 500-502 for three-node owned sets), which
reproduces every cone size reported in the original validation: 256 / 917 at
the finest level and 58 / 425 at the coarsest, at one and two layers, and
695 / 2338 / 173 / 1299 for the three-node training cones. The rollout check's
original seed was not recorded (its reported sizes were 1087 / 3837 / 4094 /
12382); grid node 500 is used instead.

    python scripts/validate_autograd.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halo_closure.graphs import build_hierarchy, grid_edges  # noqa: E402
from halo_closure.operators import (  # noqa: E402
    Operators, Rollout, iterate, neural_lam_schedule, one_pass_per_level,
)

DTYPE = torch.float64
WIDTH = 8
INITS = 8
SEED_NODE = 500


class Update(nn.Module):
    """dst' = tanh(W [dst, sum over in-edges of tanh(V [src, dst])])."""

    def __init__(self):
        super().__init__()
        self.msg = nn.Linear(2 * WIDTH, WIDTH, dtype=DTYPE)
        self.node = nn.Linear(2 * WIDTH, WIDTH, dtype=DTYPE)

    def forward(self, src_h, dst_h, src, dst):
        m = torch.tanh(self.msg(torch.cat([src_h[src], dst_h[dst]], dim=-1)))
        agg = torch.zeros_like(dst_h).index_add(0, dst, m)
        return torch.tanh(self.node(torch.cat([dst_h, agg], dim=-1)))


def replace(state, sl, new):
    return torch.cat([state[:sl.start], new, state[sl.stop:]], dim=0)


def sweep_schedule(levels, layers):
    pairs = list(zip(levels, levels[1:]))
    one = ([("mp", lv) for lv in levels] + [("up", f, c) for f, c in pairs]
           + [("down", c, f) for f, c in reversed(pairs)])
    return one * layers


class ScheduleNet(nn.Module):
    """Runs stages in order on one state tensor over every hierarchy node.
    Stages: ("mp", level), ("up", fine, coarse), ("down", coarse, fine)."""

    def __init__(self, h, schedule):
        super().__init__()
        self.updates = nn.ModuleList(Update() for _ in schedule)
        self.stages = []
        for stage in schedule:
            if stage[0] == "mp":
                edges, dst_level = h.same[stage[1]], stage[1]
            elif stage[0] == "up":
                edges, dst_level = h.up[(stage[1], stage[2])], stage[2]
            else:
                edges, dst_level = h.down[(stage[1], stage[2])], stage[2]
            self.stages.append((torch.from_numpy(edges[0]),
                                torch.from_numpy(edges[1] - h.offset[dst_level]),
                                h.level_slice(dst_level)))

    def forward(self, state):
        for update, (src, dst, sl) in zip(self.updates, self.stages):
            state = replace(state, sl, update(state, state[sl], src, dst))
        return state


class ParallelNet(nn.Module):
    """Every level updates from the previous layer's same-level, finer (up)
    and coarser (down) states at once."""

    def __init__(self, h, layers):
        super().__init__()
        self.h, self.layers = h, layers
        levels = h.levels
        self.updates = nn.ModuleList()
        self.sources = []
        for _ in range(layers):
            for i, lv in enumerate(levels):
                edge_sets = [h.same[lv]]
                if i > 0:
                    edge_sets.append(h.up[(levels[i - 1], lv)])
                if i < len(levels) - 1:
                    edge_sets.append(h.down[(levels[i + 1], lv)])
                for e in edge_sets:
                    self.updates.append(Update())
                    self.sources.append((i, torch.from_numpy(e[0]),
                                         torch.from_numpy(e[1] - h.offset[lv])))

    def forward(self, state):
        h = self.h
        k = 0
        per_layer = len(self.sources) // self.layers
        for _ in range(self.layers):
            new = [None] * len(h.levels)
            for _ in range(per_layer):
                i, src, dst = self.sources[k]
                sl = h.level_slice(h.levels[i])
                part = self.updates[k](state, state[sl], src, dst)
                new[i] = part if new[i] is None else new[i] + part
                k += 1
            state = torch.cat(new, dim=0)
        return state


class MergedNet(nn.Module):
    def __init__(self, adj, layers):
        super().__init__()
        coo = adj.tocoo()
        self.src = torch.from_numpy(coo.row.astype(np.int64))
        self.dst = torch.from_numpy(coo.col.astype(np.int64))
        self.updates = nn.ModuleList(Update() for _ in range(layers))

    def forward(self, state):
        for update in self.updates:
            state = update(state, state, self.src, self.dst)
        return state


class RolloutNet(nn.Module):
    """Encode-once model rolled out autoregressively: grid encoder per node,
    grid -> finest mesh, sweep processor, finest mesh -> grid decoder. Coarser
    levels start from position-only states that do not depend on the input."""

    def __init__(self, h, g2m, m2g, layers, steps):
        super().__init__()
        self.h, self.steps = h, steps
        finest = h.levels[0]
        self.fine = h.level_slice(finest)
        self.g2m_src = torch.from_numpy(g2m[0])
        self.g2m_dst = torch.from_numpy(g2m[1] - h.offset[finest])
        self.m2g_src = torch.from_numpy(m2g[0])
        self.m2g_dst = torch.from_numpy(m2g[1])
        self.enc = nn.Linear(WIDTH, WIDTH, dtype=DTYPE)
        self.g2m, self.m2g = Update(), Update()
        self.proc = ScheduleNet(h, sweep_schedule(h.levels, layers))
        self.register_buffer("pos", torch.randn(h.n, WIDTH, dtype=DTYPE))

    def forward(self, x):
        for _ in range(self.steps):
            grid = torch.tanh(self.enc(x))
            mesh = replace(self.pos, self.fine,
                           self.g2m(grid, self.pos[self.fine], self.g2m_src, self.g2m_dst))
            mesh = self.proc(mesh)
            x = self.m2g(mesh, grid, self.m2g_src, self.m2g_dst)
        return x


def dependency(make_net, n_in, seeds, inits=INITS, training=False):
    """Inputs with nonzero gradient, unioned over random initialisations.
    With training=True also returns the inputs on which the PARAMETER gradient
    depends (double backward through a random contraction)."""
    seeds = torch.as_tensor(seeds)
    inference = np.zeros(n_in, dtype=bool)
    train = np.zeros(n_in, dtype=bool)
    for init in range(inits):
        torch.manual_seed(init)
        net = make_net()
        x = torch.randn(n_in, WIDTH, dtype=DTYPE, requires_grad=True)
        y = net(x)
        loss = (y[seeds] * torch.randn(len(seeds), WIDTH, dtype=DTYPE)).sum()
        if not training:
            (gx,) = torch.autograd.grad(loss, x)
            inference |= (gx.abs().sum(dim=1) > 0).numpy()
            continue
        params = list(net.parameters())
        gx, *gp = torch.autograd.grad(loss, [x] + params, create_graph=True,
                                      allow_unused=True)
        inference |= (gx.detach().abs().sum(dim=1) > 0).numpy()
        contraction = sum((g * torch.randn_like(g)).sum() for g in gp if g is not None)
        (gxx,) = torch.autograd.grad(contraction, x, allow_unused=True)
        if gxx is not None:
            train |= (gxx.abs().sum(dim=1) > 0).numpy()
    return (inference, train) if training else inference


def report(label, operator, truth, results):
    equal = bool(np.array_equal(operator, truth))
    missing = int((truth & ~operator).sum())
    extra = int((operator & ~truth).sum())
    print(f"  {label:44s} operator {int(operator.sum()):6d}  autograd {int(truth.sum()):6d}"
          f"  {'EQUAL' if equal else f'DIFFER (missing {missing}, extra {extra})'}")
    results.append(equal)


def main() -> None:
    t0 = time.perf_counter()
    h = build_hierarchy([6, 5, 4])
    ops = Operators(h)
    results: list[bool] = []

    def mask(nodes, n=h.n):
        x = np.zeros(n, dtype=bool)
        x[nodes] = True
        return x

    seed = {lv: h.offset[lv] + SEED_NODE for lv in h.levels}

    print("sweep (sequential cascade)")
    for lv in (6, 4):
        for layers in (1, 2):
            truth = dependency(lambda: ScheduleNet(h, sweep_schedule(h.levels, layers)),
                               h.n, [seed[lv]])
            op = iterate(mask([seed[lv]]), ops.sequential_layer, layers)
            report(f"L{lv} node {SEED_NODE}, {layers} layer(s)", op, truth, results)

    print("parallel cascade")
    for lv in (6, 4):
        for layers in (1, 2):
            truth = dependency(lambda: ParallelNet(h, layers), h.n, [seed[lv]])
            op = iterate(mask([seed[lv]]), ops.parallel_layer, layers)
            report(f"L{lv} node {SEED_NODE}, {layers} layer(s)", op, truth, results)

    print("merged multi-mesh")
    adj, _ = ops.merged
    n_fine = adj.shape[0]
    for vertex in (0, SEED_NODE):  # vertex 0 is an icosahedron corner: edges at every level
        for layers in (1, 2, 3):
            truth = dependency(lambda: MergedNet(adj, layers), n_fine, [vertex])
            op = iterate(mask([vertex], n_fine), ops.merged_layer, layers)
            report(f"vertex {vertex}, {layers} layer(s)", op, truth, results)

    print("V-cycle, one pass per level")
    schedule = one_pass_per_level(h.levels)
    for lv in h.levels:
        for node in (SEED_NODE, SEED_NODE + 1):
            g = h.offset[lv] + node
            truth = dependency(lambda: ScheduleNet(h, schedule), h.n, [g], inits=6)
            report(f"L{lv} node {node}", ops.vcycle(mask([g]), schedule), truth, results)

    print("neural-lam Hi-LAM schedule (init up, down/up passes, read-out)")
    for layers in (1, 2):
        schedule = neural_lam_schedule(h.levels, layers)
        for lv in (6, 4):
            truth = dependency(lambda: ScheduleNet(h, schedule), h.n, [seed[lv]], inits=6)
            report(f"L{lv} node {SEED_NODE}, {layers} layer(s)",
                   ops.vcycle(mask([seed[lv]]), schedule), truth, results)

    print("training cone = inference cone (double backward, sweep)")
    for lv in (6, 4):
        for size in (1, 3):
            nodes = [seed[lv] + i for i in range(size)]
            for layers in (1, 2):
                inference, train = dependency(
                    lambda: ScheduleNet(h, sweep_schedule(h.levels, layers)),
                    h.n, nodes, inits=6, training=True)
                op = iterate(mask(nodes), ops.sequential_layer, layers)
                label = f"L{lv} {size} node(s), {layers} layer(s)"
                report(label + " inference", op, inference, results)
                report(label + " training", op, train, results)

    print("rollout (grid nodes, encode-once, sweep processor)")
    grid, g2m, m2g = grid_edges(h)
    n_grid = grid.shape[0]
    rollout = Rollout(ops, g2m, m2g, n_grid)
    for steps in (1, 2):
        for layers in (1, 2):
            truth = dependency(lambda: RolloutNet(h, g2m, m2g, layers, steps),
                               n_grid, [SEED_NODE])
            need = mask([SEED_NODE], n_grid)
            for _ in range(steps):
                need = rollout.step(need, layers)
            report(f"grid node {SEED_NODE}, R={steps}, L={layers}", need, truth, results)

    n_ok = sum(results)
    print(f"\n{n_ok}/{len(results)} checks exact  ({time.perf_counter() - t0:.0f}s)")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
