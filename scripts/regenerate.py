"""Regenerate every closure CSV in results/, or check them against disk.

    python scripts/regenerate.py               # rewrite results/*.csv
    python scripts/regenerate.py --check       # recompute and compare, write nothing
    python scripts/regenerate.py --only closure_sweep rollout_closure

Weight-free graph traversal on a CPU; the full set takes a few minutes.
--check compares every field as text, except the wall-clock `seconds` column of
closure_sweep.csv.

Conventions inherited from the original (lost) scripts, recovered by matching
their outputs:
  * medians over partitions are the UPPER median, sorted(v)[n // 2], except
    `median_owned`, which is int(numpy.median(owned counts));
  * coarsest_spacing_deg is a fixed table (NOMINAL_SPACING), NOT a measurement
    for every level: levels 6, 5, 4 and 0 are measured mean edge lengths, but
    3, 2, 1 are 4.327 * 2**(4 - level). The measured means for 3, 2, 1 are
    8.644, 17.22 and 33.86 degrees; hierarchy_sizes.csv records them;
  * single-partition files (closure_sweep, closure_growth, closure_growth_tiers,
    rollout_closure) use partition 0 of: geometric (20 base faces), lat/lon
    4 x 5 bands, METIS 20 parts;
  * the merged core is every owned hierarchy node mapped onto its finest vertex.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from halo_closure.graphs import (  # noqa: E402
    assign, assign_faces_with_grid, build_hierarchy, grid_edges,
)
from halo_closure.operators import (  # noqa: E402
    Operators, Rollout, block_schedule, iterate, neural_lam_schedule, one_pass_per_level,
    saturate,
)

RESULTS = ROOT / "results"
HIERARCHIES = [[6, 5], [6, 5, 4], [6, 5, 4, 3], [6, 5, 4, 3, 2],
               [6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1, 0]]
GAPPED = [[6, 2], [6, 4, 2], [6, 5, 4, 3, 2], [6, 0], [6, 3, 0],
          [6, 5, 4, 3, 2, 1, 0]]
COUNTS = [4, 8, 16, 32]
BANDS = {4: (2, 2), 8: (2, 4), 16: (4, 4), 20: (4, 5), 32: (4, 8)}
SWEEP_LAYOUTS = [("geometric", 20), ("latlon", 20), ("metis", 20)]


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def ops_for(levels: tuple[int, ...], radius_multiplier: float = 1.35) -> Operators:
    return Operators(build_hierarchy(list(levels), radius_multiplier=radius_multiplier))


@lru_cache(maxsize=None)
def assignment(levels: tuple[int, ...], strategy: str, n: int) -> np.ndarray:
    bands = BANDS[n] if strategy == "latlon" else None
    return assign(ops_for(levels).h, strategy, n, bands)


NOMINAL_SPACING = {6: "1.082", 5: "2.164", 4: "4.327", 3: "8.654", 2: "17.31",
                   1: "34.62", 0: "63.43"}


def spacing(ops: Operators) -> str:
    """The coarsest level's spacing as the original tables report it."""
    return NOMINAL_SPACING[ops.h.levels[-1]]


def measured_spacing(ops: Operators) -> str:
    """Mean same-level edge length of the coarsest level, degrees, 4 s.f."""
    h = ops.h
    coarsest = h.levels[-1]
    xyz = h.xyz[coarsest].double().numpy()
    e = h.same[coarsest] - h.offset[coarsest]
    cos = np.clip((xyz[e[0]] * xyz[e[1]]).sum(axis=1), -1.0, 1.0)
    return f"{np.degrees(np.arccos(cos)).mean():.4g}"


def upper_median(values):
    return sorted(values)[len(values) // 2]


def pct(count: int, total: int, digits: int) -> str:
    return str(round(100.0 * count / total, digits))


def name(levels) -> str:
    return "/".join(map(str, levels))


def core_and_layer(ops: Operators, owned: np.ndarray, family: str):
    if family == "merged":
        return ops.merged_core(np.where(owned)[0]), ops.merged_layer
    layer = {"sequential": ops.sequential_layer, "parallel": ops.parallel_layer}[family]
    return owned.copy(), layer


def l_sat(ops: Operators, owned: np.ndarray, family: str) -> int:
    return saturate(*core_and_layer(ops, owned, family))[0]


# ---------------------------------------------------------------------------
# one function per CSV: each returns (header, rows)
# ---------------------------------------------------------------------------

def closure_sweep():
    """L_sat and closure at 12 layers, partition 0 of each 20-part layout."""
    header = ["hierarchy", "coarsest_level", "coarsest_spacing_deg", "strategy",
              "architecture", "L_sat", "node_set", "closure_at_12", "pct_at_12",
              "seconds"]
    rows = []
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        for strategy, n in SWEEP_LAYOUTS:
            owned = assignment(tuple(levels), strategy, n) == 0
            for family in ("sequential", "parallel", "merged"):
                t0 = time.perf_counter()
                core, layer = core_and_layer(ops, owned, family)
                lsat, _ = saturate(core, layer)
                c12 = int(iterate(core, layer, 12).sum())
                seconds = round(time.perf_counter() - t0, 1)
                rows.append([name(levels), levels[-1], spacing(ops), strategy, family,
                             lsat, core.size, c12, pct(c12, core.size, 1), seconds])
    return header, rows


def closure_growth():
    """Closure size at every depth up to saturation; geometric partition 0."""
    header = ["hierarchy", "coarsest_level", "coarsest_spacing_deg", "architecture",
              "layer", "n_nodes", "node_set", "pct"]
    rows = []
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        owned = assignment(tuple(levels), "geometric", 20) == 0
        for family in ("sequential", "parallel", "merged"):
            core, layer = core_and_layer(ops, owned, family)
            _, sizes = saturate(core, layer)
            for depth, size in enumerate(sizes):
                rows.append([name(levels), levels[-1], spacing(ops), family, depth,
                             size, core.size, pct(size, core.size, 3)])
    return header, rows


def closure_growth_tiers():
    """Per-tier growth of the sequential closure, 6/5/4, geometric partition 0."""
    levels = (6, 5, 4)
    ops = ops_for(levels)
    h = ops.h
    header = ["layer"] + [c for lv in levels for c in (f"L{lv}", f"L{lv}_pct")]
    x = assignment(levels, "geometric", 20) == 0
    rows, depth = [], 0
    while True:
        row = [depth]
        for lv in levels:
            k = int(x[h.level_slice(lv)].sum())
            row += [k, pct(k, h.count[lv], 3)]
        rows.append(row)
        if x.all():
            return header, rows
        x = ops.sequential_layer(x)
        depth += 1


def _partition_sweep():
    """Per-partition L_sat (sequential, merged) and one-traversal V-cycle
    closure for every hierarchy, strategy and partition count."""
    out = {}
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        vcycle = one_pass_per_level(levels)
        for n in COUNTS:
            for strategy in ("latlon", "metis"):
                a = assignment(tuple(levels), strategy, n)
                owned = [a == pid for pid in range(n)]
                out[(name(levels), n, strategy)] = {
                    "owned": [int(o.sum()) for o in owned],
                    "sequential": [l_sat(ops, o, "sequential") for o in owned],
                    "merged": [l_sat(ops, o, "merged") for o in owned],
                    "unet": [100.0 * ops.vcycle(o, vcycle).sum() / ops.h.n for o in owned],
                }
    return out


@lru_cache(maxsize=1)
def partition_sweep():
    return _partition_sweep()


def closure_partition_counts():
    header = ["hierarchy", "coarsest_spacing_deg", "strategy", "bands", "n_partitions",
              "architecture", "node_set", "median_owned", "metric", "value_min",
              "value_median", "value_max"]
    data = partition_sweep()
    rows = []
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        for n in COUNTS:
            for strategy in ("latlon", "metis"):
                d = data[(name(levels), n, strategy)]
                bands = "x".join(map(str, BANDS[n])) if strategy == "latlon" else "-"
                common = [name(levels), spacing(ops), strategy, bands, n]
                owned = int(np.median(d["owned"]))
                for family, node_set in (("sequential", ops.h.n),
                                         ("merged", ops.h.count[levels[0]])):
                    v = d[family]
                    rows.append(common + [family, node_set, owned, "L_sat",
                                          min(v), upper_median(v), max(v)])
                v = d["unet"]
                rows.append(common + ["unet", ops.h.n, owned, "closure_pct_one_traversal",
                                      round(min(v), 1), round(upper_median(v), 1),
                                      round(max(v), 1)])
    return header, rows


def closure_at_published_depth():
    """Fraction of the mesh one rank of 16 must hold, each family at its
    published depth: merged 16 layers, sweep 2, V-cycle one traversal."""
    header = ["hierarchy", "coarsest_level", "coarsest_spacing_deg", "strategy",
              "n_partitions", "architecture", "published_depth", "pct_min",
              "pct_median", "pct_max"]
    rows = []
    n = 16
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        vcycle = one_pass_per_level(levels)
        for strategy in ("latlon", "metis"):
            a = assignment(tuple(levels), strategy, n)
            owned = [a == pid for pid in range(n)]
            results = {
                ("merged", 16): [iterate(*core_and_layer(ops, o, "merged"), 16).mean()
                                 for o in owned],
                ("sequential", 2): [iterate(*core_and_layer(ops, o, "sequential"), 2).mean()
                                    for o in owned],
                ("unet", "1 traversal"): [ops.vcycle(o, vcycle).mean() for o in owned],
            }
            for (family, depth), fracs in results.items():
                v = [100.0 * f for f in fracs]
                rows.append([name(levels), levels[-1], spacing(ops), strategy, n, family,
                             depth, round(min(v), 1), round(upper_median(v), 1),
                             round(max(v), 1)])
    return header, rows


def closure_vcycle_corrected():
    header = ["hierarchy", "strategy", "architecture", "metric", "pct_min",
              "pct_median", "pct_max"]
    data = partition_sweep()
    rows = []
    for levels in HIERARCHIES:
        for strategy in ("latlon", "metis"):
            v = data[(name(levels), 16, strategy)]["unet"]
            rows.append([name(levels), strategy, "unet", "closure_pct",
                         round(min(v), 1), round(upper_median(v), 1), round(max(v), 1)])
    return header, rows


def closure_counterfactual():
    """Gapped hierarchies against complete ones, 16 partitions."""
    header = ["hierarchy", "coarsest_level", "coarsest_spacing_deg", "n_levels",
              "strategy", "n_partitions", "mesh_nodes", "same_level_edges",
              "L_sat_merged", "L_sat_sweep"]
    rows = []
    n = 16
    for levels in GAPPED:
        ops = ops_for(tuple(levels))
        edges = sum(e.shape[1] for e in ops.h.same.values())
        for strategy in ("latlon", "metis"):
            a = assignment(tuple(levels), strategy, n)
            owned = [a == pid for pid in range(n)]
            rows.append([name(levels), levels[-1], spacing(ops), len(levels), strategy, n,
                         ops.h.n, edges,
                         upper_median([l_sat(ops, o, "merged") for o in owned]),
                         upper_median([l_sat(ops, o, "sequential") for o in owned])])
    return header, rows


def rollout_closure(layers: int = 12, max_steps: int = 12):
    """Grid nodes a rank must hold to run R autoregressive steps with no
    communication between them (sequential cascade, geometric partition 0;
    the step operator is `operators.Rollout`). Saturation is counted against
    every grid node, so it is never reached: 8.9% of grid nodes feed no mesh
    node."""
    header = ["hierarchy", "owned_grid", "n_grid", "R_sat", "pct_after_1", "curve"]
    rows = []
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        grid_latlon, g2m, m2g = grid_edges(ops.h)
        n_grid = grid_latlon.shape[0]
        rollout = Rollout(ops, g2m, m2g, n_grid)
        need = assign_faces_with_grid(ops.h)[0] == 0
        curve = [int(need.sum())]
        for _ in range(max_steps):
            need = rollout.step(need, layers)
            curve.append(int(need.sum()))
        r_sat = next((r for r, c in enumerate(curve) if c == n_grid), f">{max_steps}")
        rows.append([name(levels), curve[0], n_grid, r_sat, pct(curve[1], n_grid, 1),
                     " ".join(map(str, curve))])
    return header, rows


# -- files with no surviving original: paper claims that had no artifact -----

def vcycle_schedules():
    """Appendix B: V-cycle block allocation on 6/5 at 32 partitions."""
    header = ["hierarchy", "strategy", "n_partitions", "schedule", "pct_min",
              "pct_median", "pct_max"]
    levels = (6, 5)
    ops = ops_for(levels)
    schedules = {
        "one pass per level": one_pass_per_level(list(levels)),
        "1H 11L 1H": block_schedule(6, 5, [("H", 1), ("L", 11), ("H", 1)]),
        "3H 6L 3H 6L 3H": block_schedule(6, 5, [("H", 3), ("L", 6), ("H", 3),
                                                ("L", 6), ("H", 3)]),
    }
    rows = []
    for strategy in ("latlon", "metis"):
        a = assignment(levels, strategy, 32)
        for label, schedule in schedules.items():
            v = [100.0 * ops.vcycle(a == pid, schedule).mean() for pid in range(32)]
            rows.append([name(levels), strategy, 32, label, round(min(v), 1),
                         round(upper_median(v), 1), round(max(v), 1)])
    return header, rows


def reconstruction_check():
    """Appendix A: the radius graph recovers the icosphere triangulation at
    several radius multipliers, so the merged multi-mesh has no free parameter.
    L_sat is the merged family's, lat/lon 4 x 4."""
    header = ["hierarchy", "radius_multiplier", "same_level_edges",
              "equals_triangulation", "merged_L_sat_median"]
    rows = []
    for levels in ([6, 5, 4], [6, 5, 4, 3, 2, 1, 0]):
        for multiplier in (1.20, 1.35, 1.50):
            ops = ops_for(tuple(levels), multiplier)
            h = ops.h
            equal = True
            for lv in levels:
                tri = trimesh.creation.icosphere(subdivisions=lv).edges_unique
                tri = {tuple(sorted(e)) for e in tri.tolist()}
                local = h.same[lv] - h.offset[lv]
                radius = {tuple(sorted(e)) for e in local.T.tolist()}
                equal &= tri == radius
            a = assignment(tuple(levels), "latlon", 16)
            lsat = upper_median([l_sat(ops, a == pid, "merged") for pid in range(16)])
            rows.append([name(levels), f"{multiplier:.2f}",
                         sum(e.shape[1] for e in h.same.values()), equal, lsat])
    return header, rows


def closure_neural_lam(max_layers: int = 60):
    """neural-lam Hi-LAM / Graph-EFM schedule (the sweep arm of the follow-up
    study): % of mesh one rank of 16 must hold at 1-4 processor layers, and
    L_sat, min/median/max over partitions. Owned nodes at every level, as for
    the other families."""
    header = ["hierarchy", "strategy", "n_partitions", "metric", "value_min",
              "value_median", "value_max"]
    rows = []
    n = 16
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        for strategy in ("latlon", "metis"):
            a = assignment(tuple(levels), strategy, n)
            owned = [a == pid for pid in range(n)]
            for layers in (1, 2, 3, 4):
                schedule = neural_lam_schedule(levels, layers)
                v = [100.0 * ops.vcycle(o, schedule).mean() for o in owned]
                rows.append([name(levels), strategy, n, f"closure_pct_L{layers}",
                             round(min(v), 1), round(upper_median(v), 1), round(max(v), 1)])
            lsat = []
            for o in owned:
                layers = 1
                while not ops.vcycle(o, neural_lam_schedule(levels, layers)).all():
                    layers += 1
                    if layers > max_layers:
                        raise RuntimeError("closure did not saturate")
                lsat.append(layers)
            rows.append([name(levels), strategy, n, "L_sat", min(lsat), upper_median(lsat),
                         max(lsat)])
    return header, rows


def hierarchy_sizes():
    """Node and edge counts per hierarchy, for both node-set conventions."""
    header = ["hierarchy", "nominal_spacing_deg", "measured_spacing_deg", "mesh_nodes",
              "same_level_edges", "cascade_edges", "merged_vertices", "merged_edges"]
    rows = []
    for levels in HIERARCHIES:
        ops = ops_for(tuple(levels))
        h = ops.h
        same = sum(e.shape[1] for e in h.same.values())
        cascade = sum(e.shape[1] for e in h.up.values()) + sum(
            e.shape[1] for e in h.down.values())
        merged_adj, _ = ops.merged
        rows.append([name(levels), spacing(ops), measured_spacing(ops), h.n, same,
                     cascade, merged_adj.shape[0], merged_adj.nnz])
    return header, rows


TABLES = {
    "closure_sweep": closure_sweep,
    "closure_growth": closure_growth,
    "closure_growth_tiers": closure_growth_tiers,
    "closure_partition_counts": closure_partition_counts,
    "closure_at_published_depth": closure_at_published_depth,
    "closure_vcycle_corrected": closure_vcycle_corrected,
    "closure_counterfactual": closure_counterfactual,
    "rollout_closure": rollout_closure,
    "vcycle_schedules": vcycle_schedules,
    "reconstruction_check": reconstruction_check,
    "hierarchy_sizes": hierarchy_sizes,
    "closure_neural_lam": closure_neural_lam,
}
IGNORE_IN_CHECK = {"closure_sweep": {"seconds"}}


def compare(path: Path, header, rows, ignore: set[str]) -> list[str]:
    if not path.exists():
        return [f"{path.name} does not exist"]
    with path.open(newline="") as f:
        old = list(csv.reader(f))
    new = [header] + [[str(c) for c in r] for r in rows]
    if old[0] != new[0]:
        return [f"header differs: {old[0]} vs {new[0]}"]
    if len(old) != len(new):
        return [f"{len(old) - 1} rows on disk, {len(new) - 1} recomputed"]
    keep = [i for i, col in enumerate(header) if col not in ignore]
    problems = []
    for line, (a, b) in enumerate(zip(old[1:], new[1:]), start=2):
        bad = [f"{header[i]}: {a[i]} -> {b[i]}" for i in keep if a[i] != b[i]]
        if bad:
            problems.append(f"line {line}: " + "; ".join(bad))
    return problems


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="recompute and compare against results/, write nothing")
    p.add_argument("--only", nargs="+", choices=sorted(TABLES), default=sorted(TABLES))
    args = p.parse_args()

    failed = False
    for table in args.only:
        t0 = time.perf_counter()
        header, rows = TABLES[table]()
        path = RESULTS / f"{table}.csv"
        elapsed = time.perf_counter() - t0
        if args.check:
            problems = compare(path, header, rows, IGNORE_IN_CHECK.get(table, set()))
            status = "OK" if not problems else f"{len(problems)} MISMATCH(ES)"
            print(f"{table:28s} {len(rows):4d} rows  {elapsed:6.1f}s  {status}")
            for line in problems[:10]:
                print("    " + line)
            failed |= bool(problems)
        else:
            RESULTS.mkdir(exist_ok=True)
            with path.open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(rows)
            print(f"{table:28s} {len(rows):4d} rows  {elapsed:6.1f}s  wrote {path.name}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
