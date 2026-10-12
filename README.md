# halo-closure

Dependency closures for partitioned multi-resolution mesh GNNs: which nodes a
partition must hold for its owned outputs to be computed exactly, for merged
multi-meshes (GraphCast-style), per-layer hierarchy sweeps (Hi-LAM-style, as
this project's model runs them), parallel cascades and V-cycle poolings, on
icosahedral hierarchies from 6/5 down to 6/5/4/3/2/1/0.

This is the evidence behind **"Hops Are Not Distances: The Halo Cost of
Partitioning Multi-Resolution Weather Graph Neural Networks"** (Sim2Science @
NeurIPS 2026), and the closure premise of its companion paper. Everything is a
property of the architecture, not of trained weights: no data, no checkpoints,
a laptop CPU.

## Provenance

The original scripts lived in an untracked scratch directory and were lost;
only their CSV outputs survived. This repository is a rebuild (2026-10-12),
verified two ways:

- `scripts/regenerate.py --check` reproduces **all 8 surviving CSVs exactly**:
  742 rows, every field except wall-clock timing.
- `scripts/validate_autograd.py` matches **every operator against autograd
  (40/40 set equality)**, and recovers the cone sizes the original validation
  reported (256 / 917 / 58 / 425 for the sweep; 695 / 2338 / 173 / 1299 for the
  three-node training cones).

Graph construction and partitioning are vendored, logic unchanged, from
weather-gnn-partition at commit `b68523f`. That repository's
`scripts/plot_halo_closure.py` still carries the superseded operator at HEAD
(the 2026-08-04 correction was never committed there).

## Use

```bash
pip install -r requirements.txt
python scripts/regenerate.py --check    # recompute everything, compare with results/
python scripts/validate_autograd.py     # operators vs autograd, ~1 min
python scripts/plot_figures.py          # redraw the data figures into figures/regenerated/
```

`python scripts/regenerate.py` (without `--check`) rewrites `results/`; the full
set takes about a minute.

## Layout

| path | contents |
|---|---|
| `halo_closure/graphs.py` | icosphere hierarchies, k-nearest cascade edges, lat/lon, METIS and icosahedral-face partitions, grid coupling |
| `halo_closure/operators.py` | backward closure operators for each family, `Rollout`, `saturate` (L_sat) |
| `scripts/regenerate.py` | one function per CSV |
| `scripts/validate_autograd.py` | random-weight probe networks, float64, union over initialisations; includes the double-backward training cone |
| `scripts/plot_figures.py` | the six data figures from `results/` |
| `figures/` | the original figure PDFs the papers used |

## Results

Paper A uses lat/lon and METIS partitioning at 4-32 partitions and each
family's published depth. Files marked *premise* use geometric (icosahedral
face) partitioning, 20 partitions and L = 12, the companion paper's operating
point; they come from the pre-split draft and are not Paper A evidence.

| file | what | where |
|---|---|---|
| `closure_at_published_depth.csv` | % of mesh one rank of 16 must hold; merged L=16, sweep L=2, V-cycle one traversal | Paper A Table 2, Fig. 2 |
| `closure_partition_counts.csv` | L_sat (and V-cycle closure) at 4/8/16/32 partitions, min/median/max over partitions | Paper A Table 2 (L_sat column), Table 3, Figs. 3-4 |
| `closure_vcycle_corrected.csv` | V-cycle closure at 16 partitions | Paper A Table 2 |
| `vcycle_schedules.csv` | V-cycle block allocation (BSMS one pass per level, MS-MGN 1H 11L 1H, 3H 6L 3H 6L 3H), 6/5, 32 partitions | Paper A App. B |
| `reconstruction_check.csv` | radius graph equals the triangulation at multipliers 1.20/1.35/1.50; L_sat unchanged | Paper A App. A |
| `closure_counterfactual.csv` | gapped hierarchies (6/2, 6/4/2, 6/0, 6/3/0) against complete ones | Paper A Sec. 4 |
| `hierarchy_sizes.csv` | nodes and edges per hierarchy, both node-set conventions; nominal and measured spacing | Paper A Sec. 1, 4 |
| `closure_sweep.csv` | *premise*: L_sat and closure at 12 layers, partition 0, three families x three strategies | companion Sec. 3, pre-split draft |
| `closure_growth.csv` | *premise*: closure size at every depth to saturation, geometric partition 0 | pre-split draft |
| `closure_growth_tiers.csv` | *premise*: per-level growth, 6/5/4 sweep (41,040 nodes at L = 12) | companion Sec. 3 |
| `rollout_closure.csv` | *premise*: grid nodes needed for R autoregressive steps without communication | demoted; not in either paper |

The four premise files, and the `lsat_map` and `closure_growth*` figures, carry
the companion paper's operating point and should stay out of anything released
with Paper A.

## Conventions recovered from the original outputs

- Medians over partitions are the **upper median**, `sorted(v)[n // 2]`, except
  `median_owned`, which is `int(numpy.median(...))`. Lat/lon layouts have two
  sizes of partition (polar and equatorial), so the two choices differ there.
- `coarsest_spacing_deg` is a fixed table, not a measurement at every level:
  levels 6, 5, 4 and 0 are measured mean edge lengths, but 3, 2 and 1 are
  4.327 x 2^(4 - level). Measured means for 3, 2, 1 are **8.644, 17.22 and
  33.86 degrees** (`hierarchy_sizes.csv`).
- Single-partition files use partition 0 of geometric (20 faces), lat/lon 4 x 5
  and METIS 20.
- The merged core is every owned hierarchy node mapped onto its finest vertex.
- Face assignment is computed over grid and mesh nodes in one batch, as the
  source pipeline did. The 12 icosahedron corners sit exactly on face
  boundaries, so float rounding decides their face, and that rounding depends
  on the batch shape. It moves one node; nothing depends on it.

## Discrepancies with the accepted Paper A text

Found while rebuilding; candidates for the camera-ready.

- Sec. 4: "Levels 3 through 0 add 858 mesh nodes ... and cut the merged L_sat
  from 21 to 8". Table 2 gives merged L_sat 37 at 6/5/4, so 37 to 8. The 1.6%
  nodes and 3.8% edges describe the separate-node-set graph (cascade edges
  included); on the merged mesh these levels add no nodes and 1.6% of edges.
- Sec. 4: "at 6/5/4 the closure is 13% of the mesh" (V-cycle). Table 2 gives
  20.4% (lat/lon) and 15.6% (METIS); 13% is the 6/5 row.
- Sec. 4: "already at 97% one level earlier". The 97% row is 6/5/4/3, three
  levels short of the complete hierarchy.
- Table 2 and Sec. 3 call the coarsest spacings mean edge lengths; for levels
  3, 2, 1 they are nominal (see above).
