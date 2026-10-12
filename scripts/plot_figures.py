"""Redraw the closure figures from results/*.csv.

    python scripts/plot_figures.py                 # -> figures/regenerated/
    python scripts/plot_figures.py --out figures   # overwrite the committed copies

Paper A (Sim2Science 2026): closure_published (Fig. 2), unet_closure
(App. B), lsat_counts (App. C). Pre-split draft: lsat_map, closure_growth,
closure_growth_arch. The originals are committed in figures/ and are what the
papers used; this script reproduces their data and layout, not their exact
styling. cone_schematic and cone_hierarchy are hand-built illustrations whose
generator was lost, so only the originals exist.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
COARSEST = [5, 4, 3, 2, 1, 0]
COUNTS = [4, 8, 16, 32]
GREYS = {4: "#c4c4c4", 8: "#8c8c8c", 16: "#555555", 32: "#1a1a1a"}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}
FAMILY = {
    "merged": ("#b2182b", "o", "Merged multi-mesh, $L=16$ (GraphCast)"),
    "sequential": ("#2166ac", "s", "Per-layer sweep, $L=2$ (Graph-EFM)"),
    "unet": ("#1b7837", "^", "V-cycle, one traversal (BSMS-GNN)"),
}
ARCH = {"sequential": "#0072b2", "parallel": "#d55e00", "merged": "#009e73"}
DEEPER = "coarsest refinement level in the hierarchy (deeper $\\rightarrow$)"


def rows(name: str) -> list[dict]:
    with (RESULTS / f"{name}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def level_of(hierarchy: str) -> int:
    return int(hierarchy.split("/")[-1])


def level_axis(ax, label=DEEPER):
    ax.set_xlim(5.4, -0.4)
    ax.set_xticks(COARSEST)
    ax.set_xlabel(label)
    ax.grid(alpha=0.25)


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def closure_published(out: Path) -> None:
    data = defaultdict(dict)
    for r in rows("closure_at_published_depth"):
        data[(r["architecture"], r["strategy"])][level_of(r["hierarchy"])] = float(r["pct_median"])
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.axhline(100, color="#555555", ls=":", lw=1)
    ax.text(5.3, 101.5, "whole mesh", fontsize=8, color="#444444")
    ax.axvline(1, color="#999999", ls="-.", lw=1)
    for family, (colour, marker, label) in FAMILY.items():
        for strategy, style in (("latlon", dict(ls="-", lw=2, marker=marker, ms=6)),
                                ("metis", dict(ls=":", lw=1.2))):
            v = data[(family, strategy)]
            ax.plot(COARSEST, [v[k] for k in COARSEST], color=colour,
                    label=label if strategy == "latlon" else None, **style)
    level_axis(ax)
    ax.set_ylim(0, 112)
    ax.set_ylabel("closure: mesh a rank must hold (%)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, out, "closure_published")


def unet_closure(out: Path) -> None:
    data = defaultdict(dict)
    for r in rows("closure_partition_counts"):
        if r["architecture"] == "unet":
            data[(int(r["n_partitions"]), r["strategy"])][level_of(r["hierarchy"])] = float(
                r["value_median"])
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    ax.axhline(100, color="#c0392b", ls="--", lw=1)
    ax.text(0.1, 95, "whole mesh", fontsize=8, color="#c0392b", ha="center")
    for n in COUNTS:
        ax.plot(COARSEST, [data[(n, "latlon")][k] for k in COARSEST], color=GREYS[n],
                marker=MARKERS[n], lw=1.8, label=f"{n} partitions")
        ax.plot(COARSEST, [data[(n, "metis")][k] for k in COARSEST], color=GREYS[n],
                ls=":", lw=1.2)
    level_axis(ax, "coarsest refinement level in the hierarchy")
    ax.set_ylim(0, 108)
    ax.set_ylabel("closure, one traversal (% of mesh)")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, out, "unet_closure")


def lsat_counts(out: Path) -> None:
    data = defaultdict(dict)
    for r in rows("closure_partition_counts"):
        if r["metric"] == "L_sat":
            key = (r["architecture"], int(r["n_partitions"]), r["strategy"])
            data[key][level_of(r["hierarchy"])] = int(r["value_median"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
    panels = (("merged", "Merged multi-mesh (GraphCast)"),
              ("sequential", "Per-layer hierarchy sweep (Graph-EFM)"))
    for ax, (family, title) in zip(axes, panels):
        for n in COUNTS:
            ax.plot(COARSEST, [data[(family, n, "latlon")][k] for k in COARSEST],
                    color=GREYS[n], marker=MARKERS[n], lw=1.8, label=f"{n} partitions")
            ax.plot(COARSEST, [data[(family, n, "metis")][k] for k in COARSEST],
                    color=GREYS[n], ls=":", lw=1.2)
        level_axis(ax)
        ax.set_title(title, fontsize=10)
    axes[0].axhspan(15, 16, color="#b2182b", alpha=0.2)
    axes[0].text(5.35, 17.5, "GraphCast 16, X-MeshGraphNet 15", fontsize=7.5, color="#b2182b")
    axes[1].axhline(2, color="#2166ac", ls="--", lw=1.2)
    axes[1].text(2.2, 3.2, "Graph-EFM 2", fontsize=8, color="#2166ac", ha="center")
    axes[0].plot([], [], color="#555555", ls=":", label="METIS (solid: lat/lon)")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[0].set_ylabel("$L_{\\mathrm{sat}}$  (processor layers)")
    axes[0].set_ylim(0, 78)
    fig.text(0.5, -0.04, "Exact halo decomposition is available only below each curve.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    save(fig, out, "lsat_counts")


def lsat_map(out: Path) -> None:
    data = defaultdict(dict)
    spacing = {}
    for r in rows("closure_sweep"):
        if r["strategy"] == "geometric":
            data[r["architecture"]][level_of(r["hierarchy"])] = int(r["L_sat"])
            spacing[level_of(r["hierarchy"])] = r["coarsest_spacing_deg"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = {"sequential": "cascaded sequential", "parallel": "cascaded parallel",
              "merged": "merged multi-mesh"}
    for arch, colour in ARCH.items():
        ax.plot(COARSEST, [data[arch][k] for k in COARSEST], color=colour, marker="o",
                lw=2.5, ms=6, mec="white", label=labels[arch])
    ax.set_yscale("log", base=2)
    ax.set_yticks([4, 8, 16, 32, 64])
    ax.set_yticklabels(["4", "8", "16", "32", "64"])
    ax.axhline(16, color="#888888", ls=":", lw=1.2)
    ax.text(5.3, 17, "GraphCast-class depth (L = 16)", fontsize=8.5, color="#888888")
    ax.axhline(12, color="#444444", ls="--", lw=1.2)
    ax.text(5.3, 10.6, "our model runs at L = 12", fontsize=8.5, color="#444444")
    ax.set_xlim(5.25, -0.25)
    ax.set_xticks(COARSEST)
    ax.set_xticklabels([f"L{k}\n{spacing[k]}$^\\circ$" for k in COARSEST])
    ax.set_xlabel("coarsest mesh level included  (and its edge spacing)")
    ax.set_ylabel("$L_{\\mathrm{sat}}$  (layers survived)")
    ax.set_title("Exact halo decomposition is available only below each curve", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save(fig, out, "lsat_map")


def _growth():
    curves = defaultdict(list)
    for r in rows("closure_growth"):
        curves[(r["hierarchy"], r["architecture"])].append((int(r["layer"]), float(r["pct"])))
    return curves


def closure_growth(out: Path) -> None:
    curves = _growth()
    hierarchies = list(dict.fromkeys(h for h, _ in curves))
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for i, hier in enumerate(hierarchies):
        layers, pcts = zip(*curves[(hier, "sequential")])
        colour = cmap(0.5 + 0.5 * i / (len(hierarchies) - 1))
        ax.plot(layers, pcts, color=colour, lw=2.5, label=hier)
        ax.plot(layers[-1], pcts[-1], "o", color=colour, mec="white", ms=7)
    ax.axvline(12, color="#444444", ls="--", lw=1.5)
    ax.text(13, 6, "L = 12", fontsize=9, color="#444444")
    ax.set_xlim(0, 42)
    ax.set_ylim(0, 104)
    ax.set_xlabel("processor layers")
    ax.set_ylabel("halo requirement\n(% of mesh a partition must hold)")
    ax.set_title("Deeper hierarchies reach global dependency sooner", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(title="hierarchy", ncol=2, frameon=False, loc="lower right")
    save(fig, out, "closure_growth")


def closure_growth_arch(out: Path) -> None:
    curves = _growth()
    labels = {"sequential": "cascaded sequential (ours)", "parallel": "cascaded parallel",
              "merged": "merged multi-mesh"}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for arch, colour in ARCH.items():
        layers, pcts = zip(*curves[("6/5/4", arch)])
        ax.plot(layers, pcts, color=colour, lw=2.5, label=labels[arch])
    ax.axvline(12, color="#444444", ls="--", lw=1.5)
    ax.text(13, 6, "L = 12", fontsize=9, color="#444444")
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 104)
    ax.set_xlabel("processor layers")
    ax.set_ylabel("halo requirement\n(% of mesh)")
    ax.set_title("Sequential composition, not level merging, drives the cost", loc="left")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    save(fig, out, "closure_growth_arch")


FIGURES = [closure_published, unet_closure, lsat_counts, lsat_map, closure_growth,
           closure_growth_arch]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "regenerated")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        figure(args.out)
        print(f"wrote {args.out / figure.__name__}.pdf/.png")


if __name__ == "__main__":
    main()
