"""Icosahedral mesh hierarchies and the partition assignments the closures use.

Vendored from weather-gnn-partition (weather_gnn.graphs.{mesh,builder,edges,
partition,partition_strategies} at commit b68523f) with the construction logic
unchanged, so this repo runs without the training codebase. The same libraries
give the same node orderings and edge sets; that is checked end to end by
`scripts/regenerate.py --check`, which reproduces every CSV in results/.

Index space: mesh nodes only, level-major, finest level first. (The source
builder also prepends grid nodes; mesh-only closures never touch them, and the
rollout closure builds them separately with `grid_edges`.)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def xyz_to_latlon(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = torch.rad2deg(torch.asin(z.clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(y, x))
    return torch.stack([lat, lon], dim=1)


def latlon_to_xyz(latlon: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(latlon[:, 0])
    lon = torch.deg2rad(latlon[:, 1])
    return torch.stack([
        torch.cos(lat) * torch.cos(lon),
        torch.cos(lat) * torch.sin(lon),
        torch.sin(lat),
    ], dim=1)


def icosphere(level: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit-sphere icosphere at a refinement level: 10 * 4**level + 2 nodes."""
    sphere = trimesh.creation.icosphere(subdivisions=level)
    vertices = sphere.vertices / np.linalg.norm(sphere.vertices, axis=1, keepdims=True)
    xyz = torch.tensor(vertices, dtype=torch.float32)
    return xyz, xyz_to_latlon(xyz)


def mesh_radius(xyz: torch.Tensor, multiplier: float = 1.35) -> float:
    """Same-level edge radius: a multiple of the mean nearest-neighbour distance."""
    distances, _ = cKDTree(xyz.numpy()).query(xyz.numpy(), k=2)
    return float(distances[:, 1].mean() * multiplier)


def radius_edges(xyz: torch.Tensor, radius: float) -> np.ndarray:
    """Undirected radius graph as directed (2, E) pairs, both directions."""
    pairs = cKDTree(xyz.numpy()).query_pairs(r=radius, output_type="ndarray")
    src = pairs[:, 0].astype(np.int64)
    dst = pairs[:, 1].astype(np.int64)
    return np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])


def knn_edges(src_xyz: torch.Tensor, dst_xyz: torch.Tensor, k: int) -> np.ndarray:
    """Each destination node receives from its k nearest source nodes."""
    _, idx = cKDTree(src_xyz.numpy()).query(dst_xyz.numpy(), k=k)
    if k == 1:
        idx = idx.reshape(-1, 1)
    dst = np.repeat(np.arange(dst_xyz.shape[0], dtype=np.int64), k)
    return np.stack([idx.flatten().astype(np.int64), dst])


@dataclass
class Hierarchy:
    """A multi-level icosahedral mesh as separate node sets per level.

    `up[(f, c)]` holds pooling edges fine -> coarse (each coarse node receives
    from its k nearest fine nodes); `down[(c, f)]` holds coarse -> fine edges
    (each fine node receives from its k nearest coarse nodes). Neither is the
    transpose of the other.
    """

    levels: list[int]
    xyz: dict[int, torch.Tensor]
    latlon: torch.Tensor
    offset: dict[int, int]
    count: dict[int, int]
    same: dict[int, np.ndarray]
    up: dict[tuple[int, int], np.ndarray]
    down: dict[tuple[int, int], np.ndarray]

    @property
    def n(self) -> int:
        return sum(self.count.values())

    @property
    def name(self) -> str:
        return "/".join(str(level) for level in self.levels)

    def level_slice(self, level: int) -> slice:
        return slice(self.offset[level], self.offset[level] + self.count[level])


def build_hierarchy(levels: list[int], k: int = 4,
                    radius_multiplier: float = 1.35) -> Hierarchy:
    if levels != sorted(levels, reverse=True) or len(levels) < 2:
        raise ValueError("levels must be >= 2 refinement levels, finest first")
    xyz, latlon, count = {}, {}, {}
    for level in levels:
        xyz[level], latlon[level] = icosphere(level)
        count[level] = xyz[level].shape[0]
    offset, running = {}, 0
    for level in levels:
        offset[level] = running
        running += count[level]
    same = {
        level: radius_edges(xyz[level], mesh_radius(xyz[level], radius_multiplier))
        + offset[level]
        for level in levels
    }
    up, down = {}, {}
    for fine, coarse in zip(levels, levels[1:]):
        e = knn_edges(xyz[fine], xyz[coarse], k)
        up[(fine, coarse)] = np.stack([e[0] + offset[fine], e[1] + offset[coarse]])
        e = knn_edges(xyz[coarse], xyz[fine], k)
        down[(coarse, fine)] = np.stack([e[0] + offset[coarse], e[1] + offset[fine]])
    return Hierarchy(
        levels=list(levels),
        xyz=xyz,
        latlon=torch.cat([latlon[level] for level in levels], dim=0),
        offset=offset,
        count=count,
        same=same,
        up=up,
        down=down,
    )


def merged_vertex_map(h: Hierarchy) -> np.ndarray:
    """Map every hierarchy node onto its co-located finest-level vertex.

    The icosphere levels are nested, so the map is exact: the multi-mesh of a
    GraphCast-style model is the finest vertex set carrying every level's edges.
    """
    finest = h.levels[0]
    tree = cKDTree(h.xyz[finest].numpy())
    out = np.empty(h.n, dtype=np.int64)
    for level in h.levels:
        dist, idx = tree.query(h.xyz[level].numpy(), k=1)
        if dist.max() > 1e-5:
            raise RuntimeError(f"level {level} is not nested in level {finest}")
        out[h.level_slice(level)] = idx
    return out


# ---------------------------------------------------------------------------
# Partition assignments (one partition id per mesh node)
# ---------------------------------------------------------------------------

def assign_latlon(h: Hierarchy, n_lat: int, n_lon: int,
                  lon_offset: float = 0.0) -> np.ndarray:
    """Rectangular lat/lon bins, equal in angle (polar bins are small in area)."""
    lat = h.latlon[:, 0]
    lon = ((h.latlon[:, 1] - lon_offset + 180.0) % 360.0) - 180.0
    lat_edges = torch.linspace(-90, 90, n_lat + 1)
    lon_edges = torch.linspace(-180, 180, n_lon + 1)
    out = torch.zeros(h.n, dtype=torch.long)
    for i in range(n_lat):
        lat_min = lat_edges[i].item()
        lat_max = lat_edges[i + 1].item() if i < n_lat - 1 else 90.0001
        for j in range(n_lon):
            lon_min = lon_edges[j].item()
            lon_max = lon_edges[j + 1].item() if j < n_lon - 1 else 180.0001
            mask = (lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)
            out[mask] = i * n_lon + j
    return out.numpy()


def default_bands(n_parts: int) -> tuple[int, int]:
    n_lat = int(n_parts ** 0.5)
    n_lon = n_parts // n_lat
    if n_lat * n_lon != n_parts:
        raise ValueError(f"pass explicit bands for {n_parts} partitions")
    return n_lat, n_lon


def assign_metis(h: Hierarchy, n_parts: int) -> np.ndarray:
    """METIS k-way (torch-sparse backend, its default seed) over the mesh
    multigraph: every same-level edge plus the up and down cascade edges,
    symmetrised and deduplicated."""
    import torch_sparse  # noqa: F401  registers torch.ops.torch_sparse.partition

    edges = torch.from_numpy(np.concatenate(
        list(h.same.values()) + list(h.up.values()) + list(h.down.values()), axis=1))
    edges = torch.cat([edges, edges.flip(0)], dim=1)
    keys = torch.unique(edges[0] * h.n + edges[1])
    row, col = keys // h.n, keys % h.n
    rowptr = torch.zeros(h.n + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(torch.bincount(row, minlength=h.n), dim=0)
    return torch.ops.torch_sparse.partition(rowptr, col, None, n_parts, False).numpy()


@lru_cache(maxsize=1)
def _base_icosahedron():
    ico = trimesh.creation.icosphere(subdivisions=0)
    vertices = ico.vertices / np.linalg.norm(ico.vertices, axis=1, keepdims=True)
    face_vertices = vertices[ico.faces.astype(np.int64)]
    centers = face_vertices.mean(axis=1)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    edge_normals = np.stack([
        np.cross(face_vertices[:, 0], face_vertices[:, 1]),
        np.cross(face_vertices[:, 1], face_vertices[:, 2]),
        np.cross(face_vertices[:, 2], face_vertices[:, 0]),
    ], axis=1)
    orientation = np.stack([
        np.sum(edge_normals[:, 0] * face_vertices[:, 2], axis=1),
        np.sum(edge_normals[:, 1] * face_vertices[:, 0], axis=1),
        np.sum(edge_normals[:, 2] * face_vertices[:, 1], axis=1),
    ], axis=1)
    return centers, edge_normals, orientation


def assign_faces(latlon: torch.Tensor) -> np.ndarray:
    """Geometric partitioning: each node (given as lat/lon degrees) to its
    containing base-icosahedron face (20 partitions)."""
    centers, edge_normals, orientation = _base_icosahedron()
    xyz = latlon_to_xyz(latlon)
    kw = {"dtype": xyz.dtype}
    dots = xyz @ torch.tensor(centers, **kw).T
    scores = torch.einsum("nd,fed->nfe", xyz, torch.tensor(edge_normals, **kw))
    inside = ((scores * torch.tensor(orientation, **kw).unsqueeze(0)) >= -1e-6).all(dim=-1)
    out = torch.argmax(torch.where(inside, dots, torch.full_like(dots, float("-inf"))), dim=-1)
    fallback = ~inside.any(dim=-1)
    if fallback.any():
        out[fallback] = torch.argmax(dots[fallback], dim=-1)
    return out.numpy()


def assign_faces_with_grid(h: Hierarchy) -> tuple[np.ndarray, np.ndarray]:
    """Face assignment of (grid nodes, mesh nodes), computed in ONE batch.

    The 12 icosahedron corners lie exactly on face boundaries, so which face
    claims them is decided by float rounding in a batched matmul, and that
    rounding depends on the batch shape. The source pipeline assigned faces to
    grid and mesh nodes together, grid first; doing the same reproduces its
    corner assignments bit for bit (it moves one node of partition 0 in the
    6/5/4/3/2/1/0 hierarchy). Nothing scientific depends on it.
    """
    grid = grid_latlon()
    both = assign_faces(torch.cat([grid, h.latlon]))
    return both[:grid.shape[0]], both[grid.shape[0]:]


def assign(h: Hierarchy, strategy: str, n_parts: int,
           bands: tuple[int, int] | None = None) -> np.ndarray:
    if strategy == "latlon":
        return assign_latlon(h, *(bands or default_bands(n_parts)))
    if strategy == "metis":
        return assign_metis(h, n_parts)
    if strategy == "geometric":
        if n_parts != 20:
            raise ValueError("geometric partitioning is the 20 base faces")
        return assign_faces_with_grid(h)[1]
    raise ValueError(f"unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# Grid coupling, for the rollout closure only
# ---------------------------------------------------------------------------

def grid_latlon(resolution: float = 1.0) -> torch.Tensor:
    """The lat/lon grid, 181 x 360 at 1 degree, latitude-major."""
    lats = torch.arange(-90.0, 90.0 + resolution, resolution)
    lons = torch.arange(-180.0, 180.0, resolution)
    lat_grid, lon_grid = torch.meshgrid(lats, lons, indexing="ij")
    return torch.stack([lat_grid.flatten(), lon_grid.flatten()], dim=1)


def grid_edges(h: Hierarchy, k: int = 4, resolution: float = 1.0):
    """Return (grid_latlon, g2m, m2g).

    g2m: grid -> finest mesh (each fine node receives from its k nearest grid
    nodes); m2g: finest mesh -> grid (each grid node from its k nearest fine
    nodes). Mesh indices are global within `h`.
    """
    grid = grid_latlon(resolution)
    grid_xyz = latlon_to_xyz(grid)
    finest = h.levels[0]
    g2m = knn_edges(grid_xyz, h.xyz[finest], k)
    g2m[1] += h.offset[finest]
    m2g = knn_edges(h.xyz[finest], grid_xyz, k)
    m2g[0] += h.offset[finest]
    return grid, g2m, m2g
