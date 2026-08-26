"""Reconstructs hexagonal cell adjacency (crossing-free Delaunay+Gabriel graph) and one-to-one temporal links across frames (Hungarian bipartite matching) from pixel centroids, since annotation files store no grid structure. Also the single source of grid functionality for the napari frontend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional

import numpy as np
from numpy.typing import DTypeLike
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay, QhullError, cKDTree
from scipy.spatial.distance import cdist

from src.core.annotations import Annotation

if TYPE_CHECKING:
    from datetime import datetime


logger = logging.getLogger(__name__)

# A cell in a honeycomb has at most 6 direct neighbors.
MAX_NEIGHBORS = 6

# Temporal match radius as a fraction of the estimated cell spacing: generous enough for drift, tight enough that a cell can't match its neighbor.
DEFAULT_RADIUS_SCALE = 0.5

class CombGrid:
    """Hexagonal adjacency over a single frame's cell annotations, with ``labels`` as a mutable working copy so correctors never touch the originals."""

    def __init__(self, annotations: list[Annotation], neighbor_dist: Optional[float] = None) -> None:
        self.annotations = list(annotations)
        self.n = len(self.annotations)
        self.labels: list[str] = [ann.label for ann in self.annotations]

        # reshape(-1, 2) keeps shape (0, 2) even when the list is empty, so the geometry code never sees a 1-D array
        self._geometric_centers = reshape_to_2d([[ann.center_x, ann.center_y] for ann in self.annotations], dtype=np.float64)

        if self.n == 0:
            self.neighbor_dist = 0.0
            self._neighbors: list[list[int]] = []
            return

        self.neighbor_dist = (float(neighbor_dist) if neighbor_dist is not None else estimate_neighbor_dist(self._geometric_centers))
        self._neighbors = self._build_neighbors()

    def _build_neighbors(self) -> list[list[int]]:
        """Precompute for every cell the indices of its <=6 Delaunay+Gabriel neighbors within ``neighbor_dist``, closest first and self excluded."""
        pairs, lengths, _ = edge_pairs(self._geometric_centers, self.neighbor_dist)
        nb: dict[int, list[tuple[float, int]]] = {}
        for (i, j), d in zip(pairs, lengths):
            nb.setdefault(i, []).append((float(d), j))
            nb.setdefault(j, []).append((float(d), i))
        return [[j for _, j in sorted(nb.get(i, []))][:MAX_NEIGHBORS] for i in range(self.n)]

    def neighbors(self, i: int) -> list[int]:
        return self._neighbors[i]

    def neighbor_labels(self, i: int) -> list[str]:
        return [self.labels[j] for j in self._neighbors[i]]

    def set_label(self, i: int, label: str) -> None:
        self.labels[i] = label

    def to_annotations(self) -> list[Annotation]:
        """Return fresh ``Annotation`` copies that carry the current working labels."""
        return [
            Annotation(
                id=ann.id,
                center_x=ann.center_x,
                center_y=ann.center_y,
                radius=ann.radius,
                label=self.labels[i],
            )
            for i, ann in enumerate(self.annotations)
        ]


def delaunay_edges(pts: np.ndarray) -> np.ndarray:
    """Return the unique vertex-index pairs ``(n_edges, 2)`` of the Delaunay triangulation of ``pts`` (any dimension), falling back to all pairs when qhull cannot triangulate (too few or degenerate points)."""
    # Gabriel graphs are subgraphs of the Delaunay triangulation, so triangulating once gives the exact candidate set
    # QJ joggles input so regular lattices (and two parallel lattice planes) are otherwise degenerate for qhull
    try:
        simplices = Delaunay(pts, qhull_options="QJ").simplices
        ii, jj = np.triu_indices(simplices.shape[1], k=1)
        edges = np.stack([simplices[:, ii].ravel(), simplices[:, jj].ravel()], axis=1)
        return np.unique(np.sort(edges, axis=1), axis=0)
    except QhullError:
        ii, jj = np.triu_indices(len(pts), k=1)
        return np.stack([ii, jj], axis=1)


def empty_region_filter(pts: np.ndarray, edges: np.ndarray, method: str = "gabriel", lengths: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean keep-mask over ``edges`` for the Gabriel (empty diametral ball) test over ``pts`` — True where the edge is crossing-free. Pass precomputed edge ``lengths`` to skip recomputing them."""
    if len(edges) == 0:
        return np.zeros(0, dtype=bool)
    p, q = pts[edges[:, 0]], pts[edges[:, 1]]
    if lengths is None:
        lengths = np.linalg.norm(q - p, axis=1)
    mids = (p + q) / 2.0
    tree = cKDTree(pts)
    if method == "gabriel":
        # Closed diametral ball, so any third point at distance <= length/2 from midpoint blocks the edge
        hits = tree.query_ball_point(mids, lengths / 2.0)
        keep = [set(h) <= {int(a), int(b)} for h, (a, b) in zip(hits, edges)]
    else:
        raise ValueError(f"method must be 'gabriel', got {method!r}")
    return np.asarray(keep, dtype=bool)

def reshape_to_2d(arr: np.ndarray, dtype: DTypeLike = float) -> np.ndarray:
    """Reshapes an input array to a (N, 2) array."""
    return np.asarray(arr, dtype=dtype).reshape(-1, 2)

def estimate_neighbor_dist(centers: np.ndarray, scale: float = 1.3) -> float:
    """Estimate neighbor cutoff as median nearest-cell spacing times ``scale``."""
    centers = reshape_to_2d(centers)
    if len(centers) < 2:
        return 0.0
    dists, _ = cKDTree(centers).query(centers, k=2)
    return float(np.median(dists[:, 1]) * scale) # column 0 is the point itself (dist 0) -> column 1 is the nearest other cell

def edge_pairs(centers: np.ndarray, neighbor_dist: float = 0.0) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    """Crossing-free neighbor index pairs ``(i, j)``: Delaunay edges within the cutoff, Gabriel-filtered (planar by construction).
    Returns ``(pairs, lengths, centers reshaped (-1, 2))`` with ``lengths[k]`` the euclidean length of ``pairs[k]`` — computed once here so callers never re-derive distances that go stale with the centers they came from."""
    centers = reshape_to_2d(centers)
    if len(centers) < 2:
        return [], np.zeros(0), centers
    if neighbor_dist <= 0:
        neighbor_dist = estimate_neighbor_dist(centers)

    edges = delaunay_edges(centers)
    lengths = np.linalg.norm(centers[edges[:, 0]] - centers[edges[:, 1]], axis=1)
    within = lengths <= neighbor_dist
    edges, lengths = edges[within], lengths[within]
    keep = empty_region_filter(centers, edges, lengths=lengths)
    edges, lengths = edges[keep], lengths[keep]
    return [(int(i), int(j)) for i, j in edges], lengths, centers

def neighbor_map_from_pairs(pairs: list[tuple[int, int]]) -> dict[int, list[int]]:
    """Symmetric ``{cell_index: [neighbor_index, ...]}`` from undirected edge pairs."""
    neighbours: dict[int, list[int]] = {}
    for i, j in pairs:
        neighbours.setdefault(i, []).append(j)  # creates empty list when i not in neighbours dict and adds j as neighbour
        neighbours.setdefault(j, []).append(i)  # vise versa for j
    return neighbours

def connected_components(n: int, neighbors: Callable[[int], Iterable[int]], key: Callable[[int], Any]) -> list[list[int]]:
    """Connected components of cells ``0..n-1`` using iterative DFS, growing each component along
    neighbors whose ``key`` equals the start cell's. Cells with ``key(i) is None`` are skipped."""
    seen = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        own = key(start)
        if seen[start] or own is None:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in neighbors(i):
                if not seen[j] and key(j) == own:
                    seen[j] = True
                    stack.append(j)
        components.append(comp)
    return components

def neighbor_map(centers: np.ndarray, neighbor_dist: float = 0.0) -> dict[int, list[int]]:
    """Symmetric ``{cell_index: [neighbor_index, ...]}`` over the hex adjacency."""
    pairs, _, _ = edge_pairs(centers, neighbor_dist)
    return neighbor_map_from_pairs(pairs)

def _edges_from_bipartite_matching(centers_a: np.ndarray, centers_b: np.ndarray, radius: float) -> list[tuple[int, int, float]]:
    """Return one-to-one temporal matches ``(i, j, dist)`` between two frames via bipartite assignment (Crouse) on xy distance, pairs beyond ``radius`` disallowed."""
    if len(centers_a) == 0 or len(centers_b) == 0 or radius <= 0:
        return []
    distances = cdist(centers_a, centers_b)
    useable_distances = distances <= radius 
    cost_matrix = np.where(useable_distances, distances, 1e9) # scipy raises on inf-only rows, so out-of-radius pairs get a large finite penalty instead and are dropped after solving
    rows, cols = linear_sum_assignment(cost_matrix) # Crouse method
    matches = []
    for i, j in zip(rows, cols):
        if useable_distances[i, j]:
            matches.append((int(i), int(j), float(distances[i, j])))
    return matches


class SequenceGrid:
    """Spatio-temporal grid over one camera's ordered frames: a ``CombGrid`` per frame (up to 6 spatial neighbors) plus one-to-one temporal links to nearby cells in the previous/next frame."""
    #  - temporal links per frame pair, bipartite matching minimizing aggregated x-y distance
    #  - pairs beyond match_radius are not allowed
    #  - frame after breakpoint behaves like a first frame, so no previous neighbour
    def __init__(
        self,
        frames: list[list[Annotation]],
        breakpoints: Optional[list[bool]] = None,
        match_radius: Optional[float] = None,
        radius_scale: float = DEFAULT_RADIUS_SCALE,
        neighbor_dist: Optional[float] = None,
        timestamps: Optional[list[datetime]] = None,
    ) -> None:
        
        self.grids = [CombGrid(f, neighbor_dist=neighbor_dist) for f in frames]
        self.n_frames = len(self.grids)
        breakpoints = breakpoints or [False] * self.n_frames

        # temporal neighbor indices per cell, nearest first, can be empty when no match, marked for breakpoint or first/last frane
        self._prev: list[list[list[int]]] = [[[] for _ in range(g.n)] for g in self.grids]
        self._next: list[list[list[int]]] = [[[] for _ in range(g.n)] for g in self.grids]
        self._link_temporal(breakpoints, match_radius, radius_scale)

        self._calculate_gap_delta_in_hours: list[Optional[float]] = [None] * self.n_frames
        if timestamps is not None:
            for t in range(1, self.n_frames):
                if breakpoints[t]:
                    continue
                self._calculate_gap_delta_in_hours[t] = (timestamps[t] - timestamps[t - 1]).total_seconds() / 3600.0

    def _link_temporal(self, breakpoints: list[bool], match_radius: Optional[float], radius_scale: float) -> None:
        for t in range(self.n_frames - 1):
            if breakpoints[t + 1]:
                continue
            frame_a, frame_b = self.grids[t], self.grids[t + 1]
            if frame_a.n == 0 or frame_b.n == 0:
                continue
            spacing = frame_a.neighbor_dist or frame_b.neighbor_dist
            radius = match_radius if match_radius is not None else radius_scale * spacing
            edges = _edges_from_bipartite_matching(frame_a._geometric_centers, frame_b._geometric_centers, radius=radius)
            edges.sort(key=lambda e: e[2]) # Global sort by edge length -> per-cell link lists end up nearest first.
            for i, j, _ in edges:
                self._next[t][i].append(j)
                self._prev[t + 1][j].append(i)

    def labels(self, t: int) -> list[str]:
        return self.grids[t].labels

    def calculate_gap_delta_in_hours(self, t: int) -> Optional[float]:
        """Return hours between frame ``t`` and ``t-1``, or None at a boundary/breakpoint."""
        return self._calculate_gap_delta_in_hours[t]

    def prev_cell(self, t: int, i: int) -> int:
        """Return the nearest matched index of cell ``i`` in the previous frame, or -1 if none."""
        links = self._prev[t][i]
        return links[0] if links else -1

    def spatial_neighbor_labels(self, t: int, i: int) -> list[str]:
        return self.grids[t].neighbor_labels(i)

    def temporal_neighbor_labels(self, t: int, i: int) -> list[str]:
        out = [self.grids[t - 1].labels[p] for p in self._prev[t][i]]
        out += [self.grids[t + 1].labels[n] for n in self._next[t][i]]
        return out

    def set_label(self, t: int, i: int, label: str) -> None:
        self.grids[t].set_label(i, label)

    def frame_view(self, t: int, axis: str) -> FrameView:
        """Return a read-only, ``CombGrid``-shaped view of frame ``t`` along one axis (``"spatial"`` or ``"temporal"``)."""
        return FrameView(self, t, axis)

    def to_annotations(self) -> list[list[Annotation]]:
        return [g.to_annotations() for g in self.grids]


class FrameView:
    """One frame's cells exposed through the minimal rule surface (``n``, ``labels``, ``neighbor_labels``) with neighbor lookups routed to the chosen axis, so the same rule classes work spatially (2D) and temporally (3rd dimension) unchanged."""

    def __init__(self, seq: SequenceGrid, t: int, axis: str) -> None:
        axis_normalized = axis.lower() if isinstance(axis, str) else axis
        if axis_normalized not in ("spatial", "temporal"):
            raise ValueError(f"axis must be 'spatial' or 'temporal', got {axis!r}")
        self._seq = seq
        self._t = t # frame_index
        self._axis = axis_normalized
        self.n = seq.grids[t].n
        self.labels = seq.grids[t].labels

    def neighbor_labels(self, i: int) -> list[str]:
        if self._axis == "temporal":
            return self._seq.temporal_neighbor_labels(self._t, i)
        return self._seq.spatial_neighbor_labels(self._t, i)
