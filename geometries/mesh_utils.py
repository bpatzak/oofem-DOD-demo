"""Reusable helpers for structured meshes and node selection."""
import numpy as np


def subdivide(breaks, h):
    """Return sorted 1D grid coordinates passing exactly through all breakpoints.

    Each segment [b_i, b_{i+1}] is split into max(1, round(len/h)) equal parts,
    so features such as supports or load points always coincide with nodes.
    """
    breaks = np.unique(np.asarray(breaks, dtype=float))
    pts = [breaks[:1]]
    for a, b in zip(breaks[:-1], breaks[1:]):
        n = max(1, int(round((b - a) / h)))
        pts.append(np.linspace(a, b, n + 1)[1:])
    return np.concatenate(pts)


def rect_grid(xs, ys):
    """Structured quad grid on the tensor product of 1D coordinates xs, ys.

    Nodes are numbered row by row (x fastest), cells counter-clockwise.
    Returns (coords (nn,2), cells (ne,4)).
    """
    nx, ny = len(xs), len(ys)
    X, Y = np.meshgrid(xs, ys)               # shape (ny, nx)
    coords = np.column_stack([X.ravel(), Y.ravel()])
    idx = np.arange(nx * ny).reshape(ny, nx)
    n0 = idx[:-1, :-1].ravel()               # lower-left corner of each cell
    cells = np.column_stack([n0, n0 + 1, n0 + nx + 1, n0 + nx])
    return coords, cells


def breakpoints_grid(xbreaks, ybreaks, h):
    """Rectilinear quad mesh of element size ~h honouring all x/y breakpoints."""
    return rect_grid(subdivide(xbreaks, h), subdivide(ybreaks, h))


def nodes_at(coords, x=None, y=None, tol=1e-8):
    """Indices of nodes matching given x and/or y coordinate (within tol)."""
    mask = np.ones(len(coords), dtype=bool)
    if x is not None:
        mask &= np.abs(coords[:, 0] - x) < tol
    if y is not None:
        mask &= np.abs(coords[:, 1] - y) < tol
    return np.flatnonzero(mask)


def nodes_in_box(coords, xmin, xmax, ymin, ymax, tol=1e-8):
    """Indices of nodes inside an axis-aligned box (inclusive, with tol)."""
    c = coords
    mask = ((c[:, 0] >= xmin - tol) & (c[:, 0] <= xmax + tol)
            & (c[:, 1] >= ymin - tol) & (c[:, 1] <= ymax + tol))
    return np.flatnonzero(mask)


def single_node(coords, x, y, tol=1e-8):
    """Index of exactly one node at (x, y); raises if not found/ambiguous."""
    ids = nodes_at(coords, x, y, tol)
    if len(ids) != 1:
        raise ValueError(f"expected one node at ({x}, {y}), found {len(ids)}")
    return ids
