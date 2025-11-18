from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, isspmatrix, spmatrix

from fempack.spaces import FunctionSpace


def apply_dirichlet(
    A: spmatrix | NDArray[np.floating],
    b: ArrayLike,
    V: FunctionSpace,
    nodes: Sequence[int] | None = None,
    g: Callable[..., float] | ArrayLike | None = None,
) -> tuple[csr_matrix, NDArray[np.floating]]:
    """Apply Dirichlet boundary conditions by row/column modification.

    Parameters
    ----------
    A :
        Global stiffness (or system) matrix.
    b :
        Right-hand side vector.
    V :
        Finite element space.
    nodes :
        Sequence of global node indices where Dirichlet conditions are
        applied. If ``None``, all boundary nodes of the mesh are used.
    g :
        Dirichlet data. If ``None``, homogeneous conditions are applied.
        If callable, it is evaluated as ``g(x)`` in 1D or ``g(x, y)`` in
        2D at the boundary nodes. If an array is supplied, it may be
        either a full vector of length ``N`` (from which the boundary
        entries are picked) or of length ``len(nodes)``.

    Returns
    -------
    A_bc, b_bc :
        Modified matrix and right-hand side with Dirichlet conditions
        imposed strongly.

    Notes
    -----
    The implementation follows the standard approach of:

    1. Shifting the right-hand side to account for non-homogeneous
       Dirichlet conditions.
    2. Zeroing the corresponding rows and columns and setting unit
       diagonal entries for Dirichlet nodes.
    """
    if isspmatrix(A):
        A_csr = A.tocsr()  # type: ignore[union-attr]
    else:
        A_csr = csr_matrix(A)

    b_vec = np.asarray(b, dtype=float).copy()
    mesh = V.mesh
    N = A_csr.shape[0]

    if nodes is None:
        nodes_arr = mesh.boundary_nodes()
    else:
        nodes_arr = np.array(list(nodes), dtype=int)

    if nodes_arr.size == 0:
        return A_csr, b_vec

    # Evaluate Dirichlet data on boundary nodes.
    if g is None:
        g_vec = np.zeros(nodes_arr.shape[0], dtype=float)
    elif callable(g):
        coords = mesh.coords[nodes_arr]
        if mesh.dim == 1:
            g_vec = np.array([g(float(x)) for (x,) in coords], dtype=float)
        else:
            g_vec = np.array([g(float(x), float(y)) for (x, y) in coords], dtype=float)
    else:
        g_arr = np.asarray(g, dtype=float)
        if g_arr.shape == (mesh.num_vertices,):
            g_vec = g_arr[nodes_arr]
        elif g_arr.shape == (nodes_arr.shape[0],):
            g_vec = g_arr
        else:
            raise ValueError("g must have length N or len(nodes).")

    # Shift the right-hand side: b <- b - A[:, nodes] * g_vec
    delta = A_csr[:, nodes_arr] @ g_vec
    b_vec -= np.asarray(delta).ravel()

    A_lil = A_csr.tolil()
    dirichlet_set = set(int(i) for i in nodes_arr.tolist())

    # Overwrite Dirichlet rows with identity rows.
    for i, gi in zip(nodes_arr, g_vec):
        idx = int(i)
        A_lil.rows[idx] = [idx]
        A_lil.data[idx] = [1.0]
        b_vec[idx] = gi

    # Zero Dirichlet columns in the remaining rows.
    for row in range(N):
        if row in dirichlet_set:
            continue
        row_cols = A_lil.rows[row]
        row_data = A_lil.data[row]
        if not row_cols:
            continue
        keep_cols: list[int] = []
        keep_data: list[float] = []
        for c, val in zip(row_cols, row_data):
            if (c in dirichlet_set) and (c != row):
                continue
            keep_cols.append(c)
            keep_data.append(val)
        A_lil.rows[row] = keep_cols
        A_lil.data[row] = keep_data

    return A_lil.tocsr(), b_vec
