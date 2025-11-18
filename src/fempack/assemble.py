from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from fempack.quadrature import (
    gauss_legendre_interval,
    gauss_legendre_reference_interval,
    tensor_product_square_reference,
    triangle_quadrature,
)
from fempack.spaces import FunctionSpace
from fempack.local import (
    local_stiffness_p1_interval,
    local_mass_p1_interval,
    local_stiffness_p1_triangle,
    local_mass_p1_triangle,
    local_stiffness_q1_square,
    local_mass_q1_square,
    local_load_p1_interval,
    local_load_p1_triangle,
    local_load_q1_square,
)


def assemble_stiffness(V: FunctionSpace) -> csr_matrix:
    r"""Assemble the global stiffness matrix.

    Parameters
    ----------
    V :
        Finite element space.

    Returns
    -------
    scipy.sparse.csr_matrix
        Stiffness matrix corresponding to the bilinear form
        :math:`a(u, v) = \int_\Omega \nabla u \cdot \nabla v`.
    """
    mesh = V.mesh
    element = V.element
    coords = mesh.coords
    cells = mesh.cells

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    if mesh.cell_type == "interval":
        for cell in cells:
            v0, v1 = cell
            x0 = float(coords[v0, 0])
            x1 = float(coords[v1, 0])
            k_local = local_stiffness_p1_interval(x0, x1)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(k_local[a_loc, b_loc]))

    elif mesh.cell_type == "triangle":
        for cell in cells:
            verts = coords[cell, :]
            k_local = local_stiffness_p1_triangle(verts)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(k_local[a_loc, b_loc]))

    elif mesh.cell_type == "square":
        Q1d = gauss_legendre_reference_interval(2)
        Q = tensor_product_square_reference(Q1d)
        pts = Q.points
        wts = Q.weights

        for cell in cells:
            verts = coords[cell, :]
            k_local = local_stiffness_q1_square(verts, element, pts, wts)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(k_local[a_loc, b_loc]))

    else:
        raise NotImplementedError("assemble_stiffness not implemented for this cell_type.")

    N = mesh.num_vertices
    A = coo_matrix((data, (rows, cols)), shape=(N, N))
    return A.tocsr()


def assemble_mass(V: FunctionSpace) -> csr_matrix:
    r"""Assemble the global mass matrix.

    Parameters
    ----------
    V :
        Finite element space.

    Returns
    -------
    scipy.sparse.csr_matrix
        Mass matrix corresponding to the bilinear form
        :math:`m(u, v) = \int_\Omega u v`.
    """
    mesh = V.mesh
    element = V.element
    coords = mesh.coords
    cells = mesh.cells

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    if mesh.cell_type == "interval":
        for cell in cells:
            v0, v1 = cell
            x0 = float(coords[v0, 0])
            x1 = float(coords[v1, 0])
            m_local = local_mass_p1_interval(x0, x1)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(m_local[a_loc, b_loc]))

    elif mesh.cell_type == "triangle":
        for cell in cells:
            verts = coords[cell, :]
            m_local = local_mass_p1_triangle(verts)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(m_local[a_loc, b_loc]))

    elif mesh.cell_type == "square":
        Q1d = gauss_legendre_reference_interval(2)
        Q = tensor_product_square_reference(Q1d)
        pts = Q.points
        wts = Q.weights

        for cell in cells:
            verts = coords[cell, :]
            m_local = local_mass_q1_square(verts, element, pts, wts)
            for a_loc, a_glob in enumerate(cell):
                for b_loc, b_glob in enumerate(cell):
                    rows.append(int(a_glob))
                    cols.append(int(b_glob))
                    data.append(float(m_local[a_loc, b_loc]))

    else:
        raise NotImplementedError("assemble_mass not implemented for this cell_type.")

    N = mesh.num_vertices
    M = coo_matrix((data, (rows, cols)), shape=(N, N))
    return M.tocsr()


def assemble_load(
    V: FunctionSpace,
    f: Callable[..., float],
    quad_order: int = 2,
) -> NDArray[np.floating]:
    """Assemble the global load vector.

    Parameters
    ----------
    V :
        Finite element space.
    f :
        Forcing term. In 1D it is called as ``f(x)`` and in 2D as
        ``f(x, y)``.
    quad_order :
        Quadrature order for triangles and the unit interval.

    Returns
    -------
    numpy.ndarray
        One-dimensional array with the assembled load vector.
    """
    mesh = V.mesh
    element = V.element
    coords = mesh.coords
    cells = mesh.cells

    b = np.zeros(mesh.num_vertices, dtype=float)

    if mesh.cell_type == "interval":
        Q = gauss_legendre_interval(quad_order)
        for cell in cells:
            verts = coords[cell, :]
            x0 = float(verts[0, 0])
            x1 = float(verts[1, 0])
            b_local = local_load_p1_interval(x0, x1, f, element, Q.points, Q.weights)
            for a_loc, a_glob in enumerate(cell):
                b[int(a_glob)] += b_local[a_loc]

    elif mesh.cell_type == "triangle":
        Q = triangle_quadrature(order=quad_order)
        for cell in cells:
            verts = coords[cell, :]
            b_local = local_load_p1_triangle(verts, f, element, Q.points, Q.weights)
            for a_loc, a_glob in enumerate(cell):
                b[int(a_glob)] += b_local[a_loc]

    elif mesh.cell_type == "square":
        Q1d = gauss_legendre_reference_interval(2)
        Q = tensor_product_square_reference(Q1d)
        pts = Q.points
        wts = Q.weights

        for cell in cells:
            verts = coords[cell, :]
            b_local = local_load_q1_square(verts, f, element, pts, wts)
            for a_loc, a_glob in enumerate(cell):
                b[int(a_glob)] += b_local[a_loc]

    else:
        raise NotImplementedError("assemble_load not implemented for this cell_type.")

    return b
