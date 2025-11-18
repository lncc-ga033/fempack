from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fempack.mesh import CellType
from fempack.quadrature import (
    gauss_legendre_reference_interval,
    tensor_product_square_reference,
    triangle_quadrature,
)
from fempack.reference import p1_gradients, q1_gradients, q1_square_shape_functions


def l2_h1_errors(
    coords: NDArray[np.floating],
    cells: NDArray[np.integer],
    uh: NDArray[np.floating],
    u_exact: Callable[[float, float], float],
    grad_u_exact: Callable[[float, float], NDArray[np.floating]],
    cell_type: CellType = "triangle",
    order: int = 2,
) -> tuple[float, float]:
    """Compute L2 and H1-seminorm errors for P1/Q1 solutions in 2D.

    Parameters
    ----------
    coords :
        Vertex coordinates with shape ``(N, 2)``.
    cells :
        Cell connectivity with shape ``(Ne, nodes_per_cell)``. For triangles,
        nodes_per_cell=3; for squares, nodes_per_cell=4.
    uh :
        Vector of nodal values of the discrete solution.
    u_exact :
        Exact solution, called as ``u_exact(x, y)``.
    grad_u_exact :
        Exact gradient, called as ``grad_u_exact(x, y)`` and returning
        an array of length 2.
    cell_type :
        Type of cells: "triangle" for P1 elements or "square" for Q1 elements.
    order :
        Order of the quadrature rule used on each cell.

    Returns
    -------
    eL2 :
        Discrete approximation of the L2 error norm.
    eH1 :
        Discrete approximation of the H1-seminorm of the error.

    Notes
    -----
    For triangles, uses P1 (linear) elements with barycentric coordinates.
    For squares, uses Q1 (bilinear) elements with reference coordinates on [-1,1]^2.
    """
    uh = np.asarray(uh, dtype=float)

    # Select quadrature rule based on cell type
    if cell_type == "triangle":
        Q = triangle_quadrature(order=order)
        pts_ref = Q.points  # Shape (nq, 2) in [0,1] barycentric coords
        wts = Q.weights
    elif cell_type == "square":
        Q_1d = gauss_legendre_reference_interval(order)
        Q = tensor_product_square_reference(Q_1d)
        pts_ref = Q.points  # Shape (nq, 2) in [-1,1]^2
        wts = Q.weights
    else:
        raise ValueError(f"Unsupported cell_type: {cell_type}")

    err_L2_sq = 0.0
    err_H1_sq = 0.0

    for cell in cells:
        verts = coords[cell, :]
        uh_e = uh[cell]

        for q in range(len(wts)):
            if cell_type == "triangle":
                # Barycentric coordinates on [0,1]
                xi, eta = pts_ref[q]
                lam1 = 1.0 - xi - eta
                lam2 = xi
                lam3 = eta

                # Physical coordinates
                v0, v1, v2 = verts
                xq = v0 + xi * (v1 - v0) + eta * (v2 - v0)
                x_val = float(xq[0])
                y_val = float(xq[1])

                # Evaluate shape functions and gradients
                phi = np.array([lam1, lam2, lam3])
                grads, area = p1_gradients(verts)

                # Quadrature weights are defined on the reference triangle (area = 0.5),
                # so we multiply by |detJ| = 2 * area(physical) to get weights on K.
                wq = wts[q] * (2.0 * area)

            elif cell_type == "square":
                # Reference coordinates on [-1,1]^2
                xi, eta = pts_ref[q]

                # Evaluate Q1 shape functions and gradients at this quadrature point
                xi_arr = np.array([xi])
                eta_arr = np.array([eta])
                phi = q1_square_shape_functions(xi_arr, eta_arr)[0]  # Shape (4,)
                grads, detJ = q1_gradients(verts, xi, eta)  # grads shape (4, 2)

                # Physical coordinates using Q1 isoparametric mapping
                xq = verts.T @ phi
                x_val = float(xq[0])
                y_val = float(xq[1])

                # Quadrature weight scaled by Jacobian determinant
                wq = wts[q] * detJ

            # Exact solution and gradient at quadrature point
            uq_exact = u_exact(x_val, y_val)
            g_exact = grad_u_exact(x_val, y_val)

            # Approximate solution and gradient at quadrature point
            uh_q = float(phi @ uh_e)
            grad_uh = grads.T @ uh_e

            # Accumulate errors
            err_L2_sq += (uh_q - uq_exact) ** 2 * wq
            diff_grad = grad_uh - g_exact
            err_H1_sq += float(np.dot(diff_grad, diff_grad)) * wq

    eL2 = float(np.sqrt(err_L2_sq))
    eH1 = float(np.sqrt(err_H1_sq))
    return eL2, eH1
