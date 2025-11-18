"""Local element matrix and vector computations.

This module provides functions to compute element-level (local) contributions
for finite element assembly, including stiffness matrices, mass matrices, and
load vectors for different element types.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from fempack.reference import (
    interval_length,
    p1_gradients,
    p1_interval_gradients,
    p2_interval_gradients,
    q1_gradients,
    triangle_area,
)
from fempack.elements import LagrangeElement

# Type alias for cleaner function signatures
FiniteElement = LagrangeElement


def local_stiffness_p1_interval(x0: float, x1: float) -> NDArray[np.floating]:
    """Compute local stiffness matrix for P1 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.

    Returns
    -------
    numpy.ndarray
        Local stiffness matrix with shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If the element is degenerate (h <= 0).
    """
    grads, _ = p1_interval_gradients(x0, x1)
    # Stiffness matrix: K[a,b] = ∫ (dN_a/dx)(dN_b/dx) dx = h * grad_a * grad_b
    # For constant gradients on interval: K = h * outer(grads, grads)
    ke = np.outer(grads, grads) * (x1 - x0)
    return ke


def local_mass_p1_interval(x0: float, x1: float) -> NDArray[np.floating]:
    """Compute local mass matrix for P1 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.

    Returns
    -------
    numpy.ndarray
        Local mass matrix with shape ``(2, 2)``.

    Raises
    ------
    ValueError
        If the element is degenerate (h <= 0).
    """
    h = interval_length(x0, x1)
    return (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=float)


def local_stiffness_p2_interval(
    x0: float,
    x1: float,
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local stiffness matrix for P2 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.
    element :
        Finite element containing shape function information (degree=2).
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 1)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local stiffness matrix with shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If the element is degenerate (h <= 0).
    """
    h = interval_length(x0, x1)
    ke = np.zeros((3, 3), dtype=float)

    for q in range(len(quad_weights)):
        xi = quad_points[q, 0]
        grads, _ = p2_interval_gradients(x0, x1, xi)
        wq = quad_weights[q] * h

        for a in range(3):
            for b in range(3):
                ke[a, b] += wq * grads[a] * grads[b]

    return ke


def local_mass_p2_interval(
    x0: float,
    x1: float,
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local mass matrix for P2 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.
    element :
        Finite element containing shape function information (degree=2).
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 1)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local mass matrix with shape ``(3, 3)``.

    Raises
    ------
    ValueError
        If the element is degenerate (h <= 0).
    """
    h = interval_length(x0, x1)
    phi_all = element.tabulate(quad_points)
    me = np.zeros((3, 3), dtype=float)

    for q in range(len(quad_weights)):
        phi_q = phi_all[q, :]
        wq = quad_weights[q] * h

        for a in range(3):
            for b in range(3):
                me[a, b] += wq * phi_q[a] * phi_q[b]

    return me


def local_load_p2_interval(
    x0: float,
    x1: float,
    f: Callable[[float], float],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local load vector for P2 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.
    f :
        Forcing function, called as ``f(x)``.
    element :
        Finite element containing shape function information (degree=2).
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 1)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local load vector with shape ``(3,)``.

    Raises
    ------
    ValueError
        If the element is degenerate (h <= 0).
    """
    h = interval_length(x0, x1)
    xi = quad_points[:, 0]
    xq = x0 + xi * h
    fq = np.array([f(float(xx)) for xx in xq], dtype=float)
    phi = element.tabulate(quad_points)

    b_local = np.zeros(3, dtype=float)
    for q in range(len(quad_weights)):
        wq = quad_weights[q] * h
        for a_loc in range(3):
            b_local[a_loc] += fq[q] * phi[q, a_loc] * wq

    return b_local


def local_stiffness_p1_triangle(verts: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute local stiffness matrix for P1 element on a triangle.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(3, 2)``.

    Returns
    -------
    numpy.ndarray
        Local stiffness matrix with shape ``(3, 3)``.
    """
    raise NotImplementedError(
        "TODO: implement local_stiffness_p1_triangle as part of the assignment."
    )


def local_mass_p1_triangle(verts: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute local mass matrix for P1 element on a triangle.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(3, 2)``.

    Returns
    -------
    numpy.ndarray
        Local mass matrix with shape ``(3, 3)``.
    """
    raise NotImplementedError(
        "TODO: implement local_mass_p1_triangle as part of the assignment."
    )

def local_stiffness_q1_square(
    verts: NDArray[np.floating],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local stiffness matrix for Q1 element on a quadrilateral.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(4, 2)``.
    element :
        Finite element containing shape function information.
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 2)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local stiffness matrix with shape ``(4, 4)``.
    """
    ke = np.zeros((4, 4), dtype=float)

    for q in range(quad_points.shape[0]):
        xi, eta = quad_points[q, :]
        grads_phys, detJ = q1_gradients(verts, xi, eta)
        wq = quad_weights[q] * detJ

        for a in range(4):
            for b in range(4):
                ke[a, b] += wq * float(np.dot(grads_phys[a, :], grads_phys[b, :]))

    return ke


def local_mass_q1_square(
    verts: NDArray[np.floating],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local mass matrix for Q1 element on a quadrilateral.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(4, 2)``.
    element :
        Finite element containing shape function information.
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 2)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local mass matrix with shape ``(4, 4)``.
    """
    phi_all = element.tabulate(quad_points)
    me = np.zeros((4, 4), dtype=float)

    for q in range(quad_points.shape[0]):
        phi_q = phi_all[q, :]
        xi, eta = quad_points[q, :]
        _, detJ = q1_gradients(verts, xi, eta)
        wq = quad_weights[q] * detJ

        for a in range(4):
            for b in range(4):
                me[a, b] += wq * phi_q[a] * phi_q[b]

    return me


def local_load_p1_interval(
    x0: float,
    x1: float,
    f: Callable[[float], float],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local load vector for P1 element on an interval.

    Parameters
    ----------
    x0, x1 :
        Endpoints of the physical element.
    f :
        Forcing function, called as ``f(x)``.
    element :
        Finite element containing shape function information.
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 1)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local load vector with shape ``(2,)``.
    """
    h = interval_length(x0, x1)
    xi = quad_points[:, 0]
    xq = x0 + xi * h
    fq = np.array([f(float(xx)) for xx in xq], dtype=float)
    phi = element.tabulate(quad_points)

    b_local = np.zeros(2, dtype=float)
    for q in range(len(quad_weights)):
        wq = quad_weights[q] * h
        for a_loc in range(2):
            b_local[a_loc] += fq[q] * phi[q, a_loc] * wq

    return b_local


def local_load_p1_triangle(
    verts: NDArray[np.floating],
    f: Callable[[float, float], float],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local load vector for P1 element on a triangle.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(3, 2)``.
    f :
        Forcing function, called as ``f(x, y)``.
    element :
        Finite element containing shape function information.
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 2)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local load vector with shape ``(3,)``.
    """
    raise NotImplementedError(
        "TODO: implement local_load_p1_triangle as part of the assignment."
    )

def local_load_q1_square(
    verts: NDArray[np.floating],
    f: Callable[[float, float], float],
    element: FiniteElement,
    quad_points: NDArray[np.floating],
    quad_weights: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute local load vector for Q1 element on a quadrilateral.

    Parameters
    ----------
    verts :
        Vertex coordinates with shape ``(4, 2)``.
    f :
        Forcing function, called as ``f(x, y)``.
    element :
        Finite element containing shape function information.
    quad_points :
        Quadrature points in reference coordinates with shape ``(nq, 2)``.
    quad_weights :
        Quadrature weights with shape ``(nq,)``.

    Returns
    -------
    numpy.ndarray
        Local load vector with shape ``(4,)``.
    """
    phi_all = element.tabulate(quad_points)
    b_local = np.zeros(4, dtype=float)
    x_coords = verts[:, 0]
    y_coords = verts[:, 1]

    for q in range(quad_points.shape[0]):
        phi_q = phi_all[q, :]
        xi, eta = quad_points[q, :]
        _, detJ = q1_gradients(verts, xi, eta)

        xq_val = float(np.dot(phi_q, x_coords))
        yq_val = float(np.dot(phi_q, y_coords))
        fq_val = f(xq_val, yq_val)
        wq = quad_weights[q] * detJ

        for a_loc in range(4):
            b_local[a_loc] += fq_val * phi_q[a_loc] * wq

    return b_local
