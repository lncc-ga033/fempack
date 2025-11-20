from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReferenceCell:
    """Simple descriptor of a reference cell.

    Parameters
    ----------
    name :
        Human-readable name (e.g. ``"triangle"``).
    dim :
        Spatial dimension of the cell.
    """

    name: str
    dim: int


INTERVAL = ReferenceCell("interval", 1)
TRIANGLE = ReferenceCell("triangle", 2)
SQUARE = ReferenceCell("square", 2)


# ==============================================================================
# Shape Function Helper Functions
# ==============================================================================


def p1_interval_shape_functions(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate P1 shape functions on interval [0, 1].

    Parameters
    ----------
    x :
        Reference coordinates with shape (nq,).

    Returns
    -------
    numpy.ndarray
        Shape function values with shape (nq, 2).
        Column 0: N₀(x) = 1 - x (left node)
        Column 1: N₁(x) = x (right node)
    """
    phi0 = 1.0 - x
    phi1 = x
    return np.column_stack([phi0, phi1])


def p2_interval_shape_functions(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Evaluate P2 shape functions on interval [0, 1].

    P2 Lagrange basis with nodes at ξ = 0, 0.5, 1:
        N0(ξ) = 2(ξ - 0.5)(ξ - 1)     (left endpoint)
        N1(ξ) = -4ξ(ξ - 1)            (midpoint)
        N2(ξ) = 2ξ(ξ - 0.5)           (right endpoint)

    Parameters
    ----------
    x :
        Reference coordinates with shape (nq,).

    Returns
    -------
    numpy.ndarray
        Shape function values with shape (nq, 3).
    """
    phi0 = 2.0 * (x - 0.5) * (x - 1.0)  # Left node
    phi1 = -4.0 * x * (x - 1.0)  # Middle node
    phi2 = 2.0 * x * (x - 0.5)  # Right node
    return np.column_stack([phi0, phi1, phi2])


def p1_triangle_shape_functions(
    x: NDArray[np.floating], y: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Evaluate P1 shape functions on reference triangle.

    Reference triangle has vertices at (0,0), (1,0), (0,1).

    Parameters
    ----------
    x :
        x-coordinates with shape (nq,).
    y :
        y-coordinates with shape (nq,).

    Returns
    -------
    numpy.ndarray
        Shape function values with shape (nq, 3).
        Column 0: λ₁ = 1 - x - y (vertex at origin)
        Column 1: λ₂ = x (vertex at (1,0))
        Column 2: λ₃ = y (vertex at (0,1))
    """
    l1 = 1.0 - x - y
    l2 = x
    l3 = y
    return np.column_stack([l1, l2, l3])


def q1_square_shape_functions(
    xi: NDArray[np.floating], eta: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Evaluate Q1 shape functions on reference square [-1, 1]².

    Bilinear shape functions for the reference square with vertices at
    (-1,-1), (1,-1), (1,1), (-1,1).

    Parameters
    ----------
    xi :
        ξ-coordinates with shape (nq,).
    eta :
        η-coordinates with shape (nq,).

    Returns
    -------
    numpy.ndarray
        Shape function values with shape (nq, 4).
        Column 0: N₁ = ¼(1-ξ)(1-η)  (vertex at (-1,-1))
        Column 1: N₂ = ¼(1+ξ)(1-η)  (vertex at (1,-1))
        Column 2: N₃ = ¼(1+ξ)(1+η)  (vertex at (1,1))
        Column 3: N₄ = ¼(1-ξ)(1+η)  (vertex at (-1,1))
    """
    phi1 = 0.25 * (1.0 - xi) * (1.0 - eta)
    phi2 = 0.25 * (1.0 + xi) * (1.0 - eta)
    phi3 = 0.25 * (1.0 + xi) * (1.0 + eta)
    phi4 = 0.25 * (1.0 - xi) * (1.0 + eta)
    return np.column_stack([phi1, phi2, phi3, phi4])


# ==============================================================================
# Reference Gradient Helper Functions
# ==============================================================================


def p1_interval_reference_gradients() -> NDArray[np.floating]:
    """Compute P1 reference gradients on interval [0, 1].

    For P1 elements, gradients are constant:
        dN0/dξ = -1
        dN1/dξ = 1

    Returns
    -------
    numpy.ndarray
        Reference gradients with shape (2, 1).
    """
    return np.array([[-1.0], [1.0]], dtype=float)


def p1_triangle_reference_gradients() -> NDArray[np.floating]:
    """Compute P1 reference gradients on triangle.

    For P1 elements on the reference triangle, gradients are constant:
        dN0/dξ = [-1, -1]  (vertex 0)
        dN1/dξ = [1, 0]    (vertex 1)
        dN2/dξ = [0, 1]    (vertex 2)

    Returns
    -------
    numpy.ndarray
        Reference gradients with shape (3, 2).
    """
    return np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)


def q1_square_reference_gradients(
    xi: float | NDArray[np.floating],
    eta: float | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute Q1 reference gradients on square [-1, 1]**2.

    For Q1 elements on the reference square, the gradients vary with position.

    Parameters
    ----------
    xi, eta :
        Reference coordinates ξ, η ∈ [-1, 1]. Can be scalars or arrays.

    Returns
    -------
    numpy.ndarray
        Reference gradients. Shape is (4, 2) if inputs are scalars,
        or (nq, 4, 2) if inputs are arrays of length nq.
    """
    xi_arr = np.atleast_1d(xi)
    eta_arr = np.atleast_1d(eta)

    # Gradients for each of the 4 nodes
    # Format: [dN_i/dξ, dN_i/dη] for each node i
    dphi1_dxi = -0.25 * (1.0 - eta_arr)
    dphi1_deta = -0.25 * (1.0 - xi_arr)

    dphi2_dxi = 0.25 * (1.0 - eta_arr)
    dphi2_deta = -0.25 * (1.0 + xi_arr)

    dphi3_dxi = 0.25 * (1.0 + eta_arr)
    dphi3_deta = 0.25 * (1.0 + xi_arr)

    dphi4_dxi = -0.25 * (1.0 + eta_arr)
    dphi4_deta = 0.25 * (1.0 - xi_arr)

    # Stack into (nq, 4, 2) array
    gx = np.column_stack([dphi1_dxi, dphi2_dxi, dphi3_dxi, dphi4_dxi])
    gy = np.column_stack([dphi1_deta, dphi2_deta, dphi3_deta, dphi4_deta])
    grads = np.stack([gx, gy], axis=-1)

    # Return (4, 2) if inputs were scalars
    if np.isscalar(xi) and np.isscalar(eta):
        return grads[0, :, :]
    return grads


# ==============================================================================
# Geometry and Physical Gradient Functions
# ==============================================================================


def interval_length(x0: float, x1: float) -> float:
    """Compute the length of an interval element.

    Parameters
    ----------
    x0 :
        Left endpoint of the interval.
    x1 :
        Right endpoint of the interval.

    Returns
    -------
    float
        Length of the interval (h = x1 - x0).

    Raises
    ------
    ValueError
        If the interval is degenerate (h <= 0).
    """
    h = x1 - x0
    if h <= 0.0:
        raise ValueError(f"Degenerate interval (h = {h:.6e} <= 0).")
    return float(h)


def p1_interval_gradients(x0: float, x1: float) -> tuple[NDArray[np.floating], float]:
    """Compute P1 shape-function gradients and length on an interval.

    For P1 elements on the reference interval [0, 1], the shape functions are:
        N0(ξ) = 1 - ξ
        N1(ξ) = ξ

    Their gradients in reference coordinates are constant: dN0/dξ = -1, dN1/dξ = 1.
    The physical gradients are obtained by the chain rule: dN/dx = (dN/dξ) * (dξ/dx).

    Parameters
    ----------
    x0 :
        Left endpoint of the physical interval.
    x1 :
        Right endpoint of the physical interval.

    Returns
    -------
    grads :
        Array with shape ``(2,)`` containing the gradients of the two P1 shape
        functions: [dN0/dx, dN1/dx].
    h :
        Length of the interval.

    Raises
    ------
    ValueError
        If the interval is degenerate (h <= 0).
    """
    h = interval_length(x0, x1)
    # Get reference gradients from helper function
    ref_grads = p1_interval_reference_gradients()  # Shape (2, 1)
    # Physical gradients: dN/dx = (dN/dξ) / h
    grads = ref_grads[:, 0] / h  # Shape (2,)
    return grads, h


def p2_interval_reference_gradients(xi: float | NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute P2 reference gradients at given reference coordinate(s).

    Evaluates the derivatives of P2 Lagrange basis functions with respect
    to the reference coordinate ξ ∈ [0, 1].

    For P2 elements on [0, 1] with nodes at ξ = 0, 0.5, 1:
        dN0/dξ = 4ξ - 3
        dN1/dξ = -8ξ + 4
        dN2/dξ = 4ξ - 1

    Parameters
    ----------
    xi :
        Reference coordinate(s) ξ ∈ [0, 1]. Can be a scalar or array.

    Returns
    -------
    numpy.ndarray
        Reference gradients. Shape is (3,) if xi is scalar, or
        (nq, 3) if xi is an array of length nq.
    """
    xi_arr = np.atleast_1d(xi)

    # Reference gradients for P2 basis functions
    dN0_dxi = 4.0 * xi_arr - 3.0
    dN1_dxi = -8.0 * xi_arr + 4.0
    dN2_dxi = 4.0 * xi_arr - 1.0

    grads = np.column_stack([dN0_dxi, dN1_dxi, dN2_dxi])

    # Return scalar shape if input was scalar
    if np.isscalar(xi):
        return grads[0, :]
    return grads


def p2_interval_gradients(x0: float, x1: float, xi: float) -> tuple[NDArray[np.floating], float]:
    """Compute P2 shape-function gradients and length on an interval.

    For P2 elements on the reference interval [0, 1], we use three nodes
    at ξ = 0, 0.5, 1 with Lagrange basis functions:
        N0(ξ) = 2(ξ - 0.5)(ξ - 1)     (left endpoint)
        N1(ξ) = -4ξ(ξ - 1)             (midpoint)
        N2(ξ) = 2ξ(ξ - 0.5)            (right endpoint)

    Their gradients in reference coordinates are:
        dN0/dξ = 4ξ - 3
        dN1/dξ = -8ξ + 4
        dN2/dξ = 4ξ - 1

    Parameters
    ----------
    x0 :
        Left endpoint of the physical interval.
    x1 :
        Right endpoint of the physical interval.
    xi :
        Reference coordinate ξ ∈ [0, 1] at which to evaluate gradients.

    Returns
    -------
    grads :
        Array with shape ``(3,)`` containing the physical gradients of the
        three P2 shape functions: [dN0/dx, dN1/dx, dN2/dx].
    h :
        Length of the interval.

    Raises
    ------
    ValueError
        If the interval is degenerate (h <= 0) or if xi is outside [0, 1].
    """
    h = interval_length(x0, x1)

    if not (0.0 <= xi <= 1.0):
        raise ValueError(f"Reference coordinate xi = {xi} must be in [0, 1].")

    # Get reference gradients using the helper function
    ref_grads = p2_interval_reference_gradients(xi)

    # Physical gradients: dN/dx = (dN/dξ) / h
    grads = ref_grads / h
    return grads, h


def triangle_area(verts: NDArray[np.floating]) -> float:
    """Compute the area of a triangle in physical coordinates.

    Parameters
    ----------
    verts :
        Array with shape ``(3, 2)`` containing the coordinates of the
        triangle vertices.

    Returns
    -------
    float
        Area of the triangle.

    Raises
    ------
    ValueError
        If the vertices do not define a valid 2D triangle.
    """
    if verts.shape != (3, 2):
        raise ValueError(f"verts must have shape (3, 2), got {verts.shape}")

    # Vectors corresponding to two edges
    v1 = verts[1] - verts[0]
    v2 = verts[2] - verts[0]

    # Cross product in 2D (z-component of 3D cross product)
    # Area = 0.5 * (x1*y2 - x2*y1)
    area = 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])

    return area


def p1_gradients(
    verts: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """Compute P1 shape-function gradients and area on a triangle.

    For P1 elements on a triangle, the gradients are constant across the element.
    This function transforms the reference gradients to physical coordinates.

    Parameters
    ----------
    verts :
        Array with shape ``(3, 2)`` containing the vertex coordinates.

    Returns
    -------
    grads :
        Array with shape ``(3, 2)`` whose rows contain the gradients of
        the barycentric coordinates (P1 shape functions).
    area :
        Area of the triangle.

    Raises
    ------
    ValueError
        If the triangle is degenerate or has non-positive area.
    """
    if verts.shape != (3, 2):
        raise ValueError(f"verts must have shape (3, 2), got {verts.shape}")

    # Compute area (signed)
    area = triangle_area(verts)

    if area <= 1e-14:
        raise ValueError(f"Degenerate triangle or negative orientation. Area: {area}")

    # Jacobian matrix J = [x1-x0, x2-x0]
    #                     [y1-y0, y2-y0]
    J = np.column_stack((verts[1] - verts[0], verts[2] - verts[0]))

    # Reference gradients (shape 3x2)
    # phi_0 = 1 - xi - eta  => grad = [-1, -1]
    # phi_1 = xi            => grad = [ 1,  0]
    # phi_2 = eta           => grad = [ 0,  1]
    ref_grads = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    # Physical gradients: grad_phi = J^{-T} * grad_ref_phi
    # But here ref_grads has gradients as rows.
    # So we want (J^{-T} @ ref_grads.T).T = ref_grads @ J^{-1}

    J_inv = np.linalg.inv(J)
    grads = ref_grads @ J_inv

    return grads, area


def square_area(verts: NDArray[np.floating]) -> float:
    """Compute the area of a quadrilateral in physical coordinates.

    Uses the shoelace formula for a quadrilateral. Assumes vertices are
    ordered counter-clockwise.

    Parameters
    ----------
    verts :
        Array with shape ``(4, 2)`` containing the coordinates of the
        quadrilateral vertices in counter-clockwise order.

    Returns
    -------
    float
        Area of the quadrilateral.

    Raises
    ------
    ValueError
        If the vertices do not define a valid 2D quadrilateral.
    """
    if verts.shape != (4, 2):
        raise ValueError("verts must have shape (4, 2) for a 2D quadrilateral.")

    # Shoelace formula: Area = 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    x = verts[:, 0]
    y = verts[:, 1]

    # Compute cross products with wrap-around
    area = 0.5 * abs(
        x[0] * y[1]
        - x[1] * y[0]
        + x[1] * y[2]
        - x[2] * y[1]
        + x[2] * y[3]
        - x[3] * y[2]
        + x[3] * y[0]
        - x[0] * y[3]
    )

    return float(area)


def q1_gradients(
    verts: NDArray[np.floating],
    xi: float,
    eta: float,
) -> tuple[NDArray[np.floating], float]:
    """Compute Q1 shape-function gradients and Jacobian determinant.

    Evaluates the gradients of bilinear (Q1) shape functions at a point
    in reference coordinates (xi, eta) ∈ [-1, 1]^2.

    The Q1 shape functions on the reference square [-1, 1]^2 are:
        N0(ξ, η) = (1 - ξ)(1 - η) / 4
        N1(ξ, η) = (1 + ξ)(1 - η) / 4
        N2(ξ, η) = (1 + ξ)(1 + η) / 4
        N3(ξ, η) = (1 - ξ)(1 + η) / 4

    Parameters
    ----------
    verts :
        Array with shape ``(4, 2)`` containing the vertex coordinates
        in counter-clockwise order: [v0, v1, v2, v3].
    xi :
        First reference coordinate, ξ ∈ [-1, 1].
    eta :
        Second reference coordinate, η ∈ [-1, 1].

    Returns
    -------
    grads :
        Array with shape ``(4, 2)`` whose rows contain the physical-space
        gradients of the Q1 shape functions: [∂N_i/∂x, ∂N_i/∂y].
    detJ :
        Jacobian determinant at the evaluation point.

    Raises
    ------
    ValueError
        If the vertices do not define a valid quadrilateral or if the
        Jacobian is non-positive (degenerate or inverted element).
    """
    if verts.shape != (4, 2):
        raise ValueError("verts must have shape (4, 2) for a 2D quadrilateral.")

    # Get reference gradients from helper function
    # Shape: (4, 2) where rows are [dN_i/dξ, dN_i/dη]
    ref_grads = q1_square_reference_gradients(xi, eta)  # Shape (4, 2)

    # Jacobian matrix: J = [dx/dξ  dy/dξ]
    #                      [dx/dη  dy/dη]
    # Computed as J = ref_grads^T @ verts
    # Note: J = [x1-x0, x2-x0; y1-y0, y2-y0] relates reference to physical coords
    J = ref_grads.T @ verts  # Shape: (2, 2)

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    # detJ can change sign if vertices are ordered "backwards" (clockwise instead
    # of counter-clockwise). A negative detJ indicates an inverted element, which
    # we assume is a mesh error and reject.
    if detJ <= 0.0:
        raise ValueError(f"Degenerate or inverted quadrilateral (detJ = {detJ:.6e}).")

    # Inverse Jacobian
    J_inv = (
        np.array(
            [
                [J[1, 1], -J[0, 1]],
                [-J[1, 0], J[0, 0]],
            ],
            dtype=float,
        )
        / detJ
    )

    # Transform derivatives: [dN/dx, dN/dy] = [dN/dξ, dN/dη] @ J^{-T}
    grads = ref_grads @ J_inv.T

    return grads, float(detJ)
