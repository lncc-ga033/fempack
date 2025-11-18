from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fempack.reference import INTERVAL, TRIANGLE, SQUARE, ReferenceCell


@dataclass
class QuadratureRule:
    """Numerical quadrature rule on a reference cell.

    Parameters
    ----------
    cell :
        Reference cell on which the rule is defined.
    points :
        Array with shape ``(nq, dim)`` containing quadrature points in
        reference coordinates.
    weights :
        One-dimensional array of quadrature weights.
    degree :
        Algebraic degree of exactness of the rule.
    """

    cell: ReferenceCell
    points: NDArray[np.floating]
    weights: NDArray[np.floating]
    degree: int

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        wts = np.asarray(self.weights, dtype=float)
        if pts.ndim != 2:
            raise ValueError("points must have shape (nq, dim).")
        if wts.ndim != 1 or wts.shape[0] != pts.shape[0]:
            raise ValueError("weights must have shape (nq,).")
        self.points = pts
        self.weights = wts


def gauss_legendre_interval(n: int) -> QuadratureRule:
    """Gauss-Legendre rule on the physical unit interval ``[0, 1]``.

    Parameters
    ----------
    n :
        Number of quadrature points.

    Returns
    -------
    QuadratureRule
        Quadrature rule exact for polynomials up to degree ``2n - 1``.
    """
    xi, w = np.polynomial.legendre.leggauss(n)
    pts = 0.5 * (xi + 1.0)
    weights = 0.5 * w  # required since we map the quadrature from [-1, 1] to [0, 1]
    return QuadratureRule(cell=INTERVAL, points=pts[:, None], weights=weights, degree=2 * n - 1)


def gauss_legendre_reference_interval(n: int) -> QuadratureRule:
    """
    Gauss-Legendre rule on the reference interval ``[-1, 1]``.

    Parameters
    ----------
    n :
        Number of quadrature points.
    Returns
    -------
    QuadratureRule
        Quadrature rule exact for polynomials up to degree ``2n - 1``.
    """
    xi, w = np.polynomial.legendre.leggauss(n)
    return QuadratureRule(cell=INTERVAL, points=xi[:, None], weights=w, degree=2 * n - 1)


def tensor_product_square_reference(rule_1d: QuadratureRule) -> QuadratureRule:
    """Build a tensor-product rule on the reference square.

    Parameters
    ----------
    rule_1d :
        One-dimensional quadrature rule on ``[-1, 1]``.

    Returns
    -------
    QuadratureRule
        Tensor-product rule on the reference square with the same
        degree of exactness in each coordinate direction.
    """
    if rule_1d.cell != INTERVAL:
        raise ValueError("rule_1d must be defined on the interval.")
    x = rule_1d.points[:, 0]
    wx = rule_1d.weights
    X, Y = np.meshgrid(x, x, indexing="ij")
    WX, WY = np.meshgrid(wx, wx, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    weights = (WX * WY).ravel()
    return QuadratureRule(cell=SQUARE, points=pts, weights=weights, degree=rule_1d.degree)


def triangle_quadrature(order: int = 1) -> QuadratureRule:
    """Low-order quadrature rules on the reference triangle.

    Parameters
    ----------
    order :
        Requested order (1 or 2).

    Returns
    -------
    QuadratureRule
        Quadrature rule on the reference triangle with area ``1/2``.

    Raises
    ------
    NotImplementedError
        If an unsupported order is requested.

    Notes
    -----
    The order 2 rule uses quadrature points at the edge midpoints,
    following Oden, Carey, and Becker (chap. 5).
    """
    if order == 1:
        # lam here means the barycentric coordinates, the triplet
        # (lam0, lam1, lam2) in Figure 5.14 of Oden, Carey, and Becker (chap. 5)
        lam = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]], dtype=float)
        # Note that the area of the reference triangle is 0.5. Due to this,
        # the weight for the single quadrature point is 0.5 since the
        # integral of 1 over the triangle is 0.5.
        w = np.array([0.5], dtype=float)
    elif order == 2:
        # 3-point rule with quadrature points at edge midpoints
        # Following Oden, Carey, and Becker (Figure 5.14)
        lam = np.array(
            [
                [0.5, 0.0, 0.5],  # midpoint of edge between vertices 2 and 3
                [0.5, 0.5, 0.0],  # midpoint of edge between vertices 1 and 3
                [0.0, 0.5, 0.5],  # midpoint of edge between vertices 1 and 2
            ],
            dtype=float,
        )
        # For order 2 we have 3 quadrature points, each with weight 1/6. The sum of
        # the weights is 0.5, which is the area of the reference triangle.
        w = np.full(3, 1.0 / 6.0, dtype=float)
    else:
        raise NotImplementedError("triangle_quadrature is implemented only for order=1 or 2.")

    xi = lam[:, 1]
    eta = lam[:, 2]
    pts = np.column_stack([xi, eta])
    return QuadratureRule(cell=TRIANGLE, points=pts, weights=w, degree=order)
