from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fempack import reference
from fempack.mesh import CellType


@dataclass
class LagrangeElement:
    """Lagrange element of degree 1 or 2.

    Parameters
    ----------
    cell_type :
        Reference cell type: ``"interval"``, ``"triangle"`` or
        ``"square"``.
    degree :
        Polynomial degree. For intervals: degree 1 (P1) and 2 (P2) are
        supported. For triangles and squares: only degree 1 is implemented.

    Notes
    -----
    The tabulation routines follow the standard reference-element
    definitions found in most FEM textbooks (e.g. Becker-Carey-Oden).
    For quadrilaterals we use the isoparametric Q1 element on the
    reference square ``[-1, 1] x [-1, 1]``.

    For P2 on intervals, we use the reference interval [0, 1] with three
    nodes at ξ = 0, 0.5, 1, and the Lagrange basis functions:
        N0(ξ) = 2(ξ - 0.5)(ξ - 1)     (left endpoint)
        N1(ξ) = -4ξ(ξ - 1)             (midpoint)
        N2(ξ) = 2ξ(ξ - 0.5)            (right endpoint)

    **Important**: In this course, degree 2 elements are used only for
    local 1D experiments and verification. Global finite element spaces
    (:class:`~fempack.spaces.FunctionSpace`) are restricted to degree 1.
    """

    cell_type: CellType
    degree: int

    def __post_init__(self) -> None:
        if self.cell_type not in ("interval", "triangle", "square"):
            raise ValueError("Unsupported cell_type for LagrangeElement.")
        if self.degree not in (1, 2):
            raise NotImplementedError("Only degrees 1 and 2 are implemented.")
        if self.degree == 2 and self.cell_type != "interval":
            raise NotImplementedError("Degree 2 is only supported for intervals.")

    @property
    def dim(self) -> int:
        """Spatial dimension of the reference element."""
        if self.cell_type == "interval":
            return 1
        if self.cell_type in ("triangle", "square"):
            return 2
        raise ValueError("Invalid cell_type.")

    @property
    def dofs_per_cell(self) -> int:
        """Number of local degrees of freedom on the cell."""
        if self.cell_type == "interval":
            return self.degree + 1  # P1: 2 nodes, P2: 3 nodes
        if self.cell_type == "triangle":
            return 3
        if self.cell_type == "square":
            return 4
        raise ValueError("Invalid cell_type.")

    def tabulate(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate shape functions at reference points.

        Parameters
        ----------
        points :
            Array of reference coordinates with shape ``(nq, dim)``. For
            convenience a one-dimensional array of length ``nq`` is also
            accepted in the 1D case.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(nq, dofs_per_cell)`` containing the values
            of the shape functions.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, None]
        nq, dim = pts.shape
        if dim != self.dim:
            raise ValueError("Point dimension does not match element dimension.")

        if self.cell_type == "interval":
            x = pts[:, 0]
            if self.degree == 1:
                return reference.p1_interval_shape_functions(x)
            elif self.degree == 2:
                return reference.p2_interval_shape_functions(x)

        if self.cell_type == "triangle":
            x = pts[:, 0]
            y = pts[:, 1]
            return reference.p1_triangle_shape_functions(x, y)

        if self.cell_type == "square":
            xi = pts[:, 0]
            eta = pts[:, 1]
            return reference.q1_square_shape_functions(xi, eta)

        raise ValueError("Unsupported cell_type.")

    def tabulate_reference_gradients(self, points: NDArray[np.floating]) -> NDArray[np.floating]:
        """Evaluate gradients of shape functions in reference coordinates.

        Parameters
        ----------
        points :
            Array of reference coordinates with shape ``(nq, dim)``.

        Returns
        -------
        numpy.ndarray
            Array with shape ``(nq, dofs_per_cell, dim)`` containing
            the gradients of each shape function with respect to the
            reference coordinates.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts[:, None]
        nq, dim = pts.shape
        if dim != self.dim:
            raise ValueError("Point dimension does not match element dimension.")

        if self.cell_type == "interval":
            if self.degree == 1:
                # Use reference gradient computation from reference.py
                grad = reference.p1_interval_reference_gradients()
                grads = np.broadcast_to(grad, (nq,) + grad.shape)
                return grads
            elif self.degree == 2:
                x = pts[:, 0]
                # Use reference gradient computation from reference.py
                grads_ref = reference.p2_interval_reference_gradients(x)  # Shape (nq, 3)
                grads = grads_ref[:, :, None]  # Shape (nq, 3, 1)
                return grads

        if self.cell_type == "triangle":
            # Use reference gradient computation from reference.py
            grad = reference.p1_triangle_reference_gradients()
            grads = np.broadcast_to(grad, (nq,) + grad.shape)
            return grads

        if self.cell_type == "square":
            xi = pts[:, 0]
            eta = pts[:, 1]
            # Use reference gradient computation from reference.py
            grads = reference.q1_square_reference_gradients(xi, eta)
            return grads

        raise ValueError("Unsupported cell_type.")
