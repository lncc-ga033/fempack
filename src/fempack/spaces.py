from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from fempack.mesh import Mesh
from fempack.elements import LagrangeElement


@dataclass
class FunctionSpace:
    """Finite element space of Lagrange order 1 (P1/Q1).

    Parameters
    ----------
    mesh :
        Underlying :class:`~fempack.mesh.Mesh` object.
    family :
        Finite element family. At the moment only ``"Lagrange"`` is
        supported.
    degree :
        Polynomial degree. Only degree 1 is implemented.

    Notes
    -----
    The class is intentionally small: it only stores the mesh, the
    element description and the number of global degrees of freedom.
    It plays a similar role to Firedrake's :class:`FunctionSpace` but
    without the complexity of mixed spaces or higher-order elements.
    """

    mesh: Mesh
    family: str = "Lagrange"
    degree: int = 1

    def __post_init__(self) -> None:
        if self.family != "Lagrange":
            raise NotImplementedError("Only Lagrange elements are implemented.")
        if self.degree != 1:
            raise NotImplementedError("Only degree 1 elements are implemented.")
        if self.mesh.cell_type not in ("interval", "triangle", "square"):
            raise ValueError("Unsupported cell_type for FunctionSpace.")
        self.element = LagrangeElement(cell_type=self.mesh.cell_type, degree=self.degree)

    @property
    def ndofs(self) -> int:
        """Number of global degrees of freedom."""
        return int(self.mesh.num_vertices)

    @property
    def dim(self) -> int:
        """Spatial dimension of the finite element space."""
        return int(self.mesh.dim)


class Function:
    """Finite element function associated with a :class:`FunctionSpace`.

    Parameters
    ----------
    V :
        Finite element space to which the function belongs.
    values :
        Optional array of nodal values. If omitted, the function is
        initialised with zeros.

    Notes
    -----
    This is a minimal analogue of Firedrake's :class:`Function` class.
    It only stores a one-dimensional array of nodal values and does
    not implement interpolation or projection helpers.
    """

    V: FunctionSpace
    values: NDArray[np.floating]

    def __init__(self, V: FunctionSpace, values: Optional[ArrayLike] = None) -> None:
        self.V = V
        if values is None:
            self.values = np.zeros(V.ndofs, dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
            if arr.shape != (V.ndofs,):
                raise ValueError("values must be a 1D array of length ndofs.")
            self.values = arr

    def nodal_values(self) -> NDArray[np.floating]:
        """Return the nodal values as a NumPy array.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with the nodal values.
        """
        return self.values
