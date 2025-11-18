from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CellType = Literal["interval", "triangle", "square"]


@dataclass
class Mesh:
    """Structured mesh for simple FEM experiments.

    Parameters
    ----------
    coords :
        Array of vertex coordinates with shape ``(num_vertices, dim)``.
    cells :
        Array of cell connectivities with shape ``(num_cells, nloc)``.
    cell_type :
        Identifier of the reference cell: ``"interval"``, ``"triangle"``
        or ``"square"``.

    Notes
    -----
    This class intentionally mirrors the lightweight mesh containers
    used in teaching codes while keeping only the features needed
    in GA-033: structured grids on the unit interval and the unit square.
    """

    coords: NDArray[np.floating]
    cells: NDArray[np.integer]
    cell_type: CellType

    @property
    def dim(self) -> int:
        """Spatial dimension of the mesh."""
        return int(self.coords.shape[1])

    @property
    def num_vertices(self) -> int:
        """Number of mesh vertices."""
        return int(self.coords.shape[0])

    @property
    def num_cells(self) -> int:
        """Number of cells (elements) in the mesh."""
        return int(self.cells.shape[0])

    # ------------------------------------------------------------------
    # Structured mesh factories
    # ------------------------------------------------------------------
    @staticmethod
    def unit_interval(n: int) -> "Mesh":
        """Create a uniform mesh of the unit interval ``[0, 1]``.

        Parameters
        ----------
        n :
            Number of elements. The mesh will contain ``n + 1`` vertices.

        Returns
        -------
        Mesh
            Structured 1D mesh with P1 connectivity.
        """
        xs = np.linspace(0.0, 1.0, n + 1)
        coords: NDArray[np.floating] = xs[:, None].astype(float)
        cells = np.column_stack([np.arange(0, n, dtype=int), np.arange(1, n + 1, dtype=int)])
        return Mesh(coords=coords, cells=cells, cell_type="interval")

    @staticmethod
    def unit_square_triangular(nx: int, ny: int) -> "Mesh":
        """Create a P1 triangular mesh of the unit square.

        The domain is ``(0, 1) x (0, 1)`` and each rectangular cell is
        subdivided into two triangles along the main diagonal.

        Parameters
        ----------
        nx, ny :
            Number of cells in the x and y directions of the underlying
            rectangular grid.

        Returns
        -------
        Mesh
            Triangular mesh with linear (P1) connectivity.
        """
        xs = np.linspace(0.0, 1.0, nx + 1)
        ys = np.linspace(0.0, 1.0, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        coords: NDArray[np.floating] = np.column_stack([X.ravel(), Y.ravel()])

        def vid(i: int, j: int) -> int:
            return i * (ny + 1) + j

        cells_list: list[list[int]] = []
        for i in range(nx):
            for j in range(ny):
                v0 = vid(i, j)
                v1 = vid(i + 1, j)
                v2 = vid(i + 1, j + 1)
                v3 = vid(i, j + 1)
                # Lower-left triangle
                cells_list.append([v0, v1, v3])
                # Upper-right triangle
                cells_list.append([v2, v3, v1])
        cells: NDArray[np.integer] = np.array(cells_list, dtype=int)
        return Mesh(coords=coords.astype(float), cells=cells, cell_type="triangle")

    @staticmethod
    def unit_square_quadrilateral(nx: int, ny: int) -> "Mesh":
        """Create a Q1 quadrilateral mesh of the unit square.

        Parameters
        ----------
        nx, ny :
            Number of cells in the x and y directions.

        Returns
        -------
        Mesh
            Quadrilateral mesh with bilinear (Q1) connectivity.
        """
        xs = np.linspace(0.0, 1.0, nx + 1)
        ys = np.linspace(0.0, 1.0, ny + 1)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        coords: NDArray[np.floating] = np.column_stack([X.ravel(), Y.ravel()])

        def vid(i: int, j: int) -> int:
            return i * (ny + 1) + j

        cells_list: list[list[int]] = []
        for i in range(nx):
            for j in range(ny):
                v0 = vid(i, j)
                v1 = vid(i + 1, j)
                v2 = vid(i + 1, j + 1)
                v3 = vid(i, j + 1)
                cells_list.append([v0, v1, v2, v3])
        cells: NDArray[np.integer] = np.array(cells_list, dtype=int)
        return Mesh(coords=coords.astype(float), cells=cells, cell_type="square")

    # ------------------------------------------------------------------
    # Boundary nodes
    # ------------------------------------------------------------------
    def boundary_nodes(self) -> NDArray[np.int64]:
        """Return the indices of boundary vertices.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with the global indices of boundary
            vertices.

        Notes
        -----
        The implementation assumes that the physical domain is the
        unit interval or the unit square. For other domains this helper
        should be replaced by a problem-specific boundary description.
        """
        if self.dim == 1:
            return np.array([0, self.num_vertices - 1], dtype=np.int64)

        if self.dim == 2:
            x = self.coords[:, 0]
            y = self.coords[:, 1]
            tol = 1.0e-14
            mask = (
                (np.abs(x) < tol)
                | (np.abs(x - 1.0) < tol)
                | (np.abs(y) < tol)
                | (np.abs(y - 1.0) < tol)
            )
            return np.nonzero(mask)[0].astype(np.int64)

        msg = "boundary_nodes is only implemented for dim=1 or dim=2."
        raise NotImplementedError(msg)
