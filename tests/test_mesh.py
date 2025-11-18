"""Tests for mesh module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.mesh import Mesh


def test_unit_interval_basic() -> None:
    """Test basic properties of unit interval mesh."""
    mesh = Mesh.unit_interval(10)

    assert mesh.cell_type == "interval"
    assert mesh.dim == 1
    assert mesh.num_vertices == 11
    assert mesh.num_cells == 10
    assert mesh.coords.shape == (11, 1)
    assert mesh.cells.shape == (10, 2)


def test_unit_interval_coords() -> None:
    """Test coordinates of unit interval mesh."""
    mesh = Mesh.unit_interval(4)

    # Should have vertices at 0, 0.25, 0.5, 0.75, 1.0
    expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(mesh.coords[:, 0], expected)


def test_unit_interval_connectivity() -> None:
    """Test connectivity of unit interval mesh."""
    mesh = Mesh.unit_interval(3)

    # Should have cells [0,1], [1,2], [2,3]
    expected = np.array([[0, 1], [1, 2], [2, 3]], dtype=int)
    assert np.array_equal(mesh.cells, expected)


def test_unit_interval_boundary_nodes() -> None:
    """Test boundary nodes of unit interval mesh."""
    mesh = Mesh.unit_interval(5)
    boundary = mesh.boundary_nodes()

    # Should be first and last node
    assert np.array_equal(boundary, [0, 5])


def test_unit_square_triangular_basic() -> None:
    """Test basic properties of triangular unit square mesh."""
    mesh = Mesh.unit_square_triangular(2, 2)

    assert mesh.cell_type == "triangle"
    assert mesh.dim == 2
    assert mesh.num_vertices == 9  # 3x3 grid
    assert mesh.num_cells == 8  # 2 triangles per square
    assert mesh.coords.shape == (9, 2)
    assert mesh.cells.shape == (8, 3)


def test_unit_square_triangular_corners() -> None:
    """Test corner coordinates of triangular mesh."""
    mesh = Mesh.unit_square_triangular(1, 1)

    # Should have corners at (0,0), (1,0), (1,1), (0,1)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    for corner in corners:
        # Check that corner exists in coords
        found = np.any(np.all(np.isclose(mesh.coords, corner), axis=1))
        assert found, f"Corner {corner} not found in mesh"


def test_unit_square_triangular_connectivity() -> None:
    """Test that all triangles have positive area."""
    mesh = Mesh.unit_square_triangular(2, 2)

    for cell in mesh.cells:
        verts = mesh.coords[cell, :]
        # Compute area using cross product
        v0, v1, v2 = verts
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * (edge1[0] * edge2[1] - edge1[1] * edge2[0])
        assert area > 0, "Triangle has non-positive area"


def test_unit_square_triangular_boundary_nodes() -> None:
    """Test boundary nodes of triangular mesh."""
    mesh = Mesh.unit_square_triangular(2, 2)
    boundary = mesh.boundary_nodes()

    # Boundary nodes are those on edges x=0, x=1, y=0, y=1
    coords = mesh.coords
    x = coords[:, 0]
    y = coords[:, 1]

    tol = 1e-14
    expected_boundary = np.where(
        (np.abs(x) < tol) | (np.abs(x - 1.0) < tol) | (np.abs(y) < tol) | (np.abs(y - 1.0) < tol)
    )[0]

    assert np.array_equal(sorted(boundary), sorted(expected_boundary))


def test_unit_square_quadrilateral_basic() -> None:
    """Test basic properties of quadrilateral unit square mesh."""
    mesh = Mesh.unit_square_quadrilateral(3, 2)

    assert mesh.cell_type == "square"
    assert mesh.dim == 2
    assert mesh.num_vertices == 12  # 4x3 grid
    assert mesh.num_cells == 6  # 3x2 quads
    assert mesh.coords.shape == (12, 2)
    assert mesh.cells.shape == (6, 4)


def test_unit_square_quadrilateral_connectivity() -> None:
    """Test connectivity of quadrilateral mesh."""
    mesh = Mesh.unit_square_quadrilateral(1, 1)

    # Should have 4 vertices and 1 cell
    assert mesh.num_vertices == 4
    assert mesh.num_cells == 1

    # Cell should connect all 4 vertices in counter-clockwise order
    cell = mesh.cells[0]
    assert len(cell) == 4
    assert len(np.unique(cell)) == 4  # All vertices are different


def test_unit_square_quadrilateral_boundary_nodes() -> None:
    """Test boundary nodes of quadrilateral mesh."""
    mesh = Mesh.unit_square_quadrilateral(2, 2)
    boundary = mesh.boundary_nodes()

    # Check that boundary nodes are on the edges
    coords = mesh.coords[boundary]
    x = coords[:, 0]
    y = coords[:, 1]

    tol = 1e-14
    on_boundary = (
        (np.abs(x) < tol) | (np.abs(x - 1.0) < tol) | (np.abs(y) < tol) | (np.abs(y - 1.0) < tol)
    )

    assert np.all(on_boundary)


def test_mesh_boundary_nodes_3d_raises() -> None:
    """Test that boundary_nodes raises error for 3D (not implemented)."""
    # Create a dummy 3D mesh
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    cells = np.array([[0, 1, 2]], dtype=int)
    mesh = Mesh(coords=coords, cells=cells, cell_type="triangle")

    with pytest.raises(NotImplementedError):
        mesh.boundary_nodes()


def test_unit_square_triangular_uniform_refinement() -> None:
    """Test that refining gives expected number of elements."""
    mesh1 = Mesh.unit_square_triangular(2, 2)
    mesh2 = Mesh.unit_square_triangular(4, 4)

    # Doubling resolution should give 4x more triangles
    assert mesh2.num_cells == 4 * mesh1.num_cells


def test_unit_interval_single_element() -> None:
    """Test unit interval with single element."""
    mesh = Mesh.unit_interval(1)

    assert mesh.num_vertices == 2
    assert mesh.num_cells == 1
    assert np.allclose(mesh.coords[:, 0], [0.0, 1.0])
    assert np.array_equal(mesh.cells[0], [0, 1])
