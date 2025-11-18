"""Tests for spaces module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace


def test_function_space_1d_basic() -> None:
    """Test basic properties of 1D function space."""
    mesh = Mesh.unit_interval(5)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    assert V.mesh is mesh
    assert V.ndofs == 6
    assert V.element.cell_type == "interval"
    assert V.element.degree == 1


def test_function_space_2d_triangular() -> None:
    """Test 2D triangular function space."""
    mesh = Mesh.unit_square_triangular(3, 3)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    assert V.mesh is mesh
    assert V.ndofs == 16  # 4x4 grid of vertices
    assert V.element.cell_type == "triangle"
    assert V.element.degree == 1


def test_function_space_2d_quadrilateral() -> None:
    """Test 2D quadrilateral function space."""
    mesh = Mesh.unit_square_quadrilateral(2, 3)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    assert V.mesh is mesh
    assert V.ndofs == 12  # 3x4 grid of vertices
    assert V.element.cell_type == "square"
    assert V.element.degree == 1


def test_function_space_invalid_family() -> None:
    """Test that invalid family raises error."""
    mesh = Mesh.unit_interval(3)

    with pytest.raises(NotImplementedError, match="Only Lagrange"):
        FunctionSpace(mesh, family="Hermite", degree=1)


def test_function_space_invalid_degree() -> None:
    """Test that invalid degree raises error."""
    mesh = Mesh.unit_interval(3)

    with pytest.raises(NotImplementedError, match="Only degree 1"):
        FunctionSpace(mesh, family="Lagrange", degree=2)


def test_function_space_element_tabulation_1d() -> None:
    """Test element shape function tabulation in 1D."""
    mesh = Mesh.unit_interval(2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    # Evaluate at midpoint of reference interval [0, 1]
    xi = np.array([[0.5]])
    phi = V.element.tabulate(xi)

    assert phi.shape == (1, 2)
    # At midpoint, both basis functions should equal 0.5
    assert np.allclose(phi[0], [0.5, 0.5])


def test_function_space_element_tabulation_2d_triangle() -> None:
    """Test element shape function tabulation on triangle."""
    mesh = Mesh.unit_square_triangular(1, 1)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    # Evaluate at centroid (1/3, 1/3)
    xi = np.array([[1.0 / 3.0, 1.0 / 3.0]])
    phi = V.element.tabulate(xi)

    assert phi.shape == (1, 3)
    # At centroid, all three basis functions should equal 1/3
    assert np.allclose(phi[0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


def test_function_space_element_tabulation_partition_of_unity() -> None:
    """Test that shape functions sum to 1 (partition of unity)."""
    mesh = Mesh.unit_square_triangular(2, 2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    # Random points in reference triangle
    xi = np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]])
    phi = V.element.tabulate(xi)

    # Sum of basis functions at each point should be 1
    sums = np.sum(phi, axis=1)
    assert np.allclose(sums, 1.0)


def test_function_space_element_gradient_tabulation() -> None:
    """Test element gradient tabulation."""
    mesh = Mesh.unit_interval(2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    # Gradients on reference interval should be constant
    xi = np.array([[0.3], [0.7]])
    grad_phi = V.element.tabulate_reference_gradients(xi)

    assert grad_phi.shape == (2, 2, 1)
    # Gradients should be [-1, 1] at all points
    assert np.allclose(grad_phi[:, 0, 0], -1.0)
    assert np.allclose(grad_phi[:, 1, 0], 1.0)


def test_function_space_dof_count_refinement() -> None:
    """Test that DOF count scales correctly with refinement."""
    mesh1 = Mesh.unit_interval(10)
    mesh2 = Mesh.unit_interval(20)

    V1 = FunctionSpace(mesh1, family="Lagrange", degree=1)
    V2 = FunctionSpace(mesh2, family="Lagrange", degree=1)

    # DOFs should scale linearly with mesh resolution in 1D
    assert V2.ndofs == 2 * V1.ndofs - 1


def test_function_space_2d_dof_count() -> None:
    """Test DOF count for 2D meshes."""
    n = 5
    mesh_tri = Mesh.unit_square_triangular(n, n)
    mesh_quad = Mesh.unit_square_quadrilateral(n, n)

    V_tri = FunctionSpace(mesh_tri, family="Lagrange", degree=1)
    V_quad = FunctionSpace(mesh_quad, family="Lagrange", degree=1)

    # Both should have same number of DOFs (same vertices)
    assert V_tri.ndofs == V_quad.ndofs
    assert V_tri.ndofs == (n + 1) ** 2
