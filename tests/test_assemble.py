"""Tests for assemble module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_mass, assemble_load


def test_assemble_stiffness_1d() -> None:
    """Test stiffness matrix assembly in 1D."""
    mesh = Mesh.unit_interval(3)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)

    assert A.shape == (4, 4)
    # Should be symmetric
    assert np.allclose(A.toarray(), A.toarray().T)
    # Should be positive semi-definite
    eigvals = np.linalg.eigvalsh(A.toarray())
    assert np.all(eigvals >= -1e-10)


def test_assemble_stiffness_1d_structure() -> None:
    """Test structure of 1D stiffness matrix."""
    mesh = Mesh.unit_interval(2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V).toarray()

    # For uniform mesh with h=0.5, stiffness should be
    # [[2, -2, 0], [-2, 4, -2], [0, -2, 2]]
    h = 0.5
    scale = 1.0 / h
    expected = scale * np.array([[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    assert np.allclose(A, expected)


@pytest.mark.grading
def test_assemble_stiffness_2d_triangle() -> None:
    """Test stiffness matrix assembly for 2D triangular mesh."""
    mesh = Mesh.unit_square_triangular(2, 2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)

    assert A.shape == (9, 9)
    # Should be symmetric
    assert np.allclose(A.toarray(), A.toarray().T, atol=1e-12)
    # Should be positive semi-definite
    eigvals = np.linalg.eigvalsh(A.toarray())
    assert np.all(eigvals >= -1e-10)


def test_assemble_stiffness_2d_quad() -> None:
    """Test stiffness matrix assembly for 2D quadrilateral mesh."""
    mesh = Mesh.unit_square_quadrilateral(2, 2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)

    assert A.shape == (9, 9)
    # Should be symmetric
    assert np.allclose(A.toarray(), A.toarray().T, atol=1e-12)


def test_assemble_mass_1d() -> None:
    """Test mass matrix assembly in 1D."""
    mesh = Mesh.unit_interval(3)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    M = assemble_mass(V)

    assert M.shape == (4, 4)
    # Should be symmetric
    assert np.allclose(M.toarray(), M.toarray().T)
    # Should be positive definite
    eigvals = np.linalg.eigvalsh(M.toarray())
    assert np.all(eigvals > 0.0)


def test_assemble_mass_1d_sum() -> None:
    """Test that mass matrix sums to domain measure."""
    mesh = Mesh.unit_interval(10)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    M = assemble_mass(V)

    # Sum of all entries should equal length of domain (1.0)
    ones = np.ones(V.ndofs)
    integral = ones @ M @ ones
    assert np.isclose(integral, 1.0)


@pytest.mark.grading
def test_assemble_mass_2d_triangle() -> None:
    """Test mass matrix assembly for 2D triangular mesh."""
    mesh = Mesh.unit_square_triangular(2, 2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    M = assemble_mass(V)

    assert M.shape == (9, 9)
    # Should be symmetric
    assert np.allclose(M.toarray(), M.toarray().T, atol=1e-12)
    # Should be positive definite
    eigvals = np.linalg.eigvalsh(M.toarray())
    assert np.all(eigvals > 0.0)


@pytest.mark.grading
def test_assemble_mass_2d_sum() -> None:
    """Test that 2D mass matrix sums to domain area."""
    mesh = Mesh.unit_square_triangular(5, 5)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    M = assemble_mass(V)

    # Sum should equal area of unit square (1.0)
    ones = np.ones(V.ndofs)
    integral = ones @ M @ ones
    assert np.isclose(integral, 1.0, rtol=1e-10)


def test_assemble_load_1d_constant() -> None:
    """Test load vector assembly in 1D with constant forcing."""
    mesh = Mesh.unit_interval(4)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    def f(x: float) -> float:
        return 1.0

    b = assemble_load(V, f, quad_order=2)

    assert b.shape == (5,)
    # Integral of constant forcing over unit interval should be 1.0
    assert np.isclose(np.sum(b), 1.0)


def test_assemble_load_1d_linear() -> None:
    """Test load vector assembly in 1D with linear forcing."""
    mesh = Mesh.unit_interval(4)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    def f(x: float) -> float:
        return x

    b = assemble_load(V, f, quad_order=3)

    assert b.shape == (5,)
    # Entries should increase (more load at right)
    assert b[0] < b[-1]


@pytest.mark.grading
def test_assemble_load_2d_constant() -> None:
    """Test load vector assembly in 2D with constant forcing."""
    mesh = Mesh.unit_square_triangular(3, 3)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    def f(x: float, y: float) -> float:
        return 1.0

    b = assemble_load(V, f, quad_order=2)

    assert b.shape == (16,)
    # Integral of constant forcing over unit square should be 1.0
    assert np.isclose(np.sum(b), 1.0, rtol=1e-10)


@pytest.mark.grading
def test_assemble_load_2d_variable() -> None:
    """Test load vector assembly in 2D with variable forcing."""
    mesh = Mesh.unit_square_triangular(2, 2)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    def f(x: float, y: float) -> float:
        return x + y

    b = assemble_load(V, f, quad_order=2)

    assert b.shape == (9,)
    # Check that load is different at different nodes
    assert not np.allclose(b, b[0])


def test_assemble_matrices_sparsity() -> None:
    """Test that assembled matrices are sparse."""
    mesh = Mesh.unit_interval(100)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)
    M = assemble_mass(V)

    # Check that matrices are sparse (less than 10% non-zero)
    density_A = A.nnz / (A.shape[0] * A.shape[1])
    density_M = M.nnz / (M.shape[0] * M.shape[1])

    assert density_A < 0.1
    assert density_M < 0.1


@pytest.mark.grading
def test_assemble_consistency_triangle_vs_quad() -> None:
    """Test that triangle and quad meshes give similar results."""
    n = 4
    mesh_tri = Mesh.unit_square_triangular(n, n)
    mesh_quad = Mesh.unit_square_quadrilateral(n, n)

    V_tri = FunctionSpace(mesh_tri, family="Lagrange", degree=1)
    V_quad = FunctionSpace(mesh_quad, family="Lagrange", degree=1)

    # Both should have same number of DOFs
    assert V_tri.ndofs == V_quad.ndofs

    # Assemble load vectors with constant forcing
    def f(x: float, y: float) -> float:
        return 1.0

    b_tri = assemble_load(V_tri, f, quad_order=2)
    b_quad = assemble_load(V_quad, f, quad_order=2)

    # Both should sum to 1.0 (area of domain)
    assert np.isclose(np.sum(b_tri), 1.0, rtol=1e-10)
    assert np.isclose(np.sum(b_quad), 1.0, rtol=1e-10)


@pytest.mark.grading
def test_assemble_refinement_convergence() -> None:
    """Test that assembly produces consistent results under refinement."""

    def f(x: float, y: float) -> float:
        return 1.0

    # Coarse mesh
    mesh1 = Mesh.unit_square_triangular(2, 2)
    V1 = FunctionSpace(mesh1, family="Lagrange", degree=1)
    b1 = assemble_load(V1, f, quad_order=2)

    # Fine mesh
    mesh2 = Mesh.unit_square_triangular(4, 4)
    V2 = FunctionSpace(mesh2, family="Lagrange", degree=1)
    b2 = assemble_load(V2, f, quad_order=2)

    # Both should integrate to same value
    assert np.isclose(np.sum(b1), np.sum(b2), rtol=1e-10)
    assert np.isclose(np.sum(b1), 1.0, rtol=1e-10)
