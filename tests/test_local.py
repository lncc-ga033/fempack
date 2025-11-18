"""Tests for local module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.local import (
    local_stiffness_p1_interval,
    local_mass_p1_interval,
    local_stiffness_p1_triangle,
    local_mass_p1_triangle,
    local_stiffness_q1_square,
    local_mass_q1_square,
    local_load_p1_interval,
    local_load_p1_triangle,
    local_load_q1_square,
)
from fempack.elements import LagrangeElement
from fempack.quadrature import (
    gauss_legendre_interval,
    triangle_quadrature,
    gauss_legendre_reference_interval,
    tensor_product_square_reference,
)


def test_local_stiffness_p1_interval() -> None:
    """Test local stiffness matrix for P1 interval element."""
    k = local_stiffness_p1_interval(0.0, 1.0)

    assert k.shape == (2, 2)
    # For unit interval, stiffness should be [[1, -1], [-1, 1]]
    expected = np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k, expected)


def test_local_stiffness_p1_interval_scaled() -> None:
    """Test local stiffness matrix scales correctly with element size."""
    h = 0.5
    k = local_stiffness_p1_interval(0.0, h)

    # Stiffness should scale as 1/h
    expected = (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
    assert np.allclose(k, expected)


def test_local_stiffness_p1_interval_invalid() -> None:
    """Test that invalid interval raises error."""
    with pytest.raises(ValueError, match="Degenerate"):
        local_stiffness_p1_interval(1.0, 1.0)


def test_local_mass_p1_interval() -> None:
    """Test local mass matrix for P1 interval element."""
    m = local_mass_p1_interval(0.0, 1.0)

    assert m.shape == (2, 2)
    # For unit interval, mass should be (1/6)*[[2, 1], [1, 2]]
    expected = (1.0 / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    assert np.allclose(m, expected)


def test_local_mass_p1_interval_scaled() -> None:
    """Test local mass matrix scales correctly with element size."""
    h = 2.0
    m = local_mass_p1_interval(0.0, h)

    # Mass should scale as h
    expected = (h / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])
    assert np.allclose(m, expected)


@pytest.mark.grading
def test_local_stiffness_p1_triangle() -> None:
    """Test local stiffness matrix for P1 triangle element."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    k = local_stiffness_p1_triangle(verts)

    assert k.shape == (3, 3)
    # Should be symmetric
    assert np.allclose(k, k.T)
    # Row sums should be zero (for constant null space)
    assert np.allclose(np.sum(k, axis=1), 0.0)


@pytest.mark.grading
def test_local_stiffness_p1_triangle_positive_definite() -> None:
    """Test that local stiffness is positive semi-definite."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], dtype=float)
    k = local_stiffness_p1_triangle(verts)

    # All eigenvalues should be non-negative
    eigvals = np.linalg.eigvalsh(k)
    assert np.all(eigvals >= -1e-10)


@pytest.mark.grading
def test_local_mass_p1_triangle() -> None:
    """Test local mass matrix for P1 triangle element."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    m = local_mass_p1_triangle(verts)

    assert m.shape == (3, 3)
    # Should be symmetric
    assert np.allclose(m, m.T)
    # For unit right triangle (area = 0.5)
    # Mass matrix is (area/12)*[[2,1,1],[1,2,1],[1,1,2]]
    area = 0.5
    expected = (area / 12.0) * np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]])
    assert np.allclose(m, expected)


@pytest.mark.grading
def test_local_mass_p1_triangle_sum() -> None:
    """Test that mass matrix integrates constants correctly."""
    verts = np.array([[1.0, 1.0], [4.0, 1.0], [2.0, 5.0]], dtype=float)
    m = local_mass_p1_triangle(verts)

    # Sum of all entries should equal area of triangle
    area = 6.0
    assert np.isclose(np.sum(m), area)


def test_local_stiffness_q1_square() -> None:
    """Test local stiffness matrix for Q1 square element."""
    verts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    element = LagrangeElement(cell_type="square", degree=1)

    Q1d = gauss_legendre_reference_interval(2)
    Q = tensor_product_square_reference(Q1d)

    k = local_stiffness_q1_square(verts, element, Q.points, Q.weights)

    assert k.shape == (4, 4)
    # Should be symmetric
    assert np.allclose(k, k.T, atol=1e-12)
    # Row sums should be approximately zero
    assert np.allclose(np.sum(k, axis=1), 0.0, atol=1e-12)


def test_local_mass_q1_square() -> None:
    """Test local mass matrix for Q1 square element."""
    verts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    element = LagrangeElement(cell_type="square", degree=1)

    Q1d = gauss_legendre_reference_interval(2)
    Q = tensor_product_square_reference(Q1d)

    m = local_mass_q1_square(verts, element, Q.points, Q.weights)

    assert m.shape == (4, 4)
    # Should be symmetric
    assert np.allclose(m, m.T, atol=1e-12)
    # Sum should equal area (4.0 for reference square)
    assert np.isclose(np.sum(m), 4.0, rtol=1e-10)


def test_local_load_p1_interval() -> None:
    """Test local load vector for P1 interval element."""
    element = LagrangeElement(cell_type="interval", degree=1)
    Q = gauss_legendre_interval(2)

    # Constant load f(x) = 1
    def f(x: float) -> float:
        return 1.0

    b = local_load_p1_interval(0.0, 1.0, f, element, Q.points, Q.weights)

    assert b.shape == (2,)
    # For constant load on unit interval, should be [0.5, 0.5]
    assert np.allclose(b, [0.5, 0.5])


def test_local_load_p1_interval_linear() -> None:
    """Test local load vector for linear forcing."""
    element = LagrangeElement(cell_type="interval", degree=1)
    Q = gauss_legendre_interval(3)

    # Linear load f(x) = x
    def f(x: float) -> float:
        return x

    b = local_load_p1_interval(0.0, 1.0, f, element, Q.points, Q.weights)

    # Integral of x*phi_i should give different values for each node
    assert b.shape == (2,)
    assert b[0] < b[1]  # More load on right node


@pytest.mark.grading
def test_local_load_p1_triangle() -> None:
    """Test local load vector for P1 triangle element."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    element = LagrangeElement(cell_type="triangle", degree=1)
    Q = triangle_quadrature(order=2)

    # Constant load f(x, y) = 1
    def f(x: float, y: float) -> float:
        return 1.0

    b = local_load_p1_triangle(verts, f, element, Q.points, Q.weights)

    assert b.shape == (3,)
    # For constant load, each node gets 1/3 of area (area = 0.5)
    area = 0.5
    expected_per_node = area / 3.0
    assert np.allclose(b, [expected_per_node] * 3, rtol=1e-10)


@pytest.mark.grading
def test_local_load_p1_triangle_variable() -> None:
    """Test local load vector for variable forcing."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    element = LagrangeElement(cell_type="triangle", degree=1)
    Q = triangle_quadrature(order=2)

    # Load f(x, y) = x + y
    def f(x: float, y: float) -> float:
        return x + y

    b = local_load_p1_triangle(verts, f, element, Q.points, Q.weights)

    assert b.shape == (3,)
    # Should get different contributions at each node
    assert not np.allclose(b[0], b[1])


def test_local_load_q1_square() -> None:
    """Test local load vector for Q1 square element."""
    verts = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=float)
    element = LagrangeElement(cell_type="square", degree=1)

    Q1d = gauss_legendre_reference_interval(2)
    Q = tensor_product_square_reference(Q1d)

    # Constant load f(x, y) = 1
    def f(x: float, y: float) -> float:
        return 1.0

    b = local_load_q1_square(verts, f, element, Q.points, Q.weights)

    assert b.shape == (4,)
    # For constant load, each node gets 1/4 of area (area = 4.0)
    expected_per_node = 1.0
    assert np.allclose(b, [expected_per_node] * 4, rtol=1e-10)


@pytest.mark.grading
def test_local_matrices_symmetry() -> None:
    """Test that local matrices satisfy expected symmetries."""
    # Triangle
    verts_tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    k_tri = local_stiffness_p1_triangle(verts_tri)
    m_tri = local_mass_p1_triangle(verts_tri)

    assert np.allclose(k_tri, k_tri.T)
    assert np.allclose(m_tri, m_tri.T)

    # Interval
    k_int = local_stiffness_p1_interval(0.0, 1.0)
    m_int = local_mass_p1_interval(0.0, 1.0)

    assert np.allclose(k_int, k_int.T)
    assert np.allclose(m_int, m_int.T)
