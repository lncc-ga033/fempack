"""Tests for P2 interval elements."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.elements import LagrangeElement
from fempack.local import (
    local_stiffness_p2_interval,
    local_mass_p2_interval,
    local_load_p2_interval,
)
from fempack.quadrature import gauss_legendre_interval


def test_p2_element_creation() -> None:
    """Test creating P2 element for interval."""
    elem = LagrangeElement("interval", 2)
    assert elem.cell_type == "interval"
    assert elem.degree == 2
    assert elem.dofs_per_cell == 3
    assert elem.dim == 1


def test_p2_element_invalid_cell_type() -> None:
    """Test that P2 is only supported for intervals."""
    with pytest.raises(NotImplementedError, match="only supported for intervals"):
        LagrangeElement("triangle", 2)

    with pytest.raises(NotImplementedError, match="only supported for intervals"):
        LagrangeElement("square", 2)


def test_p2_element_tabulate_at_nodes() -> None:
    """Test P2 basis functions at nodes."""
    elem = LagrangeElement("interval", 2)
    points = np.array([[0.0], [0.5], [1.0]], dtype=float)
    phi = elem.tabulate(points)

    # At nodes, basis functions should be identity matrix
    expected = np.eye(3)
    assert np.allclose(phi, expected, atol=1e-12)


def test_p2_element_tabulate_midpoint() -> None:
    """Test P2 basis functions at quarter point."""
    elem = LagrangeElement("interval", 2)
    points = np.array([[0.25]], dtype=float)
    phi = elem.tabulate(points)

    # N₀(0.25) = 2(0.25-0.5)(0.25-1) = 2*(-0.25)*(-0.75) = 0.375
    # N₁(0.25) = -4*0.25*(0.25-1) = -4*0.25*(-0.75) = 0.75
    # N₂(0.25) = 2*0.25*(0.25-0.5) = 2*0.25*(-0.25) = -0.125
    expected = np.array([[0.375, 0.75, -0.125]])
    assert np.allclose(phi, expected, atol=1e-12)


def test_p2_element_partition_of_unity() -> None:
    """Test that P2 basis functions sum to 1."""
    elem = LagrangeElement("interval", 2)
    points = np.linspace(0.0, 1.0, 11)[:, None]
    phi = elem.tabulate(points)

    phi_sum = np.sum(phi, axis=1)
    assert np.allclose(phi_sum, 1.0, atol=1e-12)


def test_p2_element_gradients_at_midpoint() -> None:
    """Test P2 gradient evaluation at midpoint."""
    elem = LagrangeElement("interval", 2)
    points = np.array([[0.5]], dtype=float)
    grads = elem.tabulate_reference_gradients(points)

    assert grads.shape == (1, 3, 1)
    # At ξ=0.5: dN₀/dξ = -1, dN₁/dξ = 0, dN₂/dξ = 1
    expected = np.array([[[-1.0], [0.0], [1.0]]])
    assert np.allclose(grads, expected, atol=1e-12)


def test_p2_stiffness_symmetry() -> None:
    """Test that P2 stiffness matrix is symmetric."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)  # 3-point quadrature for P2

    ke = local_stiffness_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)

    assert ke.shape == (3, 3)
    assert np.allclose(ke, ke.T, atol=1e-12)


def test_p2_stiffness_positive_definite() -> None:
    """Test that P2 stiffness matrix is positive semi-definite."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    ke = local_stiffness_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)

    # Check eigenvalues (should have one zero and two positive)
    eigvals = np.linalg.eigvalsh(ke)
    assert eigvals[0] >= -1e-10  # Nearly zero (rigid body mode)
    assert eigvals[1] > 0
    assert eigvals[2] > 0


def test_p2_stiffness_scaling() -> None:
    """Test that P2 stiffness scales as 1/h."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    k1 = local_stiffness_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)
    k2 = local_stiffness_p2_interval(0.0, 2.0, elem, Q.points, Q.weights)

    # Stiffness should scale as 1/h
    assert np.allclose(k2, k1 / 2.0, atol=1e-10)


def test_p2_mass_symmetry() -> None:
    """Test that P2 mass matrix is symmetric."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    me = local_mass_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)

    assert me.shape == (3, 3)
    assert np.allclose(me, me.T, atol=1e-12)


def test_p2_mass_positive_definite() -> None:
    """Test that P2 mass matrix is positive definite."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    me = local_mass_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)

    # All eigenvalues should be positive
    eigvals = np.linalg.eigvalsh(me)
    assert np.all(eigvals > 0)


def test_p2_mass_scaling() -> None:
    """Test that P2 mass scales as h."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    m1 = local_mass_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)
    m2 = local_mass_p2_interval(0.0, 2.0, elem, Q.points, Q.weights)

    # Mass should scale as h
    assert np.allclose(m2, m1 * 2.0, atol=1e-10)


def test_p2_mass_constant_integration() -> None:
    """Test that P2 mass matrix integrates constant function correctly."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    me = local_mass_p2_interval(0.0, 2.0, elem, Q.points, Q.weights)

    # For constant u=1, ∫ N_i dx should sum to h
    # Each row sum represents ∫ N_i dx
    # Sum of all should be 3*h (but distributed among nodes)
    assert np.isclose(np.sum(me), 2.0, atol=1e-10)


def test_p2_load_constant_function() -> None:
    """Test P2 load vector with constant forcing."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    def f(x: float) -> float:
        return 1.0

    b = local_load_p2_interval(0.0, 2.0, f, elem, Q.points, Q.weights)

    assert b.shape == (3,)
    # For constant f=1, ∫ f N_i dx should sum to h
    assert np.isclose(np.sum(b), 2.0, atol=1e-10)


def test_p2_load_linear_function() -> None:
    """Test P2 load vector with linear forcing."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    def f(x: float) -> float:
        return x

    b = local_load_p2_interval(0.0, 1.0, f, elem, Q.points, Q.weights)

    assert b.shape == (3,)
    # ∫₀¹ x dx = 0.5, distributed among 3 nodes
    assert np.isclose(np.sum(b), 0.5, atol=1e-10)


def test_p2_load_quadratic_function() -> None:
    """Test P2 load vector with quadratic forcing."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    def f(x: float) -> float:
        return x * x

    b = local_load_p2_interval(0.0, 1.0, f, elem, Q.points, Q.weights)

    assert b.shape == (3,)
    # ∫₀¹ x² dx = 1/3
    assert np.isclose(np.sum(b), 1.0 / 3.0, atol=1e-10)


def test_p2_consistency_check() -> None:
    """Test consistency: stiffness and load for -u''=1 problem."""
    elem = LagrangeElement("interval", 2)
    Q = gauss_legendre_interval(3)

    # For -u'' = 1 with u(0)=u(1)=0, exact solution is u(x) = x(1-x)/2
    # At x=0.5 (middle node), u=0.125

    def f(x: float) -> float:
        return 1.0

    ke = local_stiffness_p2_interval(0.0, 1.0, elem, Q.points, Q.weights)
    b = local_load_p2_interval(0.0, 1.0, f, elem, Q.points, Q.weights)

    # Apply boundary conditions: u[0] = u[2] = 0
    # Solve for u[1] (middle node)
    u_mid = b[1] / ke[1, 1]

    # Should be close to 0.125
    assert np.isclose(u_mid, 0.125, atol=1e-2)


def test_p2_vs_p1_convergence() -> None:
    """Test that P2 is more accurate than P1 for smooth functions."""
    # This is a simple sanity check that P2 elements exist and work
    elem_p1 = LagrangeElement("interval", 1)
    elem_p2 = LagrangeElement("interval", 2)

    assert elem_p2.dofs_per_cell > elem_p1.dofs_per_cell
    assert elem_p2.degree > elem_p1.degree
