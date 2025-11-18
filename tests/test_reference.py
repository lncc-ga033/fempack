"""Tests for reference module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.reference import (
    interval_length,
    p1_interval_gradients,
    p1_interval_reference_gradients,
    p1_interval_shape_functions,
    p1_triangle_reference_gradients,
    p1_triangle_shape_functions,
    p2_interval_gradients,
    p2_interval_reference_gradients,
    p2_interval_shape_functions,
    q1_square_reference_gradients,
    q1_square_shape_functions,
    triangle_area,
    p1_gradients,
    square_area,
    q1_gradients,
    INTERVAL,
    TRIANGLE,
    SQUARE,
)


def test_reference_cells() -> None:
    """Test reference cell definitions."""
    assert INTERVAL.name == "interval"
    assert INTERVAL.dim == 1

    assert TRIANGLE.name == "triangle"
    assert TRIANGLE.dim == 2

    assert SQUARE.name == "square"
    assert SQUARE.dim == 2


# ============================================================================
# Shape Function Helper Tests
# ============================================================================


def test_p1_interval_shape_functions() -> None:
    """Test P1 interval shape functions helper."""
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    phi = p1_interval_shape_functions(x)

    assert phi.shape == (5, 2)
    # Check endpoints
    assert np.allclose(phi[0, :], [1.0, 0.0])
    assert np.allclose(phi[4, :], [0.0, 1.0])
    # Check partition of unity
    assert np.allclose(phi.sum(axis=1), 1.0)


def test_p1_interval_shape_functions_consistency() -> None:
    """Test P1 interval shape functions match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("interval", 1)
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    points = x[:, None]  # Shape (5, 1)

    # Get from elements.py
    elem_phi = elem.tabulate(points)  # Shape (5, 2)

    # Get from reference.py helper
    ref_phi = p1_interval_shape_functions(x)  # Shape (5, 2)

    assert np.allclose(elem_phi, ref_phi)


def test_p2_interval_shape_functions() -> None:
    """Test P2 interval shape functions helper."""
    x = np.array([0.0, 0.5, 1.0])
    phi = p2_interval_shape_functions(x)

    assert phi.shape == (3, 3)
    # Check Lagrange property: N_i(x_j) = δ_ij
    assert np.allclose(phi[0, :], [1.0, 0.0, 0.0])  # N at left node
    assert np.allclose(phi[1, :], [0.0, 1.0, 0.0])  # N at middle node
    assert np.allclose(phi[2, :], [0.0, 0.0, 1.0])  # N at right node
    # Check partition of unity
    assert np.allclose(phi.sum(axis=1), 1.0)


def test_p2_interval_shape_functions_consistency() -> None:
    """Test P2 interval shape functions match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("interval", 2)
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    points = x[:, None]  # Shape (5, 1)

    # Get from elements.py
    elem_phi = elem.tabulate(points)  # Shape (5, 3)

    # Get from reference.py helper
    ref_phi = p2_interval_shape_functions(x)  # Shape (5, 3)

    assert np.allclose(elem_phi, ref_phi)


def test_p1_triangle_shape_functions() -> None:
    """Test P1 triangle shape functions helper."""
    # Test at vertices
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    x = points[:, 0]
    y = points[:, 1]
    phi = p1_triangle_shape_functions(x, y)

    assert phi.shape == (3, 3)
    # Check Lagrange property: N_i(x_j) = δ_ij
    assert np.allclose(phi[0, :], [1.0, 0.0, 0.0])
    assert np.allclose(phi[1, :], [0.0, 1.0, 0.0])
    assert np.allclose(phi[2, :], [0.0, 0.0, 1.0])
    # Check partition of unity
    assert np.allclose(phi.sum(axis=1), 1.0)


def test_p1_triangle_shape_functions_consistency() -> None:
    """Test P1 triangle shape functions match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("triangle", 1)
    points = np.array([[0.2, 0.2], [0.5, 0.3], [0.1, 0.6]])

    # Get from elements.py
    elem_phi = elem.tabulate(points)  # Shape (3, 3)

    # Get from reference.py helper
    ref_phi = p1_triangle_shape_functions(points[:, 0], points[:, 1])

    assert np.allclose(elem_phi, ref_phi)


def test_q1_square_shape_functions() -> None:
    """Test Q1 square shape functions helper."""
    # Test at vertices: (-1,-1), (1,-1), (1,1), (-1,1)
    points = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    xi = points[:, 0]
    eta = points[:, 1]
    phi = q1_square_shape_functions(xi, eta)

    assert phi.shape == (4, 4)
    # Check Lagrange property: N_i(x_j) = δ_ij
    assert np.allclose(phi[0, :], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(phi[1, :], [0.0, 1.0, 0.0, 0.0])
    assert np.allclose(phi[2, :], [0.0, 0.0, 1.0, 0.0])
    assert np.allclose(phi[3, :], [0.0, 0.0, 0.0, 1.0])
    # Check partition of unity
    assert np.allclose(phi.sum(axis=1), 1.0)


def test_q1_square_shape_functions_consistency() -> None:
    """Test Q1 square shape functions match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("square", 1)
    points = np.array([[0.0, 0.0], [-0.5, 0.5], [0.5, -0.5]])

    # Get from elements.py
    elem_phi = elem.tabulate(points)  # Shape (3, 4)

    # Get from reference.py helper
    ref_phi = q1_square_shape_functions(points[:, 0], points[:, 1])

    assert np.allclose(elem_phi, ref_phi)


# ============================================================================
# Reference Gradient Helper Tests
# ============================================================================


def test_p1_interval_reference_gradients() -> None:
    """Test P1 interval reference gradients helper."""
    grads = p1_interval_reference_gradients()

    assert grads.shape == (2, 1)
    assert np.allclose(grads, [[-1.0], [1.0]])


def test_p1_interval_reference_gradients_consistency() -> None:
    """Test P1 interval reference gradients match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("interval", 1)
    points = np.array([[0.25], [0.5], [0.75]])

    # Get from elements.py
    elem_grads = elem.tabulate_reference_gradients(points)  # Shape (3, 2, 1)

    # Get from reference.py helper
    ref_grads = p1_interval_reference_gradients()  # Shape (2, 1)

    # Should match for all quadrature points (constant gradients)
    for i in range(3):
        assert np.allclose(elem_grads[i, :, :], ref_grads)


def test_p1_triangle_reference_gradients() -> None:
    """Test P1 triangle reference gradients helper."""
    grads = p1_triangle_reference_gradients()

    assert grads.shape == (3, 2)
    expected = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(grads, expected)


def test_p1_triangle_reference_gradients_consistency() -> None:
    """Test P1 triangle reference gradients match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("triangle", 1)
    points = np.array([[0.2, 0.2], [0.5, 0.3], [0.1, 0.6]])

    # Get from elements.py
    elem_grads = elem.tabulate_reference_gradients(points)  # Shape (3, 3, 2)

    # Get from reference.py helper
    ref_grads = p1_triangle_reference_gradients()  # Shape (3, 2)

    # Should match for all quadrature points (constant gradients)
    for i in range(3):
        assert np.allclose(elem_grads[i, :, :], ref_grads)


def test_q1_square_reference_gradients_scalar() -> None:
    """Test Q1 square reference gradients with scalar input."""
    grads = q1_square_reference_gradients(0.0, 0.0)

    assert grads.shape == (4, 2)
    # At center (0, 0), gradients should be symmetric
    expected = np.array(
        [
            [-0.25, -0.25],
            [0.25, -0.25],
            [0.25, 0.25],
            [-0.25, 0.25],
        ]
    )
    assert np.allclose(grads, expected)


def test_q1_square_reference_gradients_array() -> None:
    """Test Q1 square reference gradients with array input."""
    xi = np.array([0.0, -1.0, 1.0])
    eta = np.array([0.0, -1.0, 1.0])
    grads = q1_square_reference_gradients(xi, eta)

    assert grads.shape == (3, 4, 2)


def test_q1_square_reference_gradients_consistency() -> None:
    """Test Q1 square reference gradients match elements.py."""
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("square", 1)
    points = np.array([[0.0, 0.0], [-0.5, 0.5], [0.5, -0.5]])

    # Get from elements.py
    elem_grads = elem.tabulate_reference_gradients(points)  # Shape (3, 4, 2)

    # Get from reference.py helper
    ref_grads = q1_square_reference_gradients(points[:, 0], points[:, 1])  # Shape (3, 4, 2)

    # Should match
    assert np.allclose(elem_grads, ref_grads)


# ============================================================================
# Interval Tests
# ============================================================================


def test_interval_length_unit() -> None:
    """Test length of unit interval."""
    h = interval_length(0.0, 1.0)
    assert np.isclose(h, 1.0)


def test_interval_length_general() -> None:
    """Test length of general interval."""
    h = interval_length(2.0, 5.0)
    assert np.isclose(h, 3.0)


def test_interval_length_negative() -> None:
    """Test that negative length raises error."""
    with pytest.raises(ValueError, match="Degenerate interval"):
        interval_length(5.0, 2.0)


def test_interval_length_zero() -> None:
    """Test that zero length raises error."""
    with pytest.raises(ValueError, match="Degenerate interval"):
        interval_length(3.0, 3.0)


def test_p1_interval_gradients_unit() -> None:
    """Test P1 gradients on unit interval."""
    grads, h = p1_interval_gradients(0.0, 1.0)

    assert grads.shape == (2,)
    assert np.isclose(h, 1.0)
    # On [0,1], gradients are [-1, 1]
    assert np.allclose(grads, [-1.0, 1.0])


def test_p1_interval_gradients_general() -> None:
    """Test P1 gradients on general interval."""
    grads, h = p1_interval_gradients(1.0, 4.0)

    assert grads.shape == (2,)
    assert np.isclose(h, 3.0)
    # On interval of length 3, gradients are [-1/3, 1/3]
    assert np.allclose(grads, [-1.0 / 3.0, 1.0 / 3.0])


def test_p1_interval_gradients_sum_to_zero() -> None:
    """Test that P1 interval gradients sum to zero (partition of unity)."""
    grads, _ = p1_interval_gradients(2.0, 7.0)
    assert np.isclose(np.sum(grads), 0.0)


def test_p1_interval_gradients_constant_function() -> None:
    """Test that constant function has zero gradient."""
    grads, _ = p1_interval_gradients(0.0, 2.0)
    # For u = constant (u₀ = u₁ = c), gradient is 0
    u_nodal = np.array([5.0, 5.0])
    grad_u = np.dot(grads, u_nodal)
    assert np.isclose(grad_u, 0.0)


def test_p1_interval_gradients_linear_function() -> None:
    """Test that linear function u=x has correct gradient."""
    grads, _ = p1_interval_gradients(0.0, 1.0)
    # For u = x on [0,1], nodal values are [0, 1]
    u_nodal = np.array([0.0, 1.0])
    grad_u = np.dot(grads, u_nodal)
    # Gradient should be 1
    assert np.isclose(grad_u, 1.0)


def test_p1_interval_gradients_scaled_function() -> None:
    """Test gradient on scaled interval."""
    grads, _ = p1_interval_gradients(0.0, 2.0)
    # For u = x on [0,2], nodal values are [0, 2]
    u_nodal = np.array([0.0, 2.0])
    grad_u = np.dot(grads, u_nodal)
    # Gradient should still be 1 (du/dx = 1)
    assert np.isclose(grad_u, 1.0)


def test_p1_interval_gradients_degenerate() -> None:
    """Test that degenerate interval raises error."""
    with pytest.raises(ValueError, match="Degenerate interval"):
        p1_interval_gradients(3.0, 3.0)


# ============================================================================
# P2 Interval Tests
# ============================================================================


def test_p2_interval_gradients_unit_left() -> None:
    """Test P2 gradients at left endpoint of unit interval."""
    grads, h = p2_interval_gradients(0.0, 1.0, xi=0.0)

    assert grads.shape == (3,)
    assert np.isclose(h, 1.0)
    # At ξ=0: dN₀/dξ = -3, dN₁/dξ = 4, dN₂/dξ = -1
    assert np.allclose(grads, [-3.0, 4.0, -1.0])


def test_p2_interval_gradients_unit_middle() -> None:
    """Test P2 gradients at midpoint of unit interval."""
    grads, h = p2_interval_gradients(0.0, 1.0, xi=0.5)

    assert grads.shape == (3,)
    assert np.isclose(h, 1.0)
    # At ξ=0.5: dN₀/dξ = -1, dN₁/dξ = 0, dN₂/dξ = 1
    assert np.allclose(grads, [-1.0, 0.0, 1.0])


def test_p2_interval_gradients_unit_right() -> None:
    """Test P2 gradients at right endpoint of unit interval."""
    grads, h = p2_interval_gradients(0.0, 1.0, xi=1.0)

    assert grads.shape == (3,)
    assert np.isclose(h, 1.0)
    # At ξ=1: dN₀/dξ = 1, dN₁/dξ = -4, dN₂/dξ = 3
    assert np.allclose(grads, [1.0, -4.0, 3.0])


def test_p2_interval_gradients_general() -> None:
    """Test P2 gradients on general interval."""
    grads, h = p2_interval_gradients(1.0, 3.0, xi=0.5)

    assert grads.shape == (3,)
    assert np.isclose(h, 2.0)
    # On interval of length 2, gradients are scaled by 1/2
    assert np.allclose(grads, [-0.5, 0.0, 0.5])


def test_p2_interval_gradients_sum_to_zero() -> None:
    """Test that P2 gradients sum to zero (partition of unity)."""
    for xi in [0.0, 0.25, 0.5, 0.75, 1.0]:
        grads, _ = p2_interval_gradients(0.0, 1.0, xi=xi)
        assert np.isclose(np.sum(grads), 0.0, atol=1e-12)


def test_p2_interval_gradients_quadratic_function() -> None:
    """Test P2 can represent quadratic function exactly."""
    # u(x) = x² on [0, 1]
    # At nodes: u(0) = 0, u(0.5) = 0.25, u(1) = 1
    u_nodal = np.array([0.0, 0.25, 1.0])

    # Test at several points
    for xi in [0.2, 0.5, 0.8]:
        grads, _ = p2_interval_gradients(0.0, 1.0, xi=xi)
        grad_u = np.dot(grads, u_nodal)
        # du/dx = 2x, at physical point x = xi
        expected = 2.0 * xi
        assert np.isclose(grad_u, expected, atol=1e-10)


def test_p2_interval_gradients_linear_function() -> None:
    """Test P2 can represent linear function exactly."""
    # u(x) = 2x + 1 on [0, 1]
    # At nodes: u(0) = 1, u(0.5) = 2, u(1) = 3
    u_nodal = np.array([1.0, 2.0, 3.0])

    for xi in [0.0, 0.5, 1.0]:
        grads, _ = p2_interval_gradients(0.0, 1.0, xi=xi)
        grad_u = np.dot(grads, u_nodal)
        # du/dx = 2 everywhere
        assert np.isclose(grad_u, 2.0, atol=1e-10)


def test_p2_interval_gradients_invalid_xi() -> None:
    """Test that xi outside [0,1] raises error."""
    with pytest.raises(ValueError, match="must be in"):
        p2_interval_gradients(0.0, 1.0, xi=-0.1)

    with pytest.raises(ValueError, match="must be in"):
        p2_interval_gradients(0.0, 1.0, xi=1.1)


def test_p2_interval_gradients_degenerate() -> None:
    """Test that degenerate interval raises error."""
    with pytest.raises(ValueError, match="Degenerate interval"):
        p2_interval_gradients(3.0, 3.0, xi=0.5)


def test_p2_interval_reference_gradients_scalar() -> None:
    """Test P2 reference gradients with scalar input."""
    grads = p2_interval_reference_gradients(0.5)

    assert grads.shape == (3,)
    # At ξ=0.5: dN₀/dξ = -1, dN₁/dξ = 0, dN₂/dξ = 1
    assert np.allclose(grads, [-1.0, 0.0, 1.0])


def test_p2_interval_reference_gradients_array() -> None:
    """Test P2 reference gradients with array input."""
    xi_points = np.array([0.0, 0.5, 1.0])
    grads = p2_interval_reference_gradients(xi_points)

    assert grads.shape == (3, 3)
    # At ξ=0: [-3, 4, -1], at ξ=0.5: [-1, 0, 1], at ξ=1: [1, -4, 3]
    expected = np.array([[-3.0, 4.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -4.0, 3.0]])
    assert np.allclose(grads, expected)


def test_p2_interval_reference_gradients_consistency() -> None:
    """Test that reference gradients are used consistently."""
    # Reference gradients should match what's used in elements.py
    from fempack.elements import LagrangeElement

    elem = LagrangeElement("interval", 2)
    xi_points = np.array([[0.25], [0.5], [0.75]])

    # Get from elements.py
    elem_grads = elem.tabulate_reference_gradients(xi_points)  # Shape (3, 3, 1)

    # Get from reference.py helper
    ref_grads = p2_interval_reference_gradients(xi_points[:, 0])  # Shape (3, 3)

    # Should match
    assert np.allclose(elem_grads[:, :, 0], ref_grads)


@pytest.mark.grading
def test_triangle_area_unit() -> None:
    """Test area of unit right triangle."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    area = triangle_area(verts)

    assert np.isclose(area, 0.5)


@pytest.mark.grading
def test_triangle_area_general() -> None:
    """Test area of general triangle."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]], dtype=float)
    area = triangle_area(verts)

    # Area should be 1.0
    assert np.isclose(area, 1.0)


@pytest.mark.grading
def test_triangle_area_larger() -> None:
    """Test area of larger triangle."""
    verts = np.array([[1.0, 1.0], [4.0, 1.0], [2.0, 5.0]], dtype=float)
    area = triangle_area(verts)

    # Area = 0.5 * |3 * 4 - 0 * 1| = 6.0
    assert np.isclose(area, 6.0)


@pytest.mark.grading
def test_triangle_area_invalid_shape() -> None:
    """Test that invalid shape raises error."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="must have shape"):
        triangle_area(verts)


@pytest.mark.grading
def test_triangle_area_negative_orientation() -> None:
    """Test triangle with negative orientation."""
    # Vertices in clockwise order (negative area)
    verts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)
    area = triangle_area(verts)

    # Should return negative area for clockwise orientation
    assert area < 0


@pytest.mark.grading
def test_p1_gradients_unit_triangle() -> None:
    """Test P1 gradients on unit right triangle."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    grads, area = p1_gradients(verts)

    assert grads.shape == (3, 2)
    assert np.isclose(area, 0.5)

    # For this triangle, gradients should be:
    # grad λ₁ = (-1, -1)
    # grad λ₂ = (1, 0)
    # grad λ₃ = (0, 1)
    expected = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    assert np.allclose(grads, expected)


@pytest.mark.grading
def test_p1_gradients_sum_to_zero() -> None:
    """Test that P1 gradients sum to zero (partition of unity)."""
    verts = np.array([[1.0, 2.0], [3.0, 1.0], [2.0, 4.0]], dtype=float)
    grads, _ = p1_gradients(verts)

    # Sum of gradients should be zero
    grad_sum = np.sum(grads, axis=0)
    assert np.allclose(grad_sum, [0.0, 0.0])


@pytest.mark.grading
def test_p1_gradients_general_triangle() -> None:
    """Test P1 gradients on general triangle."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]], dtype=float)
    grads, area = p1_gradients(verts)

    assert grads.shape == (3, 2)
    assert np.isclose(area, 2.0)

    # Verify gradients are constant within element
    # and sum to zero
    assert np.allclose(np.sum(grads, axis=0), [0.0, 0.0])


@pytest.mark.grading
def test_p1_gradients_invalid_shape() -> None:
    """Test that invalid shape raises error."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="must have shape"):
        p1_gradients(verts)


@pytest.mark.grading
def test_p1_gradients_degenerate_triangle() -> None:
    """Test that degenerate triangle raises error."""
    # Collinear vertices
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="Degenerate triangle"):
        p1_gradients(verts)


@pytest.mark.grading
def test_p1_gradients_negative_orientation() -> None:
    """Test that negative orientation raises error."""
    # Clockwise vertices
    verts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="Degenerate triangle"):
        p1_gradients(verts)


@pytest.mark.grading
def test_p1_gradients_constant_on_affine_functions() -> None:
    """Test that gradients correctly represent affine functions."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    grads, _ = p1_gradients(verts)

    # For barycentric coordinates λ₁, λ₂, λ₃
    # A linear function u = a₁λ₁ + a₂λ₂ + a₃λ₃
    # has gradient ∇u = a₁∇λ₁ + a₂∇λ₂ + a₃∇λ₃

    # Test with u = x (so a = [0, 1, 0] in nodal values)
    a = np.array([0.0, 1.0, 0.0])
    grad_u = grads.T @ a

    # Should equal [1, 0]
    assert np.allclose(grad_u, [1.0, 0.0])


# ============================================================================
# Quadrilateral (Square) Tests
# ============================================================================


def test_square_area_unit() -> None:
    """Test area of unit square."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    area = square_area(verts)

    assert np.isclose(area, 1.0)


def test_square_area_rectangle() -> None:
    """Test area of rectangle."""
    verts = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]], dtype=float)
    area = square_area(verts)

    assert np.isclose(area, 6.0)


def test_square_area_general_quad() -> None:
    """Test area of general quadrilateral."""
    # Trapezoid
    verts = np.array([[0.0, 0.0], [4.0, 0.0], [3.0, 2.0], [1.0, 2.0]], dtype=float)
    area = square_area(verts)

    # Area = 0.5 * (4 + 2) * 2 = 6.0
    assert np.isclose(area, 6.0)


def test_square_area_rotated() -> None:
    """Test area of rotated square."""
    # Diamond shape (rotated unit square)
    verts = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=float)
    area = square_area(verts)

    # Area = 2.0 (diagonal of sqrt(2) gives area = 0.5 * sqrt(2) * sqrt(2) * 2 = 2)
    assert np.isclose(area, 2.0)


def test_square_area_invalid_shape() -> None:
    """Test that invalid shape raises error."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="must have shape"):
        square_area(verts)


def test_q1_gradients_unit_square_center() -> None:
    """Test Q1 gradients at center of unit square."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    grads, detJ = q1_gradients(verts, xi=0.0, eta=0.0)

    assert grads.shape == (4, 2)
    assert np.isclose(detJ, 0.25)  # For unit square, detJ = 1/4 at center

    # At center, by symmetry, gradients should be symmetric
    # Each shape function contributes equally
    expected = np.array(
        [
            [-0.5, -0.5],  # Node 0
            [0.5, -0.5],  # Node 1
            [0.5, 0.5],  # Node 2
            [-0.5, 0.5],  # Node 3
        ],
        dtype=float,
    )
    assert np.allclose(grads, expected, atol=1e-10)


def test_q1_gradients_sum_to_zero() -> None:
    """Test that Q1 gradients sum to zero (partition of unity)."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5]], dtype=float)

    # Test at several points
    for xi in [-0.5, 0.0, 0.5]:
        for eta in [-0.5, 0.0, 0.5]:
            grads, _ = q1_gradients(verts, xi=xi, eta=eta)
            grad_sum = np.sum(grads, axis=0)
            assert np.allclose(grad_sum, [0.0, 0.0], atol=1e-12)


def test_q1_gradients_rectangle() -> None:
    """Test Q1 gradients on a rectangle."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)
    grads, detJ = q1_gradients(verts, xi=0.0, eta=0.0)

    assert grads.shape == (4, 2)
    # For 2x1 rectangle, Jacobian determinant = (width/2) * (height/2) = 1 * 0.5 = 0.5
    assert np.isclose(detJ, 0.5, atol=1e-10)


def test_q1_gradients_corner_points() -> None:
    """Test Q1 gradients at corner points."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)

    # At corner (xi=-1, eta=-1), node 0 should have maximum influence
    grads, detJ = q1_gradients(verts, xi=-1.0, eta=-1.0)
    assert detJ > 0

    # Shape functions at corners should still maintain partition of unity
    assert np.allclose(np.sum(grads, axis=0), [0.0, 0.0], atol=1e-12)


def test_q1_gradients_general_quad() -> None:
    """Test Q1 gradients on general quadrilateral."""
    # Slightly distorted quadrilateral
    verts = np.array([[0.0, 0.0], [1.1, 0.1], [1.0, 1.0], [0.1, 0.9]], dtype=float)

    grads, detJ = q1_gradients(verts, xi=0.0, eta=0.0)

    assert grads.shape == (4, 2)
    assert detJ > 0

    # Gradients should still sum to zero
    assert np.allclose(np.sum(grads, axis=0), [0.0, 0.0], atol=1e-12)


def test_q1_gradients_invalid_shape() -> None:
    """Test that invalid shape raises error."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="must have shape"):
        q1_gradients(verts, xi=0.0, eta=0.0)


def test_q1_gradients_degenerate_quad() -> None:
    """Test that degenerate quadrilateral raises error."""
    # Collinear vertices (degenerate)
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="Degenerate or inverted"):
        q1_gradients(verts, xi=0.0, eta=0.0)


def test_q1_gradients_inverted_quad() -> None:
    """Test that inverted quadrilateral raises error."""
    # Vertices in wrong order (creates inverted element)
    verts = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="Degenerate or inverted"):
        q1_gradients(verts, xi=0.0, eta=0.0)


def test_q1_gradients_linear_function_x() -> None:
    """Test Q1 gradients correctly represent linear function u = x."""
    verts = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]], dtype=float)

    # Nodal values for u = x are [0, 2, 2, 0]
    u_nodal = np.array([0.0, 2.0, 2.0, 0.0])

    # Test at center
    grads, _ = q1_gradients(verts, xi=0.0, eta=0.0)
    grad_u = grads.T @ u_nodal

    # Should be approximately [1, 0] (gradient of u=x)
    assert np.allclose(grad_u, [1.0, 0.0], atol=1e-10)


def test_q1_gradients_linear_function_y() -> None:
    """Test Q1 gradients correctly represent linear function u = y."""
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]], dtype=float)

    # Nodal values for u = y are [0, 0, 2, 2]
    u_nodal = np.array([0.0, 0.0, 2.0, 2.0])

    # Test at center
    grads, _ = q1_gradients(verts, xi=0.0, eta=0.0)
    grad_u = grads.T @ u_nodal

    # Should be approximately [0, 1] (gradient of u=y)
    assert np.allclose(grad_u, [0.0, 1.0], atol=1e-10)
