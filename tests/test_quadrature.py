"""Tests for quadrature module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.quadrature import (
    gauss_legendre_interval,
    gauss_legendre_reference_interval,
    tensor_product_square_reference,
    triangle_quadrature,
)


def test_gauss_legendre_interval_basic() -> None:
    """Test Gauss-Legendre quadrature on [0, 1]."""
    Q = gauss_legendre_interval(2)

    assert Q.points.shape == (2, 1)
    assert Q.weights.shape == (2,)
    assert Q.degree == 3

    # Check that weights sum to interval length (1.0)
    assert np.isclose(np.sum(Q.weights), 1.0)

    # Check that all points are in [0, 1]
    assert np.all(Q.points >= 0.0)
    assert np.all(Q.points <= 1.0)


def test_gauss_legendre_interval_exactness() -> None:
    """Test exactness of Gauss-Legendre quadrature."""
    Q = gauss_legendre_interval(3)

    # Should integrate polynomials up to degree 5 exactly
    # Test integral of x^4 from 0 to 1 (exact = 1/5)
    x = Q.points[:, 0]
    f = x**4
    integral = np.sum(f * Q.weights)
    exact = 1.0 / 5.0

    assert np.isclose(integral, exact, rtol=1e-12)


def test_gauss_legendre_reference_interval() -> None:
    """Test Gauss-Legendre quadrature on [-1, 1]."""
    Q = gauss_legendre_reference_interval(2)

    assert Q.points.shape == (2, 1)
    assert Q.weights.shape == (2,)

    # Check that weights sum to interval length (2.0)
    assert np.isclose(np.sum(Q.weights), 2.0)

    # Check that all points are in [-1, 1]
    assert np.all(Q.points >= -1.0)
    assert np.all(Q.points <= 1.0)


def test_triangle_quadrature_order1() -> None:
    """Test order 1 triangle quadrature."""
    Q = triangle_quadrature(order=1)

    assert Q.points.shape == (1, 2)
    assert Q.weights.shape == (1,)
    assert Q.degree == 1

    # Should have one point at centroid (1/3, 1/3)
    assert np.allclose(Q.points[0], [1.0 / 3.0, 1.0 / 3.0])

    # Weight should be area of reference triangle (0.5)
    assert np.isclose(Q.weights[0], 0.5)


def test_triangle_quadrature_order2() -> None:
    """Test order 2 triangle quadrature."""
    Q = triangle_quadrature(order=2)

    assert Q.points.shape == (3, 2)
    assert Q.weights.shape == (3,)
    assert Q.degree == 2

    # Weights should sum to area of reference triangle (0.5)
    assert np.isclose(np.sum(Q.weights), 0.5)

    # All points should be in reference triangle
    xi = Q.points[:, 0]
    eta = Q.points[:, 1]
    assert np.all(xi >= 0.0)
    assert np.all(eta >= 0.0)
    assert np.all(xi + eta <= 1.0)


def test_triangle_quadrature_exactness() -> None:
    """Test exactness of triangle quadrature."""
    Q = triangle_quadrature(order=2)

    # Should integrate constant function exactly
    # Integral of 1 over reference triangle = 0.5
    integral = np.sum(Q.weights)
    assert np.isclose(integral, 0.5)

    # Should integrate linear functions exactly
    # Integral of x over reference triangle
    xi = Q.points[:, 0]
    integral_x = np.sum(xi * Q.weights)
    exact_x = 1.0 / 6.0  # ∫∫ x dA over reference triangle
    assert np.isclose(integral_x, exact_x, rtol=1e-12)


def test_triangle_quadrature_invalid_order() -> None:
    """Test that invalid order raises error."""
    with pytest.raises(NotImplementedError):
        triangle_quadrature(order=3)


def test_tensor_product_square() -> None:
    """Test tensor product quadrature on reference square."""
    Q1d = gauss_legendre_reference_interval(2)
    Q = tensor_product_square_reference(Q1d)

    assert Q.points.shape == (4, 2)
    assert Q.weights.shape == (4,)

    # Weights should sum to area of reference square (4.0)
    assert np.isclose(np.sum(Q.weights), 4.0)

    # All points should be in [-1, 1] x [-1, 1]
    assert np.all(Q.points >= -1.0)
    assert np.all(Q.points <= 1.0)


def test_tensor_product_square_exactness() -> None:
    """Test exactness of tensor product quadrature."""
    Q1d = gauss_legendre_reference_interval(2)
    Q = tensor_product_square_reference(Q1d)

    # Should integrate constant function exactly
    # Integral of 1 over [-1,1]x[-1,1] = 4
    integral = np.sum(Q.weights)
    assert np.isclose(integral, 4.0)

    # Should integrate x^2 exactly
    # Integral of x^2 over [-1,1]x[-1,1] = (2/3) * 2 = 4/3
    x = Q.points[:, 0]
    integral_x2 = np.sum(x**2 * Q.weights)
    exact_x2 = 4.0 / 3.0
    assert np.isclose(integral_x2, exact_x2, rtol=1e-12)
