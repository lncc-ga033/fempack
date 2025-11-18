"""Tests for verification module."""

from __future__ import annotations

import numpy as np
import pytest

from fempack.mesh import Mesh
from fempack.verification import l2_h1_errors


@pytest.mark.grading
def test_l2_h1_errors_triangle_exact():
    """Test error computation for P1 triangular elements with exact solution."""
    # Simple test: u(x,y) = x + y, which P1 can represent exactly

    def u_exact(x: float, y: float) -> float:
        return x + y

    def grad_u_exact(x: float, y: float) -> np.ndarray:
        return np.array([1.0, 1.0])

    mesh = Mesh.unit_square_triangular(4, 4)

    # Create exact solution at nodes
    uh = mesh.coords[:, 0] + mesh.coords[:, 1]

    eL2, eH1 = l2_h1_errors(
        mesh.coords, mesh.cells, uh, u_exact, grad_u_exact, cell_type="triangle", order=2
    )

    # Should be near machine precision since P1 represents linear exactly
    assert eL2 < 1e-12
    assert eH1 < 1e-12


def test_l2_h1_errors_square_exact():
    """Test error computation for Q1 quadrilateral elements with exact solution."""
    # Simple test: u(x,y) = x + y, which Q1 can represent exactly

    def u_exact(x: float, y: float) -> float:
        return x + y

    def grad_u_exact(x: float, y: float) -> np.ndarray:
        return np.array([1.0, 1.0])

    mesh = Mesh.unit_square_quadrilateral(4, 4)

    # Create exact solution at nodes
    uh = mesh.coords[:, 0] + mesh.coords[:, 1]

    eL2, eH1 = l2_h1_errors(
        mesh.coords, mesh.cells, uh, u_exact, grad_u_exact, cell_type="square", order=2
    )

    # Should be near machine precision since Q1 represents bilinear on each element
    assert eL2 < 1e-12
    assert eH1 < 1e-12


@pytest.mark.grading
def test_l2_h1_errors_convergence_triangle():
    """Test convergence rates for triangular elements."""

    def u_exact(x: float, y: float) -> float:
        return float(np.sin(np.pi * x) * np.sin(np.pi * y))

    def grad_u_exact(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ]
        )

    errors_L2 = []
    errors_H1 = []
    hs = []

    for n in [4, 8, 16]:
        mesh = Mesh.unit_square_triangular(n, n)

        # Project exact solution
        uh = np.array([u_exact(x, y) for x, y in mesh.coords])

        eL2, eH1 = l2_h1_errors(
            mesh.coords, mesh.cells, uh, u_exact, grad_u_exact, cell_type="triangle", order=2
        )

        errors_L2.append(eL2)
        errors_H1.append(eH1)
        hs.append(1.0 / n)

    # Check convergence rates (approximately)
    # P1 should give O(h^2) for L2, O(h) for H1
    rate_L2 = np.log(errors_L2[0] / errors_L2[1]) / np.log(hs[0] / hs[1])
    rate_H1 = np.log(errors_H1[0] / errors_H1[1]) / np.log(hs[0] / hs[1])

    assert 1.5 < rate_L2 < 2.5  # Should be ~2
    assert 0.8 < rate_H1 < 1.5  # Should be ~1


def test_l2_h1_errors_convergence_square():
    """Test convergence rates for quadrilateral elements."""

    def u_exact(x: float, y: float) -> float:
        return float(np.sin(np.pi * x) * np.sin(np.pi * y))

    def grad_u_exact(x: float, y: float) -> np.ndarray:
        return np.array(
            [
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ]
        )

    errors_L2 = []
    errors_H1 = []
    hs = []

    for n in [4, 8, 16]:
        mesh = Mesh.unit_square_quadrilateral(n, n)

        # Project exact solution
        uh = np.array([u_exact(x, y) for x, y in mesh.coords])

        eL2, eH1 = l2_h1_errors(
            mesh.coords, mesh.cells, uh, u_exact, grad_u_exact, cell_type="square", order=2
        )

        errors_L2.append(eL2)
        errors_H1.append(eH1)
        hs.append(1.0 / n)

    # Check convergence rates
    rate_L2 = np.log(errors_L2[0] / errors_L2[1]) / np.log(hs[0] / hs[1])
    rate_H1 = np.log(errors_H1[0] / errors_H1[1]) / np.log(hs[0] / hs[1])

    assert 1.5 < rate_L2 < 2.5  # Should be ~2
    assert 0.8 < rate_H1 < 1.5  # Should be ~1


def test_l2_h1_errors_invalid_cell_type():
    """Test that invalid cell type raises ValueError."""

    def u_exact(x: float, y: float) -> float:
        return x + y

    def grad_u_exact(x: float, y: float) -> np.ndarray:
        return np.array([1.0, 1.0])

    mesh = Mesh.unit_square_triangular(4, 4)
    uh = mesh.coords[:, 0] + mesh.coords[:, 1]

    with pytest.raises(ValueError, match="Unsupported cell_type"):
        l2_h1_errors(
            mesh.coords,
            mesh.cells,
            uh,
            u_exact,
            grad_u_exact,
            cell_type="invalid",  # type: ignore
            order=2,
        )
