"""Tests for solvers module."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from fempack.solvers import solve_direct, solve_cg, solve_gmres


def test_solve_direct_simple() -> None:
    """Test direct solver with simple system."""
    # 2x + 3y = 8
    # 5x + 4y = 13
    A = csr_matrix([[2.0, 3.0], [5.0, 4.0]])
    b = np.array([8.0, 13.0])

    x = solve_direct(A, b)

    # Solution should be x=1, y=2
    assert np.allclose(x, [1.0, 2.0])


def test_solve_direct_identity() -> None:
    """Test direct solver with identity matrix."""
    n = 5
    A = csr_matrix(np.eye(n))
    b = np.arange(1, n + 1, dtype=float)

    x = solve_direct(A, b)

    # Solution should equal b
    assert np.allclose(x, b)


def test_solve_direct_tridiagonal() -> None:
    """Test direct solver with tridiagonal matrix."""
    # 1D Laplacian-like matrix
    n = 10
    diag = np.full(n, 2.0)
    off_diag = np.full(n - 1, -1.0)
    A = csr_matrix(np.diag(diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1))
    b = np.ones(n)

    x = solve_direct(A, b)

    # Verify that A @ x = b
    assert np.allclose(A @ x, b)


def test_solve_direct_with_dense_matrix() -> None:
    """Test direct solver with dense matrix input."""
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    x = solve_direct(A, b)

    # Verify solution
    assert np.allclose(A @ x, b)


def test_solve_cg_simple() -> None:
    """Test CG solver with simple SPD system."""
    # Symmetric positive definite matrix
    A = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    x = solve_cg(A, b)

    # Verify solution
    assert np.allclose(A @ x, b, rtol=1e-6)


def test_solve_cg_diagonal() -> None:
    """Test CG solver with diagonal matrix."""
    n = 20
    diag = np.arange(1, n + 1, dtype=float)
    A = csr_matrix(np.diag(diag))
    b = np.ones(n)

    x = solve_cg(A, b)

    # Solution should be [1/1, 1/2, 1/3, ..., 1/n]
    expected = 1.0 / diag
    assert np.allclose(x, expected, rtol=1e-6)


def test_solve_cg_with_tolerance() -> None:
    """Test CG solver converges."""
    n = 10
    A = csr_matrix(
        np.diag(np.ones(n) * 2.0)
        + np.diag(np.ones(n - 1) * -1.0, 1)
        + np.diag(np.ones(n - 1) * -1.0, -1)
    )
    b = np.ones(n)

    x = solve_cg(A, b)

    # Check residual
    residual = np.linalg.norm(A @ x - b)
    assert residual < 1e-6


def test_solve_cg_max_iterations() -> None:
    """Test CG solver with max iterations."""
    A = csr_matrix([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])

    # Should converge even with limited iterations for this small problem
    x = solve_cg(A, b, maxiter=10)

    assert np.allclose(A @ x, b, rtol=1e-6)


def test_solve_gmres_simple() -> None:
    """Test GMRES solver with simple system."""
    A = csr_matrix([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([4.0, 3.0])

    x = solve_gmres(A, b)

    # Verify solution
    assert np.allclose(A @ x, b, rtol=1e-6)


def test_solve_gmres_nonsymmetric() -> None:
    """Test GMRES solver with non-symmetric matrix."""
    # Non-symmetric matrix
    A = csr_matrix([[3.0, 1.0], [2.0, 4.0]])
    b = np.array([5.0, 6.0])

    x = solve_gmres(A, b)

    # Verify solution
    assert np.allclose(A @ x, b, rtol=1e-6)


def test_solve_gmres_with_tolerance() -> None:
    """Test GMRES solver converges within tolerance."""
    n = 10
    A = csr_matrix(
        np.diag(np.ones(n) * 3.0)
        + np.diag(np.ones(n - 1) * 1.0, 1)
        + np.diag(np.ones(n - 1) * 0.5, -1)
    )
    b = np.ones(n)

    x = solve_gmres(A, b, rtol=1e-8)

    # Check residual
    residual = np.linalg.norm(A @ x - b)
    assert residual < 1e-7


def test_solver_comparison() -> None:
    """Test that all solvers give similar results for SPD system."""
    # Create a symmetric positive definite system
    n = 5
    A = csr_matrix(
        np.diag(np.ones(n) * 4.0)
        + np.diag(np.ones(n - 1) * -1.0, 1)
        + np.diag(np.ones(n - 1) * -1.0, -1)
    )
    b = np.random.rand(n)

    x_direct = solve_direct(A, b)
    x_cg = solve_cg(A, b)
    x_gmres = solve_gmres(A, b)

    # All solutions should be close
    assert np.allclose(x_direct, x_cg, rtol=1e-6)
    assert np.allclose(x_direct, x_gmres, rtol=1e-6)


def test_solve_direct_sparse_preserves_format() -> None:
    """Test that direct solver works with different sparse formats."""
    from scipy.sparse import coo_matrix

    # Create matrix in COO format
    row = [0, 1, 0, 1]
    col = [0, 1, 1, 0]
    data = [2.0, 3.0, 1.0, 1.0]
    A_coo = coo_matrix((data, (row, col)), shape=(2, 2))

    b = np.array([3.0, 4.0])

    x = solve_direct(A_coo, b)

    # Verify solution
    assert np.allclose(A_coo @ x, b)
