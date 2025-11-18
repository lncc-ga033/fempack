from __future__ import annotations


import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix, isspmatrix, spmatrix
from scipy.sparse.linalg import cg, gmres, spsolve


def _ensure_csr(A: spmatrix | NDArray[np.floating]) -> csr_matrix:
    """Convert a matrix to CSR format if needed."""
    if isspmatrix(A):
        return A.tocsr()  # type: ignore[union-attr]
    return csr_matrix(A)


def solve_direct(A: spmatrix | NDArray[np.floating], b: ArrayLike) -> NDArray[np.floating]:
    """Solve a linear system using a sparse direct solver.

    Parameters
    ----------
    A :
        System matrix.
    b :
        Right-hand side vector.

    Returns
    -------
    numpy.ndarray
        Solution vector.
    """
    A_csr = _ensure_csr(A)
    b_vec = np.asarray(b, dtype=float)
    x = spsolve(A_csr, b_vec)
    return np.asarray(x, dtype=float)


def solve_cg(
    A: spmatrix | NDArray[np.floating],
    b: ArrayLike,
    rtol: float = 1.0e-8,
    maxiter: int | None = None,
) -> NDArray[np.floating]:
    """Solve a symmetric positive definite system with CG.

    Parameters
    ----------
    A :
        System matrix (expected SPD).
    b :
        Right-hand side vector.
    rtol :
        Relative convergence tolerance.
    maxiter :
        Maximum number of iterations or ``None`` for the default.

    Returns
    -------
    numpy.ndarray
        Approximate solution vector.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    A_csr = _ensure_csr(A)
    b_vec = np.asarray(b, dtype=float)
    x, info = cg(A_csr, b_vec, rtol=rtol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}).")
    return np.asarray(x, dtype=float)


def solve_gmres(
    A: spmatrix | NDArray[np.floating],
    b: ArrayLike,
    rtol: float = 1.0e-8,
    maxiter: int | None = None,
) -> NDArray[np.floating]:
    """Solve a general non-symmetric system with GMRES.

    Parameters
    ----------
    A :
        System matrix.
    b :
        Right-hand side vector.
    rtol :
        Relative convergence tolerance.
    maxiter :
        Maximum number of iterations or ``None`` for the default.

    Returns
    -------
    numpy.ndarray
        Approximate solution vector.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    A_csr = _ensure_csr(A)
    b_vec = np.asarray(b, dtype=float)
    x, info = gmres(A_csr, b_vec, rtol=rtol, maxiter=maxiter)
    if info != 0:
        raise RuntimeError(f"GMRES did not converge (info={info}).")
    return np.asarray(x, dtype=float)
