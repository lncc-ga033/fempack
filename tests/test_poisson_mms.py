from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy import pi

import pytest
from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_load
from fempack.bcs import apply_dirichlet
from fempack.solvers import solve_direct
from fempack.verification import l2_h1_errors


def u_exact_1d(x: float) -> float:
    return float(np.sin(pi * x))


def f_rhs_1d(x: float) -> float:
    return float(pi**2 * np.sin(pi * x))


def solve_poisson_1d(n: int) -> float:
    mesh = Mesh.unit_interval(n)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)
    b = assemble_load(V, f_rhs_1d, quad_order=3)

    A_bc, b_bc = apply_dirichlet(A, b, V, g=u_exact_1d)
    uh = solve_direct(A_bc, b_bc)

    xs = np.linspace(0.0, 1.0, 401)
    u_ex_vals = np.sin(pi * xs)
    x_nodes = mesh.coords[:, 0]
    u_num = np.interp(xs, x_nodes, uh)
    eL2 = np.sqrt(np.trapezoid((u_num - u_ex_vals) ** 2, xs))
    return float(eL2)


def test_poisson_1d_mms_converges() -> None:
    e1 = solve_poisson_1d(16)
    e2 = solve_poisson_1d(32)
    assert e2 < e1


def u_exact_2d(x: float, y: float) -> float:
    return float(np.sin(pi * x) * np.sin(pi * y))


def grad_u_exact_2d(x: float, y: float) -> np.ndarray:
    return np.array(
        [
            pi * np.cos(pi * x) * np.sin(pi * y),
            pi * np.sin(pi * x) * np.cos(pi * y),
        ],
        dtype=float,
    )


def f_rhs_2d(x: float, y: float) -> float:
    return float(2.0 * pi**2 * np.sin(pi * x) * np.sin(pi * y))


def solve_poisson_2d(nx: int, ny: int) -> Tuple[float, float]:
    mesh = Mesh.unit_square_triangular(nx, ny)
    V = FunctionSpace(mesh, family="Lagrange", degree=1)

    A = assemble_stiffness(V)
    b = assemble_load(V, f_rhs_2d, quad_order=2)

    A_bc, b_bc = apply_dirichlet(A, b, V, g=u_exact_2d)
    uh = solve_direct(A_bc, b_bc)

    eL2, eH1 = l2_h1_errors(
        coords=mesh.coords,
        cells=mesh.cells,
        uh=uh,
        u_exact=u_exact_2d,
        grad_u_exact=grad_u_exact_2d,
        cell_type="triangle",
        order=2,
    )
    return float(eL2), float(eH1)


@pytest.mark.grading
def test_poisson_2d_mms_converges() -> None:
    eL2_1, eH1_1 = solve_poisson_2d(8, 8)
    eL2_2, eH1_2 = solve_poisson_2d(16, 16)

    assert eL2_2 < eL2_1
    assert eH1_2 < eH1_1
