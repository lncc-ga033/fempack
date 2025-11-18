# fempack.solvers

Interfaces para resolvedores de sistemas lineares.

## Funções

### solve_direct(A, b)

Resolve sistema linear usando resolvedor direto esparso.

**Parâmetros:**

- `A`: Matriz do sistema (esparsa ou densa)
- `b`: Vetor do lado direito

**Retorna:** Array NumPy com vetor solução

**Usa:** `scipy.sparse.linalg.spsolve`

**Adequado para:** Sistemas pequenos a médios onde precisão é importante

### solve_cg(A, b, rtol=1e-8, maxiter=None)

Resolve sistema simétrico positivo definido com CG (Gradiente Conjugado).

**Parâmetros:**

- `A`: Matriz do sistema (esperada SPD)
- `b`: Vetor do lado direito
- `rtol`: Tolerância de convergência relativa (padrão: 1e-8)
- `maxiter`: Número máximo de iterações ou `None` para padrão

**Retorna:** Array NumPy com vetor solução aproximado

**Lança:** `RuntimeError` se o resolvedor não convergir

**Usa:** `scipy.sparse.linalg.cg`

### solve_gmres(A, b, rtol=1e-8, maxiter=None)

Resolve sistema geral não-simétrico com GMRES.

**Parâmetros:**

- `A`: Matriz do sistema
- `b`: Vetor do lado direito
- `rtol`: Tolerância de convergência relativa (padrão: 1e-8)
- `maxiter`: Número máximo de iterações ou `None` para padrão

**Retorna:** Array NumPy com vetor solução aproximado

**Lança:** `RuntimeError` se o resolvedor não convergir

**Usa:** `scipy.sparse.linalg.gmres`

## Exemplo

```python
from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_load
from fempack.bcs import apply_dirichlet
from fempack.solvers import solve_direct, solve_cg
import numpy as np

# Configurar problema
mesh = Mesh.unit_interval(100)
V = FunctionSpace(mesh)
A = assemble_stiffness(V)
f = lambda x: 1.0
b = assemble_load(V, f)

# Aplicar BCs
A_bc, b_bc = apply_dirichlet(A, b, V)

# Resolvedor direto (exato)
u_direct = solve_direct(A_bc, b_bc)

# Gradiente conjugado (iterativo, mais rápido para sistemas grandes)
u_cg = solve_cg(A_bc, b_bc, rtol=1e-10)

# GMRES (para matrizes não-simétricas)
u_gmres = solve_gmres(A_bc, b_bc, rtol=1e-10, maxiter=1000)

print(f"Solução: {u_direct.shape}")
print(f"Erro CG vs Direto: {np.linalg.norm(u_cg - u_direct):.2e}")
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/solvers.py).
