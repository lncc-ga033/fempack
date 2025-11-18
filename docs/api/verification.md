# fempack.verification

Ferramentas para verificação via Method of Manufactured Solutions (MMS).

## Funções de Cálculo de Erro

### l2_h1_errors(coords, cells, uh, u_exact, grad_u_exact, cell_type="triangle", order=2)

Calcula erros na norma L² e seminorma H¹ para soluções P1/Q1 em 2D.

**Parâmetros:**

- `coords`: Coordenadas de vértices com shape (N, 2)
- `cells`: Conectividade de células com shape (Ne, nodes_per_cell). Para triângulos, nodes_per_cell=3; para quadrados, nodes_per_cell=4.
- `uh`: Vetor de valores nodais da solução discreta
- `u_exact`: Solução exata, chamada como `u_exact(x, y)`
- `grad_u_exact`: Gradiente exato, chamado como `grad_u_exact(x, y)` retornando array de comprimento 2
- `cell_type`: Tipo de células: `"triangle"` para elementos P1 ou `"square"` para elementos Q1 (padrão: `"triangle"`)
- `order`: Ordem da regra de quadratura usada em cada célula (padrão: 2)

**Retorna:** `(eL2, eH1)` tupla com:

- `eL2`: Aproximação discreta da norma do erro L²
- `eH1`: Aproximação discreta da seminorma H¹ do erro

**Normas calculadas:**

$$\|e\|_{L^2} = \sqrt{\int_\Omega (u - u_h)^2 \, dx}$$

$$|e|_{H^1} = \sqrt{\int_\Omega |\nabla(u - u_h)|^2 \, dx}$$

**Nota:** Para triângulos, usa elementos P1 (lineares) com coordenadas baricêntricas. Para quadrados, usa elementos Q1 (bilineares) com coordenadas de referência em [-1,1]².

## Exemplo Completo

```python
from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_load
from fempack.bcs import apply_dirichlet
from fempack.solvers import solve_direct
from fempack.verification import l2_h1_errors
import numpy as np

# Definir solução manufaturada
u_exact = lambda x, y: np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
grad_u_exact = lambda x, y: np.array([
    2*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y),
    2*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)
])

# Calcular termo fonte: f = -Δu
f = lambda x, y: 8*np.pi**2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)

# Estudo de convergência
errors_l2 = []
errors_h1 = []
h_values = []

for n in [4, 8, 16, 32]:
    # Resolver problema
    mesh = Mesh.unit_square_triangular(n, n)
    V = FunctionSpace(mesh)
    A = assemble_stiffness(V)
    b = assemble_load(V, f)
    A_bc, b_bc = apply_dirichlet(A, b, V, g=lambda x, y: 0.0)
    uh = solve_direct(A_bc, b_bc)

    # Calcular erros
    eL2, eH1 = l2_h1_errors(
        mesh.coords, mesh.cells, uh,
        u_exact, grad_u_exact,
        cell_type="triangle", order=3
    )

    h = 1.0 / n
    h_values.append(h)
    errors_l2.append(eL2)
    errors_h1.append(eH1)

    print(f"n={n:2d}, h={h:.4f}, ||e||_L2={eL2:.6e}, |e|_H1={eH1:.6e}")

# Calcular taxas de convergência
for i in range(1, len(errors_l2)):
    rate_l2 = np.log(errors_l2[i-1]/errors_l2[i]) / np.log(h_values[i-1]/h_values[i])
    rate_h1 = np.log(errors_h1[i-1]/errors_h1[i]) / np.log(h_values[i-1]/h_values[i])
    print(f"Taxa L2: {rate_l2:.2f}, Taxa H1: {rate_h1:.2f}")
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/verification.py).
