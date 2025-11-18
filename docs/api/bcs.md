# fempack.bcs

Imposição de condições de contorno de Dirichlet.

## Funções

### apply_dirichlet(A, b, V, nodes=None, g=None)

Aplica condições de contorno de Dirichlet por modificação de linhas/colunas.

**Parâmetros:**

- `A`: Matriz de rigidez (ou sistema) global - pode ser esparsa ou densa
- `b`: Vetor do lado direito
- `V`: Espaço de elementos finitos (`FunctionSpace`)
- `nodes`: Sequência de índices de nós globais onde condições de Dirichlet são aplicadas. Se `None`, todos os nós de fronteira da malha são usados.
- `g`: Dados de Dirichlet. Se `None`, condições homogêneas são aplicadas.
  - Se callable: avaliado como `g(x)` em 1D ou `g(x, y)` em 2D nos nós de fronteira
  - Se array: pode ser vetor completo de comprimento N (do qual entradas de fronteira são extraídas) ou de comprimento `len(nodes)`

**Retorna:** `(A_bc, b_bc)` - Matriz e lado direito modificados com condições de Dirichlet impostas fortemente

## Método de Imposição

A implementação segue a abordagem padrão:

1. Ajustar lado direito para condições não-homogêneas: $b \leftarrow b - A[:, \text{nodes}] \cdot g_{\text{vec}}$
2. Zerar linhas e colunas correspondentes e definir entradas diagonais unitárias para nós de Dirichlet
3. Definir $b[i] = g_i$ para cada nó de Dirichlet

## Exemplo

```python
from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_load
from fempack.bcs import apply_dirichlet
import numpy as np

# Configurar problema
mesh = Mesh.unit_interval(10)
V = FunctionSpace(mesh)
A = assemble_stiffness(V)
f = lambda x: np.pi**2 * np.sin(np.pi * x)
b = assemble_load(V, f)

# Aplicar u = 0 na fronteira (condições homogêneas)
A_bc, b_bc = apply_dirichlet(A, b, V, nodes=None, g=None)

# Aplicar condições não-homogêneas: u = sin(π*x) na fronteira
g_func = lambda x: np.sin(np.pi * x)
A_bc, b_bc = apply_dirichlet(A, b, V, g=g_func)

# Aplicar em nós específicos
boundary_nodes = mesh.boundary_nodes()
A_bc, b_bc = apply_dirichlet(A, b, V, nodes=boundary_nodes, g=0.0)
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/bcs.py).
