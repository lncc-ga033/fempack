# fempack.spaces

Definição de espaços de elementos finitos e funções de base globais.

## Classes

### FunctionSpace

Espaço de elementos finitos de Lagrange ordem 1 (P1/Q1).

**Parâmetros:**
- `mesh`: Objeto `Mesh` subjacente
- `family`: Família de elementos finitos (atualmente apenas `"Lagrange"` é suportado)
- `degree`: Grau polinomial (apenas grau 1 implementado)

**Atributos:**
- `mesh`: Malha associada
- `family`: Família do elemento (padrão: `"Lagrange"`)
- `degree`: Grau do elemento (padrão: `1`)
- `element`: Objeto `LagrangeElement` criado automaticamente

**Propriedades:**
- `ndofs`: Número de graus de liberdade globais
- `dim`: Dimensão espacial do espaço de elementos finitos

**Exemplo:**

```python
from fempack.mesh import UniformMesh1D
from fempack.spaces import FunctionSpace

mesh = UniformMesh1D(0.0, 1.0, 10)
V = FunctionSpace(mesh, family="Lagrange", degree=1)

print(f"Número de DOFs: {V.ndofs}")
print(f"Dimensão: {V.dim}")
```

### Function

Função de elementos finitos associada a um `FunctionSpace`.

**Parâmetros:**
- `V`: Espaço de elementos finitos ao qual a função pertence
- `values`: Array opcional de valores nodais (se omitido, inicializa com zeros)

**Atributos:**
- `V`: Espaço de elementos finitos
- `values`: Array NumPy dos valores nodais

**Métodos:**
- `nodal_values()`: Retorna os valores nodais como array NumPy

**Exemplo:**

```python
from fempack.mesh import UniformMesh1D
from fempack.spaces import FunctionSpace, Function
import numpy as np

mesh = UniformMesh1D(0.0, 1.0, 10)
V = FunctionSpace(mesh)

# Função com zeros
u = Function(V)

# Função com valores específicos
x = mesh.vertices
u_exact = Function(V, values=np.sin(np.pi * x))

print(u_exact.nodal_values())
```

Veja: [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/spaces.py).
