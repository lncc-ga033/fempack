# fempack.assemble

Montagem de matrizes e vetores globais a partir de contribuições locais.

## Funções Principais

### assemble_stiffness(V)

Monta matriz de rigidez global.

**Parâmetros:**

- `V`: Espaço de elementos finitos (`FunctionSpace`)

**Retorna:** `scipy.sparse.csr_matrix` correspondente à forma bilinear

$$a(u, v) = \int_\Omega \nabla u \cdot \nabla v$$

**Nota:** A quadratura é selecionada automaticamente baseada no tipo de célula.

### assemble_mass(V)

Monta matriz de massa global.

**Parâmetros:**

- `V`: Espaço de elementos finitos (`FunctionSpace`)

**Retorna:** `scipy.sparse.csr_matrix` correspondente à forma bilinear

$$m(u, v) = \int_\Omega u \cdot v$$

### assemble_load(V, f)

Monta vetor de carga global.

**Parâmetros:**

- `V`: Espaço de elementos finitos (`FunctionSpace`)
- `f`: Função fonte (callable)
  - 1D: `f(x)` aceita um float
  - 2D: `f(x, y)` aceita dois floats

**Retorna:** Array NumPy com o vetor de carga global

$$b[I] = \sum_{K} \int_K f N_I \, dx$$

## Processo de Montagem

1. Percorre todos os elementos da malha
2. Calcula matrizes/vetores locais usando funções de `fempack.local`
3. Adiciona contribuições nas posições globais corretas usando conectividade
4. Retorna estrutura global esparsa (CSR format)

## Exemplo

```python
from fempack.mesh import Mesh
from fempack.spaces import FunctionSpace
from fempack.assemble import assemble_stiffness, assemble_mass, assemble_load

# Criar malha e espaço
mesh = Mesh.unit_interval(10)
V = FunctionSpace(mesh, family="Lagrange", degree=1)

# Montar sistema
A = assemble_stiffness(V)
M = assemble_mass(V)

# Montar vetor de carga com f(x) = x^2
f = lambda x: x**2
b = assemble_load(V, f)

print(f"A: {A.shape}, esparsa: {A.nnz} elementos não-zeros")
print(f"b: {b.shape}")
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/assemble.py).
