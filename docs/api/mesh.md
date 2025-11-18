# fempack.mesh

Estruturas de dados para malhas de elementos finitos.

## Classes

### Mesh

Malha estruturada para experimentos simples de FEM.

**Parâmetros:**

- `coords`: Array de coordenadas de vértices com shape (num_vertices, dim)
- `cells`: Array de conectividades com shape (num_cells, nloc)
- `cell_type`: Identificador da célula de referência: `"interval"`, `"triangle"` ou `"square"`

**Propriedades:**

- `dim`: Dimensão espacial da malha
- `num_vertices`: Número de vértices da malha
- `num_cells`: Número de células (elementos) na malha

**Métodos estáticos (factory methods):**

### Mesh.unit_interval(n)

Cria malha uniforme do intervalo unitário [0, 1].

**Parâmetros:**

- `n`: Número de elementos (terá n+1 vértices)

**Retorna:** Objeto `Mesh` 1D estruturado com conectividade P1

### Mesh.unit_square_triangular(nx, ny)

Cria malha triangular P1 do quadrado unitário.

O domínio é (0, 1) × (0, 1) e cada célula retangular é subdividida em dois triângulos ao longo da diagonal principal.

**Parâmetros:**

- `nx`, `ny`: Número de células nas direções x e y

**Retorna:** Malha triangular com conectividade linear (P1)

### Mesh.unit_square_quads(nx, ny)

Cria malha de quadriláteros Q1 do quadrado unitário.

**Parâmetros:**

- `nx`, `ny`: Número de células nas direções x e y

**Retorna:** Malha de quadriláteros com conectividade Q1

**Métodos auxiliares:**

- `boundary_nodes()`: Retorna array de índices dos nós de fronteira

## Exemplo

```python
from fempack.mesh import Mesh

# Malha 1D com 10 elementos
mesh1d = Mesh.unit_interval(10)
print(f"Vértices: {mesh1d.num_vertices}, Células: {mesh1d.num_cells}")

# Malha 2D triangular 10x10
mesh_tri = Mesh.unit_square_triangular(10, 10)
print(f"Tipo: {mesh_tri.cell_type}")  # "triangle"

# Malha 2D com quadriláteros
mesh_quad = Mesh.unit_square_quads(10, 10)
print(f"Tipo: {mesh_quad.cell_type}")  # "square"
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/mesh.py).
