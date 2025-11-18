# fempack.local

Cálculo de matrizes e vetores locais (elementares).

## Funções de Rigidez (Stiffness)

### local_stiffness_p1_interval(x0, x1)

Calcula matriz de rigidez local para elemento P1 em intervalo.

**Parâmetros:**

- `x0`, `x1`: Extremos do elemento físico

**Retorna:** Matriz local (2, 2)

$$K[a,b] = \int (dN_a/dx)(dN_b/dx) \, dx$$

### local_stiffness_p2_interval(x0, x1, element, quad_points, quad_weights)

Calcula matriz de rigidez local para elemento P2 em intervalo.

**Retorna:** Matriz local (3, 3)

### local_stiffness_p1_triangle(vertices)

Calcula matriz de rigidez local para elemento P1 em triângulo.

**Parâmetros:**

- `vertices`: Array (3, 2) com coordenadas dos vértices

**Retorna:** Matriz local (3, 3)

### local_stiffness_q1_square(vertices, element, quad_points, quad_weights)

Calcula matriz de rigidez local para elemento Q1 em quadrilátero.

**Parâmetros:**

- `vertices`: Array (4, 2) com coordenadas dos vértices
- `element`: Objeto `LagrangeElement`
- `quad_points`: Pontos de quadratura (nq, 2)
- `quad_weights`: Pesos de quadratura (nq,)

**Retorna:** Matriz local (4, 4)

## Funções de Massa (Mass)

### local_mass_p1_interval(x0, x1)

Calcula matriz de massa local para P1 em intervalo.

**Retorna:** Matriz local (2, 2) com fórmula analítica

$$M = \frac{h}{6} \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$

### local_mass_p2_interval(x0, x1, element, quad_points, quad_weights)

Calcula matriz de massa local para P2 em intervalo.

**Retorna:** Matriz local (3, 3)

### local_mass_p1_triangle(vertices)

Calcula matriz de massa local para P1 em triângulo.

**Retorna:** Matriz local (3, 3)

### local_mass_q1_square(vertices, element, quad_points, quad_weights)

Calcula matriz de massa local para Q1 em quadrilátero.

**Retorna:** Matriz local (4, 4)

## Funções de Carga (Load)

### local_load_p1_interval(x0, x1, f, quad_points, quad_weights)

Calcula vetor de carga local para P1 em intervalo.

**Parâmetros:**

- `f`: Função fonte que aceita coordenadas 1D
- `quad_points`, `quad_weights`: Regra de quadratura

**Retorna:** Vetor local (2,)

$$b[i] = \int f N_i \, dx$$

### local_load_p1_triangle(vertices, f, quad_points, quad_weights)

Calcula vetor de carga local para P1 em triângulo.

**Parâmetros:**

- `f`: Função fonte que aceita coordenadas 2D: `f(x, y)`

**Retorna:** Vetor local (3,)

### local_load_q1_square(vertices, element, f, quad_points, quad_weights)

Calcula vetor de carga local para Q1 em quadrilátero.

**Parâmetros:**

- `f`: Função fonte que aceita coordenadas 2D: `f(x, y)`

**Retorna:** Vetor local (4,)

## Exemplo

```python
from fempack.local import local_stiffness_p1_interval, local_load_p1_interval
from fempack.quadrature import gauss_legendre_interval
import numpy as np

# Elemento no intervalo [0, 0.1]
x0, x1 = 0.0, 0.1
K_local = local_stiffness_p1_interval(x0, x1)
M_local = local_mass_p1_interval(x0, x1)

# Vetor de carga com f(x) = x^2
quad = gauss_legendre_interval(3)
f = lambda x: x**2
b_local = local_load_p1_interval(x0, x1, f, quad.points, quad.weights)

print(K_local.shape)  # (2, 2)
print(b_local.shape)  # (2,)
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/local.py).
