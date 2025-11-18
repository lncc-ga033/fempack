# fempack.quadrature

Regras de quadratura numérica para integração em elementos finitos.

## Classes

### QuadratureRule

Regra de quadratura numérica em uma célula de referência.

**Parâmetros:**

- `cell`: Célula de referência (objeto `ReferenceCell`)
- `points`: Array (nq, dim) contendo pontos de quadratura em coordenadas de referência
- `weights`: Array 1D de pesos de quadratura
- `degree`: Grau de exatidão algébrica da regra

## Funções de Quadratura

### gauss_legendre_interval(n)

Regra de Gauss-Legendre no intervalo físico [0, 1].

**Parâmetros:**

- `n`: Número de pontos de quadratura

**Retorna:** `QuadratureRule` exata para polinômios até grau 2n-1

### gauss_legendre_reference_interval(n)

Regra de Gauss-Legendre no intervalo de referência [-1, 1].

**Parâmetros:**

- `n`: Número de pontos de quadratura

**Retorna:** `QuadratureRule` exata para polinômios até grau 2n-1

### tensor_product_square_reference(rule_1d)

Constrói regra de produto tensorial no quadrado de referência.

**Parâmetros:**

- `rule_1d`: Regra de quadratura 1D em [-1, 1]

**Retorna:** `QuadratureRule` no quadrado de referência com o mesmo grau de exatidão em cada direção coordenada

### triangle_quadrature(order)

Regra de quadratura no triângulo de referência.

**Parâmetros:**

- `order`: Ordem da regra (1, 2 ou 3)

**Retorna:** `QuadratureRule` apropriada para triângulos

## Exemplo

```python
from fempack.quadrature import gauss_legendre_interval, QuadratureRule

# Regra de Gauss-3 (exata até grau 5)
quad = gauss_legendre_interval(3)
print(quad.points.shape)  # (3, 1)
print(quad.weights.shape)  # (3,)
print(quad.degree)  # 5

# Integrar função no intervalo [0,1]
import numpy as np
f = lambda x: x**2
result = np.sum(quad.weights * f(quad.points[:, 0]))
print(f"Integral: {result}")  # 0.333...
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/quadrature.py).
