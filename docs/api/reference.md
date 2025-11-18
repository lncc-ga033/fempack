# fempack.reference

Módulo de elementos de referência, fornecendo funções de forma e gradientes para elementos P1, P2, P1 triangular e Q1 quadrilateral.

## Classes

### ReferenceCell

Descritor simples de uma célula de referência.

**Parâmetros:**

- `name`: Nome legível (ex: `"triangle"`)
- `dim`: Dimensão espacial da célula

**Constantes pré-definidas:**

- `INTERVAL = ReferenceCell("interval", 1)`
- `TRIANGLE = ReferenceCell("triangle", 2)`
- `SQUARE = ReferenceCell("square", 2)`

## Funções de Forma

### p1_interval_shape_functions(x)

Avalia funções de forma P1 no intervalo [0, 1].

**Parâmetros:**

- `x`: Coordenadas de referência com shape (nq,)

**Retorna:** Array (nq, 2) com:

- Coluna 0: $N_0(x) = 1 - x$ (nó esquerdo)
- Coluna 1: $N_1(x) = x$ (nó direito)

### p2_interval_shape_functions(x)

Avalia funções de forma P2 no intervalo [0, 1] com nós em ξ = 0, 0.5, 1.

**Funções de forma:**

- $N_0(\xi) = 2(\xi - 0.5)(\xi - 1)$ (ponto extremo esquerdo)
- $N_1(\xi) = -4\xi(\xi - 1)$ (ponto médio)
- $N_2(\xi) = 2\xi(\xi - 0.5)$ (ponto extremo direito)

**Retorna:** Array (nq, 3)

### p1_triangle_shape_functions(x, y)

Avalia funções de forma P1 no triângulo de referência com vértices em (0,0), (1,0), (0,1).

**Retorna:** Array (nq, 3) com:

- Coluna 0: $\lambda_1 = 1 - x - y$ (vértice na origem)
- Coluna 1: $\lambda_2 = x$ (vértice em (1,0))
- Coluna 2: $\lambda_3 = y$ (vértice em (0,1))

### q1_square_shape_functions(xi, eta)

Avalia funções de forma Q1 no quadrado de referência [-1, 1]².

**Retorna:** Array (nq, 4) com funções bilineares

## Funções de Gradientes

### p1_interval_gradients(x0, x1)

Calcula gradientes de P1 em intervalo físico.

**Retorna:** `(gradients, h)` onde h é o comprimento do elemento

### p2_interval_gradients(x0, x1, xi)

Calcula gradientes de P2 em pontos de referência.

### p1_gradients(vertices)

Calcula gradientes de P1 para triângulo.

**Retorna:** `(gradients, area)`

### q1_gradients(vertices, xi, eta)

Calcula gradientes de Q1 em quadrilátero.

## Funções Auxiliares

- `interval_length(x0, x1)`: Comprimento de intervalo
- `triangle_area(vertices)`: Área de triângulo

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/reference.py).
