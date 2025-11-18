# Quadratura Numérica

Para calcular as integrais nas matrizes locais, usamos regras de quadratura numérica.

## Princípio Geral

Uma regra de quadratura aproxima uma integral por uma soma ponderada:

$$
\int_{\hat K} g(\hat{\mathbf{x}}) \, d\hat{\mathbf{x}} \approx \sum_{q=1}^{n_q} w_q g(\hat{\mathbf{x}}_q)
$$

onde:

- $\hat{\mathbf{x}}_q$ são os **pontos de quadratura** (quadrature points)
- $w_q$ são os **pesos de quadratura** (quadrature weights)
- $n_q$ é o número de pontos de quadratura

## Quadratura em 1D (Gauss-Legendre)

Para o intervalo $[0, 1]$, usamos pontos e pesos de Gauss-Legendre transformados.

### Gauss-1 (ponto médio)

$$
\hat x_0 = \frac{1}{2}, \quad w_0 = 1
$$

Exata para polinômios de grau $\leq 1$.

### Gauss-2

$$
\hat x_0 = \frac{1}{2} - \frac{\sqrt{3}}{6}, \quad \hat x_1 = \frac{1}{2} + \frac{\sqrt{3}}{6}
$$

$$
w_0 = \frac{1}{2}, \quad w_1 = \frac{1}{2}
$$

Exata para polinômios de grau $\leq 3$.

### Gauss-3

$$
\hat x_0 = \frac{1}{2} - \frac{\sqrt{15}}{10}, \quad \hat x_1 = \frac{1}{2}, \quad \hat x_2 = \frac{1}{2} + \frac{\sqrt{15}}{10}
$$

$$
w_0 = \frac{5}{18}, \quad w_1 = \frac{4}{9}, \quad w_2 = \frac{5}{18}
$$

Exata para polinômios de grau $\leq 5$.

## Quadratura em Triângulos

Para o triângulo de referência $\hat K = \{(\hat x, \hat y) : \hat x, \hat y \geq 0, \hat x + \hat y \leq 1\}$.

### Ordem 1 (centroide)

$$
(\hat x_0, \hat y_0) = \left(\frac{1}{3}, \frac{1}{3}\right), \quad w_0 = \frac{1}{2}
$$

Exata para polinômios de grau $\leq 1$.

### Ordem 2 (3 pontos)

$$
(\hat x_i, \hat y_i) = \left(\frac{1}{6}, \frac{1}{6}\right), \left(\frac{2}{3}, \frac{1}{6}\right), \left(\frac{1}{6}, \frac{2}{3}\right)
$$

$$
w_i = \frac{1}{6}, \quad i = 0, 1, 2
$$

Exata para polinômios de grau $\leq 2$.

### Ordem 3 (4 pontos)

$$
(\hat x_0, \hat y_0) = \left(\frac{1}{3}, \frac{1}{3}\right), \quad w_0 = -\frac{9}{32}
$$

$$
(\hat x_i, \hat y_i) = \left(\frac{1}{5}, \frac{1}{5}\right), \left(\frac{3}{5}, \frac{1}{5}\right), \left(\frac{1}{5}, \frac{3}{5}\right), \quad w_i = \frac{25}{96}
$$

Exata para polinômios de grau $\leq 3$.

## Quadratura em Quadriláteros

Para o quadrado de referência $\hat K = [0, 1] \times [0, 1]$, usamos produto tensorial de regras 1D.

### Gauss 2×2 (4 pontos)

Para Gauss-2 em cada direção:

$$
\hat x_i = \frac{1}{2} \pm \frac{\sqrt{3}}{6}, \quad \hat y_j = \frac{1}{2} \pm \frac{\sqrt{3}}{6}
$$

Pontos:

$$
(\hat x, \hat y)_{ij} = \left(\frac{1}{2} - \frac{\sqrt{3}}{6}, \frac{1}{2} - \frac{\sqrt{3}}{6}\right), \ldots
$$

Pesos:

$$
w_{ij} = \frac{1}{4}
$$

Exata para polinômios de grau $\leq 3$ em cada variável.

### Gauss 3×3 (9 pontos)

Para Gauss-3 em cada direção, obtemos 9 pontos. Exata para polinômios de grau $\leq 5$ em cada variável.

## Escolha da Ordem de Quadratura

Para integrar corretamente:

- **Matriz de rigidez**: Para P1/Q1, o integrando tem grau 0 (gradientes constantes para P1) ou 1 (Q1). Para P2, grau 2.
- **Matriz de massa**: Para P1, o integrando tem grau 2. Para P2, grau 4. Para Q1, grau 2 em cada variável.
- **Vetor de carga**: Depende do grau de $f$ e das funções de forma.

**Regra prática**: Use ordem de quadratura $\geq$ grau do integrando dividido por 2 (arredondado para cima).

## Implementação no fempack

O módulo `fempack.quadrature` fornece:

- `gauss_legendre_interval(n)`: Pontos e pesos para intervalo
- `triangle_quadrature(order)`: Pontos e pesos para triângulo
- `quad_quadrature(n)`: Pontos e pesos para quadrilátero (produto tensorial)

Exemplo:

```python
from fempack.quadrature import gauss_legendre_interval

points, weights = gauss_legendre_interval(2)  # Gauss-2
```
