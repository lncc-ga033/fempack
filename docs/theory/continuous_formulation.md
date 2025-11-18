# Formulação Variacional

## Problema de Poisson

Considere o problema de Poisson em um domínio $\Omega \subset \mathbb{R}^d$, $d = 1, 2$:

$$
\begin{cases}
-\nabla^2 u = f & \text{em } \Omega, \\
u = g & \text{em } \partial\Omega,
\end{cases}
$$

onde:

- $u: \Omega \to \mathbb{R}$ é a função desconhecida
- $f: \Omega \to \mathbb{R}$ é o termo fonte
- $g: \partial\Omega \to \mathbb{R}$ é a condição de contorno de Dirichlet

## Formulação Fraca

Multiplicando a equação diferencial por uma função teste $v \in H^1_0(\Omega)$ e integrando por partes:

$$
\int_\Omega \nabla u \cdot \nabla v \, dx = \int_\Omega f v \, dx
$$

O espaço de funções é:

$$
V = \{ v \in H^1(\Omega) : v = 0 \text{ em } \partial\Omega \}
$$

Para condições de contorno não-homogêneas, buscamos $u \in H^1(\Omega)$ tal que $u = g$ em $\partial\Omega$ e:

$$
a(u, v) = L(v), \quad \forall v \in V
$$

onde:

$$
a(u, v) = \int_\Omega \nabla u \cdot \nabla v \, dx, \quad L(v) = \int_\Omega f v \, dx
$$

## Discretização por Elementos Finitos

Substituindo $V$ por um espaço de dimensão finita $V_h \subset V$:

$$
V_h = \text{span}\{\varphi_1, \ldots, \varphi_N\}
$$

A solução aproximada é:

$$
u_h = \sum_{j=1}^N u_j \varphi_j
$$

Testando com as funções de base $\varphi_i$:

$$
\sum_{j=1}^N a(\varphi_j, \varphi_i) u_j = L(\varphi_i), \quad i = 1, \ldots, N
$$

Isso resulta no sistema linear:

$$
\mathbf{A} \mathbf{u} = \mathbf{b}
$$

onde:

$$
A_{ij} = a(\varphi_j, \varphi_i) = \int_\Omega \nabla \varphi_j \cdot \nabla \varphi_i \, dx
$$

$$
b_i = L(\varphi_i) = \int_\Omega f \varphi_i \, dx
$$
