# Condições de Contorno de Dirichlet

Após a montagem do sistema global, precisamos impor as condições de contorno de Dirichlet antes de resolver o sistema linear.

## Problema

O sistema montado é:

$$
\mathbf{A} \mathbf{u} = \mathbf{b}
$$

Para condições de contorno não-homogêneas $u = g$ nos nós de fronteira, precisamos:

1. Fixar os valores de $u_i = g_i$ para nós $i \in \partial\Omega$
2. Modificar o sistema para acomodar esses valores conhecidos

## Métodos de Imposição

### Método 1: Eliminação Direta

Separe os graus de liberdade em:

- **Livres** ($\mathcal{F}$): Nós internos onde $u$ é desconhecido
- **Fixos** ($\mathcal{D}$): Nós de fronteira onde $u = g$ é conhecido

O sistema pode ser particionado:

$$
\begin{bmatrix}
\mathbf{A}_{\mathcal{FF}} & \mathbf{A}_{\mathcal{FD}} \\
\mathbf{A}_{\mathcal{DF}} & \mathbf{A}_{\mathcal{DD}}
\end{bmatrix}
\begin{bmatrix}
\mathbf{u}_{\mathcal{F}} \\
\mathbf{u}_{\mathcal{D}}
\end{bmatrix}
=
\begin{bmatrix}
\mathbf{b}_{\mathcal{F}} \\
\mathbf{b}_{\mathcal{D}}
\end{bmatrix}
$$

Resolvemos apenas:

$$
\mathbf{A}_{\mathcal{FF}} \mathbf{u}_{\mathcal{F}} = \mathbf{b}_{\mathcal{F}} - \mathbf{A}_{\mathcal{FD}} \mathbf{u}_{\mathcal{D}}
$$

### Método 2: Penalização

Modifique as linhas correspondentes aos nós de Dirichlet:

$$
A[i, :] = [0, \ldots, 0, 1, 0, \ldots, 0], \quad b[i] = g_i
$$

onde o 1 está na posição $(i, i)$.

Alternativamente, use um valor grande $\alpha \gg 1$:

$$
A[i, i] = \alpha, \quad b[i] = \alpha g_i
$$

### Método 3: Substituição de Linhas e Colunas (Usado no fempack)

Para cada nó $i$ com condição de Dirichlet $u_i = g_i$:

1. **Modifique a linha $i$**:
   - $A[i, j] = 0$ para $j \neq i$
   - $A[i, i] = 1$
   - $b[i] = g_i$

2. **Modifique a coluna $i$** nas outras linhas:
   - Para $j \neq i$: $b[j] = b[j] - A[j, i] \cdot g_i$
   - Para $j \neq i$: $A[j, i] = 0$

Isso garante que:

- A linha $i$ fornece $u_i = g_i$ diretamente
- As outras equações são ajustadas para levar em conta o valor conhecido $u_i = g_i$

## Algoritmo de Imposição

```python
Para cada nó i com condição de Dirichlet u_i = g_i:
    # Ajustar outras linhas
    Para cada j != i:
        b[j] -= A[j, i] * g_i
        A[j, i] = 0

    # Modificar linha i
    A[i, :] = 0
    A[i, i] = 1
    b[i] = g_i
```

## Identificação dos Nós de Fronteira

Para malhas estruturadas ou simples, os nós de fronteira podem ser identificados por:

- **1D**: Nós nas extremidades do intervalo
- **2D**: Nós com $x = x_{\min}$, $x = x_{\max}$, $y = y_{\min}$, ou $y = y_{\max}$

Para malhas gerais, pode-se usar uma tolerância:

$$
\text{na fronteira} \iff |\mathbf{x} - \mathbf{x}_{\text{bnd}}| < \epsilon
$$

## Exemplo 1D

Sistema original com 4 nós:

$$
\frac{1}{h} \begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}
\begin{bmatrix}
u_0 \\ u_1 \\ u_2 \\ u_3
\end{bmatrix}
=
\begin{bmatrix}
b_0 \\ b_1 \\ b_2 \\ b_3
\end{bmatrix}
$$

Impondo $u_0 = g_0$ e $u_3 = g_3$:

1. Modificar linha 0 e ajustar coluna 0:
   - Linha 1: $b_1 \leftarrow b_1 - A[1,0] \cdot g_0 = b_1 + \frac{1}{h} g_0$

2. Modificar linha 3 e ajustar coluna 3:
   - Linha 2: $b_2 \leftarrow b_2 - A[2,3] \cdot g_3 = b_2 + \frac{1}{h} g_3$

Sistema resultante:

$$
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 2/h & -1/h & 0 \\
0 & -1/h & 2/h & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
u_0 \\ u_1 \\ u_2 \\ u_3
\end{bmatrix}
=
\begin{bmatrix}
g_0 \\ b_1 + g_0/h \\ b_2 + g_3/h \\ g_3
\end{bmatrix}
$$

## Implementação no fempack

O módulo `fempack.bcs` fornece:

- `apply_dirichlet_bc()`: Aplica condições de Dirichlet usando o método de substituição
- `get_boundary_nodes()`: Identifica nós na fronteira

Exemplo:

```python
from fempack.bcs import apply_dirichlet_bc

# Aplicar u = 0 em toda a fronteira
A_bc, b_bc = apply_dirichlet_bc(A, b, mesh, lambda x: 0.0)

# Resolver sistema modificado
u = spsolve(A_bc, b_bc)
```
