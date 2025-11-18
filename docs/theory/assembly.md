# Montagem do Sistema Global

A montagem (assembly) consiste em combinar as contribuições de todos os elementos para formar as matrizes e vetores globais.

## Numeração dos Graus de Liberdade

Considere uma malha com:

- $N_v$ vértices (ou nós)
- $N_e$ elementos

Cada elemento $K_e$ tem vértices/nós locais indexados por $i = 0, 1, \ldots, n_{\text{loc}}-1$, onde $n_{\text{loc}}$ é o número de nós por elemento (2 para P1-1D, 3 para P1-triângulo, 4 para Q1-quadrilátero, etc.).

O elemento $K_e$ tem uma conectividade $\text{conn}_e[i]$ que mapeia o índice local $i$ para o índice global $j$.

## Matriz Global de Rigidez

A matriz global $\mathbf{A} \in \mathbb{R}^{N_v \times N_v}$ é montada por:

$$
A[I, J] = \sum_{e=1}^{N_e} \sum_{\substack{i, j : \\ \text{conn}_e[i] = I \\ \text{conn}_e[j] = J}} A_{K_e}[i, j]
$$

Em outras palavras, cada entrada $A_{K_e}[i, j]$ da matriz local contribui para a entrada global $A[I, J]$, onde $I = \text{conn}_e[i]$ e $J = \text{conn}_e[j]$.

### Algoritmo de Montagem

```
Inicializar A = matriz esparsa N_v × N_v

Para cada elemento K_e:
    Calcular matriz local A_Ke
    Para cada i local:
        I = conn_e[i]
        Para cada j local:
            J = conn_e[j]
            A[I, J] += A_Ke[i, j]
```

## Vetor Global de Carga

Similarmente, o vetor global $\mathbf{b} \in \mathbb{R}^{N_v}$ é montado por:

$$
b[I] = \sum_{e=1}^{N_e} \sum_{i : \text{conn}_e[i] = I} b_{K_e}[i]
$$

### Algoritmo de Montagem

```
Inicializar b = vetor zero de tamanho N_v

Para cada elemento K_e:
    Calcular vetor local b_Ke
    Para cada i local:
        I = conn_e[i]
        b[I] += b_Ke[i]
```

## Matriz de Massa Global

A matriz de massa global $\mathbf{M}$ é montada da mesma forma que a matriz de rigidez:

$$
M[I, J] = \sum_{e=1}^{N_e} \sum_{\substack{i, j : \\ \text{conn}_e[i] = I \\ \text{conn}_e[j] = J}} M_{K_e}[i, j]
$$

## Estrutura de Dados Esparsa

Como a maioria das entradas de $\mathbf{A}$ são zero (os nós só acoplam através de elementos compartilhados), usamos formato esparso:

- **COO** (Coordinate format): Lista de triplas $(i, j, v)$
- **CSR** (Compressed Sparse Row): Formato compacto para armazenamento e operações

No `fempack`, usamos `scipy.sparse.lil_matrix` durante a montagem (eficiente para inserções) e convertemos para `csr_matrix` antes de resolver (eficiente para álgebra linear).

## Exemplo 1D

Considere 3 elementos P1 cobrindo $[0, 1]$:

- Elemento 0: nós globais 0, 1
- Elemento 1: nós globais 1, 2
- Elemento 2: nós globais 2, 3

Cada elemento contribui com uma matriz $2 \times 2$:

$$
A_{K_0} = \frac{1}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \rightarrow \text{posições } (0,0), (0,1), (1,0), (1,1)
$$

$$
A_{K_1} = \frac{1}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \rightarrow \text{posições } (1,1), (1,2), (2,1), (2,2)
$$

$$
A_{K_2} = \frac{1}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix} \rightarrow \text{posições } (2,2), (2,3), (3,2), (3,3)
$$

A matriz global resultante é:

$$
\mathbf{A} = \frac{1}{h} \begin{bmatrix}
1 & -1 & 0 & 0 \\
-1 & 2 & -1 & 0 \\
0 & -1 & 2 & -1 \\
0 & 0 & -1 & 1
\end{bmatrix}
$$

Note que o nó 1 recebe contribuições dos elementos 0 e 1 (por isso $A[1,1] = 1 + 1 = 2$).

## Implementação no fempack

O módulo `fempack.assemble` fornece:

- `assemble_stiffness()`: Monta $\mathbf{A}$
- `assemble_load()`: Monta $\mathbf{b}$
- `assemble_mass()`: Monta $\mathbf{M}$

Exemplo:

```python
from fempack.assemble import assemble_stiffness, assemble_load

A = assemble_stiffness(mesh, space)
b = assemble_load(mesh, space, f_rhs)
```

Essas funções percorrem todos os elementos, calculam as matrizes/vetores locais usando `fempack.local`, e acumulam nas estruturas globais usando a conectividade fornecida por `mesh` e `space`.
