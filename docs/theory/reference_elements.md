# Elementos de Referência

O `fempack` implementa vários elementos de referência. Esta seção descreve as funções de forma em cada elemento de referência.

## P1 no Intervalo

**Domínio de referência**: $\hat K = [0, 1]$

**Graus de liberdade**: Valores nos vértices $\hat x_0 = 0$ e $\hat x_1 = 1$

**Funções de forma**:

$$
\hat N_0(\hat x) = 1 - \hat x, \quad \hat N_1(\hat x) = \hat x
$$

**Gradientes**:

$$
\frac{d\hat N_0}{d\hat x} = -1, \quad \frac{d\hat N_1}{d\hat x} = 1
$$

## P2 no Intervalo

**Domínio de referência**: $\hat K = [0, 1]$

**Graus de liberdade**: Valores nos vértices $\hat x_0 = 0$, $\hat x_1 = 1$ e no ponto médio $\hat x_2 = 1/2$

**Funções de forma**:

$$
\hat N_0(\hat x) = (1 - \hat x)(1 - 2\hat x)
$$

$$
\hat N_1(\hat x) = \hat x(2\hat x - 1)
$$

$$
\hat N_2(\hat x) = 4\hat x(1 - \hat x)
$$

**Gradientes**:

$$
\frac{d\hat N_0}{d\hat x} = 4\hat x - 3
$$

$$
\frac{d\hat N_1}{d\hat x} = 4\hat x - 1
$$

$$
\frac{d\hat N_2}{d\hat x} = 4 - 8\hat x
$$

## P1 no Triângulo

**Domínio de referência**: $\hat K = \{(\hat x, \hat y) : \hat x \geq 0, \hat y \geq 0, \hat x + \hat y \leq 1\}$

**Vértices**: $\hat{\mathbf{x}}_0 = (0, 0)$, $\hat{\mathbf{x}}_1 = (1, 0)$, $\hat{\mathbf{x}}_2 = (0, 1)$

**Funções de forma**:

$$
\hat N_0(\hat x, \hat y) = 1 - \hat x - \hat y
$$

$$
\hat N_1(\hat x, \hat y) = \hat x
$$

$$
\hat N_2(\hat x, \hat y) = \hat y
$$

**Gradientes**:

$$
\nabla \hat N_0 = \begin{bmatrix} -1 \\ -1 \end{bmatrix}, \quad
\nabla \hat N_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
\nabla \hat N_2 = \begin{bmatrix} 0 \\ 1 \end{bmatrix}
$$

## Q1 no Quadrilátero

**Domínio de referência**: $\hat K = [0, 1] \times [0, 1]$

**Vértices**: $\hat{\mathbf{x}}_0 = (0, 0)$, $\hat{\mathbf{x}}_1 = (1, 0)$, $\hat{\mathbf{x}}_2 = (1, 1)$, $\hat{\mathbf{x}}_3 = (0, 1)$

**Funções de forma** (tensor product):

$$
\hat N_0(\hat x, \hat y) = (1 - \hat x)(1 - \hat y)
$$

$$
\hat N_1(\hat x, \hat y) = \hat x(1 - \hat y)
$$

$$
\hat N_2(\hat x, \hat y) = \hat x \hat y
$$

$$
\hat N_3(\hat x, \hat y) = (1 - \hat x)\hat y
$$

**Gradientes**:

$$
\nabla \hat N_0 = \begin{bmatrix} -(1 - \hat y) \\ -(1 - \hat x) \end{bmatrix}, \quad
\nabla \hat N_1 = \begin{bmatrix} 1 - \hat y \\ -\hat x \end{bmatrix}
$$

$$
\nabla \hat N_2 = \begin{bmatrix} \hat y \\ \hat x \end{bmatrix}, \quad
\nabla \hat N_3 = \begin{bmatrix} -\hat y \\ 1 - \hat x \end{bmatrix}
$$

## Implementação no fempack

No código, as funções de forma e gradientes são implementadas nos módulos:

- `fempack.reference`: Define elementos de referência
- `fempack.elements`: Fornece interface unificada para diferentes tipos de elementos
