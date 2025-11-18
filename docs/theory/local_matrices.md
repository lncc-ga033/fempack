# Matrizes Locais

Para cada elemento $K$, calculamos as matrizes elementares que contribuem para o sistema global.

## Matriz de Rigidez Local

A matriz de rigidez local para o elemento $K$ é definida por:

$$
A_K[i, j] = \int_K \nabla N_j \cdot \nabla N_i \, d\mathbf{x}
$$

Usando a transformação para o elemento de referência:

$$
A_K[i, j] = \int_{\hat K} (\mathbf{B}_K^{-1})^T \nabla \hat N_j \cdot (\mathbf{B}_K^{-1})^T \nabla \hat N_i \, |J_K| \, d\hat{\mathbf{x}}
$$

### Caso 1D

Para elementos em 1D:

$$
A_K[i, j] = \int_{\hat K} \frac{1}{h} \frac{d\hat N_j}{d\hat x} \cdot \frac{1}{h} \frac{d\hat N_i}{d\hat x} \cdot h \, d\hat x = \frac{1}{h} \int_{\hat K} \frac{d\hat N_i}{d\hat x} \frac{d\hat N_j}{d\hat x} \, d\hat x
$$

**Exemplo P1**:

$$
\hat A = \int_0^1 \begin{bmatrix} -1 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 1 \end{bmatrix} d\hat x = \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

Logo:

$$
A_K = \frac{1}{h} \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}
$$

### Caso 2D - Triângulos P1

Para triângulos P1, os gradientes são constantes:

$$
A_K[i, j] = |K| \cdot (\mathbf{B}_K^{-1})^T \nabla \hat N_j \cdot (\mathbf{B}_K^{-1})^T \nabla \hat N_i
$$

### Caso 2D - Quadriláteros Q1

Para Q1, os gradientes variam espacialmente, então precisamos de quadratura numérica:

$$
A_K[i, j] \approx \sum_{q} w_q \left[ (\mathbf{B}_K^{-1}(\hat{\mathbf{x}}_q))^T \nabla \hat N_j(\hat{\mathbf{x}}_q) \cdot (\mathbf{B}_K^{-1}(\hat{\mathbf{x}}_q))^T \nabla \hat N_i(\hat{\mathbf{x}}_q) \right] |J_K(\hat{\mathbf{x}}_q)|
$$

## Matriz de Massa Local

A matriz de massa local para o elemento $K$ é definida por:

$$
M_K[i, j] = \int_K N_j N_i \, d\mathbf{x} = \int_{\hat K} \hat N_j \hat N_i \, |J_K| \, d\hat{\mathbf{x}}
$$

### Caso 1D - P1

$$
M_K = h \int_0^1 \begin{bmatrix} 1 - \hat x \\ \hat x \end{bmatrix} \begin{bmatrix} 1 - \hat x & \hat x \end{bmatrix} d\hat x
$$

Calculando as integrais:

$$
M_K = h \begin{bmatrix}
\frac{1}{3} & \frac{1}{6} \\
\frac{1}{6} & \frac{1}{3}
\end{bmatrix}
$$

### Lumped Mass Matrix

A matriz de massa concentrada (lumped) é obtida somando cada linha e colocando o resultado na diagonal:

$$
M_K^{\text{lumped}} = \text{diag}\left( \sum_j M_K[i, j] \right)
$$

Para P1 em 1D:

$$
M_K^{\text{lumped}} = \frac{h}{2} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

## Vetor de Carga Local

O vetor de carga local para o elemento $K$ é definido por:

$$
b_K[i] = \int_K f N_i \, d\mathbf{x} = \int_{\hat K} f(\mathbf{F}_K(\hat{\mathbf{x}})) \hat N_i(\hat{\mathbf{x}}) \, |J_K(\hat{\mathbf{x}})| \, d\hat{\mathbf{x}}
$$

Esta integral é tipicamente calculada usando quadratura numérica:

$$
b_K[i] \approx \sum_{q} w_q f(\mathbf{F}_K(\hat{\mathbf{x}}_q)) \hat N_i(\hat{\mathbf{x}}_q) |J_K(\hat{\mathbf{x}}_q)|
$$

## Implementação no fempack

No código:

- `fempack.local.stiffness_matrix()`: Calcula $A_K$
- `fempack.local.mass_matrix()`: Calcula $M_K$ (consistente ou concentrada)
- `fempack.local.load_vector()`: Calcula $b_K$

Todas essas funções usam quadratura numérica fornecida por `fempack.quadrature`.
