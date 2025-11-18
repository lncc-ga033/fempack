# Mapeamentos Geométricos

Para conectar o elemento de referência $\hat K$ ao elemento físico $K$, usamos um mapeamento geométrico $\mathbf{F}_K: \hat K \to K$.

## Caso 1D

Para elementos P1 e P2 no intervalo, o mapeamento afim é:

$$
F_K(\hat x) = x_0 + (x_1 - x_0)\hat x = x_0 + h\hat x
$$

onde $h = x_1 - x_0$ é o comprimento do elemento.

**Jacobiano**:

$$
J_K = \frac{dF_K}{d\hat x} = h
$$

**Transformação de derivadas**:

$$
\frac{dN_i}{dx} = \frac{1}{J_K} \frac{d\hat N_i}{d\hat x} = \frac{1}{h} \frac{d\hat N_i}{d\hat x}
$$

## Caso 2D - Triângulos P1

Para triângulos com vértices $\mathbf{x}_0, \mathbf{x}_1, \mathbf{x}_2$, o mapeamento afim é:

$$
\mathbf{F}_K(\hat x, \hat y) = \mathbf{x}_0 + (\mathbf{x}_1 - \mathbf{x}_0)\hat x + (\mathbf{x}_2 - \mathbf{x}_0)\hat y
$$

**Matriz Jacobiana**:

$$
\mathbf{B}_K = \begin{bmatrix}
x_1 - x_0 & x_2 - x_0 \\
y_1 - y_0 & y_2 - y_0
\end{bmatrix}
$$

**Determinante Jacobiano**:

$$
|J_K| = \det(\mathbf{B}_K) = (x_1 - x_0)(y_2 - y_0) - (x_2 - x_0)(y_1 - y_0) = 2|K|
$$

onde $|K|$ é a área do triângulo.

**Transformação de gradientes**:

$$
\nabla N_i = (\mathbf{B}_K^{-1})^T \nabla \hat N_i
$$

## Caso 2D - Quadriláteros Q1

Para quadriláteros, o mapeamento bilinear é:

$$
\mathbf{F}_K(\hat x, \hat y) = \sum_{i=0}^3 \mathbf{x}_i \hat N_i(\hat x, \hat y)
$$

onde $\hat N_i$ são as funções de forma Q1.

**Matriz Jacobiana**:

$$
\mathbf{B}_K(\hat x, \hat y) = \begin{bmatrix}
\frac{\partial x}{\partial \hat x} & \frac{\partial x}{\partial \hat y} \\
\frac{\partial y}{\partial \hat x} & \frac{\partial y}{\partial \hat y}
\end{bmatrix}
$$

Os elementos da matriz são:

$$
\frac{\partial x}{\partial \hat x} = \sum_{i=0}^3 x_i \frac{\partial \hat N_i}{\partial \hat x}
$$

$$
\frac{\partial x}{\partial \hat y} = \sum_{i=0}^3 x_i \frac{\partial \hat N_i}{\partial \hat y}
$$

e similarmente para as derivadas de $y$.

**Determinante Jacobiano** (varia com $(\hat x, \hat y)$):

$$
|J_K(\hat x, \hat y)| = \det(\mathbf{B}_K(\hat x, \hat y))
$$

**Transformação de gradientes**:

$$
\nabla N_i(\mathbf{x}) = (\mathbf{B}_K^{-1}(\hat{\mathbf{x}}))^T \nabla \hat N_i(\hat{\mathbf{x}})
$$

onde $\mathbf{x} = \mathbf{F}_K(\hat{\mathbf{x}})$.

## Mudança de Variáveis em Integrais

Para integrar uma função $g$ sobre o elemento físico $K$:

**Caso 1D**:

$$
\int_K g(x) \, dx = \int_{\hat K} g(F_K(\hat x)) |J_K| \, d\hat x
$$

**Caso 2D**:

$$
\int_K g(\mathbf{x}) \, d\mathbf{x} = \int_{\hat K} g(\mathbf{F}_K(\hat{\mathbf{x}})) |J_K(\hat{\mathbf{x}})| \, d\hat{\mathbf{x}}
$$

Esta transformação é fundamental para calcular as integrais nas matrizes locais.
