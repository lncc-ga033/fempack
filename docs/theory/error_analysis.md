# Análise de Erros e Verificação

Para verificar a implementação do método de elementos finitos, calculamos erros entre a solução numérica $u_h$ e a solução exata $u$.

## Normas de Erro

Definimos as seguintes normas de erro:

### Norma $L^2$

$$
\|e\|_{L^2(\Omega)} = \sqrt{\int_\Omega (u - u_h)^2 \, d\mathbf{x}}
$$

Para calcular numericamente:

$$
\|e\|_{L^2}^2 \approx \sum_{K} \int_K (u - u_h)^2 \, d\mathbf{x} = \sum_{K} \sum_q w_q |u(\mathbf{x}_q) - u_h(\mathbf{x}_q)|^2 |J_K(\hat{\mathbf{x}}_q)|
$$

onde:

- $u(\mathbf{x}_q)$ é a solução exata avaliada no ponto de quadratura
- $u_h(\mathbf{x}_q) = \sum_i u_i N_i(\mathbf{x}_q)$ é a solução numérica interpolada

### Seminorma $H^1$ (Gradiente)

$$
|e|_{H^1(\Omega)} = \sqrt{\int_\Omega |\nabla(u - u_h)|^2 \, d\mathbf{x}}
$$

Numericamente:

$$
|e|_{H^1}^2 \approx \sum_{K} \sum_q w_q |\nabla u(\mathbf{x}_q) - \nabla u_h(\mathbf{x}_q)|^2 |J_K(\hat{\mathbf{x}}_q)|
$$

onde:

$$
\nabla u_h(\mathbf{x}_q) = \sum_i u_i \nabla N_i(\mathbf{x}_q)
$$

### Norma $H^1$ (Completa)

$$
\|e\|_{H^1(\Omega)} = \sqrt{\|e\|_{L^2}^2 + |e|_{H^1}^2}
$$

### Norma do Máximo ($L^\infty$)

$$
\|e\|_{L^\infty(\Omega)} = \max_{\mathbf{x} \in \Omega} |u(\mathbf{x}) - u_h(\mathbf{x})|
$$

Aproximadamente:

$$
\|e\|_{L^\infty} \approx \max_i |u(\mathbf{x}_i) - u_i|
$$

onde a maximização é feita sobre todos os nós.

## Taxas de Convergência

Para uma família de malhas com tamanho característico $h \to 0$, esperamos:

$$
\|u - u_h\|_{L^2} = \mathcal{O}(h^{p+1})
$$

$$
|u - u_h|_{H^1} = \mathcal{O}(h^p)
$$

onde $p$ é o grau polinomial do espaço de elementos finitos:

- **P1/Q1**: $p = 1$, então $L^2$ converge como $h^2$ e $H^1$ como $h^1$
- **P2**: $p = 2$, então $L^2$ converge como $h^3$ e $H^1$ como $h^2$

### Estimativa Experimental da Taxa

Para estimar a taxa de convergência experimentalmente, considere duas malhas com tamanhos $h_1 > h_2$:

$$
\text{taxa} = \frac{\log(E_1 / E_2)}{\log(h_1 / h_2)}
$$

onde $E_1$ e $E_2$ são os erros correspondentes.

Se plotarmos $\log E$ vs $\log h$, a inclinação da reta é a taxa de convergência.

## Method of Manufactured Solutions (MMS)

Para verificar a implementação:

1. **Escolha uma solução exata** $u(\mathbf{x})$ suave
2. **Calcule o termo fonte** correspondente:
   $$
   f(\mathbf{x}) = -\nabla^2 u(\mathbf{x})
   $$
3. **Resolva o problema** de elementos finitos com $f$ e condições de contorno $g = u|_{\partial\Omega}$
4. **Calcule os erros** comparando $u_h$ com $u$
5. **Verifique as taxas de convergência** refinando a malha

### Exemplo 1D

Solução manufaturada:

$$
u(x) = \sin(\pi x)
$$

Termo fonte:

$$
f(x) = -u''(x) = \pi^2 \sin(\pi x)
$$

Condições de contorno em $[0, 1]$:

$$
u(0) = 0, \quad u(1) = 0
$$

### Exemplo 2D

Solução manufaturada:

$$
u(x, y) = \sin(2\pi x) \sin(2\pi y)
$$

Termo fonte:

$$
f(x, y) = -\nabla^2 u = 8\pi^2 \sin(2\pi x) \sin(2\pi y)
$$

Condições de contorno em $[0, 1] \times [0, 1]$:

$$
u = 0 \text{ em } \partial\Omega
$$

## Teste de Convergência

Algoritmo típico:

```python
for h in [h1, h2, h3, ...]:  # Sequência de malhas cada vez mais finas
    mesh = create_mesh(h)
    A = assemble_stiffness(mesh, space)
    b = assemble_load(mesh, space, f)
    A_bc, b_bc = apply_dirichlet_bc(A, b, mesh, g)
    u_h = solve(A_bc, b_bc)

    error_L2 = compute_L2_error(u_h, u_exact, mesh, space)
    error_H1 = compute_H1_error(u_h, u_exact, grad_u_exact, mesh, space)

    print(f"h={h:.4f}, L2={error_L2:.6e}, H1={error_H1:.6e}")

# Plotar log(erro) vs log(h) e verificar inclinação
```

Para elementos P1/Q1, esperamos:

- Inclinação $\approx 2$ para erro $L^2$
- Inclinação $\approx 1$ para erro $H^1$

## Implementação no fempack

O módulo `fempack.verification` fornece:

- `compute_l2_error()`: Calcula $\|u - u_h\|_{L^2}$
- `compute_h1_error()`: Calcula $|u - u_h|_{H^1}$
- `compute_errors()`: Calcula múltiplas normas de uma vez
- `convergence_study()`: Automatiza estudos de convergência

Exemplo:

```python
from fempack.verification import compute_l2_error, compute_h1_error

error_L2 = compute_l2_error(u_h, mesh, space, u_exact)
error_H1 = compute_h1_error(u_h, mesh, space, u_exact, grad_u_exact)

print(f"||e||_L2 = {error_L2:.6e}")
print(f"|e|_H1 = {error_H1:.6e}")
```

Os notebooks em `notebooks/` demonstram estudos completos de convergência para diferentes tipos de elementos.
