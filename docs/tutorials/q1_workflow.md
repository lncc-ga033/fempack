# Workflow completo: Elementos Q1 em quadriláteros

Este tutorial demonstra o workflow completo do método de elementos finitos usando elementos bilineares (Q1) em quadriláteros para resolver o problema de Poisson 2D.

```{note}
Este tutorial está disponível como Jupyter notebook interativo em `notebooks/Intro_fempack_Q1_workflow.ipynb` no repositório.
```

## Visão geral

Vamos resolver o problema:

$$
-\Delta u = f \quad \text{em } \Omega = (0,1)^2
$$

com condições de Dirichlet $u = g$ em $\partial\Omega$, usando o método de soluções manufaturadas (MMS) para verificação.

## Etapas do workflow

1. **Definir problema modelo** com solução exata
2. **Criar malha** estruturada de quadriláteros
3. **Definir espaço** de elementos finitos
4. **Montar** matriz de rigidez e vetor de carga
5. **Impor** condições de Dirichlet
6. **Resolver** sistema linear
7. **Calcular erros** e verificar convergência

## Notebook completo

O notebook completo está disponível no repositório e inclui:

- Definição de funções para solução exata e termo fonte
- Visualização da malha
- Plots da solução numérica e exata
- Padrões de esparsidade das matrizes
- Estudo de convergência com refinamento de malha
- Gráficos log-log com taxas de convergência

## Principais conceitos cobertos

- Elementos Q1 (bilineares) em quadriláteros
- Mapeamento bilinear do elemento de referência
- Quadratura de Gauss por produto tensorial
- Montagem de matriz global esparsa
- Imposição forte de condições de Dirichlet
- Verificação por método de soluções manufaturadas
- Análise de convergência experimental

## Resultados esperados

Para elementos Q1, esperamos:

- Convergência $O(h^2)$ na norma $L^2$
- Convergência $O(h)$ na seminorma $H^1$

onde $h$ é o tamanho característico da malha.

## Ver também

- {doc}`../theory/reference_elements`: Detalhes sobre elemento Q1
- {doc}`../theory/quadrature`: Regras de quadratura usadas
- {doc}`../api/index`: Documentação das funções utilizadas
