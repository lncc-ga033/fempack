# Referência de Cálculos

Esta seção contém as fórmulas matemáticas e derivações implementadas no `fempack`.

## Visão geral

O manual de cálculos está organizado em:

1. **{doc}`continuous_formulation`**: Formulação fraca do problema de Poisson
2. **{doc}`reference_elements`**: Elementos de referência e funções de forma
3. **{doc}`mappings`**: Mapeamentos geométricos e Jacobianos
4. **{doc}`local_matrices`**: Matrizes locais de rigidez, massa e carga
5. **{doc}`quadrature`**: Regras de quadratura numérica
6. **{doc}`assembly`**: Montagem do sistema global
7. **{doc}`boundary_conditions`**: Imposição de condições de Dirichlet
8. **{doc}`error_analysis`**: Cálculo de erros e verificação

## Notação

Ao longo desta seção, usamos a seguinte notação:

- $\Omega$: Domínio computacional
- $K$: Elemento finito físico
- $\hat K$: Elemento de referência
- $N_i$: Função de forma (física)
- $\hat N_i$: Função de forma de referência
- $h$: Tamanho característico do elemento
- $|K|$: Medida (comprimento/área) do elemento $K$
- $\nabla$: Operador gradiente
- $\varphi_i$: Função de base global
