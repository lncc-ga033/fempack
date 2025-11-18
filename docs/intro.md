# Introdução ao fempack

O `fempack` é um pacote Python educacional desenvolvido para ensinar os fundamentos do **método de elementos finitos (FEM)** aplicado a problemas de equações diferenciais parciais elípticas.

## Motivação

Este pacote foi criado com os seguintes objetivos:

1. **Transparência**: Código claro e bem documentado que permite entender cada passo do FEM
2. **Educação**: Estrutura que facilita o aprendizado dos conceitos fundamentais
3. **Verificação**: Implementação de verificação por soluções manufaturadas (MMS)
4. **Modularidade**: Separação clara entre conceitos (elementos de referência, quadratura, montagem, etc.)

## Problemas suportados

O `fempack` atualmente suporta o problema de Poisson escalar:

$$
-\Delta u = f \quad \text{em } \Omega
$$

com condições de contorno de Dirichlet:

$$
u = g \quad \text{em } \partial\Omega
$$

onde $\Omega \subset \mathbb{R}^d$ é o domínio ($d=1$ ou $d=2$).

## Elementos finitos implementados

### 1D
- **P1**: Elementos lineares no intervalo
- **P2**: Elementos quadráticos no intervalo

### 2D
- **P1**: Elementos lineares em triângulos
- **Q1**: Elementos bilineares em quadriláteros

## Workflow típico

Um workflow típico com o `fempack` segue estes passos:

1. **Definir o problema**: Especificar $f$, $g$, e solução exata (para MMS)
2. **Criar malha**: Usar `Mesh.unit_square_quadrilateral()` ou similar
3. **Definir espaço**: Criar `FunctionSpace` com família e grau
4. **Montar sistema**: Usar `assemble_stiffness()` e `assemble_load()`
5. **Aplicar CCs**: Usar `apply_dirichlet()` para condições de contorno
6. **Resolver**: Usar `solve_direct()` ou métodos iterativos
7. **Verificar**: Calcular erros com `l2_h1_errors()`
8. **Visualizar**: Plotar solução e estudar convergência

## Próximos passos

- Explore os {doc}`tutorials/index` para ver exemplos práticos
- Consulte a {doc}`theory/index` para entender a matemática por trás
- Use a {doc}`api/index` como referência detalhada das funções
