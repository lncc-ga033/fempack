# API Reference

Esta seção contém a documentação completa da API do `fempack`, gerada automaticamente a partir do código-fonte.

## Módulos

### Elementos e Geometria

- [](reference.md): Elementos de referência (P1, P2, Q1) com funções de forma e gradientes
- [](elements.md): Interface unificada para diferentes tipos de elementos finitos
- [](mesh.md): Estruturas de dados para malhas e conectividade

### Espaços e Quadratura

- [](spaces.md): Espaços de elementos finitos e funções de base globais
- [](quadrature.md): Regras de quadratura numérica

### Montagem e Cálculo Local

- [](local.md): Matrizes e vetores locais (rigidez, massa, carga)
- [](assemble.md): Montagem do sistema global

### Condições de Contorno e Resolução

- [](bcs.md): Imposição de condições de contorno de Dirichlet
- [](solvers.md): Interfaces para resolvedores de sistemas lineares

### Verificação

- [](verification.md): Cálculo de erros e estudos de convergência

## Uso Básico

```python
import fempack
from fempack.mesh import UniformMesh1D
from fempack.spaces import LagrangeSpace
from fempack.assemble import assemble_stiffness, assemble_load

# Criar malha
mesh = UniformMesh1D(0, 1, 10)

# Criar espaço de elementos finitos
space = LagrangeSpace(mesh, degree=1)

# Montar sistema
A = assemble_stiffness(mesh, space)
b = assemble_load(mesh, space, source_function)
```

Para documentação detalhada de cada função e classe, consulte o código-fonte ou use `help()` no Python.
