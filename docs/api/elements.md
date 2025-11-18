# fempack.elements

Interface unificada para elementos finitos Lagrangeanos.

## Classes

### LagrangeElement

Elemento Lagrange de grau 1 ou 2.

**Parâmetros:**

- `cell_type`: Tipo de célula de referência: `"interval"`, `"triangle"` ou `"square"`
- `degree`: Grau polinomial (1 ou 2). Para intervalos: P1 e P2. Para triângulos e quadrados: apenas grau 1.

**Propriedades:**

- `dim`: Dimensão espacial do elemento de referência
- `dofs_per_cell`: Número de graus de liberdade locais na célula

**Métodos:**

- `tabulate(points)`: Avalia funções de forma em pontos de referência
  - **Parâmetros:** `points` - Array (nq, dim) de coordenadas de referência
  - **Retorna:** Array (nq, dofs_per_cell) com valores das funções de forma

- `tabulate_grad(points)`: Avalia gradientes das funções de forma
  - **Retorna:** Array (nq, dofs_per_cell, dim)

## Tipos de Elementos Suportados

- **P1 interval**: Elementos lineares em intervalos (2 DOFs)
- **P2 interval**: Elementos quadráticos em intervalos (3 DOFs)
- **P1 triangle**: Elementos lineares em triângulos (3 DOFs)
- **Q1 square**: Elementos bilineares em quadriláteros (4 DOFs)

**Importante:** Espaços globais (`FunctionSpace`) são restritos a grau 1. P2 é usado apenas para experimentos locais 1D e verificação.

## Exemplo de Uso

```python
from fempack.elements import LagrangeElement
import numpy as np

# Criar elemento P1 em 1D
element = LagrangeElement(cell_type="interval", degree=1)

# Avaliar funções de forma em pontos
xi = np.array([0.25, 0.5, 0.75])
N = element.tabulate(xi)
print(N.shape)  # (3, 2)

# Criar elemento Q1 para quadrilátero
element_q1 = LagrangeElement(cell_type="square", degree=1)
print(element_q1.dofs_per_cell)  # 4
```

Veja o [código fonte](https://github.com/lncc-ga033/fempack/blob/main/src/fempack/elements.py).
