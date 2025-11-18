# Documentação fempack

Esta pasta contém a documentação do `fempack` usando Jupyter Book.

## Estrutura

```
docs/
├── _config.yml          # Configuração do Jupyter Book
├── _toc.yml            # Tabela de conteúdos
├── index.md            # Página inicial
├── intro.md            # Introdução ao fempack
├── references.bib      # Referências bibliográficas
├── requirements.txt    # Dependências para build
├── tutorials/          # Tutoriais e notebooks
│   ├── index.md
│   └── q1_workflow.md
├── theory/             # Manual de cálculos
│   ├── index.md
│   ├── continuous_formulation.md
│   ├── reference_elements.md
│   ├── mappings.md
│   ├── local_matrices.md
│   ├── quadrature.md
│   ├── assembly.md
│   ├── boundary_conditions.md
│   └── error_analysis.md
└── api/                # Referência da API (auto-gerada)
    └── index.md
```

## Como construir

### 1. Instalar dependências

```bash
# Na raiz do projeto fempack
pip install -e ".[docs]"
```

### 2. Construir a documentação

**Opção A: Com Sphinx (recomendado para desenvolvimento)**

```bash
# Usando o script wrapper
./build-docs.sh

# Ou diretamente
python -m sphinx -b html docs docs/_build/html
```

**Opção B: Com Jupyter Book (para visualização interativa)**

```bash
# Servidor de desenvolvimento com hot-reload
jupyter-book start

# Ou apenas build
jupyter-book build docs/
```

### 3. Visualizar

**Com Sphinx:**

```bash
# macOS
open docs/_build/html/index.html

# Linux
xdg-open docs/_build/html/index.html
```

**Com Jupyter Book:**

Acesse <http://localhost:3000> após executar `jupyter-book start`.

## Limpar build

Para limpar arquivos de build anteriores:

```bash
jupyter-book clean docs/
```

## Reconstruir tudo

Para forçar reconstrução completa:

```bash
jupyter-book clean docs/
jupyter-book build docs/
```

## Conteúdo

### Tutoriais

Tutoriais práticos demonstrando o uso do fempack para resolver problemas de elementos finitos.

### Teoria

Manual de cálculos com todas as fórmulas implementadas:

- Formulação fraca do problema de Poisson
- Elementos de referência e funções de forma
- Mapeamentos geométricos
- Matrizes locais (rigidez, massa, carga)
- Quadratura numérica
- Montagem do sistema global
- Imposição de condições de contorno
- Análise de erros e convergência

### API Reference

Visão geral dos módulos e suas funcionalidades principais.

## Notas Técnicas

- **Formato**: Jupyter Book 2.x (baseado no MyST Document Engine)
- **Markup**: MyST Markdown com suporte a LaTeX math (`$$` para display, `$` para inline)
- **Configuração**: `_config.yml` (formato MyST) e `_toc.yml` (tabela de conteúdos)
- **Build**: Usa `jupyter-book build` para gerar HTML estático
- **Preview**: Use `jupyter-book start` para servidor de desenvolvimento com live reload
