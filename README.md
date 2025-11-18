# fempack -- Small FEM teaching framework

[![Tests](https://github.com/lncc-ga033/fempack/actions/workflows/tests.yml/badge.svg)](https://github.com/lncc-ga033/fempack/actions/workflows/tests.yml)

This repository accompanies the lectures for the graduate course **GA033 - Método de Elementos Finitos: Implementação Computacional** (LNCC).

A Python finite element library for teaching and learning FEM fundamentals. Implements P1 and P2 Lagrange elements for solving PDEs in 1D and 2D, with verification via the Method of Manufactured Solutions (MMS).

## Quick start

```bash
# Clone the repository
git clone https://github.com/lncc-ga033/fempack.git
cd fempack

# Create and activate environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with notebook support
pip install -e ".[dev,notebook]"

# Install pre-commit hooks (recommended for development)
pre-commit install

# Run tests
pytest

# Open notebooks
jupyter lab
```

## Features

- **Elements**: P1 and P2 Lagrange elements on intervals, triangles, and quadrilaterals
- **Reference elements**: Complete API for shape functions and gradients
- **Assembly**: Efficient local-to-global assembly routines
- **Solvers**: Direct and iterative solvers for linear systems
- **Verification**: Method of Manufactured Solutions (MMS) for convergence testing
- **Type-safe**: Full mypy type checking support

## Project structure

```text
src/fempack/       # Main package
  ├── reference.py    # Reference element geometry and basis functions
  ├── elements.py     # Lagrange finite elements (P1, P2, Q1)
  ├── mesh.py         # Mesh data structures
  ├── quadrature.py   # Numerical integration rules
  ├── spaces.py       # Function space definitions
  ├── local.py        # Element-level matrix assembly
  ├── assemble.py     # Global assembly routines
  ├── bcs.py          # Boundary conditions
  ├── solvers.py      # Linear system solvers
  └── verification.py # MMS verification tools
tests/             # Comprehensive test suite (165+ tests)
notebooks/         # Jupyter notebooks for lectures
```

## Development

```bash
# Run tests
pytest

# Type checking
mypy

# Linting and formatting
ruff check .
ruff format .

# Pre-commit hooks (automatic checks before each commit)
pre-commit run --all-files  # Run manually on all files
```

The project uses pre-commit hooks to automatically check code quality before commits. After running `pre-commit install`, the following checks run automatically:

- **ruff**: Linting and formatting Python code
- **nb-clean**: Clean notebook outputs and metadata
- **nbqa**: Format and lint notebook cells
- **trailing-whitespace** and **end-of-file-fixer**: Keep files clean


## Documentation

Complete documentation is available as a Jupyter Book, including:

- **Tutorials**: Interactive notebooks demonstrating fempack workflows
- **Theory**: Mathematical foundations with all formulas and derivations
- **API Reference**: Auto-generated documentation from source code

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build the documentation
./build-docs.sh

# View in browser
open docs/_build/html/index.html  # macOS
# or
xdg-open docs/_build/html/index.html  # Linux
```

See `docs/README.md` for more details.

## License

MIT. Feel free to reuse. Not required, but I would be happy if you let me know that `fempack` was useful for you.
