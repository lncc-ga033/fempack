# fempack Documentation

Bem-vindo à documentação do **fempack**, um pacote Python educacional para o método de elementos finitos (FEM).

## Sobre o fempack

O `fempack` é um pacote desenvolvido para o curso **GA-033 - Método de Elementos Finitos: Implementação Computacional** do [Programa de Pós-Graduação em Modelagem Computacional do LNCC](https://posgrad.lncc.br/pt-br/) (Laboratório Nacional de Computação Científica). O objetivo é fornecer uma implementação clara e didática do método de elementos finitos para problemas elípticos em 1D e 2D.

Esta abordagem é fortemente inspirada no curso [Finite elements: analysis and implementation](https://finite-element.github.io/) do Departamento de Matemática do Imperial College London, aos quais deixo meus agradecimentos.

## Características principais

- Elementos Lagrangeanos de baixa ordem (P1, P2, Q1)
- Problemas de Poisson em 1D e 2D
- Malhas estruturadas e não estruturadas
- Condições de contorno de Dirichlet
- Verificação por método de soluções manufaturadas (MMS)
- Código bem documentado com foco educacional

## Conteúdo

```{tableofcontents}
```

## Instalação

Para instalar o pacote em modo de desenvolvimento:

```bash
git clone https://github.com/lncc-ga033/fempack.git
cd fempack
pip install -e ".[dev]"
```

## Dependências

O mínimo possível, mas há outras configurações mais extensas, permitindo o uso de Jupyter Notebooks, por exemplo.

- NumPy
- SciPy
- Matplotlib (para visualização)

## Citação

Se você usar este pacote em trabalhos acadêmicos, por favor cite:

```bibtex
@misc{fempack2025,
  author = {Diego T. Volpatto},
  title = {fempack: Educational Finite Element Method Package},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/lncc-ga033/fempack}
}
```

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.
