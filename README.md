[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_-blue?logo=python&logoColor=white)](https://docs.python.org/3/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![Documentation][badge-docs]][link-docs]
[![PyPI][pypi-badge]][pypi-link]

# NetworkVI: Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Mechanistic Discovery

## Getting started

`NetworkVI` is a sparse deep generative model designed for the paired, vertical (shared cells across measurements), horizontal (shared features across datasets) or mosaic integration and interpretation of multimodal single-cell data. The model learns a rich, batch-corrected low-dimensional representation of bi- and trimodal single-cell count datasets, estimating the representation using normalized input data. Please refer to the [documentation](https://networkvi.readthedocs.io/en/latest/). We also provide [tutorials](https://networkvi.readthedocs.io/docs/build/html/tutorials/):
- [Paired integration and query-to-reference mapping](https://networkvi.readthedocs.io/docs/build/html/tutorials/paired_integration_and_query_mapping)
- [Mosaic integration](https://rtd-larnoldt.readthedocs.io/docs/build/html/tutorials/mosaic_integration)
- [Interpretability: Inference of GO importances and Gene-GO associations](https://rtd-larnoldt.readthedocs.io/docs/build/html/tutorials/go_analysis)
- [Interpretability: Infernce of GO term-specific covariate attention values](https://rtd-larnoldt.readthedocs.io/docs/build/html/tutorials/go_specific_covariate_attention)

## Installation

`NetworkVI` requires Python>3.9 on your system.

1. Install the latest release of `NetworkVI` from [PyPi](https://pypi.org/project/networkvi/):

```
pip install networkvi
```

2. Install the latest development version:

```
pip install git+https://github.com/LArnoldt/networkvi.git@main
```

Please also install the appropiate CUDA version of `torch`, `torch-scatter` and `torch-sparse` version. Here we give an example for CUDA 12.1:

```
pip install -U torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

## API

Please find the [API](https://networkvi.readthedocs.io/docs/build/html/api.html) here.

## Release notes

Please find the [release notes](https://networkvi.readthedocs.io/docs/build/html/release_notes.html) here.

## Contact

If you found a bug, please use the [issue tracker](https://github.com/LArnoldt/networkvi/issues). If you use `NetworkVI` in your research, please consider citing:

```
Arnoldt, L., Upmeier zu Belzen, J., Herrmann, L., Nguyen, K., Theis, F.J., Wild, B. , Eils, R., "Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Mechanistic Discovery", bioRxiv, June 2025.
```

## Reproducibility

Code and notebooks to reproduce the results and figues from the paper are available [here](https://github.com/LArnoldt/networkvi_reproducibility).

[badge-docs]: https://img.shields.io/readthedocs/networkvi
[link-docs]: https://networkvi.readthedocs.io/en/latest/
[pypi-badge]: https://img.shields.io/pypi/v/networkvi.svg
[pypi-link]: https://pypi.org/project/networkvi



