[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11_-blue?logo=python&logoColor=white)](https://docs.python.org/3/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![Documentation][badge-docs]][link-docs]
[![PyPI][pypi-badge]][pypi-link]
[![DOI](https://zenodo.org/badge/992124546.svg)](https://doi.org/10.5281/zenodo.21227860)

# NetworkVI: Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Discovery

## Getting started

`NetworkVI` is a sparse deep generative model designed for the paired, vertical (shared cells across measurements), horizontal (shared features across datasets) or mosaic integration and interpretation of multimodal single-cell data. The model learns a rich, batch-corrected low-dimensional representation of bi- and trimodal single-cell count datasets, estimating the representation using normalized input data. Please refer to the [documentation](https://networkvi.readthedocs.io/en/latest/). We also provide [tutorials](https://networkvi.readthedocs.io/en/latest/tutorials.html):
- [Paired integration and query-to-reference mapping](https://networkvi.readthedocs.io/en/latest/notebooks/paired_integration_and_query_mapping.html)
- [Mosaic integration](https://networkvi.readthedocs.io/en/latest/notebooks/mosaic_integration.html)
- [Interpretability: Inference of GO importances and Gene-GO associations](https://networkvi.readthedocs.io/en/latest/notebooks/go_analysis.html)
- [Interpretability: Infernce of GO term-specific covariate attention values](https://networkvi.readthedocs.io/en/latest/notebooks/go_specific_covariate_attention.html)

## Installation

`NetworkVI` supports both standard pip installation and Pixi-based reproducible environments.
We recommend Pixi for most users, as it automatically manages Python, CUDA, and PyTorch versions.

### Recommended: Installation using Pixi (reproducible, CUDA-enabled)

[Pixi](https://pixi.prefix.dev/latest/) is a modern environment manager that combines Conda and pip, making it easy to install GPU-enabled scientific software reproducibly.

1. Install Pixi

Follow the instructions at: [https://pixi.sh](https://pixi.sh)

2. Clone the repository

```
git clone https://github.com/LArnoldt/networkvi.git
cd networkvi
```

3. Create and activate the environment

```
pixi install
pixi shell
```

### Alternative: Installation using pip

If you prefer a standard pip-based installation (CPU or manually managed GPU):

1. Install the latest release of `NetworkVI` from [PyPi](https://pypi.org/project/networkvi/):

```
pip install networkvi
```

2. (Optional, GPU) Install PyTorch and PyG dependencies manually

For CUDA 12.1:

```
pip install -U torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -U torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

Other CUDA versions are available at:

- [https://pytorch.org](https://pytorch.org)
- [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

### Optional dependencies

Additional functionality can be installed via extras:

```
pip install "networkvi[tutorials]"
pip install "networkvi[docs]"
pip install "networkvi[all]"
```

When using Pixi, extras can be enabled by adjusting `pixi.toml`.

## API

Please find the [API](https://networkvi.readthedocs.io/en/latest/api.html) here.

## Release notes

Please find the [release notes](http://networkvi.readthedocs.io/en/latest/changelog.html) here.

## Contact

If you found a bug, please use the [issue tracker](https://github.com/LArnoldt/networkvi/issues). If you use `NetworkVI` in your research, please consider citing the [preprint](https://www.biorxiv.org/content/10.1101/2025.06.10.657924v2):

```
Arnoldt, L., Upmeier zu Belzen, J., Herrmann, L., Nguyen, K., Theis, F.J., Wild, B. , Eils, R., "Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Mechanistic Discovery", bioRxiv, June 2025.
```

If you are interested in genetic risk modeling with a similar architecture, check out our `OGM` [repository](https://github.com/juzb/ogm) and [preprint](https://www.medrxiv.org/content/10.64898/2026.07.28.26359187v2).

## Reproducibility

Code and notebooks to reproduce the results and figues from the paper are available [here](https://github.com/LArnoldt/networkvi_reproducibility).

[badge-docs]: https://img.shields.io/readthedocs/networkvi
[link-docs]: https://networkvi.readthedocs.io/en/latest/
[pypi-badge]: https://img.shields.io/pypi/v/networkvi.svg
[pypi-link]: https://pypi.org/project/networkvi
