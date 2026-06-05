Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Discovery
=============================================================================================

**NetworkVI** is a sparse deep generative model designed for the integration and interpretation of multimodal single-cell data.
NetworkVI models gene-gene interactions inferred from topological associated domains and utilizes the structure of gene ontology (GO) to aggregate gene embeddings to cell embeddings, enhancing the interpretability at the gene and GO-term level.
NetworkVI can be used for modality imputation, reference-to-query mapping and aids in identifying modality- and cell type-specific signatures via interpretability.
NetworkVI will support researchers in interpreting cellular disease mechanisms, guiding biomarker discovery, and ultimately aiding the development of targeted therapies in large-scale single-cell multimodal atlases.

Check out the :doc:`api`, :doc:`tutorials`, and :doc:`resources` section for further information.

Reliability guidance for interpretability outputs
-------------------------------------------------

Before drawing biological conclusions from GO importance, TAD importance, or covariate attention scores, consult the :doc:`interpretability_guidance` page. It provides a near-formal decision framework covering when results are reliable, known failure modes, sensitivity to annotation incompleteness, and a quick-reference checklist.

Citation
--------

If you use NetworkVI, please consider citing:

Arnoldt, L., Upmeier zu Belzen, J., Herrmann, L., Nguyen, K., Theis, F.J., Wild, B. , Eils, R., "Biologically Guided Variational Inference for Interpretable Multimodal Single-Cell Integration and Mechanistic Discovery", bioRxiv, June 2025.

Installation
------------

1. Install the latest release of ``NetworkVI`` from `PyPi <https://pypi.org/project/networkvi/>`_:

``pip install networkvi``

2. Install the latest development version:

``pip install git+https://github.com/LArnoldt/networkvi.git@main``

Contents
--------

.. toctree::

   api
   tutorials
   interpretability_guidance
   resources
   changelog
   references

