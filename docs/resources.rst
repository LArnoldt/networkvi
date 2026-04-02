Resources: Download and process biological prior knowledge
=====================================

NetworkVI leverages prior biological knowledge about gene-gene interactions and ontologies from multiple public databases to construct biologically informed network priors. This section documents the preprocessing pipeline for downloading and processing gene interaction data from various sources.

All preprocessing scripts are located in the ``resources/`` directory and are **not** part of the main NetworkVI package. These scripts generate standardized gene-gene interaction tables with Ensembl gene IDs and ontologies in the obo format that can be used as input for NetworkVI models.

Overview
--------

The preprocessing pipeline supports the following data sources:

1. Gene-Gene interactions:

- **Protein-Protein Interactions (PPI)**: BioGRID, STRING
- **Topologically Associating Domains (TADs)**: ENCODE Hi-C data (GM12878, K562, MCF-7, T47D)
- **Transcription Factors**: hTFtarget, TFLink
- **Gene Regulatory Networks**: GRAND (K562)

All outputs are standardized CSV files with the following format:

.. code-block:: none

   ,gene1,gene2,combined_score
   0,ENSG00000065559,ENSG00000128591,1.0
   1,ENSG00000138347,ENSG00000077522,1.0
   2,ENSG00000115170,ENSG00000168522,0.87

2. Ontologies:

- **Gene Ontology**: GO annotations (current and historical releases)
- **Pathways**: Reactome, Pathway Commons


Data Sources
------------

Protein-Protein Interactions (PPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**BioGRID**

The BioGRID database provides experimentally validated protein-protein interactions.

- **Database**: https://thebiogrid.org/
- **Script**: ``preprocessing/download_ppi.py``

**STRING**

STRING provides both experimental and predicted protein-protein interactions with confidence scores.

- **Database**: https://string-db.org/
- **Script**: ``preprocessing/download_ppi.py``

Usage:

.. code-block:: bash

   cd preprocessing
   python download_ppi.py

Outputs:

- ``data/ppi/biogrid.csv`` - BioGRID interactions
- ``data/ppi/string.csv`` - STRING interactions (all)
- ``data/ppi/string_filt.csv`` - STRING interactions (score > 250)

Topologically Associating Domains (TADs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TADs are genomic regions where DNA sequences physically interact more frequently. Genes within the same TAD are more likely to be co-regulated.

- **Database**: ENCODE (https://www.encodeproject.org/)
- **Script**: ``preprocessing/process_tads.py``

Supported cell lines:

- GM12878 (lymphoblastoid)
- K562 (erythroleukemic)
- MCF-7 (breast cancer)
- T47D (breast cancer)

Usage:

.. code-block:: bash

   cd preprocessing
   python process_tads.py

Outputs (per cell line):

- Individual TAD file interactions (CSV)
- Consensus interactions combining all files (CSV)
- Shifted benchmark controls (CSV)
- Gene groups per TAD (NPY arrays)

Transcription Factor Targets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**hTFtarget**

Database of transcription factor-target gene relationships with tissue specificity.

- **Database**: https://guolab.wchscu.cn/hTFtarget/
- **Script**: ``preprocessing/download_tf.py``

**TFLink**

Comprehensive TF-target interaction database integrating multiple sources.

- **Database**: https://tflink.net/
- **Script**: ``preprocessing/download_tf.py``

Usage:

.. code-block:: bash

   cd preprocessing
   python download_tf.py

Outputs:

- ``data/transcription_factors/htftarget_all.csv`` - All hTFtarget interactions
- ``data/transcription_factors/htftarget_blood.csv`` - Blood tissue specific
- ``data/transcription_factors/htftarget_bone_marrow.csv`` - Bone marrow specific
- ``data/transcription_factors/tflink_all.csv`` - All TFLink interactions

Gene Regulatory Networks (GRAND)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GRAND provides cell-type-specific gene regulatory networks for K562 cells, including both PPI and regulatory interactions.

- **Database**: https://granddb.s3.us-east-2.amazonaws.com/
- **Script**: ``preprocessing/download_grand.py``

Usage:

.. code-block:: bash

   cd preprocessing
   python download_grand.py

Outputs:

- ``data/grand/grand_k562_ppi_all.csv`` - All PPI interactions
- ``data/grand/grand_k562_ppi_filt_0.5.csv`` - Filtered PPI (score > 0.5)
- ``data/grand/grand_k562_grn_all.csv`` - All regulatory interactions
- ``data/grand/grand_k562_grn_filt_2.0.csv`` - Filtered GRN (|score| > 2.0)

Gene Ontology (GO)
^^^^^^^^^^^^^^^^^^

Gene Ontology provides standardized annotations for gene functions, processes, and cellular locations.

- **Database**: http://geneontology.org/, https://www.ebi.ac.uk/GOA
- **Script**: ``preprocessing/download_go.py``

The pipeline downloads both current and historical GO releases (2014, 2019, 2022) for temporal analyses.

Usage:

.. code-block:: bash

   cd preprocessing

   # Download current GO release
   python download_go.py --current

   # Download historical releases
   python download_go.py --historical

   # Process GAF files (map to Ensembl IDs)
   python download_go.py --process

   # All of the above
   python download_go.py --all

Outputs:

- ``data/go/go-basic_YYYY-MM-DD.obo`` - GO ontology files
- ``data/go/goa_human_ensembl_gene_mapping_YYYY-MM-DD.gaf`` - Protein annotations
- ``data/go/goa_human_isoform_ensembl_gene_mapping_YYYY-MM-DD.gaf`` - Isoform annotations
- ``data/go/goa_human_rna_ensembl_gene_mapping_YYYY-MM-DD.gaf`` - RNA annotations

**Shuffled Controls:**

Generate randomized GO annotations for null model testing:

.. code-block:: bash

   cd preprocessing
   python shuffle_go_annotations.py --all

This creates global and level-based shuffled versions of GO annotations.

Pathway Databases
^^^^^^^^^^^^^^^^^

**Reactome**

Reactome provides manually curated pathway annotations.

- **Database**: https://reactome.org/
- **Script**: ``preprocessing/download_pathways.py``

**Pathway Commons**

Pathway Commons integrates pathway data from multiple databases.

- **Database**: https://www.pathwaycommons.org/
- **Script**: ``preprocessing/download_pathways.py``

Usage:

.. code-block:: bash

   cd preprocessing

   # Download Reactome
   python download_pathways.py --reactome

   # Download Pathway Commons
   python download_pathways.py --pathway-commons

   # Download both
   python download_pathways.py --all

Outputs:

- ``data/pathways/reactome/reactome.obo`` - Reactome ontology
- ``data/pathways/reactome/reactome_human_ensembl_gene_mapping_YYYY-MM-DD.gaf`` - Gene-pathway associations
- ``data/pathways/pathway_commons/pathwaycommons.obo`` - Pathway Commons ontology
- ``data/pathways/pathway_commons/pathwaycommons_human_ensembl_gene_mapping.gaf`` - Gene-pathway associations

Advanced Usage
--------------

Customizing Parameters
^^^^^^^^^^^^^^^^^^^^^^

Most scripts accept parameters for customization:

**STRING score threshold:**

.. code-block:: python

   # In download_ppi.py
   download_string_database(score_threshold=250)  # Default: 250

**GRAND score thresholds:**

.. code-block:: python

   # In download_grand.py
   download_grand_ppi(score_threshold=0.5)   # Default: 0.5
   download_grand_grn(score_threshold=2.0)   # Default: 2.0

**Tissue-specific TF data:**

.. code-block:: python

   # In download_tf.py
   download_htftarget_database(tissues=["blood", "bone marrow", "liver"])


File Structure
--------------

After running the preprocessing scripts, your data directory will have the following structure:

.. code-block:: none

   data/
   ├── ppi/                      # Protein-protein interactions
   ├── tads/encode/              # Topologically associating domains
   ├── transcription_factors/    # TF-target interactions
   ├── grand/                    # K562 gene regulatory networks
   ├── go/                       # Gene Ontology annotations
   └── pathways/                 # Pathway databases
       ├── reactome/
       └── pathway_commons/
