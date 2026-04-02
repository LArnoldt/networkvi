# Resources: Download and process biological prior knowledge

This directory contains scripts for downloading and processing gene interaction data and ontologies from multiple public databases. These scripts are **not part of the NetworkVI package** but generate input data for NetworkVI models.

## Overview

The preprocessing pipeline supports the following data sources:

1. Gene-Gene interactions:

- **Protein-Protein Interactions (PPI)**: BioGRID, STRING
- **Topologically Associating Domains (TADs)**: ENCODE Hi-C data (GM12878, K562, MCF-7, T47D)
- **Transcription Factors**: hTFtarget, TFLink
- **Gene Regulatory Networks**: GRAND (K562)
- **Gene Sets**: MSigDB collections (H, C1-C5, C8)

All outputs are standardized CSV files with the following format:

```csv
,gene1,gene2,combined_score
0,ENSG00000065559,ENSG00000128591,1.0
1,ENSG00000138347,ENSG00000077522,0.87
```

2. Ontologies:

- **Gene Ontology**: GO annotations (current and historical releases)
- **Pathways**: Reactome, Pathway Commons


## Data Sources

### 1. Protein-Protein Interactions (PPI)

**BioGRID and STRING databases**

```bash
python download_ppi.py
```

**Outputs:**
- `data/ppi/biogrid.csv`
- `data/ppi/string.csv`
- `data/ppi/string_filt.csv` (score > 250)

---

### 2. Topologically Associating Domains (TADs)

**ENCODE Hi-C data for GM12878, K562, MCF-7, T47D**

```bash
python process_tads.py
```

**Outputs:**
- `data/tads/encode/ENCFF*_*.csv` (individual files)
- `data/tads/encode/*_consensus.csv` (combined per cell line)
- `data/tads/encode/*_shifted_benchmark.csv` (controls)
- `data/tads/encode/*_gene_groups.npy` (gene groups)

---

### 3. Transcription Factor Targets

**hTFtarget and TFLink databases**

```bash
python download_tf.py
```

**Outputs:**
- `data/transcription_factors/htftarget_all.csv`
- `data/transcription_factors/htftarget_blood.csv`
- `data/transcription_factors/htftarget_bone_marrow.csv`
- `data/transcription_factors/tflink_all.csv`

---

### 4. Gene Regulatory Networks (GRAND)

**K562-specific PPI and GRN**

```bash
python download_grand.py
```

**Outputs:**
- `data/grand/grand_k562_ppi_all.csv`
- `data/grand/grand_k562_ppi_filt_0.5.csv`
- `data/grand/grand_k562_grn_all.csv`
- `data/grand/grand_k562_grn_filt_2.0.csv`

---

### 6. Gene Ontology (GO)

**EBI GOA annotations (current + historical)**

```bash
# Download current GO
python download_go.py --current

# Download historical releases (2014, 2019, 2022)
python download_go.py --historical

# Process GAF files to map to Ensembl IDs
python download_go.py --process

# All of the above
python download_go.py --all
```

**Outputs:**
- `data/go/go-basic_YYYY-MM-DD.obo`
- `data/go/goa_human_ensembl_gene_mapping_YYYY-MM-DD.gaf`
- `data/go/goa_human_isoform_ensembl_gene_mapping_YYYY-MM-DD.gaf`
- `data/go/goa_human_rna_ensembl_gene_mapping_YYYY-MM-DD.gaf`

**Generate shuffled controls:**

```bash
python shuffle_go_annotations.py --all
```

Creates global and level-shuffled GO annotations for null models.

---

### 7. Pathway Databases

**Reactome and Pathway Commons**

```bash
# Reactome only
python download_pathways.py --reactome

# Pathway Commons only
python download_pathways.py --pathway-commons

# Both
python download_pathways.py --all
```

**Outputs:**
- `data/pathways/reactome/reactome.obo`
- `data/pathways/reactome/reactome_human_ensembl_gene_mapping_YYYY-MM-DD.gaf`
- `data/pathways/pathway_commons/pathwaycommons.obo`
- `data/pathways/pathway_commons/pathwaycommons_human_ensembl_gene_mapping.gaf`

---

## Script Overview

| Script | Description | Data Sources |
|--------|-------------|--------------|
| `download_ppi.py` | Protein-protein interactions | BioGRID, STRING |
| `process_tads.py` | Topologically associating domains | ENCODE (4 cell lines) |
| `download_tf.py` | TF-target interactions | hTFtarget, TFLink |
| `download_grand.py` | Gene regulatory networks | GRAND K562 |
| `download_go.py` | Gene Ontology annotations | EBI GOA |
| `download_pathways.py` | Pathway databases | Reactome, Pathway Commons |
| `shuffle_go_annotations.py` | GO shuffling for controls | - |

## Common Parameters

### Customize Score Thresholds

Edit parameters in the scripts:

```python
# In download_ppi.py
download_string_database(score_threshold=250)  # Default: 250

# In download_grand.py
download_grand_ppi(score_threshold=0.5)   # Default: 0.5
download_grand_grn(score_threshold=2.0)   # Default: 2.0
```

### Filter by Tissue

```python
# In download_tf.py
download_htftarget_database(tissues=["blood", "bone marrow", "liver"])
```

## Output Directory Structure

```
data/
├── ppi/
├── tads/encode/
├── transcription_factors/
├── grand/
├── go/
└── pathways/
    ├── reactome/
    └── pathway_commons/
```

Please see the files `ENCFF041XLP_GM12878_GRCh38_loops_juicertools.csv` and `ensembl2go.gaf` as examples for the input format.

## Further Documentation

For complete documentation, see the [NetworkVI documentation](https://networkvi.readthedocs.io/).
