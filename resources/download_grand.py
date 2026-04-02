import os
import pandas as pd
import numpy as np
from utils import (
    download_file,
    EnsemblMapper,
    create_gene_interaction_df,
    save_interaction_data,
    ensure_directory
)
from tqdm import tqdm

def download_grand_ppi(output_dir: str = "data/grand",
                       score_threshold: float = 0.5) -> None:
    """
    Download and process GRAND database PPI data for K562.
    """
    ensure_directory(output_dir)

    url = "https://granddb.s3.us-east-2.amazonaws.com/cells/ppi/EGRET_ppi.txt"
    txt_path = os.path.join(output_dir, "EGRET_ppi.txt")

    download_file(url, txt_path, extract=False)

    grand_ppi = pd.read_csv(txt_path, delimiter="\t", header=None,
                            names=["Gene1", "Gene2", "Score"])

    print(f"Loaded {len(grand_ppi)} PPI interactions from GRAND")

    mapper = EnsemblMapper()
    symbol_mapping = mapper.get_gene_id_to_ensembl_mapping('hgnc_symbol')

    symbol_to_ensembl = {}
    for _, row in symbol_mapping.iterrows():
        symbol = row['HGNC symbol']
        ensembl = row['Gene stable ID']
        if pd.notna(symbol):
            symbol_to_ensembl[str(symbol)] = ensembl

    grand_ppi['Gene1_ensembl'] = grand_ppi['Gene1'].map(symbol_to_ensembl)
    grand_ppi['Gene2_ensembl'] = grand_ppi['Gene2'].map(symbol_to_ensembl)

    grand_ppi_mapped = grand_ppi[
        grand_ppi['Gene1_ensembl'].notna() & grand_ppi['Gene2_ensembl'].notna()
        ].copy()

    print(f"Mapped {len(grand_ppi_mapped)} interactions to Ensembl IDs")

    pairs = list(zip(grand_ppi_mapped['Gene1_ensembl'], grand_ppi_mapped['Gene2_ensembl']))
    scores = list(grand_ppi_mapped['Score'])
    grand_ppi_df = create_gene_interaction_df(pairs, scores)

    output_path = os.path.join(output_dir, "grand_k562_ppi_all.csv")
    save_interaction_data(grand_ppi_df, output_path)

    grand_ppi_filt = grand_ppi_df[grand_ppi_df['combined_score'] > score_threshold].copy()
    output_path_filt = os.path.join(output_dir, f"grand_k562_ppi_filt_{score_threshold}.csv")
    save_interaction_data(grand_ppi_filt, output_path_filt)

def download_grand_grn(output_dir: str = "data/grand",
                       score_threshold: float = 2.0) -> None:
    """
    Download and process GRAND database gene regulatory network for K562.

    The GRN is stored as an adjacency matrix where rows are regulators (TFs)
    and columns are target genes. Values represent regulatory weights.
    """
    ensure_directory(output_dir)

    url = "https://granddb.s3.us-east-2.amazonaws.com/data/EGRET_K562.csv"
    csv_path = os.path.join(output_dir, "EGRET_K562.csv")

    download_file(url, csv_path, extract=False)

    grn_matrix = pd.read_csv(csv_path, index_col=0)

    print(f"Loaded GRN adjacency matrix: {grn_matrix.shape[0]} regulators x {grn_matrix.shape[1]} targets")

    regulators = grn_matrix.index.tolist()
    targets = grn_matrix.columns.tolist()

    print(f"Sample regulators: {regulators[:5]}")
    print(f"Sample targets: {targets[:5]}")

    mapper = EnsemblMapper()
    symbol_mapping = mapper.get_gene_id_to_ensembl_mapping('hgnc_symbol')

    symbol_to_ensembl = {}
    for _, row in symbol_mapping.iterrows():
        symbol = row['HGNC symbol']
        ensembl = row['Gene stable ID']
        if pd.notna(symbol):
            symbol_to_ensembl[str(symbol)] = ensembl

    edges = []
    scores = []

    for regulator in regulators:
        regulator_ensembl = symbol_to_ensembl.get(regulator)
        if not regulator_ensembl:
            continue

        for target in targets:
            score = grn_matrix.loc[regulator, target]

            if abs(score) > score_threshold:
                edges.append((regulator_ensembl, target))
                scores.append(score)

    print(f"Generated {len(edges)} regulatory interactions (|score| > {score_threshold})")

    grn_df = create_gene_interaction_df(edges, scores)

    output_path = os.path.join(output_dir, f"grand_k562_grn_filt_{score_threshold}.csv")
    save_interaction_data(grn_df, output_path)

    edges_all = []
    scores_all = []

    for regulator in tqdm(regulators):
        regulator_ensembl = symbol_to_ensembl.get(regulator)
        if not regulator_ensembl:
            continue

        for target in tqdm(targets):
            score = grn_matrix.loc[regulator, target]
            edges_all.append((regulator_ensembl, target))
            scores_all.append(score)

    grn_df_all = create_gene_interaction_df(edges_all, scores_all)
    output_path_all = os.path.join(output_dir, "grand_k562_grn_all.csv")
    save_interaction_data(grn_df_all, output_path_all)

if __name__ == "__main__":
    print("Downloading GRAND PPI data...")
    download_grand_ppi()

    print("\nDownloading GRAND GRN data...")
    download_grand_grn()

    print("\nGRAND data download complete!")
