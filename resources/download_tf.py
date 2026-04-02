import os
import pandas as pd
from utils import (
    download_file,
    EnsemblMapper,
    create_gene_interaction_df,
    save_interaction_data,
    ensure_directory
)

def download_htftarget_database(output_dir: str = "data/transcription_factors",
                                tissues: list = ["blood", "bone marrow"]) -> None:
    """
    Download and process hTFtarget database.
    """
    ensure_directory(output_dir)

    url = "https://guolab.wchscu.cn/static/hTFtarget/file_download/tf-target-infomation.txt"
    txt_path = os.path.join(output_dir, "htftarget_tf-target-information.txt")

    download_file(url, txt_path, extract=False)

    htf = pd.read_csv(txt_path, delimiter="\t", header=None,
                      names=["TF", "Target", "Tissue"])

    print(f"Loaded {len(htf)} TF-target interactions")
    print(f"Unique tissues: {htf['Tissue'].unique()[:20]}...")  # Show first 20

    mapper = EnsemblMapper()
    symbol_mapping = mapper.get_gene_id_to_ensembl_mapping('hgnc_symbol')

    symbol_to_ensembl = {}
    for _, row in symbol_mapping.iterrows():
        symbol = row['HGNC symbol']
        ensembl = row['Gene stable ID']
        if pd.notna(symbol):
            symbol_to_ensembl[str(symbol)] = ensembl

    htf['TF_ensembl'] = htf['TF'].map(symbol_to_ensembl)
    htf['Target_ensembl'] = htf['Target'].map(symbol_to_ensembl)

    htf_mapped = htf[htf['TF_ensembl'].notna() & htf['Target_ensembl'].notna()].copy()

    print(f"Mapped {len(htf_mapped)} interactions to Ensembl IDs")

    all_pairs = list(zip(htf_mapped['TF_ensembl'], htf_mapped['Target_ensembl']))
    all_df = create_gene_interaction_df(all_pairs)
    save_interaction_data(all_df, os.path.join(output_dir, "htftarget_all.csv"))

    for tissue in tissues:
        tissue_data = htf_mapped[
            htf_mapped['Tissue'].str.lower().str.contains(tissue.lower(), na=False)
        ]

        if len(tissue_data) > 0:
            tissue_pairs = list(zip(tissue_data['TF_ensembl'], tissue_data['Target_ensembl']))
            tissue_df = create_gene_interaction_df(tissue_pairs)

            safe_tissue_name = tissue.replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"htftarget_{safe_tissue_name}.csv")
            save_interaction_data(tissue_df, output_path)
        else:
            print(f"Warning: No data found for tissue '{tissue}'")

def download_tflink_database(output_dir: str = "data/transcription_factors") -> None:
    """
    Download and process TFLink database.
    """
    ensure_directory(output_dir)

    url = "https://cdn.netbiol.org/tflink/download_files/TFLink_Homo_sapiens_interactions_All_simpleFormat_v1.0.tsv.gz"
    gz_path = os.path.join(output_dir, "TFLink_Homo_sapiens_interactions.tsv.gz")

    txt_path = download_file(url, gz_path, extract=True)

    tflink = pd.read_csv(txt_path, delimiter="\t")

    print(f"Loaded {len(tflink)} TF-target interactions from TFLink")

    mapper = EnsemblMapper()
    ncbi_mapping = mapper.get_gene_id_to_ensembl_mapping('entrezgene_id')

    ncbi_to_ensembl = {}
    for _, row in ncbi_mapping.iterrows():
        ncbi_id = str(int(row['NCBI gene (formerly Entrezgene) ID'])) if pd.notna(
            row['NCBI gene (formerly Entrezgene) ID']) else None
        ensembl = row['Gene stable ID']
        if ncbi_id:
            ncbi_to_ensembl[ncbi_id] = ensembl

    tflink['TF_ensembl'] = tflink['NCBI.GeneID.TF'].astype(str).map(ncbi_to_ensembl)
    tflink['Target_ensembl'] = tflink['NCBI.GeneID.Target'].astype(str).map(ncbi_to_ensembl)

    tflink_mapped = tflink[tflink['TF_ensembl'].notna() & tflink['Target_ensembl'].notna()].copy()

    print(f"Mapped {len(tflink_mapped)} interactions to Ensembl IDs")

    pairs = list(zip(tflink_mapped['TF_ensembl'], tflink_mapped['Target_ensembl']))
    tflink_df = create_gene_interaction_df(pairs)

    output_path = os.path.join(output_dir, "tflink_all.csv")
    save_interaction_data(tflink_df, output_path)

if __name__ == "__main__":
    print("Downloading hTFtarget database...")
    download_htftarget_database()

    print("\nDownloading TFLink database...")
    download_tflink_database()

    print("\nTranscription factor data download complete!")
