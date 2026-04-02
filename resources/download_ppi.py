import os
import pandas as pd
from utils import (
    download_file,
    EnsemblMapper,
    create_gene_interaction_df,
    filter_human_only,
    save_interaction_data,
    ensure_directory
)

def download_biogrid_database(output_dir: str = "data/ppi") -> None:
    """
    Download and process BIOGRID Database.
    """
    ensure_directory(output_dir)

    url = "https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-4.4.235/BIOGRID-ALL-4.4.235.tab3.zip"
    zip_path = os.path.join(output_dir, "BIOGRID-ALL-4.4.235.tab3.zip")

    extract_dir = download_file(url, zip_path, extract=True)

    biogrid_file = os.path.join(extract_dir, "BIOGRID-ALL-4.4.235.tab3.txt")
    biogrid = pd.read_csv(biogrid_file, delimiter="\t")

    biogrid = filter_human_only(biogrid, "Organism Name Interactor A")
    biogrid = filter_human_only(biogrid, "Organism Name Interactor B")

    mapper = EnsemblMapper()
    mapping = mapper.get_gene_id_to_ensembl_mapping('uniprotswissprot')

    biogrid = biogrid.merge(
        mapping.rename(columns={'UniProtKB/Swiss-Prot ID': 'SWISS-PROT Accessions Interactor A',
                                'Gene stable ID': 'Gene stable ID A'}),
        on='SWISS-PROT Accessions Interactor A',
        how='left'
    )
    biogrid = biogrid.merge(
        mapping.rename(columns={'UniProtKB/Swiss-Prot ID': 'SWISS-PROT Accessions Interactor B',
                                'Gene stable ID': 'Gene stable ID B'}),
        on='SWISS-PROT Accessions Interactor B',
        how='left'
    )

    biogrid = biogrid[~biogrid['Gene stable ID A'].isnull()]
    biogrid = biogrid[~biogrid['Gene stable ID B'].isnull()]

    gene_pairs = list(zip(biogrid['Gene stable ID A'], biogrid['Gene stable ID B']))
    biogrid_ppi = create_gene_interaction_df(gene_pairs)

    output_path = os.path.join(output_dir, "biogrid.csv")
    save_interaction_data(biogrid_ppi, output_path)

def download_string_database(output_dir: str = "data/ppi",
                             score_threshold: int = 250) -> None:
    """
    Download and process STRING Database.
    """
    ensure_directory(output_dir)

    url = "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz"
    gz_path = os.path.join(output_dir, "9606.protein.links.full.v12.0.txt.gz")

    txt_path = download_file(url, gz_path, extract=True)

    string = pd.read_csv(txt_path, delimiter=" ")

    string['protein1'] = string['protein1'].apply(lambda x: x.split(".")[1])
    string['protein2'] = string['protein2'].apply(lambda x: x.split(".")[1])

    mapper = EnsemblMapper()
    mapping = mapper.get_gene_id_to_ensembl_mapping('ensembl_peptide_id')

    string = string.merge(
        mapping.rename(columns={'Protein stable ID': 'protein1',
                                'Gene stable ID': 'Gene stable ID 1'}),
        on='protein1',
        how='left'
    )
    string = string.merge(
        mapping.rename(columns={'Protein stable ID': 'protein2',
                                'Gene stable ID': 'Gene stable ID 2'}),
        on='protein2',
        how='left'
    )

    string = string[~string['Gene stable ID 1'].isnull()]
    string = string[~string['Gene stable ID 2'].isnull()]

    gene_pairs = list(zip(string['Gene stable ID 1'], string['Gene stable ID 2']))
    scores = list(string['combined_score'])
    string_ppi = create_gene_interaction_df(gene_pairs, scores)

    output_path = os.path.join(output_dir, "string.csv")
    save_interaction_data(string_ppi, output_path)

    string_ppi_filt = string_ppi[string_ppi['combined_score'] > score_threshold].copy()
    output_path_filt = os.path.join(output_dir, "string_filt.csv")
    save_interaction_data(string_ppi_filt, output_path_filt)

if __name__ == "__main__":
    print("Downloading BioGRID database...")
    download_biogrid_database()

    print("\nDownloading STRING database...")
    download_string_database()

    print("\nPPI data download complete!")
