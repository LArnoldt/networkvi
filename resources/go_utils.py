import os
import pandas as pd
from itertools import islice
from datetime import date
from typing import List, Tuple
from utils import EnsemblMapper, ensure_directory, get_rnacentral_mapping

def get_uniprotswissprot_mapping() -> pd.DataFrame:
    """Get mapping from UniProt/Swiss-Prot IDs to Ensembl gene IDs."""
    mapper = EnsemblMapper()
    mapping = mapper.get_gene_id_to_ensembl_mapping('uniprotswissprot')
    mapping = mapping.rename(columns={'UniProtKB/Swiss-Prot ID': 'external_identifier'})
    return mapping

def read_gaf_file(gaf_path: str, skiprows: int = None) -> Tuple[List[str], pd.DataFrame]:
    """
    Read a GAF (Gene Association File) format file.
    """
    if skiprows is None:
        skiprows = 0
        with open(gaf_path, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    skiprows += 1
                else:
                    break

    with open(gaf_path, 'r') as f:
        header = list(islice(f, skiprows))

    columns = [
        "DB", "DB_Object_ID", "DB_Object_Symbol", "Qualifier", "GO_ID",
        "DB_Reference", "Evidence_Code", "With_or_From", "Aspect",
        "DB_Object_Name", "DB_Object_Synonym", "DB_Object_Type",
        "Taxon", "Date", "Assigned_by", "Annotation_Extension",
        "Annotation_Properties"
    ]

    df = pd.read_csv(gaf_path, delimiter="\t", header=None,
                     skiprows=skiprows, comment='!')

    if len(df.columns) == len(columns):
        df.columns = columns
    elif len(df.columns) < len(columns):
        df.columns = columns[:len(df.columns)]

    return header, df

def write_gaf_file(output_path: str, header: List[str], df: pd.DataFrame) -> None:
    """
    Write a GAF format file.
    """
    ensure_directory(os.path.dirname(output_path))

    with open(output_path, 'w') as f:
        for line in header:
            f.write(line)

        df.to_csv(f, sep='\t', header=False, index=False)

    print(f"Wrote {len(df)} annotations to {output_path}")

def create_gaf_header(source: str, description: str = None) -> List[str]:
    """
    Create standard GAF 2.2 header.
    """
    header = [
        "!gaf-version: 2.2\n",
        "!\n",
    ]

    if description:
        header.append(f"!{description}\n")
        header.append("!\n")

    header.extend([
        f"!date-generated: {date.today().isoformat()}\n",
        f"!generated-by: {source}\n",
        "!\n",
    ])

    return header

def map_external_ids_to_ensembl(df: pd.DataFrame,
                                id_column: str,
                                mapping: pd.DataFrame) -> pd.DataFrame:
    """
    Map external IDs to Ensembl gene IDs in a GAF DataFrame.
    """
    df_mapped = df.merge(
        mapping.set_index('external_identifier'),
        left_on=id_column,
        right_index=True,
        how='inner'
    )

    df_mapped[id_column] = df_mapped['Gene stable ID']
    df_mapped = df_mapped.drop(columns=['Gene stable ID'])

    df_mapped = df_mapped[~df_mapped[id_column].isnull()]

    return df_mapped


def write_obo_header(f, ontology_name: str) -> None:
    """
    Write standard OBO format header.
    """
    f.write("format-version: 1.2\n")
    f.write(f"date: {date.today().isoformat()}\n")
    f.write(f"ontology: {ontology_name}\n\n")

def create_artificial_root_term(f, root_id: str, root_name: str,
                                namespace: str) -> None:
    """
    Create an artificial root term in OBO format.
    """
    f.write("[Term]\n")
    f.write(f"id: {root_id}\n")
    f.write(f"name: {root_name}\n")
    f.write(f"namespace: {namespace}\n")
    f.write(f'def: "Artificial root node for all {root_name}." []\n')
    f.write("\n")
