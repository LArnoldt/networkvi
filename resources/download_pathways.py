import os
import argparse
import gzip
import shutil
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date
from collections import defaultdict
from tqdm import tqdm
import xml.etree.ElementTree as ET

from utils import download_file, ensure_directory, EnsemblMapper
from go_utils import create_gaf_header, write_obo_header, create_artificial_root_term

def download_reactome(output_dir: str = "data/pathways/reactome") -> None:
    """
    Download and process Reactome pathway database.

    Database: https://reactome.org/

    Args:
        output_dir: Output directory
    """
    ensure_directory(output_dir)

    print("\n" + "=" * 70)
    print("DOWNLOADING REACTOME")
    print("=" * 70)

    pathways_url = "https://reactome.org/download/current/ReactomePathways.txt"
    relations_url = "https://reactome.org/download/current/ReactomePathwaysRelation.txt"
    ensembl_url = "https://reactome.org/download/current/Ensembl2Reactome_All_Levels.txt"

    pathways_file = os.path.join(output_dir, "ReactomePathways.txt")
    relations_file = os.path.join(output_dir, "ReactomePathwaysRelation.txt")
    ensembl_file = os.path.join(output_dir, "Ensembl2Reactome_All_Levels.txt")

    download_file(pathways_url, pathways_file, extract=False)
    download_file(relations_url, relations_file, extract=False)
    download_file(ensembl_url, ensembl_file, extract=False)

    print("\nProcessing Reactome data...")

    pw = pd.read_csv(
        pathways_file,
        sep="\t",
        header=None,
        names=["id", "name", "species"]
    )

    pw = pw[pw["species"] == "Homo sapiens"]
    pw = pw.set_index("id")

    print(f"Found {len(pw)} human pathways")

    rel = pd.read_csv(
        relations_file,
        sep="\t",
        header=None,
        names=["parent", "child"]
    )

    rel = rel[rel["parent"].isin(pw.index) & rel["child"].isin(pw.index)]

    parent_map = {}
    for _, r in rel.iterrows():
        parent_map.setdefault(r["child"], []).append(r["parent"])

    all_pathways = set(pw.index)
    pathways_with_parents = set(parent_map.keys())
    top_level_pathways = all_pathways - pathways_with_parents

    print(f"Found {len(top_level_pathways)} top-level pathways")

    use_artificial_root = len(top_level_pathways) > 1
    artificial_root_id = "0000000"

    if use_artificial_root:
        print(f"Creating artificial root node: REACTOME:{artificial_root_id}")
        for top_pathway in top_level_pathways:
            parent_map.setdefault(top_pathway, []).append(artificial_root_id)

    obo_file = os.path.join(output_dir, "reactome.obo")
    print(f"\nWriting {obo_file}...")

    with open(obo_file, 'w') as f:
        write_obo_header(f, "reactome_pathway")

        if use_artificial_root:
            create_artificial_root_term(
                f,
                f"REACTOME:{artificial_root_id}",
                "Reactome Pathways Root",
                "reactome"
            )

        for pid, row in pw.iterrows():
            f.write("[Term]\n")
            f.write(f"id: REACTOME:{pid}\n")
            f.write(f"name: {row['name']}\n")
            f.write("namespace: reactome\n")
            f.write('def: "Reactome pathway." []\n')
            f.write(f"xref: Reactome:{pid}\n")

            for parent in parent_map.get(pid, []):
                f.write(f"is_a: REACTOME:{parent}\n")

            f.write("\n")

    print(f"Wrote {len(pw)} pathway terms")

    ensembl = pd.read_csv(
        ensembl_file,
        sep="\t",
        header=None,
        names=["ensembl_id", "reactome_id", "url", "name", "evidence", "species"]
    )

    ensembl = ensembl[
        (ensembl["species"] == "Homo sapiens") &
        (ensembl["reactome_id"].isin(pw.index))
        ]

    print(f"Found {len(ensembl)} gene-pathway associations")

    today = date.today().isoformat()
    gaf_file = os.path.join(output_dir, f"reactome_human_ensembl_gene_mapping_{today}.gaf")

    print(f"\nWriting {gaf_file}...")

    with open(gaf_file, 'w') as f:
        header = create_gaf_header(
            "Reactome",
            "Gene associations for Reactome pathways mapped to Ensembl gene IDs."
        )
        for line in header:
            f.write(line)

        for _, row in tqdm(ensembl.iterrows(), total=len(ensembl), desc="Writing GAF"):
            ensembl_id = row["ensembl_id"]
            reactome_id = row["reactome_id"]
            pathway_name = pw.loc[reactome_id, "name"]
            evidence = row["evidence"]

            fields = [
                "Reactome",
                ensembl_id,
                ensembl_id,
                "",
                f"REACTOME:{reactome_id}",
                "REACTOME:pathway",
                evidence,
                "",
                "P",
                pathway_name,
                "",
                "protein",
                "taxon:9606",
                today.replace("-", ""),
                "Reactome",
                str(np.nan),
                str(np.nan),
            ]

            f.write("\t".join(fields) + "\n")

    print(f"Wrote {len(ensembl)} annotations")

def download_pathway_commons(output_dir: str = "data/pathways/pathway_commons") -> None:
    """
    Download and process Pathway Commons database.

    Database: https://www.pathwaycommons.org/

    Args:
        output_dir: Output directory
    """
    ensure_directory(output_dir)

    print("\n" + "=" * 70)
    print("DOWNLOADING PATHWAY COMMONS")
    print("=" * 70)

    biopax_url = "https://download.baderlab.org/PathwayCommons/PC2/v14/pc-biopax.owl.gz"
    gmt_url = "https://download.baderlab.org/PathwayCommons/PC2/v14/pc-hgnc.gmt.gz"

    biopax_gz = os.path.join(output_dir, "pc-biopax.owl.gz")
    biopax_file = os.path.join(output_dir, "pc-biopax.owl")
    gmt_gz = os.path.join(output_dir, "pc-hgnc.gmt.gz")

    if not os.path.exists(biopax_file):
        download_file(biopax_url, biopax_gz, extract=True)
        """
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with gzip.open(GZ_FILE, "rb") as f_in:
            with open(OWL_FILE, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        """

    download_file(gmt_url, gmt_gz, extract=False)

    print("\nParsing BioPAX file...")
    pathways, pathway_parents = parse_biopax(biopax_file)

    print("\nFetching HGNC to Ensembl mapping...")
    mapper = EnsemblMapper()
    mapping_df = mapper.get_gene_id_to_ensembl_mapping('hgnc_symbol')

    hgnc_to_ensembl = defaultdict(list)
    for _, row in mapping_df.iterrows():
        hgnc = row['HGNC symbol']
        ensembl = row['Gene stable ID']
        if pd.notna(hgnc) and hgnc != '':
            hgnc_to_ensembl[hgnc].append(ensembl)

    print(f"Retrieved {len(hgnc_to_ensembl)} HGNC symbols")

    print("\nParsing GMT file...")
    pathway_genes = parse_gmt_file(gmt_gz)

    pathway_genes_matched = match_pathways_to_genes(pathways, pathway_genes)

    obo_file = os.path.join(output_dir, "pathwaycommons.obo")
    write_pathway_commons_obo(pathways, pathway_parents, obo_file)

    gaf_file = os.path.join(output_dir, "pathwaycommons_human_ensembl_gene_mapping.gaf")
    write_pathway_commons_gaf(pathways, pathway_genes_matched, hgnc_to_ensembl, gaf_file)

def parse_biopax(biopax_file: str) -> tuple:
    """Parse BioPAX OWL file to extract pathways and hierarchy."""
    BP_NS = "{http://www.biopax.org/release/biopax-level3.owl#}"
    RDF_NS = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"

    pathways = {}
    pathway_components = defaultdict(list)

    print("Pass 1: Identifying pathways...")
    context = ET.iterparse(biopax_file, events=('start', 'end'))

    current_id = None
    current_name = None
    current_components = []

    for event, elem in tqdm(context, desc="Parsing BioPAX"):
        tag = elem.tag.replace(BP_NS, '')

        if event == 'start':
            if tag == 'Pathway':
                current_id = elem.get(f'{RDF_NS}about')
                current_name = None
                current_components = []

        elif event == 'end':
            if tag == 'Pathway' and current_id:
                if current_name:
                    pathways[current_id] = current_name
                else:
                    pathways[current_id] = current_id.split('/')[-1]

                pathway_components[current_id] = current_components.copy()
                current_id = None
                current_name = None
                current_components = []

            elif tag == 'displayName' and current_id:
                if elem.text:
                    current_name = elem.text

            elif tag == 'pathwayComponent' and current_id:
                resource = elem.get(f'{RDF_NS}resource')
                if resource:
                    current_components.append(resource)

            elem.clear()

    print(f"Found {len(pathways)} pathways")

    # Build hierarchy
    print("Pass 2: Building hierarchy...")
    pathway_parents = defaultdict(set)

    for parent_uri, components in tqdm(pathway_components.items(), desc="Building hierarchy"):
        for component_uri in components:
            if component_uri in pathways:
                pathway_parents[component_uri].add(parent_uri)

    print(f"Found {len(pathway_parents)} parent-child relationships")

    return pathways, pathway_parents

def parse_gmt_file(gmt_path: str) -> dict:
    """Parse GMT file to extract pathway gene sets."""
    pathway_genes = defaultdict(set)

    with gzip.open(gmt_path, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue

            pathway_id = parts[0]
            genes = parts[2:]
            pathway_genes[pathway_id] = set(genes)

    print(f"Found {len(pathway_genes)} pathways with gene mappings")
    return pathway_genes

def match_pathways_to_genes(pathways: dict, pathway_genes: dict) -> dict:
    """Match pathway URIs to gene sets."""
    matched = {}

    pathway_by_suffix = {}
    for uri, name in pathways.items():
        suffix = uri.split('/')[-1]
        pathway_by_suffix[suffix] = uri

    for gmt_id, genes in pathway_genes.items():
        if gmt_id in pathway_by_suffix:
            uri = pathway_by_suffix[gmt_id]
            matched[uri] = genes
        else:
            suffix = gmt_id.split(':')[-1] if ':' in gmt_id else gmt_id
            if suffix in pathway_by_suffix:
                uri = pathway_by_suffix[suffix]
                matched[uri] = genes

    print(f"Matched {len(matched)} pathways with gene sets")
    return matched

def write_pathway_commons_obo(pathways: dict, pathway_parents: dict,
                              obo_path: str) -> None:
    """Write Pathway Commons OBO file."""
    all_pathways = set(pathways.keys())
    root_pathways = all_pathways - set(pathway_parents.keys())

    use_artificial_root = len(root_pathways) > 1
    artificial_root_id = "PC:0000000"

    with open(obo_path, 'w') as f:
        write_obo_header(f, "pathwaycommons_pathway")

        if use_artificial_root:
            create_artificial_root_term(
                f,
                artificial_root_id,
                "Pathway Commons Root",
                "pathway"
            )

        for pathway_uri in sorted(pathways.keys()):
            pathway_id = pathway_uri.split('/')[-1]
            clean_id = f"PC:{pathway_id}"

            f.write("[Term]\n")
            f.write(f"id: {clean_id}\n")
            f.write(f"name: {pathways[pathway_uri]}\n")
            f.write("namespace: pathway\n")
            f.write('def: "Pathway Commons pathway (BioPAX-derived)." []\n')
            f.write(f"xref: PathwayCommons:{pathway_uri}\n")

            parents = pathway_parents.get(pathway_uri, set())

            if use_artificial_root and pathway_uri in root_pathways:
                f.write(f"is_a: {artificial_root_id}\n")

            for parent_uri in sorted(parents):
                parent_id = parent_uri.split('/')[-1]
                f.write(f"is_a: PC:{parent_id}\n")

            f.write("\n")

    print(f"Wrote {obo_path} with {len(pathways)} pathways")

def write_pathway_commons_gaf(pathways: dict, pathway_genes: dict,
                              hgnc_to_ensembl: dict, gaf_path: str) -> None:
    """Write Pathway Commons GAF file."""
    today = date.today().isoformat()
    annotation_count = 0

    with open(gaf_path, 'w') as f:
        header = create_gaf_header(
            "PathwayCommons",
            "Pathway Commons pathway annotations for human genes"
        )
        for line in header:
            f.write(line)

        for pathway_uri, genes in sorted(pathway_genes.items()):
            pathway_id = pathway_uri.split('/')[-1]
            clean_pathway_id = f"PC:{pathway_id}"
            pathway_name = pathways.get(pathway_uri, pathway_id)

            for gene_symbol in genes:
                ensembl_ids = hgnc_to_ensembl.get(gene_symbol, [])

                for ensembl_id in ensembl_ids:
                    fields = [
                        "PathwayCommons",
                        ensembl_id,
                        gene_symbol,
                        "",
                        clean_pathway_id,
                        "PathwayCommons:pathway",
                        "IEA",
                        "",
                        "P",
                        pathway_name,
                        "",
                        "protein",
                        "taxon:9606",
                        today.replace("-", ""),
                        "PathwayCommons",
                        str(np.nan),
                        str(np.nan),
                    ]

                    f.write("\t".join(fields) + "\n")
                    annotation_count += 1

    print(f"Wrote {gaf_path} with {annotation_count} annotations")

def main():
    parser = argparse.ArgumentParser(
        description='Download and process pathway databases'
    )
    parser.add_argument('--reactome', action='store_true',
                        help='Download Reactome')
    parser.add_argument('--pathway-commons', action='store_true',
                        help='Download Pathway Commons')
    parser.add_argument('--all', action='store_true',
                        help='Download all pathway databases')
    parser.add_argument('--output-dir', default='data/pathways',
                        help='Output directory (default: data/pathways)')

    args = parser.parse_args()

    if not any([args.reactome, args.pathway_commons, args.all]):
        parser.print_help()
        return

    if args.all or args.reactome:
        reactome_dir = os.path.join(args.output_dir, "reactome")
        download_reactome(reactome_dir)

    if args.all or args.pathway_commons:
        pc_dir = os.path.join(args.output_dir, "pathway_commons")
        download_pathway_commons(pc_dir)

    print("\nPathway processing complete!")

if __name__ == "__main__":
    main()
