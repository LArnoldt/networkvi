"""
Download and process Gene Ontology (GO) annotations from EBI.

Supports both current and historical GO releases.
"""
import os
import argparse
from datetime import date
from utils import download_file, ensure_directory
from go_utils import (
    get_uniprotswissprot_mapping,
    get_rnacentral_mapping,
    read_gaf_file,
    write_gaf_file,
    map_external_ids_to_ensembl
)

# Historical GO release versions
GO_RELEASES = {
    "2014-06-01": {
        "ontology": "https://release.geneontology.org/2014-06-01/ontology/go-basic.obo",
        "annotations": {
            "goa_human": "https://release.geneontology.org/2014-06-01/annotations/goa_human.gaf.gz"
        }
    },
    "2019-06-09": {
        "ontology": "https://release.geneontology.org/2019-06-09/ontology/go-basic.obo",
        "annotations": {
            "goa_human": "https://release.geneontology.org/2019-06-09/annotations/goa_human.gaf.gz",
            "goa_human_isoform": "https://release.geneontology.org/2019-06-09/annotations/goa_human_isoform.gaf.gz",
            "goa_human_rna": "https://release.geneontology.org/2019-06-09/annotations/goa_human_rna.gaf.gz"
        }
    },
    "2022-06-15": {
        "ontology": "https://release.geneontology.org/2022-06-15/ontology/go-basic.obo",
        "annotations": {
            "goa_human": "https://release.geneontology.org/2022-06-15/annotations/goa_human.gaf.gz",
            "goa_human_isoform": "https://release.geneontology.org/2022-06-15/annotations/goa_human_isoform.gaf.gz",
            "goa_human_rna": "https://release.geneontology.org/2022-06-15/annotations/goa_human_rna.gaf.gz"
        }
    }
}


def download_go_current(output_dir: str = "data/go") -> None:
    """
    Download current GO ontology and annotations.

    Args:
        output_dir: Output directory
    """
    ensure_directory(output_dir)

    today = date.today().isoformat()

    # Download ontology
    obo_url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
    obo_path = os.path.join(output_dir, f"go-basic_{today}.obo")
    download_file(obo_url, obo_path, extract=False)

    # Download annotations
    annotation_files = {
        "goa_human": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz",
        "goa_human_isoform": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_isoform.gaf.gz",
        "goa_human_rna": "https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human_rna.gaf.gz"
    }

    for name, url in annotation_files.items():
        gz_path = os.path.join(output_dir, f"{name}_{today}.gaf.gz")
        download_file(url, gz_path, extract=True)


def download_go_historical(output_dir: str = "data/go") -> None:
    """
    Download historical GO releases.

    Args:
        output_dir: Output directory
    """
    ensure_directory(output_dir)

    for release_date, release in GO_RELEASES.items():
        print(f"\nDownloading GO release {release_date}...")

        # Download ontology
        obo_path = os.path.join(output_dir, f"go-basic_{release_date}.obo")
        if not os.path.exists(obo_path):
            download_file(release["ontology"], obo_path, extract=False)

        # Download annotations
        for name, url in release["annotations"].items():
            gz_path = os.path.join(output_dir, f"{name}_{release_date}.gaf.gz")
            gaf_path = os.path.join(output_dir, f"{name}_{release_date}.gaf")

            if not os.path.exists(gaf_path):
                download_file(url, gz_path, extract=True)


def process_go_annotations(gaf_path: str, output_dir: str,
                           use_date: str = None) -> None:
    """
    Process GO annotations to map external IDs to Ensembl gene IDs.

    Args:
        gaf_path: Path to input GAF file
        output_dir: Output directory
        use_date: Date string to use in output filename (auto-detected if None)
    """
    ensure_directory(output_dir)

    # Determine output date
    if use_date is None:
        # Extract date from filename if possible
        import re
        match = re.search(r'(\d{4}-\d{2}-\d{2})', gaf_path)
        use_date = match.group(1) if match else date.today().isoformat()

    # Read GAF file
    print(f"Processing {gaf_path}...")
    header, gaf_df = read_gaf_file(gaf_path)

    # Determine mapping type based on filename
    if 'rna' in os.path.basename(gaf_path).lower():
        mapping = get_rnacentral_mapping()
        mapping_type = "rna"
    else:
        mapping = get_uniprotswissprot_mapping()
        mapping_type = "protein"

    # Map to Ensembl IDs
    gaf_mapped = map_external_ids_to_ensembl(gaf_df, "DB_Object_ID", mapping)

    # Create output filename
    base_name = os.path.basename(gaf_path).replace('.gaf', '')
    # Remove date if already in filename
    import re
    base_name = re.sub(r'_\d{4}-\d{2}-\d{2}', '', base_name)

    output_path = os.path.join(
        output_dir,
        f"{base_name}_ensembl_gene_mapping_{use_date}.gaf"
    )

    # Write output
    write_gaf_file(output_path, header, gaf_mapped)

    print(f"  Mapped {len(gaf_df)} -> {len(gaf_mapped)} annotations ({mapping_type})")


def process_all_go_files(input_dir: str = "data/go",
                         output_dir: str = "data/go") -> None:
    """
    Process all GAF files in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
    """
    import glob

    gaf_files = glob.glob(os.path.join(input_dir, "goa_*.gaf"))

    # Filter out already processed files
    gaf_files = [f for f in gaf_files if 'ensembl_gene_mapping' not in f]

    if not gaf_files:
        print("No GAF files found to process.")
        return

    print(f"\nFound {len(gaf_files)} GAF files to process")

    for gaf_file in gaf_files:
        try:
            process_go_annotations(gaf_file, output_dir)
        except Exception as e:
            print(f"Error processing {gaf_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Download and process Gene Ontology annotations'
    )
    parser.add_argument('--current', action='store_true',
                        help='Download current GO release')
    parser.add_argument('--historical', action='store_true',
                        help='Download historical GO releases')
    parser.add_argument('--process', action='store_true',
                        help='Process GAF files to map to Ensembl IDs')
    parser.add_argument('--all', action='store_true',
                        help='Download and process all data')
    parser.add_argument('--output-dir', default='data/go',
                        help='Output directory (default: data/go)')

    args = parser.parse_args()

    if not any([args.current, args.historical, args.process, args.all]):
        parser.print_help()
        return

    if args.all or args.current:
        print("Downloading current GO release...")
        download_go_current(args.output_dir)

    if args.all or args.historical:
        print("Downloading historical GO releases...")
        download_go_historical(args.output_dir)

    if args.all or args.process:
        print("Processing GO annotations...")
        process_all_go_files(args.output_dir, args.output_dir)

    print("\nGO processing complete!")

if __name__ == "__main__":
    main()
