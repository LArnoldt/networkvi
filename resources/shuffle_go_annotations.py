import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from go_utils import read_gaf_file, write_gaf_file
from utils import ensure_directory

try:
    from goatools.obo_parser import GODag

    GOATOOLS_AVAILABLE = True
except ImportError:
    GOATOOLS_AVAILABLE = False
    print("Warning: goatools not installed. Level-based shuffling will not be available.")
    print("Install with: pip install goatools")

def get_go_levels_and_namespace(obo_path: str) -> tuple:
    """
    Extract GO term levels and namespaces from OBO file.

    Args:
        obo_path: Path to GO OBO file

    Returns:
        Tuple of (go_level_dict, go_namespace_dict)
    """
    if not GOATOOLS_AVAILABLE:
        raise ImportError("goatools package required for level-based shuffling")

    print(f"Loading GO ontology from {obo_path}...")
    godag = GODag(obo_path, optional_attrs={"namespace"})

    go_level = {}
    go_namespace = {}

    for go_id, term in godag.items():
        if term.depth is not None:
            go_level[go_id] = term.depth
            go_namespace[go_id] = term.namespace

    print(f"Loaded {len(go_level)} GO terms")
    return go_level, go_namespace


def global_shuffle(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Globally shuffle GO term assignments.
    """
    rng = np.random.default_rng(seed)
    shuffled = df.copy()

    shuffled.iloc[:, 4] = df.iloc[:, 4].sample(frac=1, random_state=seed).values

    return shuffled

def level_shuffle(df: pd.DataFrame,
                  go_level: dict,
                  go_namespace: dict,
                  seed: int = 42) -> pd.DataFrame:
    """
    Shuffle GO terms within same ontology level and namespace.

    Preserves the level structure of the GO ontology by only shuffling
    terms that have the same depth and namespace.
    """
    rng = np.random.default_rng(seed)

    pools = defaultdict(list)

    go_id_col = df.columns[4] if hasattr(df.columns[4], 'name') else 4

    for go_id in df.iloc[:, 4].unique():
        if go_id in go_level and go_id in go_namespace:
            key = (go_namespace[go_id], go_level[go_id])
            pools[key].append(go_id)

    reassignment = {}

    for key, gos in pools.items():
        if len(gos) <= 1:
            reassignment.update({go: go for go in gos})
            continue

        shuffled = gos.copy()
        rng.shuffle(shuffled)

        for i, go in enumerate(gos):
            if shuffled[i] == go and len(gos) > 1:
                j = (i + 1) % len(gos)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

        reassignment.update(dict(zip(gos, shuffled)))

    shuffled_df = df.copy()
    shuffled_df.iloc[:, 4] = shuffled_df.iloc[:, 4].map(
        lambda g: reassignment.get(g, g)
    )

    return shuffled_df

def shuffle_gaf_file(gaf_path: str,
                     output_dir: str,
                     obo_path: str = None,
                     seed: int = 42,
                     global_only: bool = False) -> None:
    """
    Generate shuffled versions of a GAF file.
    """
    ensure_directory(output_dir)

    print(f"\nProcessing {gaf_path}...")

    header, df = read_gaf_file(gaf_path)

    base_name = Path(gaf_path).stem

    print("  Generating global shuffle...")
    df_global = global_shuffle(df, seed)
    output_path = os.path.join(output_dir, f"{base_name}_global_shuffled.gaf")
    write_gaf_file(output_path, header, df_global)

    if not global_only and obo_path and GOATOOLS_AVAILABLE:
        print("  Generating level shuffle...")
        go_level, go_namespace = get_go_levels_and_namespace(obo_path)
        df_level = level_shuffle(df, go_level, go_namespace, seed)
        output_path = os.path.join(output_dir, f"{base_name}_level_shuffled.gaf")
        write_gaf_file(output_path, header, df_level)
    elif not global_only:
        print("  Skipping level shuffle (OBO file or goatools not available)")

def shuffle_all_gaf_files(input_dir: str = "data/go",
                          output_dir: str = "data/go",
                          obo_path: str = None,
                          pattern: str = "*ensembl_gene_mapping*.gaf",
                          seed: int = 42) -> None:
    """
    Shuffle all GAF files matching a pattern.
    """
    import glob

    gaf_files = glob.glob(os.path.join(input_dir, pattern))

    gaf_files = [f for f in gaf_files if 'shuffled' not in f]

    if not gaf_files:
        print(f"No GAF files found matching pattern: {pattern}")
        return

    print(f"Found {len(gaf_files)} GAF files to shuffle")

    if obo_path is None:
        obo_candidates = glob.glob(os.path.join(input_dir, "go-basic*.obo"))
        if obo_candidates:
            obo_path = sorted(obo_candidates)[-1]
            print(f"Using OBO file: {obo_path}")

    for gaf_file in gaf_files:
        try:
            shuffle_gaf_file(gaf_file, output_dir, obo_path, seed)
        except Exception as e:
            print(f"Error shuffling {gaf_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate shuffled GO annotations for control analyses',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--gaf',
                        help='Specific GAF file to shuffle')
    parser.add_argument('--all', action='store_true',
                        help='Shuffle all processed GAF files')
    parser.add_argument('--input-dir', default='data/go',
                        help='Input directory (default: data/go)')
    parser.add_argument('--output-dir', default='data/go',
                        help='Output directory (default: data/go)')
    parser.add_argument('--obo',
                        help='Path to GO OBO file (auto-detected if not specified)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--global-only', action='store_true',
                        help='Only perform global shuffle (skip level shuffle)')

    args = parser.parse_args()

    if not args.gaf and not args.all:
        parser.print_help()
        return

    if args.gaf:
        shuffle_gaf_file(args.gaf, args.output_dir, args.obo,
                         args.seed, args.global_only)

    if args.all:
        shuffle_all_gaf_files(args.input_dir, args.output_dir,
                              args.obo, seed=args.seed)

    print("\nShuffling complete!")

if __name__ == "__main__":
    main()
