import argparse
from pathlib import Path

import numpy as np

import sys

sys.path.append("./")
from strank import datasets


def export_highly_variable(args):
    """Export higly variable genes for a datset.
    Args:
        args (argparse.Namespace): arguments.
    """
    data_dir = args.data_dir
    data_dict = datasets.load_data(data_dir)
    genes = datasets.extract_high_variance_genes(data_dict, args.ntop_genes, mean=args.mean).astype(str)
    np.savetxt(args.output_path, genes, fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export highly variable genes.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/hest1k/st_v2/feat/conch_v1",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output.",
        default="./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_all.txt",
    )
    parser.add_argument("--ntop_genes", type=int, default=50, help="Number of top genes to select.")
    parser.add_argument("--mean", action="store_true", help="Use mean expression.")
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    export_highly_variable(args)
