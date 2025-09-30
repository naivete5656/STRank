from pathlib import Path
import pickle
import scanpy as sc
import matplotlib.pyplot as plt
import h5py
import sys


def load_patches(file_name):
    with h5py.File(file_name) as f:
        barcode = f["barcode"][:]
        coords = f["coords"][:]
        patch_list = f["img"][:]
    return patch_list, barcode, coords


def preprocessing_adata(file_name, visualize=True):

    adata = sc.read_h5ad(file_name)

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
    )

    #
    # adata = adata[adata.obs.n_genes_by_counts < 1000, :]
    adata = adata[adata.obs.pct_counts_mt < 20, :].copy()

    if visualize:
        sc.pl.violin(
            adata,
            ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
            jitter=0.4,
            multi_panel=True,
        )

    #
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    if visualize:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

    # Regularized count data
    adata.layers["counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # highly variable
    # sc.pp.highly_variable_genes(adata, n_top_genes=50)

    return adata


def matching_adata_patch(adata, patch_list, barcode, visualize=False):
    code_list = []
    for bar in barcode:
        code_list.append(bar[0].decode("utf-8"))

    adata.obs["patch_index"] = None
    for code in adata.obs_names:
        try:
            idx = code_list.index(code)
            adata.obs.loc[code, "patch_index"] = idx
        except:
            pass

    adata_wpatch = adata[adata.obs["patch_index"].notnull()]

    if visualize:
        plt.rcParams["figure.figsize"] = (8, 8)
        sc.pl.spatial(
            adata_wpatch,
            img_key="downscaled_fullres",
            color=["total_counts", "n_genes_by_counts"],
            size=1,
        )

    spot_adata = adata_wpatch
    exp = spot_adata.X
    count = spot_adata.layers["counts"]

    patch_idx = spot_adata.obs["patch_index"].values
    patch = patch_list[patch_idx.astype(int)]
    img_coords = spot_adata.obsm["spatial"]
    gene_list = adata.var_names

    return exp, count, patch, img_coords, gene_list, spot_adata.obs_names


if len(sys.argv) != 2:
    print("Usage: python make_paired.py <data_directory>")
    sys.exit(1)

data_directory = sys.argv[1]
save_dir = Path(data_directory) / "paired_data"
save_dir.mkdir(parents=True, exist_ok=True)
patch_paths = sorted(Path(data_directory + "/patches").glob("*.h5"))
adata_dir = data_directory + "/st"

for patch_path in patch_paths:
    patch_list, barcode, coords = load_patches(patch_path)
    adata = preprocessing_adata(
        adata_dir + "/" + patch_path.stem + ".h5ad", visualize=False
    )
    exp, count, patch, img_coords, gene_list, barcode = matching_adata_patch(
        adata, patch_list, barcode, visualize=False
    )
    adata = sc.read_h5ad(adata_dir + "/" + patch_path.stem + ".h5ad")

    with open(
        f"{save_dir}/{patch_path.stem}.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "exp": exp,
                "count": count,
                "patch": patch,
                "img_coords": img_coords,
                "gene_list": gene_list,
                "barcode": barcode,
            },
            f,
        )
