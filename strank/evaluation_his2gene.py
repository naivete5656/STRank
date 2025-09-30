from pathlib import Path
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
import csv

# Make sure to set your PYTHONPATH:
# export PYTHONPATH="/home/tiisaishima/packages/strank/:$PYTHONPATH"
# so these imports work properly.
import datasets
from hist2gene import His2Gene


def collate_fn(batch):

    return batch


def main(args):
    # 1. Load data
    data_dict = datasets.load_patch_data(args.data_dir)

    # 2. Create a dataset for the specified sample IDs.
    #    We'll create a test dataset that contains only the requested samples.
    if args.use_gene:
        use_gene_list = np.loadtxt(args.use_gene, dtype=str).tolist()
    else:
        use_gene_list = None
    test_dataset = datasets.generate_datasets_slide_wise(
        data_dict,
        test_sample_ids=args.sample_ids,
        val_sample_ids=None,
        train_sample_ids=None,
        use_gene_list=use_gene_list,
        test=True,
    )
    # total_dataset = datasets.Slide2STDataset(data_dict, use_gene_list=use_gene_list)

    # 3. Create a DataLoader for these samples.
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=False)

    # 4. Inspect the first sample in the dataset to get model input/output size.
    #    (Ensure sample_ids match actual data; otherwise, test_dataset might be empty.)
    sample_item = test_dataset[0]
    in_features = sample_item["img_feat"].shape[0]
    out_features = sample_item["exp"].shape[0]

    _eps = 1e-6
    # mean_count_w = 1 / (test_dataset[:]["count"].mean(dim=0) + _eps)

    # set up model
    loss_params = {
        "stranklist": {"normalize_effect": True, "perm": "group"},
        "strankg": {"normalize_effect": True, "perm": "group"},
        # "strankgfw": {
        #     "normalize_effect": True,
        #     "perm": "group",
        #     "feature_weights": mean_count_w,
        # },
        "stranka": {"normalize_effect": True, "perm": "global"},
        "nb": {"feature_dim": len(use_gene_list)},
    }
    # 5. Construct the model with the same architecture and loss used during training.
    model = His2Gene(
        model_key=args.model,
        loss_key=args.loss,
        model_params={"in_features": in_features, "out_features": out_features},
        loss_params=loss_params.get(args.loss, {}),
        n_genes=len(use_gene_list),
    )

    # 6. Load the saved model parameters.
    state_dict = torch.load(args.param_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 7. Perform inference and collect predictions & targets.
    spearman_scores = []
    pearson_scores = []

    with torch.no_grad():
        for batchs in test_loader:
            for batch in batchs:
                # Since STPred calculates loss in forward(), we directly call model.model(...)

                _, outputs = model(batch)  # returns dict with 'gene_pred'

                pred = outputs["gene_pred"]  # shape: (batch_size, out_features)
                exp = batch["exp"]  # shape: (batch_size, out_features)
                count = batch["count"]  # shape: (batch_size, out_features)
                norm_count = torch.log((count / count.sum(dim=1, keepdim=True)) * 1e4 + 1)

                speaman_val = spearman_corrcoef(pred, norm_count)
                spearman_scores.append(speaman_val.detach().cpu().numpy())

                pearson_val = pearson_corrcoef(pred, norm_count)
                pearson_scores.append(pearson_val.detach().cpu().numpy())

    # 10. Calculate the average Spearman correlation.
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    avg_spearman = np.array(spearman_scores).mean(0).mean()
    with open(
        f"{args.output_csv}_mean.csv",
        mode="w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["avg_spearman"])
        writer.writerow([avg_spearman])

    # 11. Print only the average correlation.
    print(f"Average Spearman correlation across all genes: {avg_spearman:.4f}")

    # 12. Save per-gene correlations to a CSV file.
    #     The CSV will contain columns: "gene_idx" and "spearman_correlation".\
    spearman_scores = np.array(spearman_scores).mean(0)
    pearson_scores = np.array(pearson_scores).mean(0)
    gene_names = np.array(use_gene_list).astype(str)
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gene_idx",
                "gene_name",
                "spearman_correlation",
                "pearson_correlation",
            ]
        )
        for i in range(len(gene_names)):
            writer.writerow(
                [
                    i,
                    gene_names[i],
                    spearman_scores[i],
                    pearson_scores[i],
                ]
            )

    # Optionally, you could print a message indicating that the CSV was saved.
    print(f"Per-gene Spearman correlations saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute gene-wise Spearman correlation for specified samples.")
    parser.add_argument(
        "--data_dir", type=str, default="./dataset/hest1k/st_v2/paired_data", help="Directory containing the dataset."
    )
    parser.add_argument(
        "--param_path",
        type=str,
        default="dataset/hest1k/st_v2/opts/comp/stranklist/opt_param_his.pt",
        help="Path to the trained model parameters (.pth).",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        # required=True,
        default=["SPA136"],
        help="Sample IDs to evaluate.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="Model architecture (must match what was used in training).",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="strankg",
        help="Loss key (must match what was used in training).",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for inference.")
    parser.add_argument(
        "--output_csv",
        type=str,
        # required=True,
        default="output/test.csv",
        help="Path to the output CSV file where gene-wise correlations will be saved.",
    )
    parser.add_argument(
        "--use_gene",
        type=str,
        default="./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_250.txt",
        help="Specify a gene to use for evaluation. If not provided, all genes will be used.",
    )
    parser.add_argument("--no_sample_wise", action="store_true", help="Use sample-wise training")
    args = parser.parse_args()
    args.test_sample_ids = ["SPA136", "SPA135", "SPA134", "SPA133", "SPA132", "SPA131"]
    main(args)
