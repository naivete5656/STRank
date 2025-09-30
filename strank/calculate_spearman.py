import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics.functional import spearman_corrcoef
import csv

# Make sure to set your PYTHONPATH:
# export PYTHONPATH="/home/tiisaishima/packages/strank/:$PYTHONPATH"
# so these imports work properly.
import datasets
from modules import STPred


def main(args):
    # 1. Load data
    data_dict = datasets.load_data(args.data_dir)

    # 2. Create a dataset for the specified sample IDs.
    #    We'll create a test dataset that contains only the requested samples.
    test_dataset, _, _ = datasets.generate_datasets_sample_wise(
        data_dict,
        test_sample_ids=args.sample_ids[:1],
        val_sample_ids=None,
        train_sample_ids=None,
    )

    # 3. Create a DataLoader for these samples.
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 4. Inspect the first sample in the dataset to get model input/output size.
    #    (Ensure sample_ids match actual data; otherwise, test_dataset might be empty.)
    sample_item = test_dataset[0]
    in_features = sample_item["img_feat"].shape[0]
    out_features = sample_item["exp"].shape[0]

    _eps = 1e-6
    mean_count_w = 1 / (test_dataset[:]["count"].mean(dim=0) + _eps)

    # set up model
    loss_params = {
        "stranklist": {"normalize_effect": True, "perm": "group"},
        "strankg": {"normalize_effect": True, "perm": "group"},
        "strankgfw": {
            "normalize_effect": True,
            "perm": "group",
            "feature_weights": mean_count_w,
        },
        "stranka": {"normalize_effect": True, "perm": "global"},
        "nb": {"feature_dim": test_dataset[0]["exp"].shape},
    }
    # 5. Construct the model with the same architecture and loss used during training.
    model = STPred(
        model_key=args.model,
        loss_key=args.loss,
        model_params={"in_features": in_features, "out_features": out_features},
        loss_params=loss_params.get(args.loss, {}),
    )

    # 6. Load the saved model parameters.
    state_dict = torch.load(args.param_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # 7. Perform inference and collect predictions & targets.
    all_preds = []
    all_exps = []

    with torch.no_grad():
        for batch in test_loader:
            # Since STPred calculates loss in forward(), we directly call model.model(...)
            outputs = model.model(batch)  # returns dict with 'gene_pred'

            pred = outputs["gene_pred"]  # shape: (batch_size, out_features)
            exp = batch["exp"]  # shape: (batch_size, out_features)

            all_preds.append(pred)
            all_exps.append(exp)

    # 8. Concatenate predictions and expressions across all batches.
    all_preds = torch.cat(all_preds, dim=0)  # shape: (N, out_features)
    all_exps = torch.cat(all_exps, dim=0)  # shape: (N, out_features)
    mean_exps = all_exps.mean(dim=0)  # shape: (out_features,)

    # 9. Compute gene-wise Spearman correlations.
    num_genes = out_features
    spearman_scores = []

    for gene_idx in range(num_genes):
        gene_pred = all_preds[:, gene_idx]
        gene_exp = all_exps[:, gene_idx]

        # Spearman correlation for this gene
        corr_val = spearman_corrcoef(gene_exp, gene_pred)
        spearman_scores.append(float(corr_val))

    # 10. Calculate the average Spearman correlation.
    avg_spearman = np.mean(spearman_scores)

    # 11. Print only the average correlation.
    print(f"Average Spearman correlation across all genes: {avg_spearman:.4f}")

    # 12. Save per-gene correlations to a CSV file.
    #     The CSV will contain columns: "gene_idx" and "spearman_correlation".
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["gene_idx", "spearman_correlation", "mean_expression", "std_expression"]
        )
        for i, corr in enumerate(spearman_scores):
            writer.writerow([i, corr, mean_exps[i].item(), all_exps[:, i].std().item()])

    # Optionally, you could print a message indicating that the CSV was saved.
    print(f"Per-gene Spearman correlations saved to: {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute gene-wise Spearman correlation for specified samples."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset."
    )
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        help="Path to the trained model parameters (.pth).",
    )
    parser.add_argument(
        "--sample_ids",
        type=str,
        nargs="+",
        required=True,
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
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for inference."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file where gene-wise correlations will be saved.",
    )

    args = parser.parse_args()
    main(args)
