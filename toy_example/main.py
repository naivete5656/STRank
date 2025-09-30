import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import wandb

import torch
from torchmetrics.functional import spearman_corrcoef

from utils import visualize_scale, visualize_additivescale, visualize_result
from data_generation import build_function, generate_toy_data
from models import train_model


def init_seed(seed):
    """
    Set random seed for reproducibility
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example usage
def main(args):

    custom_mu = build_function(args)

    x_train, y_train, gl_train, x_val, y_val, x_test, y_test = generate_toy_data(
        custom_mu, args
    )

    dataset = {
        "x_train": x_train,
        "y_train": y_train,
        "gl_train": gl_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
    }

    method_list = [
        "MSELoss",
        "NBLoss",
        "PoissonLoss",
        "RankingLoss",
        "STRankLossPair",
        # "STRankLossStable",
        "PearsonLoss",
        "STRankLoss",
    ]

    # loss_type, custom_mu, toy_type="default"
    for loss_type in method_list:
        # Set random seed for reproducibility
        init_seed(42)

        wandb.init(
            project="simple_regression",  # プロジェクト名（自動作成可）
            name=f"run-epoch-{loss_type}_{args.data_type}_{args.sampling}_{args.func_param}",  # 実行名
        )

        # Train model on first distribution
        y_test_pred = train_model(
            dataset,
            wandb,
            loss_type=loss_type,
            epochs=args.epochs,
            learning_rate=args.lr,
            best_weight_path=f"{args.save_dir}/{args.data_type}/model_output/{loss_type}_{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}_best.pth",
        )

        scc = spearman_corrcoef(torch.Tensor(y_test), torch.Tensor(y_test_pred[:, 0]))

        np.save(
            f"{args.save_dir}/{args.data_type}/model_output/{loss_type}_{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}_test_result",
            y_test_pred,
        )
        np.save(
            f"{args.save_dir}/{args.data_type}/model_output/{loss_type}_{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}_scc",
            scc,
        )
        print(f"{loss_type}, scc:{scc:.2f}")
        wandb.finish()

    visualize_result(args, dataset, method_list, custom_mu)


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--n_sample", help="number of sample", default=10000, type=int)
    parser.add_argument("--scale", help="data scale", default=1, type=float)
    parser.add_argument("--bias", help="data bias", default=0, type=float)
    parser.add_argument("--r", help="data dispersion", default=10.0, type=float)
    parser.add_argument("--lr", help="data dispersion", default=0.001, type=float)
    parser.add_argument("--epochs", help="data dispersion", default=100, type=int)
    parser.add_argument("--func", help="function_type", default="non-linear", type=str)
    parser.add_argument("--drop_rate", help="dropout", default=0.5, type=int)
    parser.add_argument(
        "--data_type", default="multi", choices=["single", "multi"], type=str
    )
    parser.add_argument(
        "--sampling", default="imbalanced", choices=["uniform", "imbalanced"], type=str
    )
    parser.add_argument("--num_pat", help="number of patient", default=2, type=int)
    parser.add_argument("--func_param", help="function pattern", default=0, type=int)
    parser.add_argument(
        "--save_dir",
        help="number of sample",
        default="/home/hdd/kazuya/strank/strank/toy_example/outputs/rank_confirmination",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    Path(f"{args.save_dir}/{args.data_type}/model_output").mkdir(
        parents=True, exist_ok=True
    )
    Path(f"{args.save_dir}/{args.data_type}").mkdir(parents=True, exist_ok=True)
    main(args)


# # visualize_result
# custom_mu = CustomMu(scale=50)
# toy_type = "additive_scaling"
# main(
#     loss_type="STRankLoss",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="STRankLossReg",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="GroupPearson",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="MSELoss",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# visualize_additivescale(custom_mu)

# custom_mu = CustomMu(scale=100)
# toy_type = "scaling"
# main(
#     loss_type="STRankLoss",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="STRankLossReg",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="GroupPearson",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# main(
#     loss_type="MSELoss",
#     custom_mu=custom_mu,
#     n_sample=10000,
#     toy_type=toy_type,
# )
# visualize_scale(custom_mu)

# toy_type = "param_evaluation"
# for scale in [0.1, 1, 10]:
#     for dispersion in [0.1, 1, 10, 100]:
#         custom_mu = SingCustomMu(scale=scale)
#         main(
#             loss_type="STRankLoss",
#             custom_mu=custom_mu,
#             n_sample=1000,
#             toy_type=toy_type,
#             scale=scale,
#             r=dispersion,
#         )
#         main(
#             loss_type="STRankLossReg",
#             custom_mu=custom_mu,
#             n_sample=1000,
#             toy_type=toy_type,
#             scale=scale,
#             r=dispersion,
#         )
#         main(
#             loss_type="GroupPearson",
#             custom_mu=custom_mu,
#             n_sample=1000,
#             toy_type=toy_type,
#             scale=scale,
#             r=dispersion,
#         )
#         main(
#             loss_type="MSELoss",
#             custom_mu=custom_mu,
#             n_sample=1000,
#             toy_type=toy_type,
#             scale=scale,
#             r=dispersion,
#         )
# visualize(custom_mu)

# # Define a custom mu function
# for scale in [1, 5, 10, 100]:
#     custom_mu = CustomMu(scale=scale)
#     for r in [1, 5, 10, 100, 1000]:
#         for n_sample in [100, 200, 500, 1000]:
#             main(
#                 loss_type="STRankLoss",
#                 custom_mu=custom_mu,
#                 n_sample=n_sample,
#                 toy_type="reglarization",
#             )
#             main(
#                 loss_type="GroupPearson",
#                 custom_mu=custom_mu,
#                 n_sample=n_sample,
#                 toy_type="reglarization",
#             )
#             main(
#                 loss_type="MSELoss",
#                 custom_mu=custom_mu,
#                 n_sample=n_sample,
#                 toy_type="reglarization",
#             )

#     # Define colors for methods
#     color_list = ["#edae49", "#d1495b", "#00798c", "#d9bf77", "#f5cac3"]
#     methods = ["MSELoss", "GroupPearson", "STRankLoss"]
#     n_sample_list = [100, 500, 1000, 10000]

#     # Create a figure with subplots
#     fig = plt.figure(figsize=(5 * (len(n_sample_list) + 1), 5))
#     gs = GridSpec(1, len(n_sample_list) + 1, figure=fig)

#     # Loop through each n_sample value
#     for i, n_sample in enumerate(n_sample_list):
#         # for i, n_sample in enumerate([100, 500, 1000, 10000]):
#         ax = fig.add_subplot(gs[0, i])

#         result_dict = {}
#         for method in methods:
#             data = np.load(
#                 f"/home/hdd/kazuya/strank/strank/toy_example/outputs/{method}_{n_sample}_test_result.npz"
#             )

#             # Extract necessary variables
#             x1 = data["x1"]
#             y1 = data["y1"]
#             x2 = data["x2"]
#             y2 = data["y2"]
#             x3 = data["x3"]
#             y3 = data["y3"]
#             mu1 = data["mu1"]
#             mu2 = data["mu2"]
#             mu3 = data["mu3"]
#             result_dict[method] = data["y_test_pred"]

#         x = np.linspace(0, 1, 1000)
#         mu = custom_mu(x)

#         # Plot the ground truth
#         ax.plot(x, mu, color="#000000", linestyle="--", linewidth=1.5)

#         # Plot each method
#         for j, method in enumerate(methods):
#             pred = result_dict[method][:, 0]
#             scc = spearman_corrcoef(torch.Tensor(mu3), torch.Tensor(pred))

#             pred = (pred - pred.min()) / (pred.max() - pred.min()) * 0.5
#             ax.plot(
#                 x3,
#                 pred,
#                 label=f"{method}: {scc:.02f}",
#                 color=color_list[j],
#                 linewidth=2,
#             )

#         # Set title and limits
#         ax.set_title(f"n_sample = {n_sample}")
#         ax.set_ylim(0, 0.5)
#         ax.set_xlim(0, 1)
#         ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

#         # Only show y-axis label on the first subplot
#         if i == 0:
#             ax.set_ylabel("Value")

#         # Only show legend on the last subplot
#         ax.legend(loc="upper right", fontsize="small")
#         # Only show x-axis label on the middle subplots
#         if i == 2 or i == 3:
#             ax.set_xlabel("x")

#     ax = fig.add_subplot(gs[0, i + 1])

#     colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green
#     plt.scatter(x1, y1, alpha=0.2, label="Tissue 1", color=colors[0])
#     plt.scatter(x2, y2, alpha=0.2, label="Tissue 2", color=colors[1])

#     x = np.linspace(0, 1, 1000)
#     mu = custom_mu(x)
#     ax.plot(x, mu + mu1, color="#0072B2", linestyle="--", linewidth=1.5)
#     ax.plot(x, mu + mu2, color="#E69F00", linestyle="--", linewidth=1.5)

#     mu_max = max(mu1, mu2)
#     plt.ylim(0, mu_max + 3)
#     plt.xlim(0, 1)

#     # Adjust layout
#     plt.tight_layout()
#     plt.savefig("toy_example/outputs/compare.png")
