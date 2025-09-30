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
        # "STRankLoss",
        "STRankLoss2",
        # "STRankLoss4",
        # "STRankLoss8",
        # "STRankLoss16",
        # "STRankLoss32",
        "STRankLoss64",
        # "STRankLoss128",
    ]

    for loss_type in method_list:
        # Set random seed for reproducibility
        init_seed(42)

        wandb.init(
            project="stcomp",  # プロジェクト名（自動作成可）
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
    parser.add_argument("--n_sample", help="number of sample", default=50000, type=int)
    parser.add_argument("--scale", help="data scale", default=1, type=float)
    parser.add_argument("--bias", help="data bias", default=0, type=float)
    parser.add_argument("--r", help="data dispersion", default=100.0, type=float)
    parser.add_argument("--lr", help="data dispersion", default=0.001, type=float)
    parser.add_argument("--epochs", help="data dispersion", default=200, type=int)
    parser.add_argument("--func", help="function_type", default="non-linear", type=str)
    parser.add_argument(
        "--data_type", default="multi", choices=["single", "multi"], type=str
    )
    parser.add_argument(
        "--sampling", default="imbalanced", choices=["uniform", "imbalanced"], type=str
    )
    parser.add_argument("--num_pat", help="number of patient", default=2, type=int)
    parser.add_argument("--func_param", help="function pattern", default=2, type=int)
    parser.add_argument(
        "--save_dir",
        help="number of sample",
        default="/home/hdd/kazuya/strank/strank/toy_example/outputs/STcomparison",
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
