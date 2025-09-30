import argparse
from pathlib import Path

import numpy as np
import torch

from utils import visualize_scale, visualize_additivescale, visualize_result
from data_generation import build_function, generate_toy_data
from models import train_model


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train data path")
    parser.add_argument("--scale", help="data scale", default=1, type=float)
    parser.add_argument("--bias", help="data bias", default=0, type=float)
    parser.add_argument("--r", help="data dispersion", default=100, type=float)
    parser.add_argument("--func", help="function_type", default="non-linear", type=str)
    parser.add_argument("--lr", help="data dispersion", default=0.001, type=float)
    parser.add_argument(
        "--data_type", default="multi", choices=["single", "multi"], type=str
    )
    parser.add_argument(
        "--sampling", default="imbalanced", choices=["uniform", "imbalanced"], type=str
    )
    parser.add_argument("--num_pat", help="number of patient", default=2, type=int)
    parser.add_argument("--func_param", help="function pattern", default=0, type=int)
    parser.add_argument("--n_sample", help="number of sample", default=1000, type=int)
    parser.add_argument(
        "--save_dir",
        help="number of sample",
        default="/home/hdd/kazuya/strank/strank/toy_example/outputs",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    custom_mu = build_function(args)

    x_train, y_train, gl_train, x_val, y_val, x_test, y_test = generate_toy_data(
        custom_mu, args
    )

    dataset = {
        "x_train": x_train,
        "y_train": y_train,
        "gl_train": gl_train,
        "x_test": x_test,
        "y_test": y_test,
    }

    Path(f"{args.save_dir}/{args.data_type}/model_output").mkdir(
        parents=True, exist_ok=True
    )
    Path(f"{args.save_dir}/{args.data_type}").mkdir(parents=True, exist_ok=True)

    method_list = ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]
    visualize_result(args, dataset, method_list, custom_mu)
