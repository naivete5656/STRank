import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from torchmetrics.functional import spearman_corrcoef


NAMEDICT = {"additive_scaling": "addsc", "scaling": "sc"}


def visualize_additivescale(custom_mu):
    n_sample = 1000
    result_dict = {}
    for method in ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]:
        data = np.load(
            f"/home/hdd/kazuya/strank/strank/toy_example/outputs/each_result/{method}_{n_sample}_addsc_result.npz"
        )

        # 必要な変数を取り出す
        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]
        x3 = data["x3"]
        y3 = data["y3"]
        mu1 = data["mu1"]
        mu2 = data["mu2"]
        mu3 = data["mu3"]
        result_dict[method] = data["y_test_pred"]

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)
    mu = (mu - mu.min()) / (mu.max() - mu.min())

    color_list = ["#edae49", "#d1495b", "#00798c", "#d9bf77", "#f5cac3"]

    fig = plt.figure(figsize=(30, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, mu, color="#000000", linestyle="--", linewidth=1.5)

    for i, method in enumerate(
        ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]
    ):
        pred = result_dict[method][:, 0]
        scc = spearman_corrcoef(torch.Tensor(mu3), torch.Tensor(pred))

        pred = (pred - pred.min()) / (pred.max() - pred.min())
        ax.plot(
            x,
            mu,
            label=f"{method}: {scc:.02f}",
            color=color_list[i],
            linewidth=2,
        )
    ax.set_title(f"n_sample = {n_sample}")
    ax.set_ylim(0, 5)
    ax.set_xlim(0, 10)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.legend(loc="upper right", fontsize="small")

    ax = fig.add_subplot(gs[0, 1])

    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green
    plt.scatter(x1, y1, alpha=0.2, label="Tissue 1", color=colors[0])
    plt.scatter(x2, y2, alpha=0.2, label="Tissue 2", color=colors[1])

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)
    plt.plot(x, mu + mu1, color="#0072B2", linestyle="--", linewidth=1.5)
    plt.plot(x, mu + mu2, color="#E69F00", linestyle="--", linewidth=1.5)

    plt.xlabel("X")
    plt.ylabel("y")
    plt.ylim(0, 50)
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

    plt.tight_layout()
    plt.savefig("toy_example/outputs/additivescaling_training.pdf")
    plt.savefig("toy_example/outputs/additivescaling_training.png")


def visualize_scale(custom_mu):
    n_sample = 10000
    result_dict = {}
    for method in ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]:
        data = np.load(
            f"/home/hdd/kazuya/strank/strank/toy_example/outputs/each_result/{method}_{n_sample}_sc_result.npz"
        )

        # 必要な変数を取り出す
        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]
        x3 = data["x3"]
        y3 = data["y3"]
        mu1 = data["mu1"]
        mu2 = data["mu2"]
        mu3 = data["mu3"]
        result_dict[method] = data["y_test_pred"]

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)

    color_list = ["#edae49", "#d1495b", "#00798c", "#d9bf77", "#f5cac3"]
    plt.figure(figsize=(4, 4))
    plt.plot(x, mu * (1 / 100), color="#000000", linestyle="--", linewidth=1.5)
    for i, method in enumerate(
        ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]
    ):
        pred = result_dict[method][:, 0]
        scc = spearman_corrcoef(torch.Tensor(mu3), torch.Tensor(pred))

        pred = (pred - pred.min()) / (pred.max() - pred.min()) * 0.5
        plt.plot(
            x3,
            pred,
            label=f"{method}: {scc:.02f}",
            color=color_list[i],
            linewidth=2,
        )
    plt.ylim(0, 0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray")
    plt.savefig("toy_example/outputs/scaling_overall.pdf")
    plt.savefig("toy_example/outputs/scaling_overall.png")

    plt.close()
    # Visualize the generated data
    plt.figure(figsize=(5, 5))

    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green
    plt.scatter(x1, y1, alpha=0.2, label="Tissue 1", color=colors[0])
    plt.scatter(x2, y2, alpha=0.2, label="Tissue 2", color=colors[1])

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)
    plt.plot(x, mu, color="#0072B2", linestyle="--", linewidth=1.5)
    plt.plot(x, mu * mu2, color="#E69F00", linestyle="--", linewidth=1.5)

    plt.xlabel("X")
    plt.ylabel("y")
    plt.ylim(0, 50)
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

    plt.tight_layout()
    plt.savefig("toy_example/outputs/scaling_training.pdf")
    plt.savefig("toy_example/outputs/scaling_training.png")


def visualize_scale(custom_mu):
    n_sample = 10000
    result_dict = {}
    for method in ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]:
        data = np.load(
            f"/home/hdd/kazuya/strank/strank/toy_example/outputs/each_result/{method}_{n_sample}_sc_result.npz"
        )

        # 必要な変数を取り出す
        x1 = data["x1"]
        y1 = data["y1"]
        x2 = data["x2"]
        y2 = data["y2"]
        x3 = data["x3"]
        y3 = data["y3"]
        mu1 = data["mu1"]
        mu2 = data["mu2"]
        mu3 = data["mu3"]
        result_dict[method] = data["y_test_pred"]

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)

    color_list = ["#edae49", "#d1495b", "#00798c", "#d9bf77", "#f5cac3"]
    plt.figure(figsize=(4, 4))
    plt.plot(x, mu * (1 / 100), color="#000000", linestyle="--", linewidth=1.5)
    for i, method in enumerate(
        ["MSELoss", "GroupPearson", "STRankLoss", "STRankLossReg"]
    ):
        pred = result_dict[method][:, 0]
        scc = spearman_corrcoef(torch.Tensor(mu3), torch.Tensor(pred))

        pred = (pred - pred.min()) / (pred.max() - pred.min()) * 0.5
        plt.plot(
            x3,
            pred,
            label=f"{method}: {scc:.02f}",
            color=color_list[i],
            linewidth=2,
        )
    plt.ylim(0, 0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, color="gray")
    plt.savefig("toy_example/outputs/scaling_overall.pdf")
    plt.savefig("toy_example/outputs/scaling_overall.png")

    plt.close()
    # Visualize the generated data
    plt.figure(figsize=(5, 5))

    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green
    plt.scatter(x1, y1, alpha=0.2, label="Tissue 1", color=colors[0])
    plt.scatter(x2, y2, alpha=0.2, label="Tissue 2", color=colors[1])

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x)
    plt.plot(x, mu, color="#0072B2", linestyle="--", linewidth=1.5)
    plt.plot(x, mu * mu2, color="#E69F00", linestyle="--", linewidth=1.5)

    plt.xlabel("X")
    plt.ylabel("y")
    plt.ylim(0, 50)
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5, color="lightgray")

    plt.tight_layout()
    plt.savefig("toy_example/outputs/scaling_training.pdf")
    plt.savefig("toy_example/outputs/scaling_training.png")


def visualize_single(args, dataset, custom_mu, method_list):

    x_test = dataset["x_test"]
    mu = custom_mu(x_test, args.scale) + args.bias
    mu = (mu - mu.min()) / (mu.max() - mu.min())

    color_list = [
        "#edae49",
        "#d1495b",
        "#b23a48",
        "#00798c",
        "#d9bf77",
        "#f5cac3",
        "#003049",
        "#8ecae6",
        "#f5cac3",
    ]

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 2, figure=fig)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x_test, mu, color="#000000", linestyle="--", linewidth=1.5)

    for i, method in enumerate(method_list):
        pred = np.load(
            f"{args.save_dir}/{args.data_type}/model_output/{method}_{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}_test_result.npy"
        )[:, 0]
        scc = spearman_corrcoef(torch.Tensor(mu), torch.Tensor(pred))

        pred = (pred - pred.min()) / (pred.max() - pred.min())

        ax.plot(
            x_test,
            pred,
            label=f"{method}: {scc:.02f}",
            color=color_list[i],
            linewidth=2,
        )
    ax.set_title(f"n_sample = {args.n_sample}")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    ax.legend(loc="upper right", fontsize="small")

    ax = fig.add_subplot(gs[0, 1])
    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x, args.scale) + args.bias
    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green
    ax.plot(x, mu, color=colors[0], linestyle="--", linewidth=1.5)

    x = dataset["x_train"]
    y = dataset["y_train"]
    ax.scatter(x, y, alpha=0.2, label="Tissue 1", color=colors[0])

    ax.set_ylim(0, mu.max() + args.scale * 5)
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

    plt.tight_layout()
    plt.savefig(
        f"{args.save_dir}/{args.data_type}/{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}.png"
    )


def visualize_multi(args, dataset, custom_mu, method_list):

    # x_test = dataset["x_test"]
    # sorted_index = np.argsort(x_test)
    # mu = custom_mu(x_test, args.scale) + args.bias
    # mu = (mu - mu.min()) / (mu.max() - mu.min())

    # # color_list = ["#edae49", "#d1495b", "#00798c", "#8e7dbe", "#90be6d"]
    # color_list = [
    #     "#edae49",
    #     "#d1495b",
    #     "#b23a48",
    #     "#00798c",
    #     "#d9bf77",
    #     "#f5cac3",
    #     "#003049",
    #     "#8ecae6",
    #     "#f5cac3",
    # ]

    # fig = plt.figure(figsize=(10, 5))
    # gs = GridSpec(1, 2, figure=fig)

    # ax = fig.add_subplot(gs[0, 0])
    # ax.plot(
    #     x_test[sorted_index],
    #     mu[sorted_index],
    #     color="#000000",
    #     linestyle="--",
    #     linewidth=1.5,
    # )

    # for i, method in enumerate(method_list):
    #     pred = np.load(
    #         f"{args.save_dir}/{args.data_type}/model_output/{method}_{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}_test_result.npy"
    #     )[:, 0]
    #     scc = spearman_corrcoef(torch.Tensor(mu), torch.Tensor(pred))

    #     pred = (pred - pred.min()) / (pred.max() - pred.min())

    #     ax.plot(
    #         x_test[sorted_index],
    #         pred[sorted_index],
    #         label=f"{method}: {scc:.02f}",
    #         color=color_list[i],
    #         linewidth=2,
    #     )
    # ax.set_title(f"n_sample = {args.n_sample}")
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0, 1)
    # ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
    # ax.legend(loc="upper right", fontsize="small")
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0])
    colors = ["#0072B2", "#E69F00", "#009E73"]  # Blue  # Orange  # Green

    x = np.linspace(0, 1, 1000)
    mu = custom_mu(x, args.scale) + args.bias
    ax.plot(x, mu, color=colors[0], linestyle="--", linewidth=1.5)

    mu = custom_mu(x, args.scale * 10) + 10
    ax.plot(x, mu, color=colors[1], linestyle="--", linewidth=1.5)

    x = dataset["x_train"]
    y = dataset["y_train"]
    gl = dataset["gl_train"]

    for gl_label in np.unique(gl):
        ax.scatter(
            x[gl_label == gl],
            y[gl_label == gl],
            alpha=0.2,
            label=f"Tissue {gl_label + 1}",
            color=colors[int(gl_label)],
        )

    ax.set_ylim(0, mu.max() + args.scale * 5)
    ax.set_xlim(0, 1)
    # ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, linestyle="--", linewidth=0.5, color="gray")

    plt.tight_layout()
    plt.savefig(
        f"{args.save_dir}/{args.data_type}/{args.n_sample}_{args.r}_{args.scale}_{args.bias}_{args.sampling}_{args.func_param}_{args.lr}.png"
    )


def visualize_result(args, dataset, custom_mu, method_list):
    if args.data_type == "single":
        visualize_single(args, dataset, method_list, custom_mu)
    else:
        visualize_multi(args, dataset, method_list, custom_mu)
