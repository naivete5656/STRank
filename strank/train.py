import datasets, modules
import numpy as np
import os
import torch
import datetime
import argparse
import pathlib
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger


def train_model(
    data_dir,
    param_path,
    log_dir,
    test_sample_ids,
    val_sample_ids,
    train_sample_ids,
    loss,
    model,
    max_epochs,
    ngpu,
    no_sample_wise,
):
    # Load data
    data_dir = Path(data_dir)
    data_dict = datasets.load_data(data_dir)
    if args.use_gene:
        use_gene_list = np.loadtxt(args.use_gene, dtype=str).tolist()
    else:
        use_gene_list = None
    if no_sample_wise:
        test_dataset, val_dataset, train_dataset = (
            datasets.generate_datasets_all_samples(
                data_dict, val_ratio=0.1, test_ratio=0.1, use_gene_list=use_gene_list
            )
        )
    else:
        test_dataset, val_dataset, train_dataset = (
            datasets.generate_datasets_sample_wise(
                data_dict,
                test_sample_ids=test_sample_ids,
                val_sample_ids=val_sample_ids,
                train_sample_ids=train_sample_ids,
                use_gene_list=use_gene_list,
            )
        )
    if args.sampling_strategy == "pat":
        train_loader = datasets.create_balanced_dataloader(
            train_dataset,
            batch_size=256,  # 希望するバッチサイズ
        )
    elif args.sampling_strategy == "single":
        train_loader = datasets.create_imbalanced_dataloader(
            train_dataset,
            batch_size=256,  # 希望するバッチサイズ
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=True
        )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
    # setup log setting
    loss_key = loss
    model_key = model

    _eps = 1e-6
    mean_count_w = 1 / (train_dataset[:]["count"].mean(dim=0) + _eps)

    # set up model
    loss_params = {
        "stranklist": {"normalize_effect": True, "perm": "group"},
        "stranklist_v2": {"normalize_effect": True, "perm": "group", "n_pair": 256},
        "stranklist_v3": {"normalize_effect": True, "perm": "group", "n_pair": 256},
        "strankg": {"normalize_effect": True, "perm": "group"},
        "stranka": {"normalize_effect": True, "perm": "global"},
        "strankgfw": {
            "normalize_effect": True,
            "perm": "group",
            "feature_weights": mean_count_w,
        },
        "nb": {"feature_dim": train_dataset[0]["exp"].shape},
    }
    model = modules.STPred(
        model_key,
        loss_key,
        model_params={
            "in_features": train_dataset[0]["img_feat"].shape[0],
            "out_features": train_dataset[0]["exp"].shape[0],
        },
        loss_params=loss_params.get(loss_key, {}),
    )
    # train
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{model_key}_{loss_key}_{current_time}"
    monitor_loss = "val_scc"
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{log_dir}/ckpt", monitor=monitor_loss
    )
    # setup trainer
    max_epochs = max_epochs
    ngpu = ngpu
    logger = TensorBoardLogger(save_dir=log_dir, version=1, name=run_id)
    lr_finder = LearningRateFinder(num_training_steps=30)
    # trainer = pl.Trainer(max_epochs=1, devices=ngpu, accelerator="gpu", callbacks=[lr_finder], logger=logger)
    # trainer.fit(model, val_dataloaders=val_loader, train_dataloaders=train_loader)
    # print("Selected learning rate: ", model.lr)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=ngpu,
        accelerator="gpu",
        callbacks=[
            EarlyStopping(monitor=monitor_loss, patience=30),
            checkpoint_callback,
        ],
        logger=logger,
    )
    trainer.fit(model, val_dataloaders=val_loader, train_dataloaders=train_loader)
    model = modules.STPred.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        model_key=model_key,
        loss_key=loss_key,
        model_params={
            "in_features": train_dataset[0]["img_feat"].shape[0],
            "out_features": train_dataset[0]["exp"].shape[0],
        },
        loss_params=loss_params.get(loss_key, {}),
    )
    param_path = Path(param_path)
    os.makedirs(param_path.parent, exist_ok=True)
    torch.save(model.state_dict(), param_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for STPred model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset"
    )
    parser.add_argument(
        "--param_path",
        type=str,
        required=True,
        help="Path for storing the optimized parameter file",
    )
    parser.add_argument(
        "--test_sample_ids",
        type=str,
        nargs="+",
        required=False,
        help="List of test sample IDs",
        default=None,
    )
    parser.add_argument(
        "--val_sample_ids",
        type=str,
        nargs="+",
        required=False,
        help="List of validation sample IDs",
        default=None,
    )
    parser.add_argument(
        "--train_sample_ids",
        type=str,
        nargs="+",
        required=False,
        help="List of validation sample IDs",
        default=None,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--loss", type=str, help="Loss function to use", default="stranklist_v2"
    )
    parser.add_argument(
        "--model", type=str, help="Model architecture to use", default="linear"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        choices=["pat", "single", "default"],
        default="default",
        help="Sampling strategy to use for training",
    )
    parser.add_argument(
        "--use_gene",
        type=str,
        default=None,
        help="Specify a gene to use for evaluation. If not provided, all genes will be used.",
    )
    parser.add_argument(
        "--no_sample_wise", action="store_true", help="Use sample-wise training"
    )
    args = parser.parse_args()

    train_model(
        args.data_dir,
        args.param_path,
        args.log_dir,
        args.test_sample_ids,
        args.val_sample_ids,
        args.train_sample_ids,
        args.loss,
        args.model,
        args.max_epochs,
        args.ngpu,
        args.no_sample_wise,
    )


# PYTHONPATH=/home/tiisaishima/packages/strank/ python /home/tiisaishima/packages/strank/strank/train.py --data_dir data/hest1k/breat_xenium/strank_dataset --param_path data/hest1k/breat_xenium/opt_param.pt --test_sample_ids TENX97  --val_sample_ids TENX94 TENX96 --log_dir data/hest1k/breat_xenium/logs --loss mse --model linear --max_epochs 100 --ngpu 1
# PYTHONPATH=/home/tiisaishima/packages/strank/ python /home/tiisaishima/packages/strank/strank/train.py --data_dir data/hest1k/breat_xenium/strank_dataset --param_path data/hest1k/breat_xenium/opt_param.pt --test_sample_ids TENX97  --val_sample_ids TENX94 TENX96 --log_dir data/hest1k/breat_xenium/logs --loss strank --model linear --max_epochs 100 --ngpu 1 --sample_wise
# PYTHONPATH=/home/tiisaishima/packages/strank/ python /home/tiisaishima/packages/strank/strank/train.py --data_dir data/hest1k/breat_xenium/strank_dataset --param_path data/hest1k/breat_xenium/opt_param.pt   --log_dir data/hest1k/breat_xenium/logs --loss mse --model linear --max_epochs 100 --ngpu 1
# PYTHONPATH=/home/tiisaishima/packages/strank/ python /home/tiisaishima/packages/strank/strank/train.py --data_dir data/hest1k/breat_xenium/strank_dataset --param_path data/hest1k/breat_xenium/opt_param.pt   --log_dir data/hest1k/breat_xenium/logs --loss strank --model linear --max_epochs 100 --ngpu 1
