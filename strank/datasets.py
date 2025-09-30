from torch.utils.data import Dataset, Subset
import numpy as np
from PIL import Image

from tqdm import tqdm
import torch
from pathlib import Path
import pickle
import h5py
import pandas as pd
import scanpy as sc
from torchmetrics.functional.regression import spearman_corrcoef
import torchvision.transforms as transforms


def load_data(data_dir):
    data_paths = Path(data_dir).glob("*.h5")
    data_dict = {path.stem: h5py.File(path, "r") for path in data_paths}
    return data_dict


def load_patch_data(data_dir):
    data_paths = Path(data_dir).glob("*.pkl")
    data_dict = {}
    for path in data_paths:
        with open(path, "rb") as f:
            data_dict[path.stem] = pickle.load(f)

    return data_dict


def generate_datasets_all_samples(data_dict, val_ratio=0.1, test_ratio=0.1, **kwargs):
    total_dataset = IMG2STDataset(data_dict, **kwargs)
    test_num = int(len(total_dataset) * test_ratio)
    val_num = int(len(total_dataset) * val_ratio)
    train_num = len(total_dataset) - val_num - test_num
    test_dataset, val_dataset, train_dataset = torch.utils.data.random_split(
        total_dataset, [test_num, val_num, train_num]
    )
    return test_dataset, val_dataset, train_dataset


def get_sample_indeces(total_dataset, sample_ids):
    sample_ids = [total_dataset.sample_id_map[sample_id] for sample_id in sample_ids]
    sample_indeces = torch.where(
        torch.isin(total_dataset.sample_ids, torch.tensor(sample_ids))
    )[0]
    return sample_indeces


def generate_datasets_sample_wise(
    data_dict, test_sample_ids, val_sample_ids, train_sample_ids=None, **kwargs
):
    if val_sample_ids is None:
        val_sample_ids = []

    if train_sample_ids is None:
        train_sample_ids = [
            sample_id
            for sample_id in data_dict.keys()
            if sample_id not in test_sample_ids + val_sample_ids
        ]
    else:
        train_sample_ids = train_sample_ids
    total_dataset = IMG2STDataset(data_dict, **kwargs)
    test_dataset = Subset(
        total_dataset, get_sample_indeces(total_dataset, test_sample_ids)
    )
    val_dataset = Subset(
        total_dataset, get_sample_indeces(total_dataset, val_sample_ids)
    )
    train_dataset = Subset(
        total_dataset, get_sample_indeces(total_dataset, train_sample_ids)
    )
    return test_dataset, val_dataset, train_dataset


def generate_datasets_slide_wise(
    data_dict,
    test_sample_ids,
    val_sample_ids,
    train_sample_ids=None,
    test=False,
    **kwargs
):
    if val_sample_ids is None:
        val_sample_ids = []

    if train_sample_ids is None:
        train_sample_ids = [
            sample_id
            for sample_id in data_dict.keys()
            if sample_id not in test_sample_ids + val_sample_ids
        ]
    else:
        train_sample_ids = train_sample_ids
    test_dataset = Slide2STDataset(data_dict, test_sample_ids, **kwargs)
    if test:
        return test_dataset

    val_dataset = Slide2STDataset(data_dict, val_sample_ids, **kwargs)
    train_dataset = Slide2STDataset(data_dict, train_sample_ids, **kwargs)
    # test_dataset = Subset(total_dataset, get_sample_indeces(total_dataset, test_sample_ids))
    # val_dataset = Subset(total_dataset, get_sample_indeces(total_dataset, val_sample_ids))
    # train_dataset = Subset(total_dataset, get_sample_indeces(total_dataset, train_sample_ids))
    return test_dataset, val_dataset, train_dataset


def get_common_genes(data_dict):
    sample_ids = list(data_dict.keys())
    geneset_list = [
        set(data_dict[sample_id]["gene_list"][:]) for sample_id in sample_ids
    ]
    common_genes = list(set.intersection(*geneset_list))
    return common_genes


def idx_map2common_genes(data_dict, common_genes):
    idx_map_dict = {}
    for sample_id, ds in data_dict.items():
        gene_list = ds["gene_list"][:]
        gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
        idx_map_dict[sample_id] = torch.tensor(
            [gene_to_index[gene] for gene in common_genes], dtype=torch.int32
        )
    return idx_map_dict


def extract_high_variance_genes(data_dict, ntop_genes=50, mean=False):
    common_genes = np.array(get_common_genes(data_dict))
    idx_map = idx_map2common_genes(data_dict, common_genes)
    sample_ids = list(data_dict.keys())
    orig_exp = torch.cat(
        [
            torch.tensor(data_dict[sample_id]["exp"][:]).float()[:, idx_map[sample_id]]
            for sample_id in sample_ids
        ]
    )
    if mean:
        gene_variances = torch.mean(orig_exp, dim=0)
    else:
        gene_variances = torch.var(orig_exp, dim=0)
    top_genes_idx = torch.argsort(gene_variances, descending=True)[:ntop_genes].int()
    return common_genes[top_genes_idx]


class IMG2STDataset(Dataset):
    def __init__(self, data_dict, transform=None, ntop_genes=250, use_gene_list=None):
        sample_ids = list(data_dict.keys())
        self.sample_id_map = {sample_id: i for i, sample_id in enumerate(sample_ids)}
        if "img_feat" in list(list(data_dict.values())[0].keys()):
            self.img_feat = torch.cat(
                [
                    torch.tensor(data_dict[sample_id]["img_feat"][:])
                    for sample_id in sample_ids
                ]
            )
        else:
            self.img_feat = torch.cat(
                [
                    torch.tensor(data_dict[sample_id]["features"][:])
                    for sample_id in sample_ids
                ]
            )

        self.common_genes = get_common_genes(data_dict)
        self.idx_map = idx_map2common_genes(data_dict, self.common_genes)
        self.count = torch.cat(
            [
                torch.tensor(data_dict[sample_id]["count"][:]).float()[
                    :, self.idx_map[sample_id]
                ]
                for sample_id in sample_ids
            ]
        )
        if use_gene_list is None:
            orig_exp = torch.cat(
                [
                    torch.tensor(data_dict[sample_id]["exp"][:]).float()[
                        :, self.idx_map[sample_id]
                    ]
                    for sample_id in sample_ids
                ]
            )
            gene_variances = torch.var(orig_exp, dim=0)
            use_gene_idx_list = torch.argsort(gene_variances, descending=True)[
                :ntop_genes
            ]
        else:
            common_genes = list(np.array(self.common_genes).astype(str))
            use_gene_idx_list = torch.tensor(
                [common_genes.index(gene) for gene in use_gene_list]
            )
        self.count = self.count[:, use_gene_idx_list]
        self.exp = torch.log(
            1.0e4 * self.count / self.count.sum(dim=1, keepdims=True) + 1.0
        )
        self.common_genes = [list(self.common_genes)[i] for i in use_gene_idx_list]
        self.sample_ids = torch.cat(
            [
                torch.tensor(
                    [self.sample_id_map[sample_id]]
                    * len(data_dict[sample_id]["exp"][:])
                )
                for sample_id in sample_ids
            ]
        )

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        data = {
            "img_feat": self.img_feat[idx],
            "exp": self.exp[idx],
            "count": self.count[idx],
            "sample_id": self.sample_ids[idx],
        }
        return data


class Slide2STDataset(Dataset):

    def __init__(
        self, data_dict, sample_ids, transform=None, ntop_genes=250, use_gene_list=None
    ):
        # sample_ids = list(data_dict.keys())
        self.sample_id_map = {sample_id: i for i, sample_id in enumerate(sample_ids)}
        self.transforms = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.CenterCrop((112, 112)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=180),
                transforms.ToTensor(),
            ]
        )

        if "img_feat" in list(list(data_dict.values())[0].keys()):
            self.img_feat = [
                torch.tensor(data_dict[sample_id]["img_feat"][:])
                for sample_id in sample_ids
            ]
        else:
            self.img_feat = []
            for sample_id in sample_ids:
                patch_list = data_dict[sample_id]["patch"][:]

                transformed_imgs = []
                for patch in tqdm(patch_list):
                    img = patch  # shape: (224, 224, 3)
                    # numpy array (H, W, C) を uint8 に変換（必要に応じて）
                    if img.dtype != np.uint8:
                        img = (
                            (img * 255).astype(np.uint8)
                            if img.max() <= 1.0
                            else img.astype(np.uint8)
                        )

                    transformed_img = self.transforms(Image.fromarray(img))
                    transformed_imgs.append(transformed_img.flatten())

                img_feat_tensor = torch.stack(
                    transformed_imgs
                )  # shape: (510, 3, 224, 224)
                self.img_feat.append(img_feat_tensor)
        self.coords = [
            data_dict[sample_id]["img_coords"][:] for sample_id in sample_ids
        ]
        self.common_genes = get_common_genes(data_dict)
        self.idx_map = idx_map2common_genes(data_dict, self.common_genes)
        self.count = [
            torch.tensor(data_dict[sample_id]["count"][:]).float()[
                :, self.idx_map[sample_id]
            ]
            for sample_id in sample_ids
        ]
        if use_gene_list is None:
            orig_exp = [
                torch.tensor(data_dict[sample_id]["exp"][:]).float()[
                    :, self.idx_map[sample_id]
                ]
                for sample_id in sample_ids
            ]
            gene_variances = torch.var(orig_exp, dim=0)
            self.use_gene_idx_list = torch.argsort(gene_variances, descending=True)[
                :ntop_genes
            ]
        else:
            common_genes = list(np.array(self.common_genes).astype(str))
            self.use_gene_idx_list = torch.tensor(
                [common_genes.index(gene) for gene in use_gene_list]
            )

        self.common_genes = [list(self.common_genes)[i] for i in self.use_gene_idx_list]
        self.sample_ids = torch.cat(
            [torch.tensor([self.sample_id_map[sample_id]]) for sample_id in sample_ids]
        )

    def __len__(self):
        return len(self.img_feat)

    def __getitem__(self, idx):
        img_data = self.img_feat[idx]

        coords = (torch.Tensor(self.coords[idx]) / 300).int()
        coords = coords - coords.min(0)[0]
        count = self.count[idx][:, self.use_gene_idx_list]
        exp = torch.log(1.0e4 * count / count.sum(dim=1, keepdims=True) + 1.0)

        data = {
            "img_feat": img_data,
            "coords": coords,
            "exp": exp,
            "count": count,
            "sample_id": self.sample_ids[idx].expand(len(self.img_feat[idx])),
        }
        return data


from torch.utils.data import Sampler
import random
from collections import defaultdict


class BalancedSampleSampler(Sampler):
    """
    患者（sample）毎に同じ数のデータポイントを含むミニバッチを作成するサンプラー
    """

    def __init__(self, dataset, batch_size, spots_per_sample=None):
        self.dataset = dataset
        self.batch_size = batch_size

        # sample_id毎にインデックスをグループ化
        self.sample_indices = defaultdict(list)
        for idx in range(len(dataset)):
            if hasattr(dataset, "dataset"):  # Subsetの場合
                sample_id = dataset.dataset.sample_ids[dataset.indices[idx]].item()
            else:  # 通常のデータセットの場合
                sample_id = dataset.sample_ids[idx].item()
            self.sample_indices[sample_id].append(idx)

        self.sample_ids = list(self.sample_indices.keys())
        self.num_samples = len(self.sample_ids)

        # 各サンプルから取得するスポット数を決定
        if spots_per_sample is None:
            # バッチサイズをサンプル数で割った数
            self.spots_per_sample = max(1, batch_size // self.num_samples)
        else:
            self.spots_per_sample = spots_per_sample

        # 実際のバッチサイズを調整
        self.actual_batch_size = self.spots_per_sample * self.num_samples

    def __iter__(self):
        while True:
            batch_indices = []

            # 各サンプルから指定数のスポットをランダムサンプリング
            for sample_id in self.sample_ids:
                available_indices = self.sample_indices[sample_id]
                if len(available_indices) >= self.spots_per_sample:
                    sampled_indices = random.sample(
                        available_indices, self.spots_per_sample
                    )
                else:
                    # 利用可能なインデックスが足りない場合は復元抽出
                    sampled_indices = random.choices(
                        available_indices, k=self.spots_per_sample
                    )
                batch_indices.extend(sampled_indices)

            yield batch_indices

    def __len__(self):
        # 各サンプルの最小スポット数に基づいて計算
        min_spots = min(len(indices) for indices in self.sample_indices.values())
        return (min_spots // self.spots_per_sample) * self.num_samples


def balanced_collate_fn(batch):
    """
    バランス取れたバッチ用のコレート関数
    """

    return {
        "img_feat": batch[0]["img_feat"],
        "exp": batch[0]["exp"],
        "count": batch[0]["count"],
        "sample_id": batch[0]["sample_id"],
    }


def create_balanced_dataloader(
    dataset, batch_size, spots_per_sample=None, num_workers=0, **kwargs
):
    """
    バランス取れたデータローダーを作成

    Args:
        dataset: データセット
        batch_size: 希望するバッチサイズ
        spots_per_sample: 各サンプルから取得するスポット数（Noneの場合自動計算）
        num_workers: ワーカー数
        **kwargs: その他のDataLoaderパラメータ

    Returns:
        DataLoader: バランス取れたデータローダー
    """
    sampler = BalancedSampleSampler(dataset, batch_size, spots_per_sample)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=balanced_collate_fn,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader


def create_imbalanced_dataloader(
    dataset, batch_size, spots_per_sample=None, num_workers=0, **kwargs
):
    """
    不均衡なデータローダーを作成

    Args:
        dataset: データセット
        batch_size: 希望するバッチサイズ
        spots_per_sample: 各サンプルから取得するスポット数（Noneの場合自動計算）
        num_workers: ワーカー数
        **kwargs: その他のDataLoaderパラメータ

    Returns:
        DataLoader: 不均衡なデータローダー
    """
    sampler = ImbalancedSampleSampler(dataset, batch_size, spots_per_sample)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=balanced_collate_fn,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader


class ImbalancedSampleSampler(Sampler):
    """
    患者（sample）毎に同じ数のデータポイントを含むミニバッチを作成するサンプラー
    """

    def __init__(self, dataset, batch_size, spots_per_sample=None):
        self.dataset = dataset
        self.batch_size = batch_size

        # sample_id毎にインデックスをグループ化
        self.sample_indices = defaultdict(list)
        for idx in range(len(dataset)):
            if hasattr(dataset, "dataset"):  # Subsetの場合
                sample_id = dataset.dataset.sample_ids[dataset.indices[idx]].item()
            else:  # 通常のデータセットの場合
                sample_id = dataset.sample_ids[idx].item()
            self.sample_indices[sample_id].append(idx)

        self.sample_ids = list(self.sample_indices.keys())
        self.num_samples = len(self.sample_ids)

        # 各サンプルから取得するスポット数を決定
        if spots_per_sample is None:
            # バッチサイズをサンプル数で割った数
            self.spots_per_sample = max(1, batch_size // self.num_samples)
        else:
            self.spots_per_sample = spots_per_sample

        # 実際のバッチサイズを調整
        self.actual_batch_size = self.spots_per_sample * self.num_samples

    def __iter__(self):
        while True:
            # 各サンプルから指定数のスポットをランダムサンプリング
            # for sample_id in self.sample_ids:
            sample_id = np.random.choice(self.sample_ids)
            available_indices = self.sample_indices[sample_id]
            if len(available_indices) >= self.batch_size:
                sampled_indices = random.sample(available_indices, self.batch_size)
            else:
                # 利用可能なインデックスが足りない場合は復元抽出
                sampled_indices = random.choices(available_indices, k=self.batch_size)

            yield sampled_indices

    def __len__(self):
        # 各サンプルの最小スポット数に基づいて計算
        min_spots = min(len(indices) for indices in self.sample_indices.values())
        return (min_spots // self.spots_per_sample) * self.num_samples


if __name__ == "__main__":
    data_dict = load_data("./dataset/hest1k/st_v2/feat/conch_v1")
    use_gene_list = np.loadtxt(
        "./dataset/hest1k/st_v2/opts/comp/highly_variable_genes_250.txt", dtype=str
    ).tolist()
    _, _, dataset = generate_datasets_sample_wise(
        data_dict,
        test_sample_ids=["SPA142", "SPA141", "SPA140", "SPA139", "SPA138", "SPA137"],
        val_sample_ids=["SPA136", "SPA135", "SPA134", "SPA133", "SPA132", "SPA131"],
        train_sample_ids=None,
        use_gene_list=use_gene_list,
    )
    # train_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=256, shuffle=True
    # )
    # imbalanced sampler
    train_dataloader = create_imbalanced_dataloader(
        dataset,
        batch_size=256,  # 希望するバッチサイズ
    )
    for data in train_dataloader:
        print(
            data["img_feat"].shape,
            data["exp"].shape,
            data["count"].shape,
            data["sample_id"],
        )
        break

    # balanced sampler
    train_dataloader = create_balanced_dataloader(
        dataset,
        batch_size=256,  # 希望するバッチサイズ
    )

    for data in train_dataloader:
        print(
            data["img_feat"].shape,
            data["exp"].shape,
            data["count"].shape,
            data["sample_id"],
        )
        break
