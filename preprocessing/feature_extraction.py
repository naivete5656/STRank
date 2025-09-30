# ------------------------------------------------------------------------------
# Copyright (c) 2007 Free Software Foundation, Inc. https://fsf.org/
# Licensed under the GPL-3.0 License.
# The code is based on CLAM.
# (https://github.com/mahmoodlab/CLAM)
# ------------------------------------------------------------------------------

from pathlib import Path
import time
import argparse
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import timm
from torch.utils.data import DataLoader
from PIL import Image
import h5py
import openslide
from tqdm import tqdm

import numpy as np

import sys
import dotenv

dotenv.load_dotenv()

sys.path.append("../preprocessing/utils")
from utils.file_utils import save_hdf5
from models import get_encoder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_w_loader(output_path, loader, model, gene_list, verbose=0):
    """
    args:
            output_path: directory to save computed features (.h5 file)
            model: pytorch model
            verbose: level of feedback
    """
    if verbose > 0:
        print(f"processing a total of {len(loader)} batches".format(len(loader)))

    mode = "w"
    asset_dict = {"gene_list": np.array(gene_list)}
    attr_dict = {"gene_list": {"dtype": h5py.string_dtype()}}
    with h5py.File(output_path, mode) as f:
        f.create_dataset("gene_list", data=np.array(gene_list, dtype=h5py.string_dtype()))
        init_daata = loader.dataset[0]
        dtype_dict = {
            "img": np.float32,
            "coords": np.int32,
            "exp": np.float32,
            "count": np.float32,
            "barcode": h5py.string_dtype(),
        }

        for count, data in enumerate(tqdm(loader)):
            with torch.inference_mode():
                batch = data["img"]
                coords = data["coords"].numpy().astype(np.int32)
                batch = batch.to(device, non_blocking=True)
                features = model(batch)
                features = features.cpu().numpy().astype(np.float32)
                if count == 0:
                    f.create_dataset(
                        "features",
                        data=features,
                        maxshape=(None,) + features.shape[1:],
                        dtype=np.float32,
                    )
                else:
                    f["features"].resize(f["features"].shape[0] + features.shape[0], axis=0)
                    f["features"][-features.shape[0] :] = features
                for key in ["barcode", "exp", "count", "coords"]:
                    val = np.array(data[key])
                    if count == 0:
                        if key == "barcode":
                            f.create_dataset(
                                key,
                                data=list(val),
                                maxshape=(None,) + val.shape[1:],
                                dtype=dtype_dict[key],
                            )
                        else:
                            f.create_dataset(
                                key,
                                data=val,
                                maxshape=(None,) + val.shape[1:],
                                dtype=dtype_dict[key],
                            )
                    else:
                        f[key].resize(f[key].shape[0] + val.shape[0], axis=0)
                        f[key][-val.shape[0] :] = val
    return output_path


class PatchDataLoader(Dataset):
    def __init__(self, data_dir, img_transforms):
        super().__init__()

        with data_dir.open("rb") as f:
            data = pickle.load(f)
        self.patch = data["patch"]
        self.coords = data["img_coords"]
        self.exp = data["exp"]
        self.count = data["count"]
        self.gene_list = data["gene_list"]
        self.barcode = data["barcode"]
        self.roi_transforms = img_transforms

    def __len__(self):
        return len(self.patch)

    def __getitem__(self, idx):
        img = Image.fromarray(self.patch[idx])
        img = self.roi_transforms(img)
        coord = self.coords[idx]
        exp = self.exp[idx].toarray().flatten()
        count = self.count[idx].toarray().flatten()
        barcode = self.barcode[idx]
        return {
            "img": img,
            "coords": coord,
            "exp": exp,
            "count": count,
            "barcode": barcode,
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument(
        "--model_name",
        type=str,
        default="densenet121",
        choices=["resnet50", "resnet50_trunc", "uni_v1", "conch_v1", "hopt", "dinov2", "densenet121"],
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--no_auto_skip", default=False, action="store_true")
    parser.add_argument("--target_patch_size", type=int, default=224)
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save computed features",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .pkl files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("initializing dataset")

    model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)

    _ = model.eval()
    model = model.to(device)

    loader_kwargs = {"num_workers": 8, "pin_memory": True} if device.type == "cuda" else {}

    save_dir = Path(args.save_dir)
    h5_file_dir = save_dir
    h5_file_dir.mkdir(parents=True, exist_ok=True)
    data_dirs = sorted(Path(args.input_dir).glob("*.pkl"))
    total = len(data_dirs)

    for bag_candidate_idx, data_dir in enumerate(data_dirs):
        print("\nprogress: {}/{}".format(bag_candidate_idx, total))
        print(data_dir.stem)

        h5_file_path = Path(f"{h5_file_dir}/{data_dir.stem}.h5")
        # if h5_file_path.exists():
        #     print("skipped {}".format(data_dir.stem))
        #     continue

        time_start = time.time()
        dataset = PatchDataLoader(data_dir, img_transforms)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
        output_file_path = compute_w_loader(
            h5_file_path,
            loader=loader,
            model=model,
            gene_list=dataset.gene_list,
            verbose=1,
        )

        time_elapsed = time.time() - time_start
        print("\ncomputing features for {} took {} s".format(output_file_path, time_elapsed))

        with h5py.File(output_file_path, "r") as file:
            features = file["features"][:]
            print("features size: ", features.shape)
            print("coordinates size: ", file["coords"].shape)

        # features = torch.from_numpy(features)
        # bag_base, _ = os.path.splitext(bag_name)
        # torch.save(features, os.path.join(args.feat_dir, "pt_files", bag_base + ".pt"))
