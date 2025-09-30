import pandas as pd
import sys

import dotenv

dotenv.load_dotenv()

task_id = sys.argv[1]

df = pd.read_csv("./dataset/hest1k/HEST_v1_1_0.csv")

from huggingface_hub import login


import os

login(os.environ["HUGGINGFACE_TOKEN"])

import datasets

df = pd.read_csv("./dataset/hest1k/HEST_v1_1_0.csv")
# breast_st_ids = df[(df["organ"] == "Breast") & (df["st_technology"] == "Spatial Transcriptomics")].id.tolist()
stnet = df[
    (df["organ"] == "Breast")
    & (df["st_technology"] == "Spatial Transcriptomics")
    & (df["oncotree_code"] == "IDC")
    & (
        df["dataset_title"]
        == "Integrating spatial gene expression and breast tumour morphology via deep learning"
    )
].id.tolist()
breast_st_ids = df[
    (df["organ"] == "Breast")
    & (df["st_technology"] == "Spatial Transcriptomics")
    & (df["patient"].str.contains("Patient", na=False))
].id.tolist()

# task3 xenium
tasks_id_dict = {
    "task_1": ["TENX95", "TENX99", "NCBI783", "NCBI785"],
    "task_3": ["TENX116", "TENX126", "TENX140"],
    "task_5": ["TENX111", "TENX147", "TENX148", "TENX149"],
    "task_6": ["ZEN36", "ZEN40", "ZEN48", "ZEN49"],
    "task_9": ["NCBI681", "NCBI682", "NCBI683", "NCBI684"],
    "task_2": [f"MEND{i}" for i in range(139, 163)],
    "task_7": [f"INT{i}" for i in range(1, 25)],
    "st_v2": breast_st_ids,
    "st_v3": stnet,
}


def download_hest_data(ids_to_query, local_dir):
    list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
    dataset = datasets.load_dataset(
        "MahmoodLab/hest", cache_dir=local_dir, patterns=list_patterns
    )
    sub_meta_df = df.query("id in @ids_to_query")
    sub_meta_df.to_csv(f"{local_dir}/meta.csv", index=False)
    sub_meta_df.patient.value_counts().to_csv(f"{local_dir}/patient_counts.csv")


local_dir = f"./dataset/hest1k/{task_id}"
download_hest_data(tasks_id_dict[task_id], local_dir)
