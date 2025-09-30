import subprocess

# targets = ["task_1", "task_2", "task_3", "task_5", "task_6", "task_7", "task_9"]
targets = ["task_1"]
# # targets = [
# #     'breat_xenium'
# ]
val_id_dict = {
    "task_1": "TENX95",
    "task_5": "TENX111",
    "task_6": "ZEN36",
    "task_9": "NCBI681",
    "task_2": "MEND139",
    "task_7": "INT1",
    "task_3": "TENX116",
    "breat_xenium": '"TENX95,TENX97"',
    "breat_visium": '"TENX13,TENX14"',
}
test_id_dict = {
    "task_1": "NCBI783",
    "task_5": "TENX147",
    "task_6": "ZEN40",
    "task_9": "NCBI682",
    "task_2": "MEND140",
    "task_3": "TENX126",
    "task_7": "INT2",
    "breat_xenium": '"TENX94,TENX96"',
    "breat_visium": "TENX53",
}
for target in targets:
    subprocess.run(f"python ./preprocessing/download_hest_benchmarks.py {target}",shell=True)
    for feature in ["conch_v1", "dinov2"]:
        subprocess.run(f"bash ./scripts/prepare_features.sh ./dataset/hest1k/{target} {feature}",shell=True)
        subprocess.run(
            f"bash ./scripts/comp_eval_general_id.sh comp_{feature} ./dataset/hest1k/{target} {feature} {val_id_dict[target]} {test_id_dict[target]} linear 50",
            shell=True,
        )
