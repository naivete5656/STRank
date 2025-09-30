
# Base directories
COMP_NAME=$1
BASE_DIR=$2
IMG_MODEL=$3
VAL_IDS=$4
TEST_IDS=$5
PRED_MODEL=$6
GENE_NUM=$7
MEAN_OPTION=$8

DATA_DIR="${BASE_DIR}/feat/${IMG_MODEL}"
OUT_DIR="${BASE_DIR}/opts/${COMP_NAME}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}"
# Select genes
USE_GENE_PATH="${OUT_DIR}/highly_variable_genes.txt"
PYTHONPATH=./strank/ \
python ./script/export_highly_variable.py \
    "${DATA_DIR}" \
    "${USE_GENE_PATH}" \
    --ntop_genes ${GENE_NUM} ${MEAN_OPTION} 
# Specify sample IDs
META="${BASE_DIR}/meta.csv"
# Split TEST_IDS and VAL_IDS by comma
IFS=',' read -ra TEST_IDS <<< "${TEST_IDS}"
IFS=',' read -ra VAL_IDS <<< "${VAL_IDS}"

echo "Validation IDs: ${VAL_IDS[@]}"
echo "Test IDs: ${TEST_IDS[@]}"
echo "Training IDs: ${TRAIN_IDS[@]}"
LOSS_NAMES=(stranklist strankg pearsona pearsong strankgfw stranka mse nb poisson ranking)
# LOSS_NAMES=(strank32)
# LOSS_NAMES=(strankgfw pearsong)
for loss in "${LOSS_NAMES[@]}"; do
    echo "=== Training with loss: ${loss} ==="
    # 1. Train the model
    python ./strank/train.py \
        --data_dir "${DATA_DIR}" \
        --param_path "${OUT_DIR}/${loss}/opt_param.pt" \
        --test_sample_ids "${TEST_IDS[@]}" \
        --val_sample_ids "${VAL_IDS[@]}" \
        --log_dir "${LOG_DIR}" \
        --loss "${loss}" \
        --model "${PRED_MODEL}" \
        --max_epochs 1000 \
        --use_gene "${USE_GENE_PATH}" \
        --ngpu 1
    # 2. Calculate Spearman correlation for the trained model
    #    - This assumes you have `calculate_spearman.py` accessible.
    #    - Adjust the path to the script if needed.
    echo "=== Calculating Spearman correlation for loss: ${loss} ==="
    SPEARMAN_CSV="${OUT_DIR}/${loss}/corr_results.csv"
    python ./strank/evaluation.py \
        --data_dir "${DATA_DIR}" \
        --param_path "${OUT_DIR}/${loss}/opt_param.pt" \
        --sample_ids "${TEST_IDS[@]}" \
        --model "${PRED_MODEL}" \
        --loss "${loss}" \
        --batch_size 1024 \
        --use_gene "${USE_GENE_PATH}" \
        --output_csv "${SPEARMAN_CSV}"

    echo "Spearman CSV saved to: ${SPEARMAN_CSV}"
    echo "============================================"
done