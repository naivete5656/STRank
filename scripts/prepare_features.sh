
# Base directories
BASE_DIR=$1
MODEL=$2
PAIRED_DIR="${BASE_DIR}/paired_data"
FEAT_DIR="${BASE_DIR}/feat/${MODEL}"
python ./preprocessing/make_paired.py ${BASE_DIR}
python ./preprocessing/feature_extraction.py --model_name ${MODEL} --save_dir ${FEAT_DIR} --input_dir ${PAIRED_DIR}

