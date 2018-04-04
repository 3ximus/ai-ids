#!/bin/bash
set -e

NN1_TRAIN_DIR="csv/train/layer1/"
FEATURES_FILE="scripts/features/17.txt"

[[ -d "${NN1_TRAIN_DIR}/individual/" ]] || mkdir -p "${NN1_TRAIN_DIR}/individual/"
python scripts/data_selector.py csv/base/cicfl_used_format/*.csv "${NN1_TRAIN_DIR}/individual/" -r
python scripts/compact_flows.py ${NN1_TRAIN_DIR}/individual/*.csv "${NN1_TRAIN_DIR}/trainingNN1.csv" -f "$FEATURES_FILE"
rm -r "${NN1_TRAIN_DIR}/individual/"
