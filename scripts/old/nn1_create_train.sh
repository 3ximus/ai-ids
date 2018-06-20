#!/bin/bash
set -e

NN1_TRAIN_DIR="csv/train/layer1/"
NN1_EXTRACTED_TRAIN="csv/extracted/train"
FEATURES_FILE="scripts/features/all.txt"

[[ -d "${NN1_TRAIN_DIR}/individual/" ]] || mkdir -p "${NN1_TRAIN_DIR}/individual/"

if [ "$1" = "dataset" ]; then # create train from original dataset
	python scripts/data_selector.py csv/base/*.csv "${NN1_TRAIN_DIR}/individual/" -r
else
	python scripts/data_selector.py ${NN1_EXTRACTED_TRAIN}/portscan/*.csv ${NN1_EXTRACTED_TRAIN}/fastdos/*.csv ${NN1_EXTRACTED_TRAIN}/bruteforce/*.csv "${NN1_TRAIN_DIR}/individual/" -r
fi
python scripts/compact_flows.py ${NN1_TRAIN_DIR}/individual/*.csv "${NN1_TRAIN_DIR}/trainingNN1.csv" -f "$FEATURES_FILE"
rm -r "${NN1_TRAIN_DIR}/individual/"