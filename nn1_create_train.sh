#!/bin/bash
set -e
# USAGE: nn1_create_train.sh 17
python scripts/nn1_data_selector.py csv/base/cicfl_used_format/*.csv csv/train/${1}features/layer1/individual/ -r
python scripts/compact_flows.py csv/train/${1}features/layer1/individual/*.csv csv/train/${1}features/layer1/trainingNN1.csv -f scripts/features/${1}.txt