#!/bin/bash
set -e
cd csv/selected-compacted-datasets
mkdir -p 21features/distributed/individual 21features/raw/individual 21features/raw/benign
cd ../..
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/Malign-Dataset.csv -f21 --csv
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/benign/Benign-Dataset.csv -f21 --csv --benign
python scripts/utils/nn1_individualize.py --default
python scripts/utils/nn1_distribute.py --default
python scripts/utils/nn1_createTrain.py
python scripts/utils/nn1_createTests.py
mv csv/selected-compacted-datasets/21features/raw/individual/*.test csv/selected-compacted-datasets/21features/distributed/individual/