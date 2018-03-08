#!/bin/bash
set -e
cd csv/selected-compacted-datasets
mkdir -p 21features/distributed/individual 21features/raw/individual 21features/raw/benign
cd ../..
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/Malign-Dataset.csv --csv
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/benign/Benign-Dataset.csv --csv --benign
python scripts/utils/fast_csv_individualize.py --default
python scripts/utils/fast_csv_distribute.py --default
python scripts/utils/fast_csv_createTrain.py
python scripts/utils/fast_csv_createTests.py
mv csv/selected-compacted-datasets/21features/raw/individual/*.test csv/selected-compacted-datasets/21features/distributed/individual/