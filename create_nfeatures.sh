#!/bin/bash
set -e
mkdir -p csv/selected-compacted-datasets/{21features/distributed/individual,21features/raw/individual,21features/raw/benign}
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/Malign-Dataset.csv -f21
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/21features/raw/benign/Benign-Dataset.csv -f21 --benign
python scripts/utils/nn1_data_selector.py csv/selected-compacted-datasets/21features/raw/Malign-Dataset.csv csv csv/selected-compacted-datasets/21features/distributed/individual/DDoS.csv 
python scripts/utils/nn1_createTrain.py
python scripts/utils/nn1_createTests.py
mv csv/selected-compacted-datasets/21features/raw/individual/*.test csv/selected-compacted-datasets/21features/distributed/individual/
