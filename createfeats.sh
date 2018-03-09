#!/bin/bash
set -e
if [ -z "$1" ] ; then
	echo "Number of features missing"
	exit 1
fi

cd csv/selected-compacted-datasets
mkdir -p ${1}features/distributed/individual ${1}features/raw/individual ${1}features/raw/benign
cd ../..
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/${1}features/raw/Malign-Dataset.csv -f${1}
python scripts/utils/compact_flows.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/${1}features/raw/benign/Benign-Dataset.csv -f${1} --benign
python scripts/utils/fast_csv_individualize.py --default
python scripts/utils/fast_csv_distribute.py --default
python scripts/utils/fast_csv_createTrain.py
python scripts/utils/fast_csv_createTests.py
mv csv/selected-compacted-datasets/${1}features/raw/individual/*.test csv/selected-compacted-datasets/${1}features/distributed/individual/
