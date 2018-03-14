#!/bin/bash
set -e
# USAGE: nn1_create_train.sh 17
if [ -z "$1" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi

[[ -d "csv/train/${1}features/layer1/individual/" ]] || mkdir -p "csv/train/${1}features/layer1/individual/"
python scripts/data_selector.py csv/base/cicfl_used_format/*.csv "csv/train/${1}features/layer1/individual/" -r
python scripts/compact_flows.py csv/train/${1}features/layer1/individual/*.csv "csv/train/${1}features/layer1/trainingNN1.csv" -f "scripts/features/${1}.txt"
