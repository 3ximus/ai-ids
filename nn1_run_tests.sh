#!/bin/bash
set -e
# USAGE: nn1_run_tests.sh 17
if [ -z "$1" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi

for test in csv/test/${1}features/*.csv; do
	python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" "$test" "${@:2:$#}"
done
