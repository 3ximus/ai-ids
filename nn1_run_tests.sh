#!/bin/bash
set -e
# USAGE: nn1_run_tests.sh 17
for test in csv/test/compacted/${1}features/*.test; do
	python classifiers/layer1-classifier.py csv/train/${1}features/layer1/trainingNN1.csv $test
done