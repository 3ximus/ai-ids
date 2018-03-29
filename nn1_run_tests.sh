#!/bin/bash
set -e
# USAGE: nn1_run_tests.sh 17
if [ -z "$1" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi

echo "Malign:" 1>&2
for file in csv/test/${1}features/dos*.csv; do
	python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" ${file}
done
python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" "csv/test/${1}features/portscan-nmap-5min.csv"
python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" "csv/test/${1}features/portscan-nmap-23min-alloptions.csv"
python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" "csv/test/${1}features/ftp-patator-5min.csv"
python classifiers/layer1-classifier.py "csv/train/${1}features/layer1/trainingNN1.csv" "csv/test/${1}features/ssh-patator-5min.csv"
