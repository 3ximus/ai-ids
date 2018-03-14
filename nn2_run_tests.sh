#!/bin/bash
set -e
# USAGE: nn2_run_tests.sh 9
if [ -z "$1" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi

for file in csv/test/${1}features/dos*.csv; do
	python classifiers/layer2-classifier.py "csv/train/${1}features/layer2/benign-DoS-Attack.csv" ${file}
done
python classifiers/layer2-classifier.py "csv/train/${1}features/layer2/benign-PortScan.csv" "csv/test/${1}features/portscan-nmap-5min.csv"
python classifiers/layer2-classifier.py "csv/train/${1}features/layer2/benign-FTP-Patator.csv" "csv/test/${1}features/ftp-patator-5min.csv"
python classifiers/layer2-classifier.py "csv/train/${1}features/layer2/benign-SSH-Patator.csv" "csv/test/${1}features/ssh-patator-5min.csv"

