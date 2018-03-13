#!/bin/bash
set -e
# arg1: abspath to pcap file, arg2: label, arg3: how many features or 'all'
if [ -z "$1" ] ; then
	echo "Pcap test-file missing" 1>&2 && exit 1
elif [ -z "$2" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi
cd dist/bin
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
filename=$(basename "$1")
name=${filename%.*}
./CICFlowMeter "$1" ../../csv/test/extracted/
cd ../..
[[ -d "csv/test/${2}features/" ]] || mkdir "csv/test/${2}features/"
if [ "$3" == "BENIGN" ] ; then
	python scripts/compact_flows.py "csv/test/extracted/${name}.csv" "csv/test/${2}features/${name}.csv" -f "scripts/features/${2}.txt" --benign
else
	python scripts/compact_flows.py "csv/test/extracted/${name}.csv" "csv/test/${2}features/${name}.csv" -f "scripts/features/${2}.txt"
fi
python classifiers/layer2-classifier.py "csv/train/${2}features/benign-individual/benign-dos-attack.csv"  "csv/test/${2}features/${name}.csv"
