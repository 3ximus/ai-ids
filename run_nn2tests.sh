#!/bin/bash
set -e
# arg1: abspath to pcap file, arg2: label, arg3: how many features or 'all'
if [ -z "$1" ] ; then
	echo "Pcap test-file missing"
	exit 1
elif [ -z "$2" ] ; then
	echo "Number of features missing"
	exit 1
fi
cd dist/bin
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
filename=$(basename $1)
name=${filename%.*}
./CICFlowMeter $1 ../../csv/real-datasets/extracted/
cd ../..
[[ -d "csv/real-datasets/compacted/${2}features/" ]] || mkdir csv/real-datasets/compacted/${2}features/
if [ "$3" == "BENIGN" ] ; then
	python scripts/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${2}features/${name}.test -f${2} --benign
else
	python scripts/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${2}features/${name}.test -f${2}
fi
python classifiers/layer2-classifier.py csv/selected-compacted-datasets/${2}features/benign-individual/benign-dos-attack.csv  csv/real-datasets/compacted/${2}features/${name}.test
