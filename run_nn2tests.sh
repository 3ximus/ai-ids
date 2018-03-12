#!/bin/bash
set -e
# arg1: abspath to pcap file, arg2: label, arg3: how many features or 'all'
if [ -z "$1" ] ; then
	echo "Pcap test-file missing"
	exit 1
fi
if [ -z "$2" ] ; then
	echo "Label missing"
	exit 1
fi
if [ -z "$3" ] ; then
	echo "Number of features missing"
	exit 1
fi
cd dist/bin
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
filename=$(basename $1)
name=${filename%.*}
./CICFlowMeter $1 ../../csv/real-datasets/extracted/
cd ../..
sed -i "s/No Label/$2/" csv/real-datasets/extracted/${name}.csv
if [ "$2" == "BENIGN" ] ; then
	python scripts/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${3}features/${name}.test -f${3} --benign
else
	python scripts/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${3}features/${name}.test -f${3}
fi
python classifiers/layer2-classifier.py csv/selected-compacted-datasets/${3}features/distributed/benign-individual/benign-dos-attack.csv  csv/real-datasets/compacted/${3}features/${name}.test
