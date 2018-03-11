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
filename=$(basename $1)
name=${filename%.*}
./CICFlowMeter $1 ../../csv/real-datasets/extracted/
cd ../..
python scripts/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${2}features/${name}.test -f${2}
python classifiers/layer1-classifier.py csv/selected-compacted-datasets/${2}features/distributed/trainingNN1.csv  csv/real-datasets/compacted/${2}features/${name}.test
