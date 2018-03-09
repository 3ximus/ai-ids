#!/bin/bash
set -e
# arg1: abspath to pcap file, arg2: label, arg3: how many features or 'all'

if [ -z "$2" ] ; then
	exit 1
fi
cd dist/bin
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
filename=$(basename $1)
name=${filename%.*}
./CICFlowMeter $1 ../../csv/real-datasets/extracted/
cd ../..
sed -i "s/No Label/$2/" csv/real-datasets/extracted/${name}.csv
python scripts/utils/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/${3}features/${name}.test -f${3} --csv
python scripts/layer1-classifier.py csv/selected-compacted-datasets/${3}features/distributed/training.csv  csv/real-datasets/compacted/${3}features/${name}.test
