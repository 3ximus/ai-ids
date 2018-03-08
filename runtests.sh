#!/bin/bash
set -e
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
python scripts/utils/compact_flows.py csv/real-datasets/extracted/${name}.csv csv/real-datasets/compacted/9features/${name}.test --csv
python scripts/simple-nn.py csv/selected-compacted-datasets/9features/distributed/training.csv  csv/real-datasets/compacted/9features/${name}.test
