#!/bin/bash
set -e
#usage: ./ids.sh <pcap full path> <benign(b)|malign(m)|unkown(u)>

pcap_filename=$(basename ${1})
csv_filename=${pcap_filename%.*}.csv

#dataset conversion
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
cd dist/bin
./CICFlowMeter "$1" ../../csv/extracted/test/
cd ../..

[[ -d "csv/test/malign" ]] || mkdir "csv/test/malign"
[[ -d "csv/test/benign" ]] || mkdir "csv/test/benign"

if [ "$2" == "m" ] || [ "$2" == "u" ] ; then
	python scripts/compact_flows.py "csv/extracted/test/$csv_filename" "csv/test/malign/" -f "scripts/features/all.txt"
elif [ "$2" == "b" ] ; then
	python scripts/compact_flows.py "csv/extracted/test/$csv_filename" "csv/test/benign/" -f "scripts/features/all.txt" --benign
else
	echo "Give option <benign(b)|malign(m)|unkown(u)>"
	exit 1
fi

#classification
python classifiers/ids.py "csv/test/malign/$csv_filename"

rm "csv/test/malign/$csv_filename"
