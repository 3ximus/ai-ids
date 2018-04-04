#!/bin/bash
set -e
#usage: ./ids.sh <pcap full path> <benign(b)|malign(m)|unkown(u)>

pcap_filename=$(basename ${1})
csv_filename=${pcap_filename%.*}.csv

#dataset conversion
JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
cd dist/bin
./CICFlowMeter "$1" ../../csv/test/extracted/others/ &>/dev/null
cd ../..

[[ -d "csv/test/allfeatures/others/" ]] || mkdir "csv/test/allfeatures/others/"
[[ -d "csv/test/17features/others/" ]] || mkdir "csv/test/17features/others/"

if [ "$2" == "m" ] || [ "$2" == "u" ] ; then
	python scripts/compact_flows.py "csv/test/extracted/others/$csv_filename" "csv/test/allfeatures/others/" -f "scripts/features/all.txt"
	python scripts/compact_flows.py "csv/test/extracted/others/$csv_filename" "csv/test/17features/others/" -f "scripts/features/17.txt"
elif [ "$2" == "b" ] ; then
	python scripts/compact_flows.py "csv/test/extracted/others/$csv_filename" "csv/test/allfeatures/others/" -f "scripts/features/all.txt" --benign
	python scripts/compact_flows.py "csv/test/extracted/others/$csv_filename" "csv/test/17features/others/" -f "scripts/features/17.txt" --benign
else
	echo "Error. Exiting"
	exit 1
fi

#classification
python classifiers/ids.py "csv/test/17features/others/$csv_filename" "csv/test/allfeatures/others/$csv_filename"

rm "csv/test/17features/others/$csv_filename" "csv/test/allfeatures/others/$csv_filename"