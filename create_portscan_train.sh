#!/bin/bash
set -e

rm pcap/real/tekever_pcap_splitted/*
editcap -c 100000 pcap/real/others/tekever-portscan-dump.pcap pcap/real/tekever_pcap_splitted/

counter=0
for file in pcap/real/tekever_pcap_splitted/*; do
	mv "${file}" "pcap/real/tekever_pcap_splitted/portscan"${counter}
	counter=$((counter+1))
done

JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
for file in pcap/real/tekever_pcap_splitted/*; do
	cd dist/bin
	./CICFlowMeter "$file" ../../csv/train/extracted/
	cd ../..
done

cat csv/train/extracted/*.csv > csv/train/extracted/tekever-portscan-train.csv
rm csv/train/extracted/portscan*.csv

echo "$(tail -n +2 csv/train/extracted/tekever-portscan-train.csv)" > csv/train/extracted/tekever-portscan-train.csv
python scripts/compact_flows.py csv/base/cicfl_used_format/*.csv /tmp/compacted-benign --benign -f "scripts/features/all.txt"

shuf /tmp/compacted-benign -n$(wc -l < csv/train/allfeatures/layer2/PortScan.csv) > $(echo csv/train/allfeatures/layer2/PortScan.csv | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
cat csv/train/allfeatures/layer2/PortScan.csv >> $(echo csv/train/allfeatures/layer2/PortScan.csv | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
rm csv/train/allfeatures/layer2/PortScan.csv