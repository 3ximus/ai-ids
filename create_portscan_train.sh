#!/bin/bash
set -e
shopt -s extglob

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

python scripts/compact_flows.py csv/train/extracted/!(tekever-portscan-train.csv) csv/train/extracted/tekever-portscan-train.csv -f "scripts/features/all.txt"
rm csv/train/extracted/!(tekever-portscan-train.csv)

cp csv/train/extracted/tekever-portscan-train.csv csv/train/allfeatures/layer2/PortScan.csv
python scripts/compact_flows.py csv/base/cicfl_used_format/*.csv /tmp/compacted-benign --benign -f "scripts/features/all.txt"

shuf /tmp/compacted-benign -n$(wc -l < csv/train/allfeatures/layer2/PortScan.csv) > $(echo csv/train/allfeatures/layer2/PortScan.csv | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
cat csv/train/allfeatures/layer2/PortScan.csv >> $(echo csv/train/allfeatures/layer2/PortScan.csv | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
rm csv/train/allfeatures/layer2/PortScan.csv
