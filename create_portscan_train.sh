#!/bin/bash
set -e
shopt -s extglob


TMP_DIR="/tmp/portscanfiles"
mkdir -p $TMP_DIR
find "$TMP_DIR" -maxdepth 1 -type f  -exec rm '{}' \;
editcap -c 100000 pcap/real/others/tekever-portscan-dump.pcap $TMP_DIR/

counter=0
for file in $TMP_DIR/* ; do
	mv "${file}" "${TMP_DIR}/portscan${counter}"
	counter=$((counter+1))
done

JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
[[ -d  "$directory" ]] || mkdir "$TMP_DIR/extracted"
for file in $TMP_DIR/portscan*; do
	cd dist/bin
	./CICFlowMeter "$file" "$TMP_DIR/extracted"
	cd ../..
done

python scripts/compact_flows.py $TMP_DIR/extracted/!(tekever-portscan-train.csv) $TMP_DIR/extracted/tekever-portscan-train.csv -f "scripts/features/all.txt"
mv $TMP_DIR/extracted/tekever-portscan-train.csv csv/base/tekever/
rm -r $TMP_DIR
