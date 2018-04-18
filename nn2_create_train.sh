#!/bin/bash
set -e
shopt -s extglob

NN2_EXTRACTED_TRAIN="csv/extracted/train"
directory="csv/train/layer2"
#find "$directory" -maxdepth 1 -type f  -exec rm '{}' \;

echo "Compacting Malign flows..."	# obtained from static dirs ${NN2_EXTRACTED_TRAIN}/*

#bruteforce
#python scripts/compact_flows.py ${NN2_EXTRACTED_TRAIN}/bruteforce/*.csv "$directory/tekever-bruteforce.csv" -f "scripts/features/all.txt"

#fastdos
#python scripts/compact_flows.py ${NN2_EXTRACTED_TRAIN}/fastdos/*.csv "$directory/tekever-fastdos.csv" -f "scripts/features/all.txt"

#portscan
#python scripts/compact_flows.py ${NN2_EXTRACTED_TRAIN}/portscan/tekever-portscan.csv "$directory/tekever-portscan.csv.tmp" -f "scripts/features/all.txt"
#head -n 60000 "${directory}/tekever-portscan.csv.tmp" > "$directory/tekever-portscan.csv"

echo "Filtering BENIGN..."
TMP_FILE=/tmp/compacted-benign
[[ -f $TMP_FILE ]] || python scripts/compact_flows.py csv/base/*.csv $TMP_FILE --benign -f "scripts/features/all.txt"

echo "Shuffling..."
for file in ${directory}/*.csv; do
	echo "$file"
	if [ $(wc -l < "${file}") -gt $(wc -l < "${TMP_FILE}") ];then
		shuf "$file" -n$(wc -l < "$TMP_FILE") > $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
		cat "$TMP_FILE" >> $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
	else
		shuf "$TMP_FILE" -n$(wc -l < "${file}") > $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
		cat "$file" >> $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
	fi
done

rm ${directory}/!(benign*.csv)
