#!/bin/bash
set -e
shopt -s extglob

directory="csv/train/layer2"
[[ -d  "$directory" ]] || mkdir -p "$directory"
find "$directory" -maxdepth 1 -type f  -exec rm '{}' \;

echo "Compacting Malign flows..."	# obtained from static dirs 'csv/train/extracted/*'

#bruteforce
python scripts/compact_flows.py csv/train/extracted/bruteforce/*.csv "${directory}" -f "scripts/features/all.txt"
cat $directory/tekever-*patator.csv > $directory/tekever-bruteforce.csv
rm $directory/tekever-*patator.csv

#dos
python scripts/compact_flows.py csv/train/extracted/dos/*.csv "${directory}" -f "scripts/features/all.txt"
cat $directory/tekever-dos-*.csv > $directory/tekever-dos.csv
rm $directory/tekever-dos-*.csv

#portscan
python scripts/compact_flows.py csv/train/extracted/pscan/tekever-portscan.csv "${directory}" -f "scripts/features/all.txt"
#cp csv/train/extracted/pscan/tekever-portscan-2.csv "${directory}/tekever-portscan.csv"

echo "Filtering BENIGN..."
TMP_FILE=/tmp/compacted-benign
python scripts/compact_flows.py csv/base/cicfl_used_format/*.csv $TMP_FILE --benign -f "scripts/features/all.txt"

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

#rm $TMP_FILE
rm ${directory}/!(benign*.csv)