#!/bin/bash
set -e
shopt -s extglob

directory="csv/train/layer2"
[[ -d  "$directory" ]] || mkdir -p "$directory"
find "$directory" -maxdepth 1 -type f  -exec rm '{}' \;

echo "Compacting Malign flows..."	# obtained from static dirs 'csv/train/extracted/*'

#bruteforce
cp csv/train/extracted/bruteforce/*.csv "$directory"
python scripts/compact_flows.py ${directory}/tekever-*patator.csv "${directory}" -f "scripts/features/all.txt" -s ".tmp"	# sendo agora compactado
for f in ${directory}/*.tmp; do
	mv ${f} ${f%.*}.csv;
done
cat $directory/tekever-*patator.csv >> $directory/tekever-bruteforce.csv
rm $directory/tekever-*patator.csv

#portscan
cp csv/train/extracted/pscan/*.csv "$directory"			# ja compactado (all features) usando todos os portscan cortados e extraídos anteriormente

#dos
cp csv/train/extracted/dos/*.csv "$directory"

cp csv/base/tekever/tekever-portscan-train.csv ${directory}/PortScan.csv

echo "Filtering BENIGN..."
TMP_FILE=/tmp/compacted-benign
#python scripts/compact_flows.py csv/train/extracted/benign/*.csv $TMP_FILE --benign -f "scripts/features/all.txt"
echo "Shuffling..."
for file in ${directory}/*.csv; do
	echo "$file"
	shuf $TMP_FILE -n$(wc -l < "${file}") > $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
	cat "$file" >> $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
done

#rm $TMP_FILE

# dos
#cat ${directory}/benign-D*.csv > "${directory}/benign-DoS-Attack.csv"
rm ${directory}/!(benign*.csv) #${directory}/benign-*Patator.csv ${directory}/benign-D!(oS-Attack.csv)
