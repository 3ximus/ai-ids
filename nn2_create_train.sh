#!/bin/bash
set -e
shopt -s extglob
# USAGE: create_nn2train.sh 9
if [ -z "$1" ] ; then
	echo "Number of features missing" 1>&2 && exit 1
fi

directory="csv/train/${1}features/layer2"
[[ -d  "$directory" ]] || mkdir -p "$directory"
find "$directory" -maxdepth 1 -type f  -exec rm '{}' \;

echo "Compacting Malign flows... (all features)"	# obtained from static dirs 'csv/train/extracted/*'
#bruteforce
cp csv/train/extracted/bruteforce/*.csv "$directory"
python scripts/compact_flows.py ${directory}/tekever-*patator.csv "${directory}" -f "scripts/features/${1}.txt" -s ".tmp"	# sendo agora compactado
for f in ${directory}/*.tmp; do
	mv ${f} ${f%.*}.csv;
done
cat $directory/tekever-*patator.csv >> $directory/tekever-bruteforce.csv
rm $directory/tekever-*patator.csv

#portscan
cp csv/train/extracted/pscan/*.csv "$directory"			# ja compactado (all features) usando todos os portscan cortados e extraídos anteriormente

#dos
cp csv/train/extracted/dos/*.csv "$directory"			# ja compactado (all features) usando todos os portscan cortados e extraídos anteriormente


echo "Filtering BENIGN..."
TMP_FILE=/tmp/compacted-benign
#python scripts/compact_flows.py csv/base/cicfl_used_format/*.csv $TMP_FILE --benign -f "scripts/features/${1}.txt"
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