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

echo "Selecting Malign flows..."
python scripts/data_selector.py csv/base/cicfl_used_format/*.csv "$directory" -i
echo "Compacting Malign flows..."
python scripts/compact_flows.py ${directory}/*.csv "${directory}" -f "scripts/features/${1}.txt" -s ".tmp"
for f in ${directory}/*.tmp; do
	mv ${f} ${f%.*}.csv;
done


echo "Filtering BENIGN..."
TMP_FILE=/tmp/compacted-benign
python scripts/compact_flows.py csv/base/cicfl_used_format/*.csv $TMP_FILE --benign -f "scripts/features/${1}.txt"
echo "Shuffling..."
for file in ${directory}/*.csv; do
	echo "$file"
	shuf $TMP_FILE -n$(wc -l < "${file}") > $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
	cat "$file" >> $(echo "$file" | sed 's@\(.*\)\/\(.*\)@\1/benign-\2@')
done
rm $TMP_FILE

# dos
cat ${directory}/benign-D*.csv > "${directory}/benign-DoS-Attack.csv"
rm ${directory}/!(benign*.csv) ${directory}/benign-D!(oS-Attack.csv)

