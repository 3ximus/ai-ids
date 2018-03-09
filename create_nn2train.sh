#!/bin/bash
set -e
directory=csv/selected-compacted-datasets/9features
if find ${directory}/distributed/benign-individual/ -mindepth 1 | read; then
	rm ${directory}/distributed/benign-individual/*.csv
fi
cp ${directory}/raw/individual/*.csv ${directory}/distributed/benign-individual/
for file in ${directory}/distributed/benign-individual/*.csv; do
	filename=$(basename ${file})
	name=${filename%.*}
	if [[ name = *"dos"* ]]; then
  		head -n $(wc -l < ${file}) ${directory}/raw/benign/Benign-Dataset.csv >> ${file}
		mv ${file} ${directory}/distributed/benign-individual/benign-dos.csv
	else
		head -n $(wc -l < ${file}) ${directory}/raw/benign/Benign-Dataset.csv >> ${file}
		mv ${file} ${directory}/distributed/benign-individual/benign-${name,,}.csv
	fi
done