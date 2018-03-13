#!/bin/bash
set -e
# USAGE: create_nn2train.sh 9
directory=csv/selected-compacted-datasets/${1}features
[[ -d "${directory}/layer2/benign-individual7" ]] || mkdir ${directory}/layer2/benign-individual7
if find ${directory}/layer2/benign-individual/ -mindepth 1 | read; then
	rm ${directory}/layer2/benign-individual/*.csv
fi
python scripts/nn1_data_selector.py csv/base-datasets/cicfl_used_format/*.csv csv/selected-compacted-datasets/${1}features/layer2/benign-individual/ -i
#for file in ${directory}/layer2/benign-individual/*.csv; do
#	filename=$(basename ${file})
#	name=${filename%.*}
#	head -n $(wc -l < ${file}) ${directory}/layer2/Benign-Dataset.csv >> ${file}
#	mv ${file} ${directory}/layer2/benign-individual/benign-${name,,}.csv
#done

#dos
#for file in ${directory}/distributed/benign-individual/benign-d*.csv; do#
#	cat $file >> ${directory}/distributed/benign-individual/benign-dos-attack.csv
#	rm $file
#done
