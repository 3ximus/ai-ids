#!/bin/bash
set -e
[[ -d results ]] || mkdir -p results/
outfile=results/results-$(date +%s)
echo "MALIGN" | tee -a "${outfile}"
for file in csv/test/malign/*.csv; do python classifiers/ids.py $file | grep RESULTS -A 3 | tee -a "${outfile}";done
echo "BENIGN" | tee -a "${outfile}"
for file in csv/test/benign/*.csv; do python classifiers/ids.py $file | grep RESULTS -A 3 | tee -a "${outfile}";done
echo >> "${outfile}"
cat classifiers/options.cfg >> "${outfile}"