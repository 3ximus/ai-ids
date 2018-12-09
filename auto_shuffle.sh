#!/bin/bash
set -e

malign_traindir="DATA/train/malign/"
benign_traindir="DATA/train/benign/"
l1dir="DATA/train/layer1/"
l2dir="DATA/train/layer2/"


# selected malign/benign for training
[[ -d $l1dir ]] || mkdir -p $l1dir	# need benigns
[[ -d $l2dir ]] || mkdir -p $l2dir	# need benigns

find "$l1dir" "$l2dir" -maxdepth 1 -type f -exec rm '{}' \;

smallest_file() {
	wc -l $1 | grep -v "[0-9]\+ total$" | awk 'BEGIN {min = 99999999999} {if ($1 < min) min = $1} END {print min-1}'
}

attacks=( dos patator portscan ) 
map_attacks=( fastdos bruteforce portscan ) 
for attack in "${attacks[@]}" ; do
	echo $attack
	for file in $malign_traindir/*-$attack*.csv ; do
		echo $file $(smallest_file "$malign_traindir/*-$attack*.csv")
		# extract by smallest in each group to later be used in layer 2
		tail -n+2 "$file" | shuf -n $(smallest_file "$malign_traindir/*-$attack*.csv") > "$l1dir/$(basename $file)"
	done
done

echo "Joining files..."


# number of flows in each attack category
attack_flows=($(for attack in "${attacks[@]}" ; do wc -l $l1dir/*-$attack*.csv | tail -n1 | awk '{print $1}'; done))
min_attack_flows=$(echo ${attack_flows[@]} | tr ' ' '\n' | awk 'BEGIN {min = 99999999999} {if ($1 < min) min = $1} END {print min-1}')

# Create layer 1 file header
head -n1 "$benign_traindir/Monday-WorkingHours.csv" > $l1dir/training_L1.csv
for index in ${!attacks[@]}; do
	echo ${attacks[index]}
	# Create layer 2 file header
	head -n1 "$benign_traindir/Monday-WorkingHours.csv" > $l2dir/benign-${map_attacks[index]}.csv

	for file in $l1dir/*-${attacks[index]}*.csv ; do
		# L1
		files_in_attack=$(ls $l1dir/*-${attacks[index]}*.csv | wc -l)
		cat $file | shuf -n $(bc <<< "$min_attack_flows / $files_in_attack") >> $l1dir/training_L1.csv
		# L2 attacks
		min_benign_malign=$(bc <<< "$(echo ${attack_flows[index]} $(wc -l <"$benign_traindir/Monday-WorkingHours.csv") | awk '{if ($1 < $2) print $1; else print($2 -1)}') / $files_in_attack")
		shuf -n $min_benign_malign $file >> $l2dir/benign-${map_attacks[index]}.csv
	done
	# L2 benign
	tail -n+2 "$benign_traindir/Monday-WorkingHours.csv" | shuf -n $(wc -l < $l2dir/benign-${map_attacks[index]}.csv) >> $l2dir/benign-${map_attacks[index]}.csv
done


