#!/bin/bash
set -e

echo "Selecting samples for layer1/layer2 training"

malign_traindir="new/train/malign/"
benign_traindir="new/train/benign/"
l1dir="new/train/layer1/"
l2dir="new/train/layer2/"
n_fastdos="1110"
n_bruteforce="740"
n_portscan="2220"

# selected malign/benign for training
[[ -d $l1dir ]] || mkdir -p $l1dir	# need benigns
[[ -d $l2dir ]] || mkdir -p $l2dir	# need benigns

tail -n+2 "$malign_traindir/tekever-ftp-patator.csv" | shuf -n $n_bruteforce > "$l1dir/tekever-ftp-patator.csv"

tail -n+2 "$malign_traindir/tekever-ssh-patator.csv" | shuf -n $n_bruteforce > "$l1dir/tekever-ssh-patator.csv"

tail -n+2 "$malign_traindir/tekever-telnet-patator.csv" | shuf -n $n_bruteforce > "$l1dir/tekever-telnet-patator.csv"

tail -n+2 "$malign_traindir/tekever-dos-goldeneye.csv" | shuf -n $n_fastdos > "$l1dir/tekever-dos-goldeneye.csv"

tail -n+2 "$malign_traindir/tekever-dos-hulk.csv" | shuf -n $n_fastdos > "$l1dir/tekever-dos-hulk.csv"

tail -n+2 "$malign_traindir/tekever-portscan.csv" | shuf -n $n_portscan > "$l1dir/tekever-portscan.csv"

head -n 1 "$malign_traindir/tekever-portscan.csv" > "$l1dir/trainingNN1.csv"

cat $l1dir/tekever-* >> "$l1dir/trainingNN1.csv"

shuf "$benign_traindir/Monday-WorkingHours.csv" -n 2220 > "$l2dir/iscx-benign.csv"
cat "$l1dir/tekever-ftp-patator.csv" "$l1dir/tekever-ssh-patator.csv" "$l1dir/tekever-telnet-patator.csv" > "$l2dir/tekever-bruteforce.csv"
cat "$l1dir/tekever-dos-goldeneye.csv" "$l1dir/tekever-dos-hulk.csv" > "$l2dir/tekever-fastdos.csv"
cat "$l1dir/tekever-portscan.csv" > "$l2dir/tekever-portscan.csv"
cat "$l2dir/iscx-benign.csv" "$l2dir/tekever-bruteforce.csv" > "$l2dir/benign-bruteforce.csv"
cat "$l2dir/iscx-benign.csv" "$l2dir/tekever-fastdos.csv" > "$l2dir/benign-fastdos.csv"
cat "$l2dir/iscx-benign.csv" "$l2dir/tekever-portscan.csv" > "$l2dir/benign-portscan.csv"
echo "Done"