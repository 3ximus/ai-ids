#!/bin/bash
set -e

echo "Creating all training and testing datasets"

if [ "$1" == "-v" ];then
	verbose="-v"
else
	verbose=""
fi

# all malign/benign
[[ -d new/train/malign ]] || mkdir -p new/train/malign
[[ -d new/train/benign ]] || mkdir -p new/train/benign 	# need benigns

# selected malign/benign for testing
[[ -d new/test/malign ]] || mkdir -p new/test/malign
[[ -d new/test/benign ]] || mkdir -p new/test/benign 	# need benigns

python tk_flowmeter.py pcap/train/bruteforce/tekever-*.pcap -l bruteforce $verbose -o new/train/malign
python tk_flowmeter.py pcap/train/portscan/tekever-portscan.pcap -l portscan $verbose -o new/train/malign
python tk_flowmeter.py pcap/train/fastdos/tekever-*.pcap -l fastdos $verbose -o new/train/malign
python tk_flowmeter.py pcap/train/benign/Monday-WorkingHours.pcap -l benign $verbose -o new/train/benign

python tk_flowmeter.py pcap/test/bruteforce/*.pcap -l bruteforce $verbose -o new/test/malign
python tk_flowmeter.py pcap/test/portscan/*.pcap -l portscan $verbose -o new/test/malign
python tk_flowmeter.py pcap/test/fastdos/*.pcap -l fastdos $verbose -o new/test/malign
python tk_flowmeter.py pcap/test/benign/*.pcap -l benign $verbose -o new/test/benign

echo "Done"