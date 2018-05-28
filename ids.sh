#!/bin/bash
set -e

interface="$1"

pcap_dir="new/running/pcap/"
csv_dir="new/running/csv/"
alert_dir="new/running/alerts/"
cap_duration=600

[[ -d $pcap_dir ]] || mkdir -p $pcap_dir
[[ -d $csv_dir ]] || mkdir -p $csv_dir
[[ -d $alert_dir ]] || mkdir -p $alert_dir

pid_buf=()
counter=1
while [ 1==1 ]
do
	now_date=$(echo $(date) | sed -e 's/\ /-/g')
	./capture.sh "$interface" "$pcap_dir" "$cap_duration" "$counter" "$now_date"
	echo "Background task: detecting intrusions"
	./detect_intrusions.sh "$pcap_dir" "$csv_dir" "$alert_dir" "$counter" "$now_date" &> /dev/null & 	# background task with no output
	pid_buf+=("$!")
	counter=$(($counter+1))
done

echo "Finished"