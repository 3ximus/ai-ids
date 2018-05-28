#!/bin/bash
set -e

pcap_dir="$1"
csv_dir="$2"
alert_dir="$3"
counter="$4"
now_date="$5"
pcap_file="${pcap_dir}/${counter}-${now_date}.pcap"
csv_file="${csv_dir}/${counter}-${now_date}.csv"
alert_file="${alert_dir}/${counter}-${now_date}.txt"

python tk_flowmeter.py "$pcap_file" --verbose --out-dir "${csv_dir}/"		# default label: unknown
if [ -e "$csv_file" ]
then
	echo "Testing capture file ${counter} for intrusions..."
	python ids.py "$csv_file" --show-comms-only --alert-file "$alert_file"
	if [ -s "$alert_file" ] 									# check if alert file has content
	then
		echo "Alert saved in: $alert_file"
	else
		echo "There were no recorded alerts for this capture, so all files will be deleted."
		rm "$pcap_file"
		rm "$csv_file"
		rm "$alert_file"
	fi
	counter=$(($counter+1))
else
	echo "CSV wasn't generated, which means that there were no tcp flows captured in this pcap. Alert file not created."
	echo "Deleting pcap..."
	rm "$pcap_file"
	counter=$(($counter+1))
fi
