#!/bin/bash
set -e

if [ $# -ne 5 ]
then
	echo "Usage: capture <interface> <duration> <pcap directory> <pcap file name 1> <pcap file name 2>"
	exit
fi

interface="$1"
cap_duration="$2"
pcap_dir="$3"
counter="$4"
now_date="$5"
pcap_file="${pcap_dir}/${counter}-${now_date}.pcap"
trap 'rm "$pcap_file";exit;' SIGINT

echo "Capturing on ${interface}..."
tshark -i "$interface" -a "duration:${cap_duration}" -w "$pcap_file" -F pcap &> /dev/null
echo "Capture ${counter} ended."
