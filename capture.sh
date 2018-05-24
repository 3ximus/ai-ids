#!/bin/bash
set -e

interface="$1"
pcap_dir="$2"
cap_duration="$3"
counter="$4"
now_date="$5"
pcap_file="${pcap_dir}/${counter}-${now_date}.pcap"

echo "Capturing on ${interface}..."
tshark -i "$interface" -a "duration:${cap_duration}" -w "$pcap_file" -F pcap &> /dev/null
echo "Capture ${counter} ended."
