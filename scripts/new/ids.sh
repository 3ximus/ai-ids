#!/bin/bash
set -e

if [ $# -ne 5 ]
then
	echo "Usage: ids <interface> <duration> <pcap directory> <csv directory> <alerts directory>"
	exit
fi

interface="$1"
cap_duration="$2"

pcap_dir="$3"
csv_dir="$4"
alert_dir="$5"


[[ -d $pcap_dir ]] || mkdir -p $pcap_dir
[[ -d $csv_dir ]] || mkdir -p $csv_dir
[[ -d $alert_dir ]] || mkdir -p $alert_dir

trap 'break;' SIGINT
pid_buf=()
counter=1
while [ 1==1 ]
do
	now_date=$(echo $(date) | sed -e 's/\ /-/g')
	./capture "$interface" "$cap_duration" "$pcap_dir" "$counter" "$now_date"
	echo "Background task: detecting intrusions"
	./detect_intrusions "$pcap_dir" "$csv_dir" "$alert_dir" "$counter" "$now_date" &> /dev/null & 	# background task with no output
	pid_buf+=("$!")
	counter=$(($counter+1))
done

trap '' SIGINT 			# forcefully ignore SIGINT from here on

echo "Shutting down Intrusion Detection System..."
echo "Checking if all ${#pid_buf[@]} processes have finished. If not, wait for them..."
RESULT=0
c=0
while [ $c -lt ${#pid_buf[@]} ]
do
	echo "Waiting for process with pid ${pid_buf[$c]} to finish..."
	wait "${pid_buf[$c]}" || let "RESULT=1"
	c=$(($c+1))
done

echo "Done"
