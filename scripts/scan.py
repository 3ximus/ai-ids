"""Script to scan ISCX-IDS2012 dataset for Attack IPs"""
import sys
tags = {}
with open(sys.argv[1], "r") as fp:
	for line in fp:
		if "<source>" in line:
			ip = line.split('>')[1].split('<')[0]
			tags[ip] = []
			for line in fp:
				if "<Tag>" in line:
					tag = line.split('>')[1].split('<')[0]
					tags[ip].append(tag)
					break
				if "<source>" in line:
					print(ip + " without tag")
					exit(1)

print([(ip, tags[ip]) for ip in tags if 'Attack' in tags[ip]])
