"""Script to scan ISCX-IDS2012 dataset for Attack IPs and label the generated dataset

Usage

python scan-and-label-ISCX2012.py <xml-file> <csv-file> <label-to-replace-with>"""

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

with open(sys.argv[2], "r") as main_file:
	with open(sys.argv[2]+'.new','w') as replacement_file:
		replacement_file.write(next(main_file)) # write header
		for line in main_file:
			if any([line.startswith(k) for k in tags if 'Attack' in tags[ip]]):
				line = line.replace('unknown', sys.argv[3])
			else:
				line = line.replace('unknown', 'BENIGN')
			replacement_file.write(line)
