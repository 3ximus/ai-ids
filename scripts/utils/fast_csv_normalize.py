#!/usr/bin/env python
import sys
import argparse

op = argparse.ArgumentParser()
op.add_argument('files', metavar='file', nargs='*', help='')
op.add_argument('-d', '--default', action='store_true', help="default behavior", dest='default')
args = op.parse_args()

if args.default:
	defaults = [
	['csv/selected-compacted-datasets/all_features/raw/individual/DDoS.csv','csv/selected-compacted-datasets/all_features/normalized/individual/DDoS.csv',400],
	['csv/selected-compacted-datasets/all_features/raw/individual/DoS-GoldenEye.csv','csv/selected-compacted-datasets/all_features/normalized/individual/DoS-GoldenEye.csv',400],
	['csv/selected-compacted-datasets/all_features/raw/individual/DoS-Hulk.csv','csv/selected-compacted-datasets/all_features/normalized/individual/DoS-Hulk.csv',400],
	['csv/selected-compacted-datasets/all_features/raw/individual/DoS-Slowhttptest.csv','csv/selected-compacted-datasets/all_features/normalized/individual/DoS-Slowhttptest.csv',400],
	['csv/selected-compacted-datasets/all_features/raw/individual/DoS-slowloris.csv','csv/selected-compacted-datasets/all_features/normalized/individual/DoS-slowloris.csv',400],
	['csv/selected-compacted-datasets/all_features/raw/individual/FTP-Patator.csv','csv/selected-compacted-datasets/all_features/normalized/individual/FTP-Patator.csv',2000],
	['csv/selected-compacted-datasets/all_features/raw/individual/PortScan.csv','csv/selected-compacted-datasets/all_features/normalized/individual/PortScan.csv',2000],
	['csv/selected-compacted-datasets/all_features/raw/individual/SSH-Patator.csv','csv/selected-compacted-datasets/all_features/normalized/individual/SSH-Patator.csv',2000]
	]

	for default in defaults:
		f = open(default[0],'r')
		of = open(default[1],'w')
		n_samples = default[2]
		
		i=0
		for line in f:
			if i<n_samples:
				of.write(line)
				i+=1
		of.close()
		f.close()
		print "Done,", i, "lines written to file:", of.name

else:
	f = open (sys.argv[1], 'r')
	of = open(sys.argv[2], 'w')
	n_samples = int(sys.argv[3])

	i=0
	for line in f:
		if i<n_samples:
			of.write(line)
			i+=1

	of.close()
	f.close()
	print "Done,", i, "lines written to file:", of.name

