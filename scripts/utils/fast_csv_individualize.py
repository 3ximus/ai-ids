#!/usr/bin/env python
import sys
import argparse

op = argparse.ArgumentParser()
op.add_argument('files', metavar='file', nargs='*', help='')
op.add_argument('-d', '--default', action='store_true', help="default behavior", dest='default')
args = op.parse_args()

if args.default:
	f = open('csv/selected-compacted-datasets/21features/raw/Malign-Dataset.csv', 'r')
	defaults = {'DDoS':'csv/selected-compacted-datasets/21features/raw/individual/DDoS.csv',
	'DoS GoldenEye':'csv/selected-compacted-datasets/21features/raw/individual/DoS-GoldenEye.csv',
	'DoS Hulk':'csv/selected-compacted-datasets/21features/raw/individual/DoS-Hulk.csv',
	'DoS Slowhttptest':'csv/selected-compacted-datasets/21features/raw/individual/DoS-Slowhttptest.csv',
	'DoS slowloris':'csv/selected-compacted-datasets/21features/raw/individual/DoS-slowloris.csv',
	'FTP-Patator':'csv/selected-compacted-datasets/21features/raw/individual/FTP-Patator.csv',
	'PortScan':'csv/selected-compacted-datasets/21features/raw/individual/PortScan.csv',
	'SSH-Patator':'csv/selected-compacted-datasets/21features/raw/individual/SSH-Patator.csv',
	}
	for key in defaults:
		f.seek(0)
		of = open(defaults[key], 'w')
		i=0
		for line in f:
			if key in line:
				of.write(line)
				i+=1
		of.close()
		print "Done,", i, "lines written to file:", of.name
	f.close()
else:
	f = open(sys.argv[1], 'r')
	of = open(sys.argv[2], 'w')
	lbl = sys.argv[3]
	i=0
	for line in f:
		if (line.find(lbl)!=-1):
			of.write(line)
			i+=1

	f.close()
	of.close()
	print "Done,", i, "lines written to file:", of.name