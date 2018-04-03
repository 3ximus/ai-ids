#!/usr/bin/env python
import sys, argparse
from os import path
import random
from loading import progress_bar

op = argparse.ArgumentParser(description="Setup train or test data with fixed amount of features. Uses the file 'train_data.config' to read ammount of data to extract into separate datasets")
op.add_argument('files', metavar='file', nargs='+', help='input files. Last file is the output directory')
op.add_argument('-i', '--ignore-counts', action='store_true', help="ignore count of tests from 'train_data.config' file and dump all the data", dest='ignore_count')
op.add_argument('-r', '--random', action='store_true', help="random samples", dest='random')
args = op.parse_args()

if not len(args.files) >= 2:
    op.print_usage()
    print('Input and output files must be given as argument')
    sys.exit(1)

# read settings
with open(path.dirname(sys.argv[0])+'/train_data.config', 'r') as conf:
    conf_lines = [[i.strip() for i in line.rsplit(' ', 1)] for line in conf.read().splitlines() if not line.startswith('#')]

# open output files
of_names = ['%s/%s.csv' % (args.files[-1], line[0].replace(' ','-')) for line in conf_lines]
of_descriptors = [open(of, 'w') for of in of_names]
[of.write(open(args.files[0],'r').readline()) for of in of_descriptors]
counters = [0]*len(conf_lines)
for x, in_file in enumerate(args.files[:-1]):
    progress_bar((x+1) / len(args.files[:-1]) * 100, initial_text=in_file+' ', bar_body="\033[34m-\033[m", bar_arrow="\033[34m>\033[m", align_right=True)
    with open(in_file, 'r') as if_:
        if args.random:
            for i, c in enumerate(conf_lines): # for each attack
                 filtered = [line for line in if_ if "Infinity" not in line and c[0] in line]
                 if_.seek(0)
                 if filtered:
                     for line in random.sample(filtered, int(c[1])): # c[1] is label count
                        of_descriptors[i].write(line)
        else:
            for line in if_:
                for i, c in enumerate(conf_lines): # for each attack
                    if "Infinity" not in line and counters[i] < int(c[1]) and c[0] in line: # check attack type against line
                        of_descriptors[i].write(line)
                        if not args.ignore_count: counters[i] += 1
for of in of_descriptors:
    of.close()
