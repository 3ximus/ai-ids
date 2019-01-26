#!/usr/bin/env python3

"""This file contains a single classifier node

AUTHOR:

Fabio Almeida <fabio4335@gmail.com>
"""

import os, argparse, re, sys
import numpy as np
from lib.node import NodeModel
import threading
import configparser

# =====================
#     CLI OPTIONS
# =====================

op = argparse.ArgumentParser(description="Multilayered AI traffic classifier")
op.add_argument('-i', '--input', metavar='FILE', dest='input', help='csv file with test data. If none is given stdin is read')
op.add_argument('-d', '--disable-load', action='store_true', help="disable loading of previously created models", dest='disable_load')
op.add_argument('-v', '--verbose', action='store_true', help="Verbose output.", dest='verbose')
op.add_argument('-c', '--config-file', help="configuration file", dest='config_file', default='configs/ids.cfg')
args = op.parse_args()

# =====================
#    CONFIGURATION
# =====================
flow_results = dict()
# load config file settings
conf = configparser.ConfigParser(allow_no_value=True)
conf.optionxform=str
conf.read(args.config_file)

# load train files
L1_TRAIN_FILE = conf.get('ids', 'l1')
CHUNK_SIZE = conf.getint('ids', 'chunk-size')

# =====================
#   CREATE AND TRAIN
# =====================

# LAYER 1
l1 = NodeModel('l1', conf, verbose=args.verbose)
l1.train(L1_TRAIN_FILE, args.disable_load)

if args.verbose: print("Reading Test Dataset in chunks...")
fd = open(args.input, 'r') if args.input else sys.stdin
for test_data in l1.yield_csvdataset(fd, CHUNK_SIZE): # launch threads
	l1.predict(test_data)

if fd != sys.stdin: fd.close()

# =====================
#   PRINT FINAL STATS
# =====================

if args.input: print(os.path.basename(args.input))
print("\033[1;36m    LAYER 1\033[m")
print(l1.stats)
l1.logger.log("%s\n" % l1.node_name + str(l1.stats))
