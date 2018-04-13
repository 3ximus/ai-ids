#!/usr/bin/env python
from __future__ import print_function
import os, argparse, re
import numpy as np
import layer1, layer2
try: import configparser
except ImportError: import ConfigParser as configparser # for python2

# =====================
#     CLI OPTIONS
# =====================

op = argparse.ArgumentParser(description="Multilayered AI traffic classifier")
op.add_argument('files', metavar='file', nargs='+', help='csv file with all features to test')
op.add_argument('-d', '--disable-load', action='store_true', help="disable loading of previously created models", dest='disable_load')
op.add_argument('-v', '--verbose', action='store_true', help="verbose output", dest='verbose')
op.add_argument('-c', '--config-file', help="configuration file", dest='config_file', default=os.path.dirname(__file__) + '/options.cfg')
args = op.parse_args()

# =====================
#    CONFIGURATION
# =====================

# load config file settings
conf = configparser.ConfigParser(allow_no_value=True)
conf.optionxform=str
conf.read(args.config_file)

# load train files
L1_TRAIN_FILE = conf.get('ids', 'l1')
L2_NODE_NAMES = [op for op in conf.options('ids') if re.match('l2-.+', op)]
L2_TRAIN_FILES = [conf.get('ids', node_name) for node_name in L2_NODE_NAMES]

# verifiy configuration integrity
l2_sections = [s for s in conf.sections() if re.match('l2-.+', s)]
if not len(L2_NODE_NAMES) == len(conf.options('labels-l1')) == len(l2_sections):
    print("Number of l1 output labels and l2 nodes don't match in config file %s" % args.config_file)
    exit()
if not all([(item[0] == item[1][item[1].find('-')+1:] == item[2][item[2].find('-')+1:]) for item in zip(conf.options('labels-l1'), L2_NODE_NAMES, l2_sections)]):
    print("Names of l1 output labels do not match l2 node names in config file %s" % args.config_file)
    exit()

# setup temp directory and l1 output / l2 input files
TMP_DIR = '/tmp/ids.py.tmp/'
if not os.path.isdir(TMP_DIR): os.makedirs(TMP_DIR)
TMP_L1_OUTPUT_FILES = [TMP_DIR + out_label + ".csv" for out_label in L2_NODE_NAMES]


# =====================
#       LAYER 1
# =====================

print("\n\033[1;36m    LAYER 1\033[m")
y1_predicted = layer1.classify(L1_TRAIN_FILE, args.files[0], conf, args.disable_load, args.verbose)
exit(1)
y1_predicted = (y1_predicted == y1_predicted.max(axis=1, keepdims=True)).astype(int)
#for i,elem in enumerate(y1_predicted):
#    y1_predicted[i]=[1,0,0]
# OUTPUT DATA PARTITION TO FEED LAYER 2

l2_input_files = [open(fd, 'w') for fd in TMP_L1_OUTPUT_FILES]
l2_data_count = [0] * len(L2_NODE_NAMES)
with open(args.files[0],"r") as fd: # TODO dont read test data so many times #9
# write each data entry from test file to some l2 input file based on l1 prediction
    for i, entry in enumerate(fd.read().splitlines()):
        x = np.argmax(y1_predicted[i]) # speedup
        l2_input_files[x].write(entry + '\n')
        l2_data_count[x] += 1
[fd.close() for fd in l2_input_files]


# =====================
#       LAYER 2
# =====================

print("\n\033[1;36m    LAYER 2\033[m")

# output counter for l2
output_counter = [0] * len(conf.options('labels-l2'))

for node in range(len(L2_NODE_NAMES)):
    if l2_data_count[node] != 0:
        y2_dos_predicted = layer2.classify(L2_TRAIN_FILES[node], TMP_L1_OUTPUT_FILES[node],
                                           L2_NODE_NAMES[node], conf, args.disable_load, args.verbose)
        for prediction in y2_dos_predicted:
            output_counter[np.argmax(prediction)] += 1

print("\n\033[1;35m    RESULTS\033[m [%s]\n           \033[1;32mBENIGN\033[m | \033[1;31mMALIGN\033[m" % os.path.basename(args.files[0]))
print("Count:  \033[1;32m%9d\033[m | \033[1;31m%d\033[m" % tuple(output_counter))
print("Ratio:  \033[1;32m%9f\033[m | \033[1;31m%f\033[m" % (output_counter[0]*100./sum(output_counter), output_counter[1]*100./sum(output_counter)))
