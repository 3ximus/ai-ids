#!/usr/bin/env python3

"""This file contains the double layered classifier

AUTHORS:

Joao Meira <joao.meira@tekever.com>
Fabio Almeida <fabio4335@gmail.com>
"""

import os, argparse, re, sys, time
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
op.add_argument('-a', '--alert-file', help="alert file", dest='alert_file', default='alerts')
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
L2_NODE_NAMES = [op for op in conf.options('ids') if re.match('l2-.+', op)]
L2_TRAIN_FILES = [conf.get('ids', node_name) for node_name in L2_NODE_NAMES]

CHUNK_SIZE = conf.getint('ids', 'chunk-size')
MAX_THREADS = conf.getint('ids', 'max-threads')
ALERT_LOWER_BOUND_FLOWS = conf.getint('ids', 'lower-bound-flows')

# verifiy configuration integrity
l2_sections = [s for s in conf.sections() if re.match('l2-.+', s)]
if l2_sections and not len(L2_NODE_NAMES) == len(conf.options('labels-l1')) == len(l2_sections):
    print("Number of l1 output labels and l2 nodes don't match in config file %s" % args.config_file)
    exit()
if not all([(item[0] == item[1][item[1].find('-')+1:] == item[2][item[2].find('-')+1:]) for item in zip(conf.options('labels-l1'), L2_NODE_NAMES, l2_sections)]):
    print("Names of l1 output labels do not match l2 node names in config file %s" % args.config_file)
    exit()

# =====================
#   CREATE AND TRAIN
# =====================

# LAYER 1
l1 = NodeModel('l1', conf, verbose=args.verbose)
l1.train(L1_TRAIN_FILE, args.disable_load)

# LAYER 2
l2_nodes = [NodeModel(node_name, conf, verbose=args.verbose) for node_name in L2_NODE_NAMES]
for node in range(len(l2_nodes)):
    l2_nodes[node].train(L2_TRAIN_FILES[node], args.disable_load)

# =====================
#   THREAD TEST CHUNK
# =====================

def predict_chunk(test_data):
    thread_semaphore.acquire()
    # LAYER 1
    y_predicted, flow_ids = l1.predict(test_data)

    # OUTPUT DATA PARTITION TO FEED LAYER 2
    labels_index = np.argmax(y_predicted, axis=1) if not l1.use_regressor else y_predicted

    # ignore test_data[1] since its only used for l1 crossvalidation
    filter_labels = lambda x: [np.take(test_data[0], np.where(labels_index == x)[0], axis=0), # x
                               np.take(test_data[2], np.where(labels_index == x)[0], axis=0), # labels
                               np.take(test_data[3], np.where(labels_index == x)[0], axis=0)] # flow_ids
    l2_inputs = [filter_labels(x) for x in range(len(L2_NODE_NAMES))]

    # LAYER 2
    for node in range(len(l2_nodes)):
        if len(l2_inputs[node][0]) != 0:
            current_test_data = l2_nodes[node].process_data(l2_inputs[node][0], l2_inputs[node][1],l2_inputs[node][2])
            y_predicted, flow_ids = l2_nodes[node].predict(current_test_data)
            labels_index = np.argmax(y_predicted, axis=1)
    thread_semaphore.release()


# =====================
#  LAUNCH TEST THREADS
# =====================

thread_semaphore = threading.BoundedSemaphore(value=MAX_THREADS)

if args.verbose: print("Reading Test Dataset in chunks...")
fd = open(args.input, 'r') if args.input else sys.stdin
start_time = time.time()
acc = 0
for test_data in l1.yield_csvdataset(fd, CHUNK_SIZE): # launch threads
    acc += len(test_data[0])
    thread = threading.Thread(target=predict_chunk,args=(test_data,))
    thread.start()

if fd != sys.stdin: fd.close()
for t in threading.enumerate(): # wait for the remaining threads
    if t.getName()!="MainThread":
        t.join()
print("%d flows predicted in " % acc + str(time.time() - start_time) + " seconds", file=sys.stderr)

# =====================
#   PRINT FINAL STATS
# =====================

if args.input: print(os.path.basename(args.input))
print("\033[1;36m    LAYER 1\033[m")
print(l1.stats)
l1.logger.log("%s\n" % l1.node_name + str(l1.stats))
# output counter for l2
print("\033[1;36m    LAYER 2\033[m")
total = total_correct = total_fp = 0
for node in range(len(l2_nodes)):
    if l2_nodes[node].stats.n > 0:
        total += l2_nodes[node].stats.n
        total_correct += l2_nodes[node].stats.total_correct
        print(L2_NODE_NAMES[node])
        print(l2_nodes[node].stats)
        l2_nodes[node].logger.log("%s\n" % l2_nodes[node].node_name + str(l2_nodes[node].stats))
