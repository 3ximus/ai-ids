#!/usr/bin/env python
from __future__ import print_function
import os, argparse, re, curses
import numpy as np
from classifiers.node_model import NodeModel
import threading
try: import configparser
except ImportError: import ConfigParser as configparser # for python2

# can be useful later
class ThreadWithReturnValue(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs, Verbose)
        self._return = None
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args,
                                                **self._Thread__kwargs)
    def join(self):
        threading.Thread.join(self)
        return self._return

# =====================
#     CLI OPTIONS
# =====================

op = argparse.ArgumentParser(description="Multilayered AI traffic classifier")
op.add_argument('files', metavar='file', nargs='+', help='csv file with all features to test')
op.add_argument('-s', '--select', nargs='+', help='select on layer/node to test from config file')
op.add_argument('-d', '--disable-load', action='store_true', help="disable loading of previously created models", dest='disable_load')
op.add_argument('-z', '--show-comms-only', action='store_true', help="show communication information only", dest='show_comms')
op.add_argument('-v', '--verbose', action='store_true', help="verbose output. Disables curses interface", dest='verbose')
op.add_argument('-c', '--config-file', help="configuration file", dest='config_file', default='classifiers/options.cfg')
op.add_argument('-a', '--alert-file', help="alert file", dest='alert_file', default='alerts')
args = op.parse_args()

# =====================
#    CONFIGURATION
# =====================
flow_results = dict()
benign_flows = [np.array([[]]),np.array([], dtype='int8'),np.array([], dtype='int8')]
benign_flows = [[],[],[]]
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

ALERT_LOWER_BOUND_FLOWS = 150           # "heuristically" chosen value (must come from a probabilistic study on the upper bound of no. of flows usually present in benign communications. 
                                        # It should take into consideration the capture time (current "classification window size") and the most probable number of benign flows per communication (2nd module - a view on bulks of flows)
# verifiy configuration integrity
l2_sections = [s for s in conf.sections() if re.match('l2-.+', s)]
if not len(L2_NODE_NAMES) == len(conf.options('labels-l1')) == len(l2_sections):
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

def print_curses_stats(): # meant to be used inside each thread to update its results
    with curses_lock:
        stdscr.addstr(os.path.basename(args.files[0]) + "\n")
        stdscr.addstr("    LAYER 1\n", curses.color_pair(7) | curses.A_BOLD)
        l1.stats.update_curses_screen(stdscr, curses)
        if any([node.stats.n for node in l2_nodes]):
            stdscr.addstr("    LAYER 2\n", curses.color_pair(7) | curses.A_BOLD)
        for node in range(len(l2_nodes)):
            if l2_nodes[node].stats.n > 0:
                stdscr.addstr(L2_NODE_NAMES[node]+'\n')
                l2_nodes[node].stats.update_curses_screen(stdscr, curses)
        stdscr.refresh()
        stdscr.clear()

def predict_chunk(test_data):
    thread_semaphore.acquire()
    # LAYER 1
    y_predicted, flow_ids = l1.predict(test_data)
    labels_index = np.argmax(y_predicted, axis=1) if not l1.use_regressor else y_predicted
    for i, fl_id in enumerate(flow_ids):
        flow_results[fl_id] = [labels_index[i]]
        
    if not args.verbose and not args.show_comms: print_curses_stats()
    # OUTPUT DATA PARTITION TO FEED LAYER 2
    # ignore test_data[1] since its only used for l1 crossvalidation
    filter_labels = lambda x: [np.take(test_data[0], np.where(labels_index == x)[0], axis=0), # x
                               np.take(test_data[2], np.where(labels_index == x)[0], axis=0), # labels
                               np.take(test_data[3], np.where(labels_index == x)[0], axis=0)] # flow_ids
    l2_inputs = [filter_labels(x) for x in range(len(L2_NODE_NAMES))]
    if not args.verbose and not args.show_comms: print_curses_stats()

    # LAYER 2
    for node in range(len(l2_nodes)):
        if len(l2_inputs[node][0]) != 0:
            current_test_data = l2_nodes[node].process_data(l2_inputs[node][0], l2_inputs[node][1],l2_inputs[node][2])
            y_predicted, flow_ids = l2_nodes[node].predict(current_test_data)
            labels_index = np.argmax(y_predicted, axis=1)
            for i, fl_id in enumerate(flow_ids):
                flow_results[fl_id].append(labels_index[i])                    # the order wasn't being kept because we are not classifying flows sequentially. Now this fixes it...
            benign_flows[0].extend(list(l2_inputs[node][0]))
            benign_flows[1].extend(["benign" if x==0 else "malign" for x in labels_index])
            benign_flows[2].extend(list(l2_inputs[node][2]))
        if not args.verbose and not args.show_comms: print_curses_stats()
    thread_semaphore.release()


# =====================
#  LAUNCH TEST THREADS
# =====================

if not args.verbose and not args.show_comms:
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

# Start colors in curses
    curses.start_color()
    curses.use_default_colors()
    for i in range(0, curses.COLORS):
        curses.init_pair(i + 1, i, -1)
    curses_lock = threading.Lock()


thread_semaphore = threading.BoundedSemaphore(value=MAX_THREADS)

try:
    if args.verbose: print("Reading Test Dataset in chunks...")
    for test_data in l1.yield_csvdataset(args.files[0], CHUNK_SIZE): # launch threads
        thread = threading.Thread(target=predict_chunk,args=(test_data,))
        thread.start()
    for t in threading.enumerate(): # wait for the remaining threads
        if t.getName()!="MainThread":
            t.join()
finally:
    if not args.verbose and not args.show_comms:
        curses.echo()
        curses.nocbreak()
        curses.endwin()

# =====================
#   PRINT FINAL STATS
# ====================j

def flow_id_to_communication_id(flow_id):
    splitted_flow_id = flow_id.split('-')
    return splitted_flow_id[0] + '-' + splitted_flow_id[2]

if not args.show_comms:
    print(os.path.basename(args.files[0]))
    print("\033[1;36m    LAYER 1\033[m")
    print(l1.stats)
# output counter for l2
    print("\033[1;36m    LAYER 2\033[m")
    total = total_correct = total_fp = 0
    for node in range(len(l2_nodes)):
        if l2_nodes[node].stats.n > 0:
            total += l2_nodes[node].stats.n
            total_correct += l2_nodes[node].stats.total_correct
            print(L2_NODE_NAMES[node])
            print(l2_nodes[node].stats)
else:
    communications = dict()
    for flow_id in flow_results:
        communication_id = flow_id_to_communication_id(flow_id)
        if communication_id in communications.keys():
            communications[communication_id].append(flow_results[flow_id])
        else:
            communications[communication_id] = [flow_results[flow_id]]
    of1 = open(args.alert_file+"_level1.txt","w")
    of2 = open(args.alert_file+"_level2.txt","w")
    of3 = open(args.alert_file+"_level3.txt","w")
    for comm in communications:
        fastdos_count = communications[comm].count([0,1])
        portscan_count = communications[comm].count([1,1])
        bruteforce_count = communications[comm].count([2,1])
        malign_count = fastdos_count + portscan_count + bruteforce_count
        benign_count = communications[comm].count([0,0]) + communications[comm].count([1,0]) + communications[comm].count([2,0])
        benign_ratio = benign_count*1.0/(benign_count+malign_count)
        if args.verbose:
            print("%s:\nFastdos: %d\nPortscan: %d\nBruteforce: %d\nBenign: %d\nBenign ratio: %f" %
                    (comm, fastdos_count, portscan_count, bruteforce_count, benign_count, benign_ratio))
        # alerts level 1 (very suspicious)
        if benign_ratio<=0.2 and (benign_count+malign_count)>=ALERT_LOWER_BOUND_FLOWS:
            of1.write("%s:\nFastdos: %d\nPortscan: %d\nBruteforce: %d\nCertainty: %f%%\n" %
                    (comm, fastdos_count, portscan_count, bruteforce_count, (1-benign_ratio)*100))
        # alerts level 2 (suspicious)
        elif malign_count>=ALERT_LOWER_BOUND_FLOWS:
            of2.write("%s:\nFastdos: %d\nPortscan: %d\nBruteforce: %d\nCertainty: %f%%\n" %
                    (comm, fastdos_count, portscan_count, bruteforce_count, (1-benign_ratio)*100))
        # alerts level 3 (unusual communication rate), this alert level doesn't rely on the flow classifier to get everything right
        elif (malign_count+benign_count)>=ALERT_LOWER_BOUND_FLOWS:
            of3.write("%s:\nFastdos: %d\nPortscan: %d\nBruteforce: %d\nBenign: %d\nCertainty: %f%%\n" %
                    (comm, fastdos_count, portscan_count, bruteforce_count, benign_count, (1-benign_ratio)*100))

    of1.close()
    of2.close()
    of3.close()

#L3_TRAIN_FILE = conf.get('ids','l3')
#l3 = NodeModel('l3', conf, verbose=args.verbose)
#l3.train(L3_TRAIN_FILE, args.disable_load)
#benign = [[],[],[]]
#for i, elem in enumerate(benign_flows[1]):
#    if elem=="benign":
#        benign[0].append(benign_flows[0][i])
#        benign[1].append(benign_flows[1][i])
#        benign[2].append(benign_flows[2][i])
#del(benign_flows)

#current_test_data = l3.process_data(benign[0],benign[1],benign[2])
#y_predicted, flow_ids = l3.predict(current_test_data)
#labels_index = np.argmax(y_predicted, axis=1) if not l3.use_regressor else y_predicted
#print("Benign:",(labels_index==0).sum())
#print("Malign/Backdoor:",(labels_index==1).sum())

