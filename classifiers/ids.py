#!/usr/bin/env python
from __future__ import print_function
import sys, os, argparse
import numpy as np
import layer1, layer2

# =====================
#       OPTIONS
# =====================

op = argparse.ArgumentParser(description="Multilayered AI traffic classifier")
op.add_argument('files', metavar='file', nargs='+', help='csv file with all features to test')
op.add_argument('-d', '--disable-load', action='store_true', help="disable loading of previously created models", dest='disable_load')
op.add_argument('-v', '--verbose', action='store_true', help="verbose output", dest='verbose')
op.add_argument('-c', '--config-file', help="configuration file", dest='config_file', default=os.path.dirname(__file__) + '/options.cfg')
args = op.parse_args()

# =====================
#     CONFIGURATION
# =====================

# TODO CHANGE THIS HARDCODING
L1_TRAIN_FILE = "csv/train/layer1/trainingNN1.csv"
L2_TRAIN_FILE_DOS = "csv/train/layer2/benign-tekever-dos.csv"
L2_TRAIN_FILE_PORTSCAN = "csv/train/layer2/benign-tekever-portscan.csv"
L2_TRAIN_FILE_BRUTEFORCE = "csv/train/layer2/benign-tekever-bruteforce.csv"

# setup temp directory
TMP_DIR = '/tmp/ids.py.tmp'
if not os.path.isdir(TMP_DIR): os.makedirs(TMP_DIR)
TMP_L1_OUTPUT_DOS = TMP_DIR +"/dos.csv"
TMP_L1_OUTPUT_PORTSCAN = TMP_DIR +"/portscan.csv"
TMP_L1_OUTPUT_BRUTEFORCE = TMP_DIR +"/bruteforce.csv"

# Layer 1
print("Layer 1: 'Attack-Profiling'")
y1_predicted = layer1.classify(L1_TRAIN_FILE, args.files[0], config_file=args.config_file, disable_load=args.disable_load, verbose=args.verbose)
y1_predicted = (y1_predicted == y1_predicted.max(axis=1, keepdims=True)).astype(int)

dos=[]
pscan=[]
bforce=[]
for i,prediction in enumerate(y1_predicted):
	if np.argmax(prediction)==0: #dos
		dos.append(i)
	elif np.argmax(prediction)==1: #portscan
		pscan.append(i)
	elif np.argmax(prediction)==2: #bruteforce
		bforce.append(i)
	else:
		print("Error.")

# Layer 2
print("Layer 2: 'Flow Classification'")

fd = open(args.files[0],"r")
content = fd.readlines()
content = [x.strip('\n') for x in content]
fd.close()

dos = set(dos)
pscan = set(pscan)
bforce = set(bforce)
dos_of = open(TMP_L1_OUTPUT_DOS,"w")
pscan_of = open(TMP_L1_OUTPUT_PORTSCAN,"w")
bforce_of = open(TMP_L1_OUTPUT_BRUTEFORCE,"w")

print("Selecting layer1 selected flows...")
for i,elem in enumerate(content):
	if i in dos:
		dos_of.write(elem + "\n")
	elif i in pscan:
		pscan_of.write(elem + "\n")
	elif i in bforce:
		bforce_of.write(elem + "\n")
dos_of.close()
pscan_of.close()
bforce_of.close()

benign=[]
malign=[]
if len(dos)!=0:
	y2_dos_predicted = layer2.classify(L2_TRAIN_FILE_DOS, TMP_L1_OUTPUT_DOS, disable_load=args.disable_load, verbose=args.verbose)
	for prediction in y2_dos_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)
if len(pscan)!=0:
	y2_pscan_predicted = layer2.classify(L2_TRAIN_FILE_PORTSCAN, TMP_L1_OUTPUT_PORTSCAN, disable_load=args.disable_load, verbose=args.verbose)
	for prediction in y2_pscan_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)
if len(bforce)!=0:
	y2_bforce_predicted = layer2.classify(L2_TRAIN_FILE_BRUTEFORCE, TMP_L1_OUTPUT_BRUTEFORCE, disable_load=args.disable_load, verbose=args.verbose)
	for prediction in y2_bforce_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)

benign_length = len(benign)
malign_length = len(malign)
print("\033[35m\n -----------\n | RESULTS |\n -----------\033[0;0m")
print("\033[1;32m#Flows classified as benign:\033[0;0m")
print(benign_length)
print("\033[1;31m#Flows classified as malign:\033[0;0m")
print(malign_length)
print("\033[1;32mBenign/(Malign + Benign) ratio\033[0;0m")
print(benign_length*100*1.0/(benign_length+malign_length),"%")
print("\033[1;31mMalign/(Malign + Benign) ratio\033[0;0m")
print(malign_length*100*1.0/(benign_length+malign_length),"%")
