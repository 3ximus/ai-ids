#!/usr/bin/env python
from __future__ import print_function
import sys, argparse, re, glob
import numpy as np, pandas
from os import path
from loading import progress_bar

try: # beacause python 2 -.-
   input = raw_input
except NameError:
   pass

op = argparse.ArgumentParser( description="Select features or filter MALIGN / BENIGN flows")
op.add_argument('files', metavar='file', nargs='*', help='list of input files and output [last file is the output, if this file is a directory all files are processed separately into their respective output files]')
op.add_argument('-b', '--benign', action='store_true', help="print only benign flows", dest='benign')
op.add_argument('-f9', '--features9', action='store_const', help="9 features", dest='features', const=9)
op.add_argument('-f21', '--features21', action='store_const', help="21 features", dest='features', const=21)
op.add_argument('-fall', '--featuresall', action='store_const', help="all features (64)", dest='features', const=64)
args = op.parse_args()

CHUNKSIZE = 10 ** 4
if not args.features: args.features = 64 # give a default value of all features

# READ FEATURES FROM features/*.txt files
# list of files containing the features
features_files = glob.glob(path.dirname(sys.argv[0])+'/features/*.txt')
# count of #features in each feature file
features_count = [sum([1 for l in fd if l not in ('\n','',' ')])-1 for fd in (open(x,'r') for x in features_files)]
# list of features in each file
features_list = [f.read().splitlines() for f in (open(x,'r') for x in features_files)]
FEATURES = dict(zip(features_count, features_list)) # zip everything into a dictionary
KNOWN_LABELS = ("portscan", "ftp.?patator", "ssh.?patator", "bot", "infiltration", "heartbleed", "dos.?hulk", "dos.?goldeneye", "dos.?slowloris", "dos.?slowhttptest", "ddos")

if path.isdir(args.files[-1]): # directory output, process files separately
    of_names = [args.files[-1] + '/' + path.splitext(path.basename(in_file))[0] + '.test' for in_file in args.files[:-1]]
else: # file is a regular file, process all inputs into this file
    of_names = [args.files[-1],]

of_descriptors = [open(of, 'w') for of in of_names]
print("Processing...")
for i, in_file in enumerate(args.files[:-1]):
    new_label = None
    for k in KNOWN_LABELS:
        if re.search(k, in_file, re.I):
            new_label = k
            break
    if not new_label: # ask user for new label
        new_label = input('Unable to choose label for %s, give me one > ' % in_file)
    total_chunks = sum(1 for row in open(in_file,'r')) / CHUNKSIZE
    for c, chunk in enumerate(pandas.read_csv(in_file,chunksize=CHUNKSIZE)):
        progress_bar((c+1) / total_chunks * 100, initial_text=in_file+' ', bar_body="\033[34m-\033[m", bar_arrow="\033[34m>\033[m", align_right=True)
        chunk['Label'] = new_label # assign new label
        df = chunk[(chunk["Label"] != "BENIGN")] if not args.benign else chunk[(chunk["Label"] == "BENIGN")]
        df = df[df["Flow Byts/s"].notnull()]
        df = df[df["Flow Pkts/s"].notnull()]
        df[FEATURES[args.features]].to_csv(of_names[i%len(of_names)], mode='a', header=False)
    for of in of_descriptors: of.close() # close all the files

# print("Filtering...")
# of_descriptors = [(open(of, 'r'), open(of, 'w')) for of in of_names] # open files for Infinity filtering
# for i, ofrw in enumerate(of_descriptors):
#     ofrw[1].write(''.join([line for line in ofrw[0] if 'Infinity' not in line])) # filter Infinity lines
# for ofrw in of_descriptors: # close files
#     ofrw[0].close() # close readable
#     ofrw[1].close() # close writable


# NOTA: nao funciona diretamente com os .csv disponibilizados, so pelos criados pelo CICFlowMeter.
# Para funcionar com os .csv criados por eles tem que se editar os titulos dos .csv para corresponderem
# aos criados pelo programa, ou seja, com abreviaturas, sem espacos e sem a coluna ' Fwd Header Length'

# DATASETS DISPONIBILIZADOS: 84 features + 1 feature (no caso do DDoS, 'External IP')
# DATASETS A SER CRIADOS: 83 features, nao incluem em caso algum o atributo 'External IP'
