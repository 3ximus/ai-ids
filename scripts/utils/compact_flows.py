#!/usr/bin/env python
from __future__ import print_function
import sys, argparse, pandas
from os import path
import numpy as np
from loading import progress_bar
import re

op = argparse.ArgumentParser( description="Select features or filter MALIGN / BENIGN flows")
op.add_argument('files', metavar='file', nargs='*', help='list of input files and output [last file is the output, if this file is a directory all files are processed separately into their respective output files]')
op.add_argument('-b', '--benign', action='store_true', help="print only benign flows", dest='benign')
op.add_argument('-r', '--replace-label', action='store_true', help="replace labels, if a known label is found its done automaticaly otherwise the user is prompted to choose a label", dest='rlabel')
op.add_argument('-f9', '--features9', action='store_const', help="9 features", dest='features', const=9)
op.add_argument('-f21', '--features21', action='store_const', help="21 features", dest='features', const=21)
op.add_argument('-fall', '--featuresall', action='store_const', help="all features", dest='features', const=65)
args = op.parse_args()

CHUNKSIZE = 10 ** 4

# READ FEATURES

features_list = [f.read().splitlines() for f in [open(x,'r') for x in ['features/9.txt', 'features/21.txt', 'features/all.txt']]]
FEATURES = dict(zip((9,21,65), features_list)
KNOWN_LABELS = ["portscan", "ftp.?patator", "ssh.?patator", "bot", "infiltration", "heartbleed", "dos.?hulk", "dos.?goldeneye", "dos.?slowloris", "dos.?slowhttptest", "ddos"]

out_file = args.files[-1]
if path.isdir(out_file): # directory output, process files separately
    of_names = [path.splitext(path.basename(in_file))[0] + '.test' for in_file in args.files[:-1]]
else: # file is a regular file, process all inputs into this file
    of_names = [out_file,]

n_inputs = len(args.files[:-1])
of_descriptors = [open(of, 'w') for of in of_names]
print("Processing...")
for i, in_file in enumerate(args.files[:-1]):
    progress_bar(i / n_inputs * 100, initial_text=in_file, bar_body="\033[34m-\033[m", bar_empty=" ", bar_arrow="\033[34m>\033[m")
# process dataframes chunk by chunk for memory optimization because my pc is great
    for chunk in pandas.read_csv(in_file,chunksize=CHUNKSIZE):
        if args.rlabel:
            # TODO REPLACE LABEL
        df = chunk[(chunk["Label"] != "BENIGN")] if not args.benign else chunk[(chunk["Label"] == "BENIGN")]
        df = df[df["Flow Byts/s"].notnull()]
        df = df[df["Flow Pkts/s"].notnull()]
        df[args.features].to_csv(of_names[i%len(of_names)], mode='a', header=False)
    for of in of_descriptors: of.close() # close all the files

print("Filtering...")
of_descriptors = [(open(of, 'r'), open(of, 'w')) for of in of_names] # open files for Infinity filtering
for i, ofrw in enumerate(of_descriptors):
    ofrw[1].write(''.join([line for line in ofrw[0] if 'Infinity' not in line])) # filter Infinity lines
for ofrw in of_descriptors: # close files
    ofrw[0].close() # close readable
    ofrw[1].close() # close writable


# NOTA: nao funciona diretamente com os .csv disponibilizados, so pelos criados pelo CICFlowMeter.
# Para funcionar com os .csv criados por eles tem que se editar os titulos dos .csv para corresponderem
# aos criados pelo programa, ou seja, com abreviaturas, sem espacos e sem a coluna ' Fwd Header Length'

# DATASETS DISPONIBILIZADOS: 84 features + 1 feature (no caso do DDoS, 'External IP')
# DATASETS A SER CRIADOS: 83 features, nao incluem em caso algum o atributo 'External IP'
