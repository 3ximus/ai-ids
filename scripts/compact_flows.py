#!/usr/bin/env python
from __future__ import print_function
import sys, argparse, re, glob
import pandas
from os import path

# beacause python 2 -.-
try: input = raw_input
except NameError: pass

op = argparse.ArgumentParser(description="Select features, filter and label flows. Uses the .txt files inside features/ to choose wich features to extract")
op.add_argument('files', metavar='file', nargs='+', help='list of input files and output [last file is the output, if this file is a directory all files are processed separately into their respective output files]')
op.add_argument('-m', '--manual-label', action='store_true', help="always manually label data", dest='manual_label')
op.add_argument('-b', '--benign', action='store_true', help="write only benign flows, in absence of this flag writes only non benign flows", dest='benign')
op.add_argument('-f', '--features', help="number of features", dest='features')
args = op.parse_args()

CHUNKSIZE = 10 ** 4
if not args.features: args.features = 64 # give a default value of all features
args.features = int(args.features)
if not len(args.files) >= 2:
    op.print_usage()
    print('Input and output files must be given as argument')
    sys.exit(1)

# READ FEATURES FROM features/*.txt files
# list of files containing the features
features_files = glob.glob(path.dirname(sys.argv[0])+'/features/*.txt')
# count of #features in each feature file
features_count = [sum([1 for l in fd if l not in ('\n','',' ')])-1 for fd in (open(x,'r') for x in features_files)]
# list of features in each file
features_list = [f.read().splitlines() for f in (open(x,'r') for x in features_files)]
FEATURES = dict(zip(features_count, features_list)) # zip everything into a dictionary
KNOWN_LABELS = {"portscan": "PortScan", "ftp.?patator": "FTP-Patator", "ssh.?patator": "SSH-Patator", "bot": "Bot", "infiltration": "Infiltration", "heartbleed": "Heartbleed", "dos.?hulk": "DoS Hulk", "dos.?goldeneye": "DoS GoldenEye", "dos.?slowloris": "DoS slowloris", "dos.?slowhttptest": "DoS Slowhttptest", "ddos": "DDoS"}
print(FEATURES.keys())
if path.isdir(args.files[-1]): # directory output, process files separately
    of_names = [args.files[-1] + '/' + path.splitext(path.basename(in_file))[0] + '.test' for in_file in args.files[:-1]]
else: # file is a regular file, process all inputs into this file
    of_names = [args.files[-1],]

of_descriptors = [open(of, 'w') for of in of_names]
print("Processing...")
for i, in_file in enumerate(args.files[:-1]):
    print(in_file)
    nlabel = None
    for k in KNOWN_LABELS: # regex
        if re.search(k, in_file, re.I):
            nlabel = KNOWN_LABELS[k]
            break
    if args.manual_label or not nlabel: # ask user for new label
        nlabel = input('Choose label for %s %s > ' % (in_file, '[PRESS ENTER: %s]' % nlabel if nlabel else '')) or nlabel
    total_chunks = sum(1 for row in open(in_file,'r')) / CHUNKSIZE
    for c, chunk in enumerate(pandas.read_csv(in_file,chunksize=CHUNKSIZE)):
        chunk['Label'] = nlabel # assign new label
        df = chunk[(chunk["Label"] != "BENIGN")] if not args.benign else chunk[(chunk["Label"] == "BENIGN")]
        df = df[df["Flow Byts/s"].notnull()]
        df = df[df["Flow Pkts/s"].notnull()]
        try: df[FEATURES[args.features]].to_csv(of_names[i%len(of_names)], mode='a', header=False)
        except IndexError:
            print('Number of features is not correct according to the flag given (%d), please fix this.' % args.features)
            sys.exit(1)
    for of in of_descriptors: of.close() # close all the files

print("Filtering...")
read_f, write_f =  (open(of, 'r') for of in of_names), (open(of, 'w') for of in of_names) # open files for reading
for i, ofr in enumerate(read_f):
    write_me = ''.join([line for line in ofr if 'Infinity' not in line])
    ofr.close() # close read
    ofw = next(write_f)
    ofw.write(write_me)
    ofw.close() # close

# DATASETS DISPONIBILIZADOS: 84 features + 1 feature (no caso do DDoS, 'External IP')
# DATASETS A SER CRIADOS: 83 features, nao incluem em caso algum o atributo 'External IP'
