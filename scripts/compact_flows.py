#!/usr/bin/env python
from __future__ import print_function
import sys, argparse, pandas
from loading import progress_bar

def write_to_file(of, line, title):
    of.write("-------------- Flow " + str(i+1) + " --------------\n")
    splitted = line.split(',')
    for j,elem in enumerate(splitted):
        of.write(title[j].strip(' \n') + ": " + elem + "\n")

op = argparse.ArgumentParser( description="Neural Network for poker card combinations")
op.add_argument('files', metavar='file', nargs='*', help='')
op.add_argument('-b', '--benign', action='store_true', help="print Only Benign flows", dest='benign')
op.add_argument('-x', '--csv', action='store_true', help="print selected columns in csv format", dest='csv')
args = op.parse_args()

with open(args.files[-1], 'w') as of:
    # example usage: python compact_flows.py ../csv/*.csv ../flows/compacted_flows.txt
    for in_file in args.files[:-1]:
        if args.csv:
            df = pandas.read_csv(in_file)
            df = df[(df[' Label'] != "BENIGN")] if not args.benign else df[(df[' Label'] == "BENIGN")]
            df[[" Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets", "Fwd Packets/s", " Bwd Packets/s", " Fwd IAT Mean", " Bwd IAT Mean", " Label"]].to_csv(args.files[-1], mode='a', header=False)
        else:
            fd = open(in_file, 'r')
            ln = sum(1 for x in fd)
            fd.seek(0)
            title = fd.readline().split(',')
            print("FILE:", in_file)

            for i,line in enumerate(fd):
                progress_bar(i / (ln-2)*100, initial_text="Spitting: ", bar_body="\033[34m-\033[m", bar_empty=" ", bar_arrow="\033[34m>\033[m", show_percentage=True)
                if not args.benign and "BENIGN" not in line:
                    write_to_file(of, line, title)
                elif args.benign and "BENIGN" in line:
                    write_to_file(of, line, title)
            fd.close()

