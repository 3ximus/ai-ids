#!/usr/bin/env python
from __future__ import print_function
import sys, argparse, pandas
import numpy as np
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
op.add_argument('-f9', '--features9', action='store_true', help="9 feats", dest='f9')
op.add_argument('-f21', '--features21', action='store_true', help="21 feats", dest='f21')
op.add_argument('-fall', '--featuresall', action='store_true', help="all feats", dest='fall')
args = op.parse_args()
chunksize = 10 ** 4

#USAGE: python compact_flows.py ../csv/datasets/*.csv ../csv/other-datasets/compacted.csv
of = open(args.files[-1], 'w')
for in_file in args.files[:-1]:
    print("FILE:", in_file)
    if args.csv:
        for chunk in pandas.read_csv(in_file,chunksize=chunksize):          # process dataframes chunk by chunk for memory optimization because my pc is great
            df = chunk[(chunk["Label"] != "BENIGN")] if not args.benign else chunk[(chunk["Label"] == "BENIGN")]
            df = df[df["Flow Byts/s"].notnull()]
            df = df[df["Flow Pkts/s"].notnull()]
            # 9 features
            if args.f9:
                df[["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd IAT Mean", "Bwd IAT Mean", "Label"]].to_csv(args.files[-1], mode='a', header=False)
            # more features
            elif args.f21:
                df[["Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkts/s", "Bwd Pkts/s", "Fwd IAT Mean", "Bwd IAT Mean", "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts", "Label"]].to_csv(args.files[-1], mode='a', header=False)
            # all usable features
            elif args.fall:
                df[["Flow Duration","Tot Fwd Pkts","Tot Bwd Pkts","TotLen Fwd Pkts","TotLen Bwd Pkts","Fwd Pkt Len Max","Fwd Pkt Len Min","Fwd Pkt Len Mean","Fwd Pkt Len Std","Bwd Pkt Len Max","Bwd Pkt Len Min","Bwd Pkt Len Mean","Bwd Pkt Len Std","Flow Byts/s","Flow Pkts/s","Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Tot","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Tot","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min","Fwd Header Len","Bwd Header Len","Fwd Pkts/s","Bwd Pkts/s","Pkt Len Min","Pkt Len Max","Pkt Len Mean","Pkt Len Std","Pkt Len Var","Down/Up Ratio","Pkt Size Avg","Fwd Seg Size Avg","Bwd Seg Size Avg","Fwd Byts/b Avg","Fwd Pkts/b Avg","Fwd Blk Rate Avg","Bwd Byts/b Avg","Bwd Pkts/b Avg","Bwd Blk Rate Avg","Subflow Fwd Pkts","Subflow Fwd Byts","Subflow Bwd Pkts","Subflow Bwd Byts","Init Fwd Win Byts","Init Bwd Win Byts","Fwd Act Data Pkts","Fwd Seg Size Min","Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min","Label"]].to_csv(args.files[-1], mode='a', header=False)
            # Removed:
            # Flow ID,Src IP,Src Port,Dst IP,Dst Port,Protocol,Timestamp,Fwd PSH Flags,Bwd PSH Flags,Fwd URG Flags,Bwd URG Flags,FIN Flag Cnt,SYN Flag Cnt,RST Flag Cnt,PSH Flag Cnt,ACK Flag Cnt,URG Flag Cnt,CWE Flag Count,ECE Flag Cnt
    else:
        fd = open(in_file, 'r')
        ln = sum(1 for x in fd)
        fd.seek(0)
        title = fd.readline().split(',')
        of = open(args.files[-1], 'w')
        for i,line in enumerate(fd):
            progress_bar(i / (ln-2)*100, initial_text="Spitting: ", bar_body="\033[34m-\033[m", bar_empty=" ", bar_arrow="\033[34m>\033[m", show_percentage=True)
            if not args.benign and "BENIGN" not in line:
                write_to_file(of, line, title)
            elif args.benign and "BENIGN" in line:
                write_to_file(of, line, title)
        fd.close()

if args.csv:
    of.close()
    of = open(args.files[-1], 'r')
    noinf_str = ""
    for line in of:
        if line.find("Infinity")==-1:
            noinf_str+=line
    of.close()
    of = open(args.files[-1],'w')
    of.write(noinf_str)

of.close()
# NOTA: nao funciona diretamente com os .csv disponibilizados, so pelos criados pelo CICFlowMeter. 
# Para funcionar com os .csv criados por eles tem que se editar os titulos dos .csv para corresponderem
# aos criados pelo programa, ou seja, com abreviaturas, sem espacos e sem a coluna ' Fwd Header Length'

# DATASETS DISPONIBILIZADOS: 84 features + 1 feature (no caso do DDoS, 'External IP')
# DATASETS A SER CRIADOS: 83 features, nao incluem em caso algum o atributo 'External IP'