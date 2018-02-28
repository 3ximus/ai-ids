#!/usr/bin/env python

import pandas, sys
for fd in sys.argv[1:]:
    df = pandas.read_csv(fd)
    labels = df[" Label"]
    print(fd)
    print(labels.value_counts())

# FINflag = df["FIN Flag Count"]
# SYNflag = df[" SYN Flag Count"]
# RSTflag = df[" RST Flag Count"]
# PSHflag = df[" PSH Flag Count"]
# ACKflag = df[" ACK Flag Count"]
# URGflag = df[" URG Flag Count"]
# ECEflag = df[" ECE Flag Count"]
# CWRflag = df[" CWE Flag Count"]

# outfile = open('csv/csvdataset.txt','w')
# dataset_length = len(labels)
# i=0
# while(i<dataset_length):
# 	outfile.write((str(1) if labels[i]=="BENIGN" else str(0)) + str(FINflag[i]) + str(SYNflag[i]) + str(RSTflag[i]) + str(PSHflag[i]) + str(ACKflag[i]) + str(URGflag[i])  + str(ECEflag[i]) + str(CWRflag[i]) + '\n')
# 	i+=1
# outfile.close()


