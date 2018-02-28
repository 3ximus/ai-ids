#!/usr/bin/env python
import sys
import re

def spit(in_file, out_file):
    fd = open(in_file, 'r')
    of = open(out_file, 'w')
    title = fd.readline().split(',')
    i = 0
    line = ""
    try:
        for i,line in enumerate(fd):
            of.write("-------------- Flow " + str(i+1) + " --------------\n")
            splitted = line.split(',')
            for j,elem in enumerate(splitted):
                of.write(title[j].strip(' \n') + ": " + elem + "\n")
    except UnicodeDecodeError:
        print("ERROR:" , line)
    fd.close()
    of.close()

if __name__ == "__main__":
    spit(sys.argv[1], sys.argv[2])
