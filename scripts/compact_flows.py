#!/usr/bin/env python
import sys
from loading import progress_bar

with open(sys.argv[-1], 'w') as of:
    # example usage: python compact_flows.py ../csv/*.csv ../flows/compacted_flows.txt
    for in_file in sys.argv[1:-1]:
        fd = open(in_file, 'r')
        ln = sum(1 for x in fd)
        fd.seek(0)
        title = fd.readline().split(',')
        print "FILE:", in_file
        i = 1
        line = ""
        for i,line in enumerate(fd):
            of.write("-------------- Flow " + str(i+1) + " --------------\n")
            splitted = line.split(',')
            progress_bar(i / (ln-2)*100, initial_text="Spitting: ", bar_body="\033[34m-\033[m", bar_empty=" ", bar_arrow="\033[34m\033[m", show_percentage=True)
            if "BENIGN" not in line:
                for j,elem in enumerate(splitted):
                    of.write(title[j].strip(' \n') + ": " + elem + "\n")
        fd.close()
