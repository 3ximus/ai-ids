#!/usr/bin/env python
import sys, pandas
df = pandas.read_csv(sys.argv[1])
df[[" Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets", "Fwd Packets/s", " Bwd Packets/s", " Fwd IAT Mean", " Bwd IAT Mean", " Label"]].to_csv(sys.argv[2])


