#!/usr/bin/env python

from __future__ import print_function
from scapy.all import *
import time
import sys

tcp_flags = {
    'F': 'FIN',			# Main flag (finish connection)
    'S': 'SYN',			# Main flag (ask for connection)
    'R': 'RST',			# Main flag (reset connection)
    'P': 'PSH',			# Urgent (don't buffer the packet, process it)
    'A': 'ACK',			# Main flag (acknowledge connection)
    'U': 'URG',			# Urgent (don't buffer the packet, process it)
    'E': 'ECE',			# Used to check if the receiving host is ECN capable
    'C': 'CWR',			# Used to Ack the receival of a ECE packet
}

ip_flags = {
	'R': 'Reserved',
    'DF': 'Don\'t Fragment',
	'MF': 'More Fragments',
}

infile = open(sys.argv[1],"r")
pkts = rdpcap(infile)
infile.close()
outfile = open(sys.argv[2],"w")
npkt=0
for i,p in enumerate(pkts):
    # TCP FLAGS
    # The first 8 elements of a line in dataset ([0,7])
    formatted_feature_lst = [0,0,0,0,0,0,0,0]
    current_flags = p.sprintf('%TCP.flags%')
    for tcp_flag in current_flags:
        if(tcp_flag=='F'):
            formatted_feature_lst[0]=1
        elif(tcp_flag=='S'):
            formatted_feature_lst[1]=1
        elif(tcp_flag=='R'):
            formatted_feature_lst[2]=1
        elif(tcp_flag=='P'):
            formatted_feature_lst[3]=1
        elif(tcp_flag=='A'):
            formatted_feature_lst[4]=1
        elif(tcp_flag=='U'):
            formatted_feature_lst[5]=1
        elif(tcp_flag=='E'):
            formatted_feature_lst[6]=1
        elif(tcp_flag=='C'):
            formatted_feature_lst[7]=1
        else:
            print("????? (error: TCP flags)")
    '''
    # IP FLAGS
    # The [8,10]th elements
    current_flags = p.sprintf('%IP.flags%')
    if(current_flags.find("R")==0):
        formatted_feature_lst[8]=1
    if(current_flags.find("DF")==0):
        formatted_feature_lst[9]=1
    if(current_flags.find("MF")==0):
        formatted_feature_lst[10]=1
    #packet_length = p.sprintf('%IP.len%')
    
    pkt_hour = time.strftime('%H',time.localtime(p.time))
    if not (pkt_hour>=8 and pkt_hour<=23):
        formatted_feature_lst[11] = 1
    '''
    for elem in formatted_feature_lst:
        outfile.write(str(elem))
    outfile.write("\n")
outfile.close()
