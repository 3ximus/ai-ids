#!/usr/bin/env python3

"""This file contains the extraction tool to convert pcap files into a dataset format

AUTHORS:

Joao Meira <joao.meira@tekever.com>
Fabio Almeida <fabio4335@gmail.com>
"""

import dpkt
import numpy as np
import os, sys, time, datetime, socket, argparse

from dpkt.compat import compat_ord
from collections import OrderedDict


# =====================
#     CLI OPTIONS
# =====================

op = argparse.ArgumentParser(description='PCAP flow parser')
op.add_argument('files', metavar='file', nargs='+', help='pcap file to parse flows from')
op.add_argument('-l', '--label', help="label all the flows", dest='label', default='unknown')
op.add_argument('-o', '--out-dir', help="output directory", dest='outdir')
op.add_argument('-c', '--check-transport-data-length', action='store_true', help='verbose output', dest='check_transport_data_length')
op.add_argument('-v', '--verbose', action='store_true', help='verbose output', dest='verbose')

args = op.parse_args()

datetime_format1 = "%Y-%m-%d %H:%M:%S.%f"
datetime_format2 = "%Y-%m-%d %H:%M:%S"
scale_factor = 0.001    # milliseconds --> seconds
packet_len_minimum = 64

def flow_id_to_communication_id(flow_id):
    splitted_flow_id = flow_id.split('-')
    return splitted_flow_id[0] + '-' + splitted_flow_id[2]

def gen_flow_str(flow_features):
    return flow_id_to_str(flow_features[0]) + ',' + ','.join(map(str,flow_features[1:])) + '\n'

def flow_id_to_str(flow_id):
    return '-'.join(map(str,flow_id))

def unix_time_millis(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

def mac_addr(address):
    '''Convert a MAC address to a readable/printable string
       Args:
           address (str): a MAC address in hex form (e.g. '\x01\x02\x03\x04\x05\x06')
       Returns:
           str: Printable/readable MAC address
    '''
    return ':'.join('%02x' % compat_ord(b) for b in address)


def inet_to_str(inet):
    '''Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    '''
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

# PROCESS PCAP
def process_pcap(file):
    pcap = dpkt.pcap.Reader(file)
    n_tcp=0
    n_udp=0
    packet_properties=[]

    for timestamp, buf in pcap:
        # Unpack the Ethernet frame (mac src/dst, ethertype)
        eth = dpkt.ethernet.Ethernet(buf)

        # Check if the Ethernet data contains an IP packet. If it doesn't, ignore it
        if not isinstance(eth.data, dpkt.ip.IP):
            continue

        # Unpack the data within the Ethernet frame (the IP packet)
        ip = eth.data

        # Pull out fragment information
        do_not_fragment = bool(ip.off & dpkt.ip.IP_DF)
        more_fragments = bool(ip.off & dpkt.ip.IP_MF)
        # fragment_offset = ip.off & dpkt.ip.IP_OFFMASK

        transport_layer=ip.data
        transport_protocol_name=type(transport_layer).__name__

        if transport_protocol_name in ('TCP'):
            if transport_protocol_name=='TCP':
                n_tcp+=1
                transport_protocol_code=6
                fin_flag = ( transport_layer.flags & dpkt.tcp.TH_FIN ) != 0
                syn_flag = ( transport_layer.flags & dpkt.tcp.TH_SYN ) != 0
                rst_flag = ( transport_layer.flags & dpkt.tcp.TH_RST ) != 0
                psh_flag = ( transport_layer.flags & dpkt.tcp.TH_PUSH) != 0
                ack_flag = ( transport_layer.flags & dpkt.tcp.TH_ACK ) != 0
                urg_flag = ( transport_layer.flags & dpkt.tcp.TH_URG ) != 0
                ece_flag = ( transport_layer.flags & dpkt.tcp.TH_ECE ) != 0
                cwr_flag = ( transport_layer.flags & dpkt.tcp.TH_CWR ) != 0
                # tcp_seq = transport_layer.seq               # tcp seq number: not used to separate/select flows as the implemented rules alone seem to be working really fine
            elif transport_protocol_name=='UDP':
                n_udp+=1
                transport_protocol_code=17

            ip_header_len = (ip.__hdr_len__ + len(ip.opts))
            transport_header_len = transport_layer.__hdr_len__ + len(transport_layer.opts)
            header_len = 14 + ip_header_len + transport_header_len    # header definition includes all except tcp.data (ip header, ip options, tcp header, tcp options)
            pkt_len = len(buf)

            # ethernet zero-byte padding until 64 bytes are reached
            if pkt_len>=packet_len_minimum:                                       # ethernet frame minimum size (minimum packet length)
                pkt_size = pkt_len - header_len                                   # packet size (tcp data length)
            else:
                eth_padding_bytes = pkt_len - header_len
                # header len will ignore eth padding bytes
                pkt_len = pkt_len - eth_padding_bytes
                pkt_size = pkt_len - header_len                         # ethernet zero-byte padding until 64 bytes are reached

            if pkt_size!=len(transport_layer.data) and args.check_transport_data_length:
                print("Error on packet no." + str(n_udp + n_tcp) + ".Packet size should always correspond to tcp data length.", file=sys.stderr)
                print(len(transport_layer.data),'!=',pkt_size, file=sys.stderr)
                exit()

            direction_id=(inet_to_str(ip.src),transport_layer.sport,inet_to_str(ip.dst),transport_layer.dport,transport_protocol_code,0)          # src ip, src port, dst ip, dst port, protocol, inflow_counter
            packet_info = (direction_id,str(datetime.datetime.utcfromtimestamp(timestamp)),pkt_len,header_len,pkt_size,do_not_fragment,more_fragments,          \
                fin_flag,syn_flag,rst_flag,psh_flag,ack_flag,urg_flag,ece_flag,cwr_flag) if transport_protocol_name=='TCP'\
                else (direction_id,str(datetime.datetime.utcfromtimestamp(timestamp)),pkt_len,header_len,pkt_size,do_not_fragment,more_fragments)
            packet_properties.append(packet_info)
            # eventually_useful = (mac_addr(eth.src),mac_addr(eth.dst),eth.type,fragment_offset)
    if args.verbose:
        print('Total number of packets: %d [TCP %d | UDP %d ]' % (n_udp + n_tcp, n_tcp, n_udp), file=sys.stderr)
    return packet_properties

def build_uniflows(packet_properties):
    #associate uniflow_ids to packets
    uniflows = dict()
    uniflow_ids = []
    for prop in packet_properties:
        uniflow_ids.append(prop[0])
        if prop[0] in uniflows:
            uniflows[prop[0]].append(prop)
        else:
            uniflows[prop[0]]=[prop]
    uniflow_ids=list(OrderedDict.fromkeys(uniflow_ids))             #remove duplicates mantaining order
    if args.verbose:
        print('Number of unidirectional flows (w/o flag separation):',len(uniflow_ids), file=sys.stderr)
    return uniflows,uniflow_ids

def parse_duplicates(uniflow_ids):
    #join unidirectional flows with their counterpart (flows/conversations)
    duplicates_parsed = []
    for uniflow_id in uniflow_ids:
        try:
            custom_items = [ duplicates_parsed[i] for i in range(5) ]
        except IndexError:
            custom_items = []
        if uniflow_id[0:-1] not in custom_items:
            duplicates_parsed.append(uniflow_id)
            duplicates_parsed.append((uniflow_id[2],uniflow_id[3],uniflow_id[0],uniflow_id[1],uniflow_id[4],uniflow_id[5]))
    return list(OrderedDict.fromkeys(duplicates_parsed))

def build_nsp_flows(uniflows, duplicates_parsed):
    #join unidirectional flow information into its bidirectional flow equivalent
    nsp_flows=dict()
    #non-separated flow ids (flows that haven't yet taken into account the begin/end flow flags)
    nsp_flow_ids=[]
    j=0
    while(j<len(duplicates_parsed)):
        nsp_flow_id = duplicates_parsed[j]
        duplicate_id = duplicates_parsed[j+1]
        # have in mind every flow_id in this list will constitute the first packet ever recorded in that flow,
        # which is assumed to be the first request, i.e., a 'forward' packet
        nsp_flow_ids.append(nsp_flow_id)
        try:
            nsp_flows[nsp_flow_id] = uniflows[nsp_flow_id] + uniflows[duplicate_id]
        except KeyError:
            nsp_flows[nsp_flow_id] = uniflows[nsp_flow_id]
        j+=2
    if args.verbose:
        print('Number of bidirectional flows (w/o flag separation):',len(nsp_flow_ids), file=sys.stderr)
    return nsp_flows, nsp_flow_ids

def build_tcpflows(nsp_flows,nsp_flow_ids):
    # TODO: separate using tcp_seq too
    # fin,syn,rst,psh,ack,urg,ece,cwr (2,...,9)
    flows=dict()
    flow_ids=[]         # ordered flow keys (by flow start time)

    # create conventionally correct flows (conversations)
    for key in nsp_flow_ids:
        flow = nsp_flows[key]
        flow.sort(key=lambda x: x[1])       # sorting the packets in each flow by date-and-time
        if key[4]==17: #udp flow
            flows[key] = flow
            flow_ids.append(key)
            continue
        flow_n_pkts = len(flow)

        if flow_n_pkts==0:
            raise ValueError('The flow can\'t have 0 packets.')
        elif flow_n_pkts in (1,2,3):     #1/2/3 pacotes num so nsp_flow_id perfazem no maximo 1 e 1 so flow
            flows[key] = flow
            flow_ids.append(key)
        else:
            i=0
            last_i=0
            flow_begin=False
            inflow_counter=0
            while i<flow_n_pkts:
                fin1,syn1,rst1,psh1,ack1,urg1,ece1,cwr1=flow[i][-8:]
                if i==flow_n_pkts-2:   # penultimate packet
                    fin2,syn2,rst2,psh2,ack2,urg2,ece2,cwr2=flow[i+1][-8:]
                    fin3,syn3,rst3,psh3,ack3,urg3,ece3,cwr3=[False]*8
                elif i==flow_n_pkts-1: # last packet
                    fin2,syn2,rst2,psh2,ack2,urg2,ece2,cwr2=[False]*8
                    fin3,syn3,rst3,psh3,ack3,urg2,ece3,cwr3=[False]*8
                else:               # other packets
                    fin2,syn2,rst2,psh2,ack2,urg2,ece2,cwr2=flow[i+1][-8:]
                    fin3,syn3,rst3,psh3,ack3,urg3,ece3,cwr3=flow[i+2][-8:]

                ###### TCP FLOW RULES ######
                # r1,r2: begin flow
                r1 = (syn1 and not ack1) and (syn2 and ack2) and ack3           # 3-way handshake (full-duplex), syn+syn-ack+ack / syn+syn-ack+syn-ack
                r2 = (syn1 and not ack1) and ack2                               # 2-way handshake (half-duplex), syn+syn-ack / syn+ack
                # r3,r4: end flow
                r3 = fin1 and (fin2 and ack2) and ack3
                r4 = rst1 and not rst2

                # consider flow begin or ignore it (considering it is safer, but not considering it will leave out flows that have started before the capture)
                # the only rule used will be the half-duplex handshake rule because it is inclusive of the full-duplex handshake rule
                if r2:
                    flow_begin=True

                # we consider flows only the ones that start with a 2 or 3-way handshake (r1,r2)
                # the flow end conditions are r3 and r4, (fin,fin-ack,ack)/(rst,!rst,---), or if the packet is the last one of the existing communication
                if flow_begin:
                    if r3:
                        new_key=(key[0],key[1],key[2],key[3],key[4],key[5]+inflow_counter)
                        flows[new_key] = flow[last_i:i+3]
                        flow_ids.append(new_key)
                        flow_begin=False
                        last_i=i+3
                        inflow_counter+=1
                    elif r4 or i==flow_n_pkts-1:
                        new_key=(key[0],key[1],key[2],key[3],key[4],key[5]+inflow_counter)
                        flows[new_key] = flow[last_i:i+1]
                        flow_ids.append(new_key)
                        flow_begin=False
                        last_i=i+1
                        inflow_counter+=1
                i+=1
    return flows,flow_ids

def calculate_flows_features(flows,flow_ids,label):
    '''This function is a generator'''
    flow_properties=[]
    for flow_id in flow_ids:
        flow_n_pkts = len(flows[flow_id])
        direction_id = flows[flow_id][0][0]
        flow_iats = []
        fwd_iats = []
        bwd_iats = []
        flow_pkt_lens = []
        fwd_pkt_lens = []
        bwd_pkt_lens = []
        flow_header_lens = []
        fwd_header_lens = []
        bwd_header_lens = []
        flow_pkt_sizes = []
        fwd_pkt_sizes = []
        bwd_pkt_sizes = []
        flow_n_data_pkts = 0
        fwd_n_data_pkts = 0
        bwd_n_data_pkts = 0
        flow_flags = []

        i = 0
        while i<flow_n_pkts:
            if i>=1:
                try:
                    first_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][i-1][1], datetime_format1))
                except ValueError:
                    first_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][i-1][1], datetime_format2))
                try:
                    second_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][i][1], datetime_format1))
                except ValueError:
                    second_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][i][1], datetime_format2))
                current_iat = scale_factor*(second_pkt_time - first_pkt_time)
                flow_iats.append(current_iat)
                if flows[flow_id][i-1][0]==direction_id:
                    fwd_iats.append(current_iat)
                else:
                    bwd_iats.append(current_iat)

            current_pkt_len = flows[flow_id][i][2]
            current_header_len = flows[flow_id][i][3]
            current_pkt_size = flows[flow_id][i][4]
            current_flags = flows[flow_id][i][-10:]

            flow_flags.append(current_flags)
            flow_pkt_lens.append(current_pkt_len)
            flow_header_lens.append(current_header_len)
            flow_pkt_sizes.append(current_pkt_size)

            if flows[flow_id][i][0]==direction_id:
                fwd_pkt_lens.append(current_pkt_len)
                fwd_header_lens.append(current_header_len)
                fwd_pkt_sizes.append(current_pkt_size)
                if current_header_len != current_pkt_len:
                    flow_n_data_pkts+=1
                    fwd_n_data_pkts+=1
            else:
                bwd_pkt_lens.append(current_pkt_len)
                bwd_header_lens.append(current_header_len)
                bwd_pkt_sizes.append(current_pkt_size)
                if current_header_len != current_pkt_len:
                    flow_n_data_pkts+=1
                    bwd_n_data_pkts+=1
            i+=1

        # number of packets (all times in seconds)
        try:
            first_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][0][1], datetime_format1))
        except ValueError:
            first_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][0][1], datetime_format2))

        try:
            last_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][flow_n_pkts-1][1], datetime_format1))
        except ValueError:
            last_pkt_time = unix_time_millis(datetime.datetime.strptime(flows[flow_id][flow_n_pkts-1][1], datetime_format2))
        flow_duration = scale_factor*(last_pkt_time - first_pkt_time)

        fwd_n_pkts = len(fwd_pkt_lens)
        bwd_n_pkts = len(bwd_pkt_lens)

        if flow_duration==0:
            flow_pkts_per_sec = fwd_pkts_per_sec = bwd_pkts_per_sec = 0
        else:
            flow_pkts_per_sec = flow_n_pkts/flow_duration
            fwd_pkts_per_sec = fwd_n_pkts/flow_duration
            bwd_pkts_per_sec = bwd_n_pkts/flow_duration

        # packet lengths
        flow_pkt_len_total = float(np.sum(flow_pkt_lens))           # to add and test x
        flow_bytes_per_sec = 0 if flow_duration==0 else float(flow_pkt_len_total/flow_duration)
        flow_pkt_len_mean = float(np.mean(flow_pkt_lens))
        flow_pkt_len_std = float(np.std(flow_pkt_lens))
        flow_pkt_len_var = float(np.var(flow_pkt_lens))
        flow_pkt_len_max = float(np.max(flow_pkt_lens))
        flow_pkt_len_min = float(np.min(flow_pkt_lens))

        fwd_pkt_len_total = float(np.sum(fwd_pkt_lens))
        fwd_pkt_len_mean = float(np.mean(fwd_pkt_lens))
        fwd_pkt_len_std = float(np.std(fwd_pkt_lens))
        fwd_pkt_len_var = float(np.var(fwd_pkt_lens))               # to add and test x
        fwd_pkt_len_max = float(np.max(fwd_pkt_lens))
        fwd_pkt_len_min = float(np.min(fwd_pkt_lens))

        if len(bwd_pkt_lens)!=0:
            bwd_pkt_len_total = float(np.sum(bwd_pkt_lens))
            bwd_pkt_len_mean = float(np.mean(bwd_pkt_lens))
            bwd_pkt_len_std = float(np.std(bwd_pkt_lens))
            bwd_pkt_len_var = float(np.var(bwd_pkt_lens))           # to add and test x
            bwd_pkt_len_max = float(np.max(bwd_pkt_lens))
            bwd_pkt_len_min = float(np.min(bwd_pkt_lens))
        else:
            bwd_pkt_len_total = bwd_pkt_len_mean = bwd_pkt_len_std = bwd_pkt_len_var = bwd_pkt_len_max = bwd_pkt_len_min = 0

        # header lengths
        fwd_header_len_total = float(np.sum(fwd_header_lens))                                    # 14 byte Ether header + ip header + tcp/udp header
        bwd_header_len_total = float(np.sum(bwd_header_lens)) if len(bwd_header_lens)!=0 else 0  # 14 byte Ether header + ip header + tcp/udp header

        # packet size
        flow_pkt_size_mean = float(np.mean(flow_pkt_sizes))
        flow_pkt_size_std = float(np.std(flow_pkt_sizes))       # to add and test x
        flow_pkt_size_max = float(np.max(flow_pkt_sizes))       # to add and test x
        flow_pkt_size_min = float(np.min(flow_pkt_sizes))       # to add and test x

        fwd_pkt_size_mean = float(np.mean(fwd_pkt_sizes))
        fwd_pkt_size_std = float(np.std(fwd_pkt_sizes))         # to add and test x
        fwd_pkt_size_max = float(np.max(fwd_pkt_sizes))         # to add and test x
        fwd_pkt_size_min = float(np.min(fwd_pkt_sizes))

        if len(bwd_pkt_sizes)!=0:
            bwd_pkt_size_mean = float(np.mean(bwd_pkt_sizes))
            bwd_pkt_size_std = float(np.std(bwd_pkt_sizes))     # to add and test x
            bwd_pkt_size_max = float(np.max(bwd_pkt_sizes))     # to add and test x
            bwd_pkt_size_min = float(np.min(bwd_pkt_sizes))     # to add and test x
        else:
            bwd_pkt_size_mean = bwd_pkt_size_std = bwd_pkt_size_max = bwd_pkt_size_min = 0


        # packet inter-arrival times
        if len(flow_iats)!=0:
            flow_iat_total = float(np.sum(flow_iats))           # to add and test x
            flow_iat_mean = float(np.mean(flow_iats))
            flow_iat_std = float(np.std(flow_iats))
            flow_iat_max = float(np.max(flow_iats))
            flow_iat_min = float(np.min(flow_iats))
        else:
            flow_iat_total = flow_iat_mean = flow_iat_std = flow_iat_max = flow_iat_min = 0


        if len(fwd_iats)!=0:
            fwd_iat_total = float(np.sum(fwd_iats))
            fwd_iat_mean = float(np.mean(fwd_iats))
            fwd_iat_std = float(np.std(fwd_iats))
            fwd_iat_max = float(np.max(fwd_iats))
            fwd_iat_min = float(np.min(fwd_iats))
        else:
            fwd_iat_total = fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0


        if len(bwd_iats)!=0:
            bwd_iat_total = float(np.sum(bwd_iats))
            bwd_iat_mean = float(np.mean(bwd_iats))
            bwd_iat_std = float(np.std(bwd_iats))
            bwd_iat_max = float(np.max(bwd_iats))
            bwd_iat_min = float(np.min(bwd_iats))
        else:
            bwd_iat_total = bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0


        # flag counts (ip/tcp)
        flow_flag_counts = [0]*10           # to add and test x

        for flags in flow_flags:
            for i,flag in enumerate(flags):
                if flag:
                    flow_flag_counts[i]+=1

        flow_properties = \
            [flow_id,fwd_header_len_total,bwd_header_len_total,flow_pkt_size_mean,flow_pkt_size_std,flow_pkt_size_max,flow_pkt_size_min,\
            fwd_pkt_size_mean,fwd_pkt_size_std,fwd_pkt_size_max,bwd_pkt_size_mean,bwd_pkt_size_std,bwd_pkt_size_max,bwd_pkt_size_min,fwd_pkt_size_min,flow_duration,\
            fwd_n_pkts,bwd_n_pkts,flow_pkts_per_sec,fwd_pkts_per_sec,bwd_pkts_per_sec,flow_bytes_per_sec,\
            flow_pkt_len_total,flow_pkt_len_mean,flow_pkt_len_std,flow_pkt_len_var,flow_pkt_len_max,flow_pkt_len_min,\
            fwd_pkt_len_total,fwd_pkt_len_mean,fwd_pkt_len_std,fwd_pkt_len_var,fwd_pkt_len_max,fwd_pkt_len_min,\
            bwd_pkt_len_total,bwd_pkt_len_mean,bwd_pkt_len_std,bwd_pkt_len_var,bwd_pkt_len_max,bwd_pkt_len_min,\
            flow_iat_total,flow_iat_mean,flow_iat_std,flow_iat_max,flow_iat_min,\
            fwd_iat_total,fwd_iat_mean,fwd_iat_std,fwd_iat_max,fwd_iat_min,\
            bwd_iat_total,bwd_iat_mean,bwd_iat_std,bwd_iat_max,bwd_iat_min,\
            flow_n_data_pkts,fwd_n_data_pkts,bwd_n_data_pkts] + flow_flag_counts + [label]
        yield flow_properties

def generate_dataset(filename, flow_features_generator):
    features_header = 'flow_id,fwd_header_len_total,bwd_header_len_total,flow_pkt_size_mean,flow_pkt_size_std,flow_pkt_size_max,flow_pkt_size_min,fwd_pkt_size_mean,fwd_pkt_size_std,fwd_pkt_size_max,bwd_pkt_size_mean,bwd_pkt_size_std,bwd_pkt_size_max,bwd_pkt_size_min,fwd_pkt_size_min,flow_duration,fwd_n_pkts,bwd_n_pkts,flow_pkts_per_sec,fwd_pkts_per_sec,bwd_pkts_per_sec,flow_bytes_per_sec,flow_pkt_len_total,flow_pkt_len_mean,flow_pkt_len_std,flow_pkt_len_var,flow_pkt_len_max,flow_pkt_len_min,fwd_pkt_len_total,fwd_pkt_len_mean,fwd_pkt_len_std,fwd_pkt_len_var,fwd_pkt_len_max,fwd_pkt_len_min,bwd_pkt_len_total,bwd_pkt_len_mean,bwd_pkt_len_std,bwd_pkt_len_var,bwd_pkt_len_max,bwd_pkt_len_min,flow_iat_total,flow_iat_mean,flow_iat_std,flow_iat_max,flow_iat_min,fwd_iat_total,fwd_iat_mean,fwd_iat_std,fwd_iat_max,fwd_iat_min,bwd_iat_total,bwd_iat_mean,bwd_iat_std,bwd_iat_max,bwd_iat_min,flow_n_data_pkts,fwd_n_data_pkts,bwd_n_data_pkts,flow_df_count,flow_mf_count,flow_fin_count,flow_syn_count,flow_rst_count,flow_psh_count,flow_ack_count,flow_urg_count,flow_ece_count,flow_cwr_count,label\n'

    outfilename, _ = os.path.splitext(os.path.basename(filename))
    of = open('%s/%s.csv' % (args.outdir, outfilename),'w') if args.outdir else sys.stdout
    of.write(features_header)
    for flow_features in flow_features_generator:
        of.write(gen_flow_str(flow_features))
    if of != sys.stdout: of.close()


# PRINT FLOWS
def print_flows(file):
    start_time = time.time()

    packet_properties = process_pcap(file)
    print("File processed in \033[34m" + str(time.time() - start_time) + "\033[m seconds", file=sys.stderr)
    start_time = time.time()
    uniflows,uniflow_ids = build_uniflows(packet_properties)
    del(packet_properties)
    duplicates_parsed = parse_duplicates(uniflow_ids)
    del(uniflow_ids)
    flows,flow_ids = build_nsp_flows(uniflows, duplicates_parsed)
    del(uniflows)
    del(duplicates_parsed)
    flows,flow_ids = build_tcpflows(flows, flow_ids) # At this point, flow_ids are ordered by the flow start time and the packets in each flow are internally ordered by their timestamp

    # Print some information about the selected flows
    if args.verbose:
        all_pkts=0
        for flow_id in flow_ids:
            all_pkts+=len(flows[flow_id])
        print('Number of packets included in the flows\' analysis:',all_pkts, file=sys.stderr)
        print('Number of bidirectional flows (w/ flag separation):',len(flows), file=sys.stderr)

    # Error case
    if len(flows)==0:
        print('This pcap doesn\'t have any communication that satisfies our flow definition. Abort.', file=sys.stderr)
        return

    flow_features_generator = calculate_flows_features(flows, flow_ids, args.label)
    # Generate csv file
    generate_dataset(file.name, flow_features_generator)

    print("Dataset generated in \033[34m" + str(time.time() - start_time) + "\033[m seconds", file=sys.stderr)

if __name__ == '__main__':
    filenames = args.files
    for filename in filenames:
        print("Parsing " + filename + "...", file=sys.stderr)
        with open(filename, 'rb') as f:
            print_flows(f)
