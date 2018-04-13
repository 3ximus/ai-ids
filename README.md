# Artificially Inteligent Intrusion Detection System

### ISCX DATASET

`DATASETS/ISCX/Friday-WorkingHours-Afternoon-DDos-flow.csv`

|Type|Count|
|---|---|
|BENIGN   | 183910|
|DDoS     |  41835|

`DATASETS/ISCX/Friday-WorkingHours-Afternoon-PortScan-flow.csv`

|Type|Count|
|---|---|
|PortScan   | 158930|
|BENIGN     | 127537|

`DATASETS/ISCX/Friday-WorkingHours-Morning-flow.csv`

|Type|Count|
|---|---|
|BENIGN   | 189067|
|Bot      |   1966|

`DATASETS/ISCX/Monday-WorkingHours-flow.csv`

|Type|Count|
|---|---|
|BENIGN   |    529918|

`DATASETS/ISCX/Thursday-WorkingHours-Afternoon-Infilteration-flow.csv`

|Type|Count|
|---|---|
|BENIGN         | 288566|
|Infiltration   |     36|

`DATASETS/ISCX/Tuesday-WorkingHours-flow.csv`

|Type|Count|
|---|---|
|BENIGN        | 432074|
|FTP-Patator   |   7938|
|SSH-Patator   |   5897|

`DATASETS/ISCX/Wednesday-workingHours-flow.csv`

|Type|Count|
|---|---|
|BENIGN            |  440031|
|DoS Hulk          |  231073|
|DoS GoldenEye     |   10293|
|DoS slowloris     |    5796|
|DoS Slowhttptest  |    5499|
|Heartbleed        |      11|

### TOTAL NUMBER OF FLOWS

- malign 469274
- benign 2191103

## Useful Labels

- Flow Duration
- Total Fwd Packets
- Total Backward Packets
- Total Length of Fwd Packets
- Total Length of Bwd Packets
- Fwd Packets/s
- Bwd Packets/s
- Fwd IAT Mean
- Bwd IAT Mean


## Label Meanings
Idle and Active times only saved for flows with >500s

# Tests
## Benign:
<em><span style="color:green">dnsqueries-fb-google-youtube.csv: 835 flows</em></span>  
<em><span style="color:green">dnsqueries-searchengines.csv: 396 flows</em></span>  
<em><span style="color:green">download-2.csv: 117 flows</em></span>  
<em><span style="color:green">fastdownload.csv: 40 flows</em></span>  
<em><span style="color:green">Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv: 225711 flows</em></span>  
<em><span style="color:green">Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv: 286096 flows</em></span>  
<em><span style="color:green">Friday-WorkingHours-Morning.pcap_ISCX.csv: 190911 flows</em></span>  
<em><span style="color:green">Monday-WorkingHours.pcap_ISCX.csv: 529481 flows</em></span>  
<em><span style="color:green">randomnormal2.csv: 593 flows</em></span>  
<em><span style="color:green">randomnormal.csv: 1378 flows</em></span>  
<em><span style="color:green">Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv: 288395 flows</em></span>  
<em><span style="color:green">Tuesday-WorkingHours.pcap_ISCX.csv: 445645 flows</em></span>  
<em><span style="color:green">Wednesday-workingHours.pcap_ISCX.csv: 691406 flows</em></span>  
<em><span style="color:green">youtube-all.csv: 6669 flows</em></span>  
## Malign:
- **Received:**  
<span style="color:red"><em>received-dos-goldeneye.csv: 4013 flows</em></span>  
<span style="color:red"><em>received-dos-hulk.csv: 90791 flows</em></span>  
<span style="color:red"><em>received-portscan.csv: 9009 flows</em></span>  
<span style="color:red"><em>received-ftp-patator.csv: 1380 flows</em></span>  
<span style="color:red"><em>received-ssh-patator.csv: 1020 flows</em></span>  
<span style="color:red"><em>received-telnet-patator.csv: 82698 flows</em></span> 
- **Sent:**  
<span style="color:red"><em>sent-dos-goldeneye.csv: 9995 flows</em></span>  
<span style="color:red"><em>sent-dos-hulk.csv: 89320 flows</em></span>  
<span style="color:red"><em>sent-portscan.csv: 9009 flows</em></span>  
<span style="color:red"><em>dos-goldeneye-1min.csv: 11981 flows</em></span>  
<span style="color:red"><em>dos-hulk-1min.csv: 31902 flows</em></span>  
<span style="color:red"><em>portscan-ncat-filtered-1min.csv: 65537 flows</em></span>  
<span style="color:red"><em>portscan-nmap-23min-alloptions.csv: 9003 flows</em></span>  
<span style="color:red"><em>portscan-nmap-5min.csv: 67751 flows</em></span>  
<span style="color:red"><em>ftp-patator-5min.csv: 968 flows</em></span>  
<span style="color:red"><em>ssh-patator-5min.csv: 490 flows</em></span>  
<span style="color:red"><em>telnet-patator-5min.csv: 45208 flows</em></span>   

# Results  
## Layer 1 (kNN classifier) accuracies:  
- **Malign tests:**  
<span style="color:red"><em>received-dos-goldeneye.csv: 96.810366%</em></span>  
<span style="color:red"><em>received-dos-hulk.csv: 99.288476%</em></span>   
<span style="color:red"><em>received-portscan.csv: 0.366300% (the majority was classified as bruteforce)</em></span>  
<span style="color:red"><em>received-ftp-patator.csv: 99.927536%</em></span>  
<span style="color:red"><em>received-ssh-patator.csv: 100.000000%</em></span>  
<span style="color:red"><em>received-telnet-patator.csv: 63.850395% (most of the rest was classified as portscan)</em></span>  
<span style="color:red"><em>sent-dos-goldeneye.csv: 95.707854%</em></span>  
<span style="color:red"><em>sent-dos-hulk.csv: 98.919615%</em></span>  
<span style="color:red"><em>sent-portscan.csv: 99.900100%</em></span>  
<span style="color:red"><em>dos-goldeneye-1min.csv: 99.749604%</em></span>  
<span style="color:red"><em>dos-hulk-1min.csv: 99.276860%</em></span>  
<span style="color:red"><em>portscan-ncat-filtered-1min.csv: 100.000000%</em></span>  
<span style="color:red"><em>portscan-nmap-5min.csv: 99.865685%</em></span>  
<span style="color:red"><em>portscan-nmap-23min-alloptions.csv: 99.966678%</em></span>  
<span style="color:red"><em>ftp-patator-5min.csv: 99.793388%</em></span>  
<span style="color:red"><em>ssh-patator-5min.csv: 99.795918%</em></span>  
<span style="color:red"><em>telnet-patator-5min.csv: 28.008317% (the majority was classified as portscan)</em></span>  
## Layer 2 ('fastdos'-> kNN classifier, 'portscan'-> kNN classifier, 'bruteforce'-> RF regressor) accuracies:  
- **Benign tests:**    
<em><span style="color:green">dnsqueries-fb-google-youtube.csv: 89.820359%</em></span>  
<em><span style="color:green">dnsqueries-searchengines.csv: 83.080808%</em></span>  
<em><span style="color:green">download-2.csv: 89.743590%</em></span>  
<em><span style="color:green">fastdownload.csv: 100.000000%</em></span>  
<em><span style="color:green">Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv: 98.389977%</em></span>  
<em><span style="color:green">Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv: 97.771377%</em></span>  
<em><span style="color:green">Friday-WorkingHours-Morning.pcap_ISCX.csv: 95.696948%</em></span>  
<em><span style="color:green">Monday-WorkingHours.pcap_ISCX.csv: 95.405312%</em></span>  
<em><span style="color:green">randomnormal2.csv: 89.038786%</em></span>  
<em><span style="color:green">randomnormal.csv: 92.017417%</em></span>  
<em><span style="color:green">Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv: 94.778342%</em></span>  
<em><span style="color:green">Tuesday-WorkingHours.pcap_ISCX.csv: 87.432598%</em></span>  
<em><span style="color:green">Wednesday-workingHours.pcap_ISCX.csv: 85.552483%</em></span>  
<em><span style="color:green">youtube-all.csv: 93.252362%</em></span>  
- **Malign tests:**  
<span style="color:red"><em>received-dos-goldeneye.csv: 89.783205%</em></span>  
<span style="color:red"><em>received-dos-hulk.csv: 96.631825%</em></span>   
<span style="color:red"><em>received-portscan.csv: 99.755800%</em></span>  
<span style="color:red"><em>received-ftp-patator.csv: 99.927536%</em></span>  
<span style="color:red"><em>received-ssh-patator.csv: 99.019608%</em></span>  
<span style="color:red"><em>received-telnet-patator.csv: 93.765266%</em></span>  
<span style="color:red"><em>sent-dos-goldeneye.csv: 87.183592%</em></span>  
<span style="color:red"><em>sent-dos-hulk.csv: 98.753918%</em></span>  
<span style="color:red"><em>sent-portscan.csv: 99.589300%</em></span>  
<span style="color:red"><em>dos-goldeneye-1min.csv: 99.332276%</em></span>  
<span style="color:red"><em>dos-hulk-1min.csv: 95.448561%</em></span>  
<span style="color:red"><em>portscan-ncat-filtered-1min.csv: 99.993897%</em></span>  
<span style="color:red"><em>portscan-nmap-5min.csv: 90.603829%</em></span>  
<span style="color:red"><em>portscan-nmap-23min-alloptions.csv: 99.944463%</em></span>  
<span style="color:red"><em>ftp-patator-5min.csv: 99.793388%</em></span>  
<span style="color:red"><em>ssh-patator-5min.csv: 97.959184%</em></span>  
<span style="color:red"><em>telnet-patator-5min.csv: 95.640152%</em></span>  

## Average accuracy per file:
<span style="color:green"><em>**Benign:**</em></span> 92.284311%  
<span style="color:red"><em>**Malign:**</em></span> 96.654459%  


**Notes:** 
See current options.cfg file for the optimal layer-configuration. We have moved our layers to classify with kNN mostly since it's the algorithm that achieves the best results, but it comes with a disadvantage: its performance is not the best. Having this in mind, we have yet to consider if we should use MLP in layer1 and layer2-portscan. As of layer2-dos and layer2-bruteforce, the chosen algorithms (k-nearest neighbors classifier for l2-dos and random forest regressor for l2-bruteforce) are undoubtly the best we could find among the wholesome group of algorithms that were tested.