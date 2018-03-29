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

# Results  
## Layer 1 (MLP - [40,16], 17 features):  
- Malign:  
Classifier Accuracy (dos-goldeneye-1min.csv):0.994240881396  
Classifier Accuracy (dos-hulk-1min.csv):0.920725973293  
Classifier Accuracy (dos-slowhttptestB-10min.csv):0.487940630798  
Classifier Accuracy (dos-slowhttptestH-10min.csv):0.483333333333  
Classifier Accuracy (dos-slowhttptestX-10min.csv):0.828205128205  
Classifier Accuracy (dos-slowloris-10min.csv):1.0  
Classifier Accuracy (portscan-nmap-5min.csv):0.790453277442  
Classifier Accuracy (portscan-nmap-23min-alloptions.csv):0.959457958458  
Classifier Accuracy (ftp-patator-5min.csv):0.948347107438  
Classifier Accuracy (ssh-patator-5min.csv):0.508163265306  
  
## Layer 2 (RFE, 64 features):  
- Malign:  
Classifier Accuracy (dos-goldeneye-1min.csv):0.938319005091  
Classifier Accuracy (dos-hulk-1min.csv):0.960660773619  
Classifier Accuracy (dos-slowhttptestB-10min.csv):0.556586270872  
Classifier Accuracy (dos-slowhttptestH-10min.csv):0.490740740741  
Classifier Accuracy (dos-slowhttptestX-10min.csv):0.569230769231  
Classifier Accuracy (dos-slowloris-10min.csv):0.833333333333  
Classifier Accuracy (portscan-nmap-5min.csv):0.999837640773  
Classifier Accuracy (portscan-nmap-23min-alloptions.csv):0.999888925914  
Classifier Accuracy (ftp-patator-5min.csv):0.323347107438  
Classifier Accuracy (ssh-patator-5min.csv):0.95306122449  
- Benign:  
DoS-Attack classifier:  
Classifier Accuracy (Monday-WorkingHours.pcap_ISCX.csv):0.951858518058  
PortScan classifier:  
Classifier Accuracy (Monday-WorkingHours.pcap_ISCX.csv):0.991419899864  
FTP-Patator classifier:  
Classifier Accuracy (Monday-WorkingHours.pcap_ISCX.csv):0.997692079602  
SSH-Patator classifier:  
Classifier Accuracy (Monday-WorkingHours.pcap_ISCX.csv):0.993954457289  
