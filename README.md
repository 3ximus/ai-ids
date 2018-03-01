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

Flow Duration
Total Fwd Packets
Total Backward Packets
Total Length of Fwd Packets
Total Length of Bwd Packets
Fwd Packets/s
Bwd Packets/s
Fwd IAT Mean
Bwd IAT Mean


## Label Meanings
Idle and Active times only saved for flows with >500s
