from __future__ import print_function
import layer1_classifier
from layer1_classifier import *
import layer2_classifier

# Usage: ids.py <17 features csv to test> <same all features csv to test>
# Future Usage: ids.py <pcap file>

TMP_DIR="/tmp/ai-ids"

# Layer 1
print("Layer 1: 'Attack-Profiling'")
print("Layer 1 predicting...")
y1_predicted = layer1_classifier.layer1_classify("csv/train/17features/layer1/trainingNN1.csv",sys.argv[1],testing=True)
y1_predicted = (y1_predicted == y1_predicted.max(axis=1, keepdims=True)).astype(int)

dos=[]
pscan=[]
ftp=[]
for i,prediction in enumerate(y1_predicted):
	if np.argmax(prediction)==0: #DoS
		dos.append(i)
	elif np.argmax(prediction)==1: #PortScan
		pscan.append(i)
	elif np.argmax(prediction)==2: #Bruteforce
		ftp.append(i)
	else:
		print("Error.")

# Layer 2
print("Layer 2: 'Flow Classification'")

fd = open(sys.argv[2],"r")
content = fd.readlines()
content = [x.strip('\n') for x in content]
fd.close()

dos = set(dos)
pscan = set(pscan)
ftp = set(ftp)
dos_of = open(TMP_DIR + "/dos.csv","w")
pscan_of = open(TMP_DIR + "/pscan.csv","w")
ftp_of = open(TMP_DIR + "/bruteforce.csv","w")

print("Selecting layer1 selected flows...")
for i,elem in enumerate(content):
	if i in dos:
		dos_of.write(elem + "\n")
	elif i in pscan:
		pscan_of.write(elem + "\n")
	elif i in ftp:
		ftp_of.write(elem + "\n")
dos_of.close()
pscan_of.close()
ftp_of.close()

benign=[]
malign=[]
print("Layer 2 predicting...")
if len(dos)!=0:
	y2_dos_predicted = layer2_classifier.layer2_classify("csv/train/allfeatures/layer2/benign-DoS-Attack.csv",TMP_DIR + "/dos.csv",testing=True)
	for prediction in y2_dos_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)
if len(pscan)!=0:
	y2_pscan_predicted = layer2_classifier.layer2_classify("csv/train/allfeatures/layer2/benign-PortScan.csv",TMP_DIR + "/pscan.csv",testing=True)
	for prediction in y2_pscan_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)
if len(ftp)!=0:
	y2_ftp_predicted = layer2_classifier.layer2_classify("csv/train/allfeatures/layer2/benign-Bruteforce.csv",TMP_DIR + "/bruteforce.csv",testing=True)
	for prediction in y2_ftp_predicted:
		if np.argmax(prediction)==0: #Benign
			benign.append(1)
		elif np.argmax(prediction)==1: #Malign
			malign.append(1)

benign_length = len(benign)
malign_length = len(malign)
print("\033[1;35m\n---------\nRESULTS |\n---------\033[0;0m")
print("\033[1;32m#Flows classified as benign:\033[0;0m")
print(benign_length)
print("\033[1;31m#Flows classified as malign:\033[0;0m")
print(malign_length)
print("\033[1;32mBenign/(Malign + Benign) ratio\033[0;0m")
print(benign_length*100*1.0/(benign_length+malign_length),"%")
print("\033[1;31mMalign/(Malign + Benign) ratio\033[0;0m")
print(malign_length*100*1.0/(benign_length+malign_length),"%")
