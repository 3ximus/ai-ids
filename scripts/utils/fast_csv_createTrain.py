import os

dirpath = "/root/Desktop/PCAP_SANDBOX/ai-ids/csv/selected-compacted-datasets/21features/distributed/individual/"
dirlist = os.listdir(dirpath)
of = open("/root/Desktop/PCAP_SANDBOX/ai-ids/csv/selected-compacted-datasets/21features/distributed/training.csv","w")
for filename in dirlist:
	absfilename = dirpath + filename
	if(filename.find("csv")!=-1):
		f = open(absfilename, "r")
		of.write(f.read())
		f.close()

of.close()