import os

dirpath = "/root/Desktop/PCAP_SANDBOX/ai-ids/csv/selected-compacted-datasets/all_features/raw/individual/"
dirlist = os.listdir(dirpath)
for filename in dirlist:
	absfilename = dirpath + filename
	if(filename.find("csv")!=-1 and filename.find("DoS")!=-1):
		of = open(os.path.dirname(absfilename) + os.sep + "test_" + os.path.splitext(os.path.basename(absfilename))[0] + ".test","w")
		f = open(absfilename, "r")
		i=0
		for line in f:
			if i>=400:
				of.write(line)
			else:
				i+=1
		f.close()
		of.close()
	elif(filename.find("csv")!=-1):
		of = open(os.path.dirname(absfilename) + os.sep + "test_" + os.path.splitext(os.path.basename(absfilename))[0] + ".test","w")
		f = open(absfilename, "r")
		i=0
		for line in f:
			if i>=2000:
				of.write(line)
			else:
				i+=1
		f.close()
		of.close()