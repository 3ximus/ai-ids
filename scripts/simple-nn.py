#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pickle, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

#USAGE: simple-nn training_dataset.csv testing_dataset.csv

param_grid = [
    {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
        (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        ]
    }
]

def save_model(filename, clfmodel):
    # save the model to disk
    filename = 'clf.sav'
    model_file = open(filename,'wb')
    pickle.dump(clfmodel, model_file)
    model_file.close()
    return

def load_model(filename):
    # load the model from disk
    model_file = open(filename, 'rb')
    loaded_model = pickle.load(model_file)
    model_file.close()
    return loaded_model

def parse_csvdataset(filename):
    x_in = []
    y_in = []
    #outputs = {"DDoS": 0, "PortScan": 1, "Bot": 2, "Infiltration": 3, "FTP-Patator": 4, "SSH-Patator": 5, "DoS Hulk": 6, "DoS GoldenEye": 7, "DoS slowloris": 8, "DoS Slowhttptest": 9, "Heartbleed": 10}
    outputs = {"DoS-Attack": 0, "PortScan": 1, "Bot": 2, "Infiltration": 3, "FTP-Patator": 4, "SSH-Patator": 5, "Heartbleed": 6}
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])       # the 9 extracted features
            #print(tmp[14])
            #y_tmp = [0] * 11
            y_tmp = [0] * 7
            tmp_last = tmp[-1]
            if tmp_last=="BENIGN": tmp[-1]="Heartbleed"      # testing purposes
            if tmp_last=="DDoS" or tmp_last=="DoS Hulk" or tmp_last=="DoS GoldenEye" or tmp_last=="DoS slowloris" or tmp_last=="DoS Slowhttptest": tmp[-1]="DoS-Attack"
            y_tmp[outputs[tmp[-1]]] = 1  # choose result based on label
            y_in.append(y_tmp)
    return x_in, y_in

def print_stats(y_predicted_lst, y_test_lst):
    '''
    DDoS = [1] + [0]*10
    PortScan = [0] + [1] + [0]*9
    Bot = [0]*2 + [1] + [0]*8
    Infiltration = [0]*3 + [1] + [0]*7
    FTP_Patator = [0]*4 + [1] + [0]*6
    SSH_Patator = [0]*5 + [1] + [0]*5
    DoS_Hulk = [0]*6 + [1] + [0]*4
    DoS_GoldenEye = [0]*7 + [1] + [0]*3
    DoS_slowloris = [0]*8 + [1] + [0]*2
    DoS_Slowhttptest = [0]*9 + [1] + [0]
    Heartbleed = [0]*10 + [1]

    print("DDoS: ", y_predicted_lst.count(DDoS), "predicted out of", y_test_lst.count(DDoS), "test values")
    print("PortScan: ", y_predicted_lst.count(PortScan), "predicted out of", y_test_lst.count(PortScan), "test values")
    print("Bot: ", y_predicted_lst.count(Bot), "predicted out of", y_test_lst.count(Bot), "test values")
    print("Infiltration: ", y_predicted_lst.count(Infiltration), "predicted out of", y_test_lst.count(Infiltration), "test values")
    print("FTP-Patator: ", y_predicted_lst.count(FTP_Patator), "predicted out of", y_test_lst.count(FTP_Patator), "test values")
    print("SSH-Patator: ", y_predicted_lst.count(SSH_Patator), "predicted out of", y_test_lst.count(SSH_Patator), "test values")
    print("DoS-Hulk: ", y_predicted_lst.count(DoS_Hulk), "predicted out of", y_test_lst.count(DoS_Hulk), "test values")
    print("DoS-GoldenEye: ", y_predicted_lst.count(DoS_GoldenEye), "predicted out of", y_test_lst.count(DoS_GoldenEye), "test values")
    print("DoS-slowloris: ", y_predicted_lst.count(DoS_slowloris), "predicted out of", y_test_lst.count(DoS_slowloris), "test values")
    print("DoS-Slowhttptest: ", y_predicted_lst.count(DoS_Slowhttptest), "predicted out of", y_test_lst.count(DoS_Slowhttptest), "test values")
    print("Heartbleed: ", y_predicted_lst.count(Heartbleed), "predicted out of", y_test_lst.count(Heartbleed), "test values")
    '''
#                Type        Flow count         Network Output
    attacks = (("DoS-Attack",    294496, [1, 0, 0, 0, 0, 0, 0]),
               ("PortScan",      158930, [0, 1, 0, 0, 0, 0, 0]),
               ("Bot",             1966, [0, 0, 1, 0, 0, 0, 0]),
               ("Infiltration",      36, [0, 0, 0, 1, 0, 0, 0]),
               ("FTP-Patator",     7938, [0, 0, 0, 0, 1, 0, 0]),
               ("SSH-Patator",     5897, [0, 0, 0, 0, 0, 1, 0]),
               ("Heartbleed",        11, [0, 0, 0, 0, 0, 0, 1]))
    print("# Flows             Type  Predicted / TOTAL")
    for x in attacks:
        predict, total = y_predicted_lst.count(x[2]), y_test_lst.count(x[2])
        print("\033[1;3%dm% 7d %16s     % 6d / %d\033[m" % (1 if predict > total else 2, x[1], x[0], predict, total))

    print("")
    i=0
    for elem in y_predicted_lst:
        if(elem.count(1)!=1):
            i+=1
        #if(elem!=[1] + [0]*10 and elem!=[0] + [1] + [0]*9):
        #    print(elem)
    if i!=0:
        print("The NN behavior must be dealt with. The NN should only turn on one output node on each output. Wrong output count: ", i)

# PARSE DATA AND GET TRAINING VALUES
# input_lst = parse_pcapdataset("pcap/pcapdataset.txt")
# testpcap_input = parse_pcapdataset("pcap/pcapdataset.txt")
X_train, y_train = parse_csvdataset(sys.argv[1])
#print("------")
#print(len(X_train))
#print(X_train)
X_train = np.array(X_train, dtype='float64')
y_train = np.array(y_train, dtype='float64')
#print(X_train.tolist())
# NORMALIZE TRAINING VALUES
scaler = preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)    # normalize

# train_test_split is not working as expected
# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42,stratify=y_train)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
#MultilayerPerceptron = MLPClassifier()
#print("Searching Grid")
#clf = GridSearchCV(MultilayerPerceptron, param_grid, cv=3, scoring='accuracy')

#print("Best parameters set found on development set:")
#print(clf)

# DEFINE MODEL
clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(64), random_state=1) # adam porque o dataset tem milhares de exemplos
#print(clf)

# TRAIN MODEL
print("Training...")
clf.fit(X_train, y_train)

# PREDICT VALUES BASED ON THE GIVEN INPUT
print("Predicting...\n")

X_test, y_test = parse_csvdataset(sys.argv[2])
X_test = np.array(X_test, dtype='float64')
#X_test = scaler.transform(X_test)      # normalize testing values
y_test = np.array(y_test, dtype='float64')
y_predicted = clf.predict(X_test)
#print(y_predicted)

# SAVE MODEL, LOAD MODEL
#save_model('clf.sav', clf)
#clfmodel = load_model('clf.sav')
#y_predicted = clfmodel.predict(X_test)
#result = clfmodel.score(X_test, y_test)
#print("MLP Accuracy (Pickle): " + str(result))

# PRINT RESULTS
print("MLP Correctly Classified:" , accuracy_score(y_test, y_predicted,normalize=False) , "/" , len(y_predicted))
print("MLP Accuracy (sklearn):" , accuracy_score(y_test, y_predicted,normalize=True))

# LOOK AT PREDICTED VALUES AND PRINT STATS
print_stats(y_predicted.tolist(), y_test.tolist())
