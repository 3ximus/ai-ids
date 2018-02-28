#!/usr/bin/env python

import numpy as np
import pickle, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer

#from keras.models import Sequential
#from keras.layers import Activation,Dense

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
	outputs = {"DDoS": 0, "PortScan": 1, "Bot": 2, "SSH-Patator": 3, "FTP-Patator": 4, "Infiltration": 5, "DoS Hulk": 6, "DoS GoldenEye": 7, "DoS slowloris": 8, "DoS Slowhttptest": 9, "Heartbleed": 10}
	with open(filename, 'r') as fd:
		for line in fd:
			tmp = line.strip('\n').split(',')
			x_in.append(tmp[1:-1])
			y_tmp = [0] * 11
			y_tmp[outputs[tmp[-1]]] = 1
			y_in.append(y_tmp)
	return x_in, y_in

#input_lst = parse_pcapdataset("pcap/pcapdataset.txt")
# testpcap_input = parse_pcapdataset("pcap/pcapdataset.txt")
input_lst, result_lst = parse_csvdataset(sys.argv[1])
input_lst_X = np.array(input_lst, dtype='float64')
input_lst_y = np.array(result_lst, dtype='float64')
X_train, X_test, y_train, y_test = train_test_split(input_lst_X, input_lst_y, test_size=0.25, random_state=42,stratify=input_lst_y)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
MultilayerPerceptron = MLPClassifier()
print("Searching Grid")
clf = GridSearchCV(MultilayerPerceptron, param_grid, cv=3, scoring='accuracy')

print("Best parameters set found on development set:")
print(clf)

# DEFINE MODEL
# clf = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(16), random_state=1) # adam porque o dataset tem milhares de exemplos
#print(clf)

# TRAIN MODEL
print("Training")
clf.fit(X_train, y_train)

# PREDICT VALUES BASED ON THE GIVEN INPUT
print("Predicting")

y_predicted = clf.predict(X_test)
#print(y_predicted)

# SAVE MODEL, LOAD MODEL
save_model('clf.sav', clf)
#clfmodel = load_model('clf.sav')
#y_predicted = clfmodel.predict(X_test)
#result = clfmodel.score(X_test, y_test)
#print("MLP Accuracy (Pickle): " + str(result))

# PRINT RESULTS
print("MLP Correctly Classified: " + str(accuracy_score(y_test, y_predicted,normalize=False)) + "/" + str(len(y_predicted)))
print("MLP Accuracy (sklearn): " + str(accuracy_score(y_test, y_predicted,normalize=True)))
