#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys, hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
#new
from sklearn.ensemble import RandomForestRegressor

# =====================
#     CONFIGURATION
# =====================

# Attack Mapping for output encoding, DoS-Attack is used when refering to all DoS type attacks. DDoS value is 0 for easier output
LABELS = 3
ATTACK_KEYS = ["DoS-Attack", "PortScan", "Bruteforce"]
ATTACK_INDEX = [0, 1, 2]
ATTACKS = dict(zip(ATTACK_KEYS,ATTACK_INDEX))
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]
PARAM_GRID = [{
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [
    (16,),(32,),(64,),(128,)]}]
scaler=0

# =====================
#       FUNCTIONS
# =====================

def parse_csvdataset(filename, output_labels_known=False):
    x_in = []
    y_in = []
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])
            if output_labels_known:
                try:
                    if tmp[-1]=="BENIGN": tmp[-1]="DoS-Attack" # in case we're testing benign and in test mode, we need to assign a known label
                    if tmp[-1]=="FTP-Patator" or tmp[-1]=="SSH-Patator" or tmp[-1]=="TELNET-Patator": tmp[-1]="Bruteforce"
                    if tmp[-1].find("DoS")!=-1: tmp[-1]="DoS-Attack"
                    y_in.append(OUTPUTS[ATTACKS[tmp[-1]]]) # choose result based on label
                except IndexError:
                    print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
                except KeyError:
                    print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
    return x_in, y_in


def train_new_network(filename):
    print('Reading Training Dataset...')
    X_train, y_train = parse_csvdataset(filename, True) # true, we know our labels
    label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)    # normalize
    save_model("saved_neural_networks/layer1/scalerX",scaler)
    neural_network1 = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(40,16), random_state=1)
    #neural_network1=RandomForestRegressor(max_depth=3, random_state=0)
    print("Training... (" + filename + ")")
    neural_network1.fit(X_train, y_train)
    return label_count, neural_network1

def predict(neural_network1, filename, testing=False):
    if testing: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename,testing)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    scaler = load_model("saved_neural_networks/layer1/scalerX", testing)
    X_test = scaler.transform(X_test)      # normalize
    if testing: print("Predicting... (" + filename + ")\n")
    y_predicted = neural_network1.predict_proba(X_test)
    return y_test, y_predicted

def layer1_classify(train_filename, test_filename, load=False, testing=False):
    digester = hashlib.md5()
    with open(train_filename, 'rb') as tf: digester.update(tf.read())
    saved_path = 'saved_neural_networks/layer1/%s-%s' % (train_filename.strip('/.csv').replace('/','-'), digester.hexdigest())
    if path.isfile(saved_path) and not load: # default if it exists
        neural_network1 = load_model(saved_path, testing)
    else: # create a new network
        label_count, neural_network1 = train_new_network(train_filename)
        save_model(saved_path, neural_network1)

    y_test, y_predicted = predict(neural_network1, test_filename, testing)

    if testing:
        print_stats(y_predicted, y_test, LABELS, OUTPUTS,
                    lambda i: ATTACK_KEYS[i],
                    test_filename, None if not load else label_count)

    return y_predicted

# train_test_split is not working as expected
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42,stratify=y_train)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
    #MultilayerPerceptron = MLPClassifier()
    #print("Searching Grid")
    #clf = GridSearchCV(MultilayerPerceptron, PARAM_GRID, cv=3, scoring='accuracy')

    #print("Best parameters set found on development set:")
    #print(clf)
