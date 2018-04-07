#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys, argparse, hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
from sklearn.ensemble import RandomForestRegressor

# =====================
#     CONFIGURATION
# =====================

LABELS = 2
ATTACK_TYPES = ("DoS-Attack", "PortScan", "FTP-Patator", "SSH-Patator", "TELNET-Patator","Bot", "Infiltration", "Heartbleed", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS")
CLASSIFICATIONS = {"BENIGN": 0, "MALIGN": 1}
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]
PARAM_GRID = [{
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [
    (16,),(32,),(64,),(128,)]}]

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
                if tmp[-1] in ATTACK_TYPES: tmp[-1] = "MALIGN"
                try:
                    y_in.append(OUTPUTS[CLASSIFICATIONS[tmp[-1]]]) # choose result based on label
                except IndexError:
                    print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
                except KeyError:
                    print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
    return x_in, y_in

def train_new_network(filename):
    print('Reading Training Dataset... (' + filename + ')')
    X_train, y_train = parse_csvdataset(filename, True)
    label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)    # normalize
    classifier = RandomForestRegressor(max_depth=3, random_state=0)
    print("Training... (" + filename + ")")
    classifier.fit(X_train, y_train)
    return label_count, classifier

def predict(classifier, filename, testing=False):
    if testing: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename,testing)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    #X_test = scaler.transform(X_test)      # normalize
    if testing: print("Predicting... (" + filename + ")\n")
    y_predicted = classifier.predict(X_test)
    return y_test, y_predicted

def layer2_classify(train_filename, test_filename, load=False, testing=False):
    digester = hashlib.md5()
    with open(train_filename, 'rb') as tf: digester.update(tf.read())
    saved_path = 'saved_neural_networks/layer2/%s-%s' % (train_filename.strip('/.csv').replace('/','-'), digester.hexdigest())
    if path.isfile(saved_path) and not load: # default if it exists
        classifier = load_model(saved_path, testing)
    else: # create a new network
        label_count, classifier = train_new_network(train_filename)
        save_model(saved_path, classifier)

    #print(classifier.feature_importances_)
    y_test, y_predicted = predict(classifier, test_filename, testing)

    if testing:
        print_stats(y_predicted, y_test, LABELS, OUTPUTS,
                    lambda i: list(CLASSIFICATIONS.keys())[i],
                    test_filename, None if not load else label_count)

    return y_predicted
