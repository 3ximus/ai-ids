#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys, hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
from sklearn import preprocessing
try: import configparser
except ImportError: import ConfigParser as configparser # for python2

# =====================
#     CONFIGURATION
# =====================

# load config file settings
conf = configparser.ConfigParser(allow_no_value=True)
conf.optionxform=str
conf.read(path.dirname(__file__) + '/options.cfg')

# set options
LABELS = conf.getint('l1', 'labels')
ATTACK_KEYS = conf.options('l1-labels')
ATTACKS = dict(zip(ATTACK_KEYS, range(len(ATTACK_KEYS))))
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]

# do necessary imports and load the module
if conf.get('l1', 'classifier_module'): exec('import '+ conf.get('l1', 'classifier_module'))
MODEL = eval(conf.get('l1', 'classifier'))
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
    print("Training... (" + filename + ")")
    MODEL.fit(X_train, y_train)
    return label_count, MODEL

def predict(classifier, filename, testing=False):
    if testing: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename,testing)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    scaler = load_model("saved_neural_networks/layer1/scalerX", testing)
    X_test = scaler.transform(X_test)      # normalize
    if testing: print("Predicting... (" + filename + ")\n")
    y_predicted = classifier.predict_proba(X_test)
    return y_test, y_predicted

def layer1_classify(train_filename, test_filename, load=False, testing=False):
    digester = hashlib.md5()
    with open(train_filename, 'rb') as tf: digester.update(tf.read())
    saved_path = 'saved_neural_networks/layer1/%s-%s' % (train_filename.strip('/.csv').replace('/','-'), digester.hexdigest())
    if path.isfile(saved_path) and not load: # default if it exists
        classifier = load_model(saved_path, testing)
    else: # create a new network
        label_count, classifier = train_new_network(train_filename)
        save_model(saved_path, classifier)

    y_test, y_predicted = predict(classifier, test_filename, testing)

    if testing:
        print_stats(y_predicted, y_test, LABELS, OUTPUTS,
                    lambda i: ATTACK_KEYS[i],
                    test_filename, None if not load else label_count)

    return y_predicted
