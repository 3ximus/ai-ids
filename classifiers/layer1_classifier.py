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
                    if tmp[-1]=="BENIGN": tmp[-1]="dos" # in case we're testing benign and in test mode, we need to assign a known label
                    if tmp[-1] in ("ftpbruteforce", "sshbruteforce", "telnetbruteforce"): tmp[-1]="bruteforce"
                    if tmp[-1].find("dos")!=-1: tmp[-1]="dos"
                    y_in.append(OUTPUTS[ATTACKS[tmp[-1]]]) # choose result based on label
                except IndexError:
                    print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
                except KeyError:
                    print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]))
                    sys.exit(1)
    return x_in, y_in


def train_new_network(test_filename, verbose=False):
    '''Train a new Neural Network model from given test file'''

    if verbose: print('Reading Training Dataset...')
    X_train, y_train = parse_csvdataset(test_filename, True) # true, we know our labels
    label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')

# scaler setup
    if conf.has_option('l1', 'scaler_module'):
        exec('import '+ conf.get('l1', 'scaler_module')) # import scaler module
    if conf.has_option('l1', 'scaler'):
        scaler = eval(conf.get('l1', 'scaler')).fit(X_train)
        X_train = scaler.transform(X_train)    # normalize
        save_model("saved_neural_networks/layer1/scalerX", scaler)

# classifier setup
    if conf.has_option('l1', 'classifier_module'):
        exec('import '+ conf.get('l1', 'classifier_module')) # import classifier module
    model = eval(conf.get('l1', 'classifier'))
    if verbose: print("Training... (" + test_filename + ")")
    model.fit(X_train, y_train)
    return label_count, model

def predict(classifier, filename, verbose=False):
    if verbose: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename, verbose)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    if path.isfile("saved_neural_networks/layer1/scalerX"):
        scaler = load_model("saved_neural_networks/layer1/scalerX")
        X_test = scaler.transform(X_test) # normalize
    if verbose: print("Predicting... (" + filename + ")\n")
    y_predicted = classifier.predict_proba(X_test)
    return y_test, y_predicted

def layer1_classify(train_filename, test_filename, load=False, testing=False):
    used_model_md5 = hashlib.md5()
    used_model_md5.update(conf.get('l1', 'classifier').encode('utf-8'))
    train_file_md5 = hashlib.md5()
    with open(train_filename, 'rb') as tf: train_file_md5.update(tf.read())
    saved_path = 'saved_neural_networks/layer1/%s-%s-%s' % (train_filename.strip('/.csv').replace('/','-'), train_file_md5.hexdigest()[:7], used_model_md5.hexdigest()[:7])
    if path.isfile(saved_path) and not load: # default if it exists
        classifier = load_model(saved_path)
    else: # create a new network
        label_count, classifier = train_new_network(train_filename)
        save_model(saved_path, classifier)

    y_test, y_predicted = predict(classifier, test_filename, testing)

    if testing:
        print_stats(y_predicted, y_test, LABELS, OUTPUTS,
                    lambda i: ATTACK_KEYS[i],
                    test_filename, None if not load else label_count)

    return y_predicted
