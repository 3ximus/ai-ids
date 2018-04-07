#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys, argparse, hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
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
LABELS = conf.getint('l2', 'labels')
CLASSIFICATIONS = dict(zip(conf.options('l2-labels'), range(len(conf.options('l2-labels')))))
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]
ATTACK_TYPES = conf.options('l2-malign')

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

def train_new_network(filename, verbose=False):
    '''Train a new Neural Network model from given test file'''

    if verbose: print('Reading Training Dataset... (' + filename + ')')
    X_train, y_train = parse_csvdataset(filename, True)
    label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')

# scaler setup
    if conf.has_option('l2', 'scaler_module'):
        exec('import '+ conf.get('l2', 'scaler_module')) # import scaler module
    if conf.has_option('l2', 'scaler'):
        scaler = eval(conf.get('l2', 'scaler')).fit(X_train)
        X_train = scaler.transform(X_train) # normalize
        save_model("saved_neural_networks/layer1/scalerX",scaler)

# classifier setup
    if conf.has_option('l2', 'classifier_module'):
        exec('import '+ conf.get('l2', 'classifier_module')) # import classifier module
    model = eval(conf.get('l2', 'classifier'))
    if verbose: print("Training... (" + filename + ")")
    model.fit(X_train, y_train)
    return label_count, model

def predict(classifier, filename, testing=False):
    if testing: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename, testing)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    #X_test = scaler.transform(X_test) # normalize
    if testing: print("Predicting... (" + filename + ")\n")
    y_predicted = classifier.predict(X_test)
    return y_test, y_predicted

def layer2_classify(train_filename, test_filename, load=False, testing=False):
    used_model_md5 = hashlib.md5()
    used_model_md5.update(conf.get('l2', 'classifier').encode('utf-8'))
    train_file_md5 = hashlib.md5()
    with open(train_filename, 'rb') as tf: train_file_md5.update(tf.read())
    saved_path = 'saved_neural_networks/layer2/%s-%s-%s' % (train_filename.strip('/.csv').replace('/','-'), train_file_md5.hexdigest()[:7], used_model_md5.hexdigest()[:7])
    if path.isfile(saved_path) and not load: # default if it exists
        classifier = load_model(saved_path)
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
