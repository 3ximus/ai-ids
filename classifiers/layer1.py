#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
try: import configparser
except ImportError: import ConfigParser as configparser # for python2



def parse_csvdataset(filename, attacks, outputs):
    '''Parse a dataset

        Parameters
        ----------
        - filename       filename of the dataset
        - attacks        dictionary that maps attack names to their index
        - outputs        list of output encodings, maps each index to a discrete binary output
    '''

    x_in, y_in = [], []
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])
            if tmp[-1] == "BENIGN": tmp[-1] = "dos" # FIXME testing benign, we need to assign a known label
            if tmp[-1] in ("ftpbruteforce", "sshbruteforce", "telnetbruteforce"): tmp[-1] = "bruteforce"
            if "dos" in tmp[-1]: tmp[-1] = "dos"
            try:
                y_in.append(outputs[attacks[tmp[-1]]]) # choose result based on label
            except IndexError:
                print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]))
                exit()
            except KeyError:
                print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]))
                exit()
    return x_in, y_in



def train_new_network(test_filename, attacks, outputs, saved_model_file, config, verbose=False):
    '''Train a new Neural Network model from given test dataset file

        Parameters
        ----------
        - test_filename       filename of the test dataset
        - attacks             dictionary that maps attack names to their index
        - outputs             list of output encodings, maps each index to a discrete binary output
        - saved_model_file    file path to save the model (including filename)
        - config                ConfigParser object from which the following options are loaded (section.option):
                                  l1.scaler,  l1.scaler_module
                                  l1.classifier, l1.classifier_module
                              classifier options are mandatory
    '''

    if verbose: print('Reading Training Dataset...')
    X_train, y_train = parse_csvdataset(test_filename, attacks, outputs)
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')

# scaler setup
    if config.has_option('l1', 'scaler_module'):
        exec('import '+ config.get('l1', 'scaler_module')) # import scaler module
    if config.has_option('l1', 'scaler'):
        scaler = eval(config.get('l1', 'scaler')).fit(X_train)
        X_train = scaler.transform(X_train)    # normalize
        save_model(path.dirname(saved_model_file) + "/scalerX", scaler)

# classifier setup
    if config.has_option('l1', 'classifier_module'):
        exec('import '+ config.get('l1', 'classifier_module')) # import classifier module
    model = eval(config.get('l1', 'classifier'))

# train and save the model
    if verbose: print("Training... (" + test_filename + ")")
    model.fit(X_train, y_train)
    save_model(saved_model_file, model)
    return model



def predict(classifier, test_filename, attacks, outputs, saved_model_path=None, verbose=False):
    '''Apply the given classifier model to a test dataset

        Parameters
        ----------
        - classifier          classifier model
        - test_filename       filename of the test dataset
        - attacks             dictionary that maps attack names to their index
        - outputs             list of output encodings, maps each index to a discrete binary output
        - saved_model_path    directory path to save the scaler model
        - verbose             print actions
    '''

    if verbose: print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(test_filename, attacks, outputs)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')

    if saved_model_path and path.isfile(saved_model_path + "/scalerX"):
        scaler = load_model(saved_model_path + "/scalerX")
        X_test = scaler.transform(X_test) # normalize

    if verbose: print("Predicting... (" + test_filename + ")\n")
    y_predicted = classifier.predict_proba(X_test)
    return y_test, y_predicted



def classify(train_filename, test_filename, config, disable_load=False, verbose=False):
    '''Create or load train model from given dataset and apply it to the test dataset

        If there is already a created model with the same classifier and train dataset
            it will be loaded, otherwise a new one is created and saved

        Parameters
        ----------
        - train_filename      filename of the train dataset
        - test_filename       filename of the test dataset
        - config              ConfigParser object to get layer settings
        - disable_load        list of output encodings, maps each index to a discrete binary output
    '''

# get options
    labels = config.getint('l1', 'labels')
    attack_keys = config.options('l1-labels')
    attacks = dict(zip(attack_keys, range(len(attack_keys))))
    outputs = [[1 if j == i else 0 for j in range(labels)] for i in range(labels)]

    saved_model_path = config.get('l1', 'saved_model_path')

# generate model filename
    used_model_md5 = hashlib.md5()
    used_model_md5.update(config.get('l1', 'classifier').encode('utf-8'))
    train_file_md5 = hashlib.md5()
    with open(train_filename, 'rb') as tf: train_file_md5.update(tf.read())
    saved_model_file = saved_model_path + '/%s-%s-%s' % (
            train_filename.strip('/.csv').replace('/','-'), train_file_md5.hexdigest()[:7], used_model_md5.hexdigest()[:7])

# train or load the network
    if path.isfile(saved_model_file) and not disable_load:
        classifier = load_model(saved_model_file)
    else: # create a new network
        classifier = train_new_network(train_filename, attacks, outputs, saved_model_file, config, verbose)
        # save_model(saved_model_file, classifier)

# apply network to the test data
    y_test, y_predicted = predict(classifier, test_filename, attacks, outputs, path.dirname(saved_model_file), verbose)

    print_stats(y_predicted, y_test, labels, outputs, lambda i: attack_keys[i], test_filename)
    return y_predicted

