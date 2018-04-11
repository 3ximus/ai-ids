#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from os import path
from classifier_functions import save_model, load_model, print_stats, gen_saved_model_pathname
try: import configparser
except ImportError: import ConfigParser as configparser # for python2


def process_dataset(data, attacks, outputs):
    '''Process a dataset

        Parameters
        ----------
        - data             tuple with data input and data labels
        - attacks          dictionary that maps attack names to their index
        - outputs          list of output encodings, maps each index to a discrete binary output
    '''
    x_in, y_in = data
    for i, label in enumerate(y_in):
        if label not in attacks: y_in[i] = "MALIGN"
        y_in[i] = outputs[attacks[y_in[i]]] # choose result based on label
    x_in = np.array(x_in, dtype='float64')
    y_in = np.array(y_in, dtype='float64')
    return x_in, y_in



def train_new_network(train_data, saved_model_file, node_name, classifier, classifier_module=None, scaler=None, scaler_module=None, verbose=False):
    '''Train a new Neural Network model from given test dataset file

        Parameters
        ----------
        - train_data          tuple with data input and data labels
        - saved_model_file    file path to save the model (including filename)
        - classifier          string to be evaluated as the classifier
        - classifier_module   string containing the classifier module if needed
        - scaler              string to be evaluated as the scaler
        - scaler_module       string containing the scaler module if needed
    '''

    X_train, y_train = train_data

# scaler setup
    if scaler_module:
        exec('import '+ scaler_module) # import scaler module
    if scaler:
        scaler = eval(scaler).fit(X_train)
        X_train = scaler.transform(X_train) # normalize
        save_model(path.dirname(saved_model_file) + "/scalerX" + node_name,scaler)

# classifier setup
    if classifier_module:
        exec('import '+ classifier_module) # import classifier module
    model = eval(classifier)

# train and save the model
    if verbose: print("Training... ")
    model.fit(X_train, y_train)
    save_model(saved_model_file, model)
    return model



def predict(classifier, test_data, node_name, scaler_path=None, verbose=False):
    '''Apply the given classifier model to a test dataset

        Parameters
        ----------
        - classifier          classifier model
        - test_data           tuple with data input and data labels
        - scaler_path         directory path to save the scaler model
        - verbose             print actions
    '''

    X_test, y_test = test_data

    if scaler_path and path.isfile(scaler_path + "/scalerX" + node_name):
        scaler = load_model(scaler_path + "/scalerX" + node_name)
        X_test = scaler.transform(X_test) # normalize

    if verbose: print("Predicting... ")
    y_predicted = classifier.predict(X_test)
    return y_test, y_predicted



def classify(train_data, test_data, train_filename, node_name, config, disable_load=False, verbose=False):
    '''Create or load train model from given dataset and apply it to the test dataset

        If there is already a created model with the same classifier and train dataset
            it will be loaded, otherwise a new one is created and saved

        Parameters
        ----------
        - train_data          tuple with data input and data labels
        - test_data           tuple with data input and data labels
        - train_filename      filename of the train dataset
        - test_filename       filename of the test dataset
        - node_name           name of this node (used to load config options)
        - config              ConfigParser object to get layer settings
        - disable_load        list of output encodings, maps each index to a discrete binary output
    '''

# set options
    attacks = dict(zip(config.options('labels-l2'), range(len(config.options('labels-l2')))))
    n_labels = len(attacks)
    outputs = [[1 if j == i else 0 for j in range(n_labels)] for i in range(n_labels)]

# generate model filename
    saved_model_file = gen_saved_model_pathname(config.get(node_name, 'saved-model-path'), train_filename, config.get(node_name, 'classifier'))

# train or load the network
    if path.isfile(saved_model_file) and not disable_load and not config.has_option(node_name, 'force_train'):
        classifier = load_model(saved_model_file)
    else: # create a new network
        classifier = train_new_network(process_dataset(train_data, attacks, outputs), saved_model_file, node_name,
                classifier=config.get(node_name, 'classifier'),
                classifier_module=config.get(node_name, 'classifier-module') if config.has_option(node_name, 'classifier-module') else None,
                scaler=config.get(node_name, 'scaler') if config.has_option(node_name, 'scaler') else None,
                scaler_module=config.get(node_name, 'scaler-module') if config.has_option(node_name, 'scaler-module') else None,
                verbose=verbose)
        save_model(saved_model_file, classifier)

# apply network to the test data
    y_test, y_predicted = predict(classifier, process_dataset(test_data, attacks, outputs), node_name, path.dirname(saved_model_file) if config.has_option(node_name, 'scaler') else None, verbose)

    print_stats(y_predicted, y_test, n_labels, outputs, lambda i: list(attacks.keys())[i])
    return y_predicted

