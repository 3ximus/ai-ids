from __future__ import print_function
import numpy as np
from os import path
from classifier_functions import *


class NodeModel:
    '''Class to train and apply classifier/regressor models'''

    def __init__(self, node_name, config, verbose=False):
        '''Create a model object with given node_name, configuration and labels gathered from label_section in config options'''

        self.verbose = verbose
        self.node_name = node_name

        # get options
        attack_keys        = config.options(config.get(node_name, 'labels'))
        n_labels           = len(attack_keys)
        outputs            = [[1 if j == i else 0 for j in range(n_labels)] for i in range(n_labels)]

        self.outputs       = dict(zip(attack_keys, outputs))
        self.force_train   = config.has_option(node_name, 'force_train')
        self.use_regressor = config.has_option(node_name, 'regressor')
        self.save_path     = config.get(node_name, 'saved-model-path')

        self.label_map     = dict(config.items(config.get(node_name, 'labels-map'))) if config.has_option(node_name, 'labels-map') else dict()

        # model settings
        self.classifier        = config.get(node_name, 'classifier')
        self.classifier_module = config.get(node_name, 'classifier-module') \
                                    if config.has_option(node_name, 'classifier-module') else None
        self.scaler            = config.get(node_name, 'scaler') if config.has_option(node_name, 'scaler') else None
        self.scaler_module     = config.get(node_name, 'scaler-module') if config.has_option(node_name, 'scaler-module') else None

        self.model = None # leave uninitialized (run self.train)


    def process_dataset(self, data):
        '''Process data, where data is a list with x and y (y must be a list of labels)
            x is left untouched and should be an np.array
        '''

        for i, label in enumerate(data[1]): # data[1] is y data (labels)
            # try to apply label to elf.outputs, if not existent use label mapping to find valid label conversion
            if label in self.outputs:
                data[1][i] = self.outputs[label] # encode label into categorical ouptut classes
            elif label in self.label_map:
                data[1][i] = self.outputs[self.label_map[label]] # if an error ocurrs here your label conversion is wrong
            else:
                print("\033[1;31mERROR\033[m: Unknown label %s. Add it to correct mapping section in config file")
                exit()
        data[0] = np.array(data[0], dtype='float64')
        data[1] = np.array(data[1], dtype='int8')
        return data


    def train(self, train_filename, disable_load=False):
        '''Create or load train model from given dataset and apply it to the test dataset

            If there is already a created model with the same classifier and train dataset
                it will be loaded, otherwise a new one is created and saved

            Parameters
            ----------
            - train_filename      filename of the train dataset
            - disable_load        list of output encodings, maps each index to a discrete binary output
        '''
        # generate model filename
        self.saved_model_file = gen_saved_model_pathname(self.save_path, train_filename, self.classifier)
        self.saved_scaler_file = (path.dirname(self.saved_model_file) + '/scalerX_' + self.node_name) if self.scaler else None

        if path.isfile(self.saved_model_file) and not disable_load and not self.force_train:
            if self.verbose: print("Loading model: %s" % self.saved_model_file)
            self.model = load_model(self.saved_model_file)
        else: # create a new network
            train_data = parse_csvdataset(train_filename)
            self.model = train_model(self.process_dataset(train_data), self.saved_model_file, self.classifier,
                                     self.classifier_module, self.scaler, self.scaler_module, self.saved_scaler_file,
                                     self.use_regressor, self.verbose)
        return self.model

    def predict(self, test_data):
        '''Apply a created model to given test_data and return predicted classification

            Parameters
            ----------
            - test_data           tuple with data input and data labels
        '''

        if not self.model:
            print("ERROR: A model hasn't been trained or loaded yet. Run L1_Classifier.train")
            exit()

        # apply network to the test data
        X_test, y_test = self.process_dataset(test_data)
        y_predicted = predict(self.model, X_test, self.saved_scaler_file, self.verbose)

        if self.use_regressor:
            y_test = [np.argmax(x) for x in y_test]
            outputs = [np.argmax(x) for x in outputs] # FIXME
        else:
            y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)

        print_stats(y_predicted, y_test, self.outputs, self.use_regressor)

        return y_predicted


