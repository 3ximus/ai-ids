from __future__ import print_function
import os, pickle, hashlib
from sklearn.metrics import accuracy_score
import numpy as np

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
        self.scaler            = config.get(node_name, 'scaler') if config.has_option(node_name, 'scaler') else None
        self.scaler_module     = config.get(node_name, 'scaler-module') if config.has_option(node_name, 'scaler-module') else None
        self.classifier        = config.get(node_name, 'classifier')
        self.classifier_module = config.get(node_name, 'classifier-module') \
                                    if config.has_option(node_name, 'classifier-module') else None

        self.model = None # leave uninitialized (run self.train)


    @staticmethod
    def save_model(filename, clfmodel):
        '''Save the model to disk'''
        if not os.path.isdir(os.path.dirname(filename)) and os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename))
        model_file = open(filename,'wb')
        pickle.dump(clfmodel, model_file)
        model_file.close()
        return

    @staticmethod
    def load_model(filename):
        '''Load the model from disk'''
        if not os.path.isfile(filename):
            print("File %s does not exist." % filename)
            exit()
        model_file = open(filename, 'rb')
        loaded_model = pickle.load(model_file)
        model_file.close()
        return loaded_model

    @staticmethod
    def gen_saved_model_pathname(base_path, train_filename, classifier_settings):
        '''Generate name of saved model file

            Parameters
            ----------
            - base_path                base path name
            - train_filename           train file name
            - classifier_settings      string for with classifier training settings
        '''
        used_model_md5 = hashlib.md5()
        used_model_md5.update(classifier_settings.encode('utf-8'))
        train_file_md5 = hashlib.md5()
        with open(train_filename, 'rb') as tf:
            train_file_md5.update(tf.read())
        return base_path + '/%s-%s-%s' % (train_filename[:-4].replace('/','-'), train_file_md5.hexdigest()[:7], used_model_md5.hexdigest()[:7])


    @staticmethod
    def parse_csvdataset(filename):
        '''Parse a dataset'''
        x_in, y_in = [], []
        with open(filename, 'r') as fd:
            for line in fd:
                tmp = line.strip('\n').split(',')
                x_in.append(tmp[1:-1])
                y_in.append(tmp[-1]) # choose result based on label
        return [x_in, y_in]


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
        self.saved_model_file = self.gen_saved_model_pathname(self.save_path, train_filename, self.classifier)
        self.saved_scaler_file = (os.path.dirname(self.saved_model_file) + '/scalerX_' + self.node_name) if self.scaler else None

        if os.path.isfile(self.saved_model_file) and not disable_load and not self.force_train:
            # LOAD MODEL
            if self.verbose: print("Loading model: %s" % self.saved_model_file)
            self.model = self.load_model(self.saved_model_file)
        else:
            # CREATE NEW MODEL

            X_train, y_train = self.process_dataset(self.parse_csvdataset(train_filename))
            if self.use_regressor:
                y_train = [np.argmax(x) for x in y_train]

            # scaler setup
            if self.scaler_module:
                exec('import '+ self.scaler_module) # import scaler module
            if self.scaler:
                if self.verbose: print("Using scaler... ")
                scaler = eval(self.scaler).fit(X_train)
                X_train = scaler.transform(X_train)    # normalize
                self.save_model(self.saved_scaler_file, scaler)

            # classifier setup
            if self.classifier_module:
                exec('import '+ self.classifier_module) # import classifier module
            self.model = eval(self.classifier)

            # train and save the model
            if self.verbose: print("Training... ")
            try:
                self.model.fit(X_train, y_train)
            except ValueError as err:
                print("\n\033[1;31mERROR\033[m: Problem found when training model in L2.")
                print("This classifier might be a regressor:\n%s\nIf it is use 'regressor' option in configuration file" % model)
                print("ValueError:", err)
                exit()
            self.save_model(self.saved_model_file, self.model)
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

        X_test, y_test = self.process_dataset(test_data)
        # apply network to the test data
        if self.saved_scaler_file and os.path.isfile(self.saved_scaler_file):
            if self.verbose: print("Loading scaler: %s" % self.saved_scaler_file)
            scaler = self.load_model(self.saved_scaler_file)
            X_test = scaler.transform(X_test) # normalize

        if self.verbose: print("Predicting... ")
        y_predicted = self.model.predict(X_test)

        if self.use_regressor:
            y_test = [np.argmax(x) for x in y_test]
            self.outputs = {k:np.argmax(self.outputs[k]) for k in self.outputs}
        else:
            y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)

        # TODO SEPARATE CROSS VALIDATION #13

        self.print_stats(y_predicted, y_test, self.outputs, self.use_regressor)

        return y_predicted

    @staticmethod
    def print_stats(y_predicted, y_test, outputs, use_regressor=False):
        '''Print Classifier Statistics on a test dataset

            Parameters
            ----------
            - y_predicted     numpy list of predict NN outputs
            - y_test          numpy list of target outputs
            - outputs         categorical ouput classes (binary class array)
        '''

        print("            Type  Predicted / TOTAL")
        y_predicted_lst = y_predicted.tolist()
        y_test_lst = y_test.tolist() if not use_regressor else y_test
        for label in outputs:
            predict, total = y_predicted_lst.count(outputs[label]), y_test_lst.count(outputs[label])
            color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
            print("%s%16s     % 6d / %d\033[m" % (color, label, predict, total))
        print('    \033[1;34m->\033[m %f%% [%d/%d]' % (accuracy_score(y_test, y_predicted, normalize=True)*100, accuracy_score(y_test, y_predicted, normalize=False) , len(y_predicted)))

        # TODO REMOVE THIS CODE
        # non_desc = sum((1 for elem in y_predicted if elem.count(1) != 1))
        # if non_desc: print("Non-descriptive output count:\033[1;33m", non_desc,"\033[mtest values")




