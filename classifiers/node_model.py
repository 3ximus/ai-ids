from __future__ import print_function
import os, pickle, hashlib, threading
from sklearn.metrics import accuracy_score
import numpy as np


class NodeModel:
    '''Class to train and apply classifier/regressor models'''

    def __init__(self, node_name, config, verbose=False):
        '''Create a model object with given node_name, configuration and labels gathered from label_section in config options'''

        self.verbose = verbose
        self.node_name = node_name
        self.message_buffer = [] # buffer for error messages

        # get options
        attack_keys        = config.options(config.get(node_name, 'labels'))
        n_labels           = len(attack_keys)
        outputs            = [[1 if j == i else 0 for j in range(n_labels)] for i in range(n_labels)]

        self.outputs       = dict(zip(attack_keys, outputs))
        self.force_train   = config.has_option(node_name, 'force_train')
        self.use_regressor = config.has_option(node_name, 'regressor')
        self.unsupervised = config.has_option(node_name, 'unsupervised')
        self.save_path     = config.get(node_name, 'saved-model-path')

        if self.use_regressor or self.unsupervised: # generate outputs for unsupervised or regressor models
            self.outputs = {k:np.argmax(self.outputs[k]) for k in self.outputs}

        self.label_map     = dict(config.items(config.get(node_name, 'labels-map'))) if config.has_option(node_name, 'labels-map') else dict()

        # model settings
        self.scaler            = config.get(node_name, 'scaler') if config.has_option(node_name, 'scaler') else None
        self.scaler_module     = config.get(node_name, 'scaler-module') if config.has_option(node_name, 'scaler-module') else None
        self.classifier        = config.get(node_name, 'classifier')
        self.classifier_module = config.get(node_name, 'classifier-module') \
                                    if config.has_option(node_name, 'classifier-module') else None

        self.model = None # leave uninitialized (run self.train)
        self.stats = Stats(self)


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
            self.log(MessageType.error, "File %s does not exist." % filename)
            exit()
        model_file = open(filename, 'rb')
        loaded_model = pickle.load(model_file)
        model_file.close()
        return loaded_model

    def log(self, m_type, message, force_log=False):
        if self.verbose or force_log:
            print('%s %13s [%s]: %s' % (m_type, self.node_name, threading.current_thread().getName(), message))
        if m_type == MessageType.error:
            self.message_buffer.append('%13s [%s]: %s' % (self.node_name, threading.current_thread().getName(), message))

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
        return base_path + '/%s-%s' % (train_filename[:-4].replace('/','-'), used_model_md5.hexdigest()[:7])


    def parse_csvdataset(self, filename):
        '''Parse entire dataset and return processed np.array with x and y'''
        x_in, y_in = [], []
        with open(filename, 'r') as fd:
            for line in fd:
                tmp = line.strip('\n').split(',')
                x_in.append(tmp[1:-1])
                y_in.append(tmp[-1]) # choose result based on label
        return self.process_data(x_in, y_in)

    def yield_csvdataset(self, filename, n_chunks):
        '''Iterate over dataset, yielding np.array with x and y in chunks of size n_chunks'''
        x_in, y_in = [], []
        with open(filename, 'r') as fd:
            for i, line in enumerate(fd):
                tmp = line.strip('\n').split(',')
                x_in.append(tmp[1:-1])
                y_in.append(tmp[-1]) # choose result based on label
                if (i+1) % n_chunks == 0:
                    yield self.process_data(x_in, y_in)
                    x_in, y_in = [], []
        yield self.process_data(x_in, y_in)


    def process_data(self, x, labels):
        '''Process data, y must be a list of labels, returns list with both lists converted to np.array'''
        y = []
        for i, label in enumerate(labels):
            # try to apply label to elf.outputs, if not existent use label mapping to find valid label conversion
            if label in self.outputs:
                y.append(self.outputs[label]) # encode label into categorical ouptut classes
            elif label in self.label_map:
                y.append(self.outputs[self.label_map[label]]) # if an error ocurrs here your label conversion is wrong
            else:
                self.log(MessageType.error, "Unknown label %s. Add it to correct mapping section in config file" %label)
                exit()
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='int8')
        return [x, y, labels]

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
            self.log(MessageType.log, "Loading model: %s" % os.path.basename(self.saved_model_file))
            self.model = self.load_model(self.saved_model_file)
        else:
            # CREATE NEW MODEL

            X_train, y_train, _ = self.parse_csvdataset(train_filename)

            # scaler setup
            if self.scaler_module:
                exec('import '+ self.scaler_module) # import scaler module
            if self.scaler:
                scaler = eval(self.scaler).fit(X_train)
                X_train = scaler.transform(X_train)    # normalize
                self.save_model(self.saved_scaler_file, scaler)

            # classifier setup
            if self.classifier_module:
                exec('import '+ self.classifier_module) # import classifier module
            self.model = eval(self.classifier)

            # train and save the model
            self.log(MessageType.log,"Training %s" % self.node_name, force_log=True)
            try:
                if(self.unsupervised):
                    self.model.fit(X_train)
                else:
                    self.model.fit(X_train, y_train)
            except ValueError as err:
                self.log(MessageType.error, "Problem found when training model, this classifier might be a regressor:\n%s\nIf it is, use 'regressor' option in configuration file" % self.model)
                exit()
            except TypeError:
                self.log(MessageType.error, "Problem found when training model, this classifier might not be unsupervised:\n%s" % self.model)
                exit()
            self.save_model(self.saved_model_file, self.model)
        return self.model

    def predict(self, test_data):
        '''Apply a created model to given test_data (tuple with data input and data labels) and return predicted classification'''

        if not self.model:
            self.log(MessageType.error, "ERROR: A model hasn't been trained or loaded yet for %s. Run NodeModel.train" % self.node_name)
            exit()

        X_test, y_test, _ = test_data
        # apply network to the test data
        if self.saved_scaler_file and os.path.isfile(self.saved_scaler_file):
            scaler = self.load_model(self.saved_scaler_file)
            try:
                X_test = scaler.transform(X_test) # normalize
            except ValueError as e:
                self.log(MessageType.error, 'Transforming with scaler\n'+ str(e))
                exit()

        self.log(MessageType.log, "Predicting on #%d samples" % len(X_test))
        try:
            y_predicted = self.model.predict(X_test)
        except ValueError as e:
            self.log(MessageType.error, 'Predicting\n'+ str(e))
            exit()

        if not self.use_regressor and not self.unsupervised:
            y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)

        if self.unsupervised:
            y_predicted[y_predicted == 1] = 0
            y_predicted[y_predicted == -1] = 1
        self.stats.update(y_predicted, y_test, self.outputs)
        return y_predicted


class Stats:
    '''Holds stats from predictions. Can be updated multiple times to include more stats on tests with same labels'''

    def __init__(self, node):
        self.node = node
        self.stats = dict()
        self.total = 0
        self.total_correct = 0
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.stats)

    def get_label_predicted(self, label):
        return self.stats[label][0]

    def get_label_total(self, label):
        return self.stats[label][1]

    def get_total(self):
        return self.total

    def get_total_correct(self):
        return self.total_correct

    def update(self, y_predicted, y_test, outputs):
        '''Update stats values with more results. Thread safe.

            Parameters
            ----------
            - y_predicted     numpy list of predict NN outputs
            - y_test          numpy list of target outputs
            - outputs         dictionary where keys are labels and values are encoded outputs of each label
        '''

        predict_uniques, predict_counts = np.unique(y_predicted, axis=0, return_counts=True)
        test_uniques, test_counts = np.unique(y_test, axis=0, return_counts=True)

        with self.lock:
            for label in outputs:
                tmp = [np.array_equal(outputs[label], x) for x in predict_uniques]
                label_predicted = predict_counts[tmp.index(True)] if any(tmp) else 0

                tmp = [np.array_equal(outputs[label], x) for x in test_uniques]
                label_total = test_counts[tmp.index(True)] if any(tmp) else 0

                if label not in self.stats:
                    self.stats[label] = [label_predicted, label_total]
                else:
                    self.stats[label][0] += label_predicted
                    self.stats[label][1] += label_total

            self.total += len(y_predicted)
            self.total_correct += accuracy_score(y_test, y_predicted, normalize=False)

    def __repr__(self):
        rep_str = "            Type  Predicted / TOTAL\n"
        with self.lock:
            for label in self.stats:
                predict = self.stats[label][0]
                total = self.stats[label][1]
                color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
                rep_str += "%s%16s     % 6d / %d\033[m\n" % (color, label, predict, total)
            rep_str += '    \033[1;34m->\033[m %f%% [%d/%d]\n' % (float(self.total_correct)/self.total*100, self.total_correct , self.total)
            for err_msg in self.node.message_buffer:
                rep_str += '[\033[1;31mERROR\033[m]%s\n'%err_msg
        return rep_str

    def update_curses_screen(self, curses_screen, curses):
        with self.lock:
            curses_screen.addstr("            Type  Predicted / TOTAL\n")
            for label in self.stats:
                predict = self.stats[label][0]
                total = self.stats[label][1]
                color = curses.color_pair(2) if predict == total == 0 else curses.color_pair(2 if predict > total else 3)
                curses_screen.addstr("%16s     % 6d / %d\n" % (label, predict, total), color | curses.A_BOLD)
            curses_screen.addstr('    -> %f%% [%d/%d]\n' % (float(self.total_correct)/self.total*100, self.total_correct , self.total))
            for err_msg in self.node.message_buffer:
                curses_screen.addstr('[')
                curses_screen.addstr('ERROR', curses.color_pair(2) | curses.A_BOLD)
                curses_screen.addstr(']%s\n' + err_msg)

class MessageType:
    error = '[\033[1;31mERROR\033[m]'
    warning = '[\033[1;33mWARNING\033[m]'
    log = '[\033[1;34m LOG \033[m]'

