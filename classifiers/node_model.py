from __future__ import print_function
import os, pickle, hashlib, threading
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

class NodeModel:
    '''Class to train and apply classifier/regressor models'''

    def __init__(self, node_name, config, verbose=False):
        '''Create a model object with given node_name, configuration and labels gathered from label_section in config options'''

        self.verbose = verbose
        self.node_name = node_name
        self.message_buffer = [] # buffer for error messages

        # get options
        self.attack_keys   = config.options(config.get(node_name, 'labels'))
        n_labels           = len(self.attack_keys)
        outputs            = [[1 if j == i else 0 for j in range(n_labels)] for i in range(n_labels)]

        self.outputs       = dict(zip(self.attack_keys, outputs))
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
        self.saved_model_file = None
        self.saved_scaler_file = None
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
            print("File %s does not exist." % filename)
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

    # normal csvs
    def parse_csvdataset(self, filename):
        '''Parse entire dataset and return processed np.array with x and y'''
        flow_ids, x_in, y_in = [], [], []
        with open(filename, 'r') as fd:
            feature_string = fd.readline()
            feature_string = feature_string.split(',')
            index_subset=[]
            for elem in feature_string:
                if elem.find('iat')==-1 and elem.find('sec')==-1 and elem.find('duration')==-1 and elem!='label\n' and elem!='flow_id':
                    index_subset.append(feature_string.index(elem))
            for line in fd:
                tmp = line.strip('\n').split(',')
                #x_in.append(tmp[1:-1])
                x_in.append([tmp[i] for i in index_subset])
                y_in.append(tmp[-1]) # choose result based on label
                flow_ids.append(tmp[0])
        return self.process_data(x_in, y_in, flow_ids)

    def yield_csvdataset(self, filename, n_chunks):
        '''Iterate over dataset, yielding np.array with x and y in chunks of size n_chunks'''
        flow_ids, x_in, y_in = [], [], []
        with open(filename, 'r') as fd:
            feature_string = fd.readline()
            feature_string = feature_string.split(',')
            index_subset=[]
            for elem in feature_string:
                if elem.find('iat')==-1 and elem.find('sec')==-1 and elem.find('duration')==-1 and elem!='label\n' and elem!='flow_id':
                    index_subset.append(feature_string.index(elem))
            for i, line in enumerate(fd):
                tmp = line.strip('\n').split(',')
                #x_in.append(tmp[1:-1])
                x_in.append([tmp[j] for j in index_subset])
                y_in.append(tmp[-1]) # choose result based on label
                flow_ids.append(tmp[0])
                if (i+1) % n_chunks == 0:
                    yield self.process_data(x_in, y_in, flow_ids)
                    x_in, y_in, flow_ids = [], [], []
        yield self.process_data(x_in, y_in, flow_ids)

    def process_data(self, x, labels, flow_ids):
        '''Process data, y must be a list of labels, returns list with both lists converted to np.array'''
        y = []
        for label in labels:
            # try to apply label to elf.outputs, if not existent use label mapping to find valid label conversion
            if label in self.outputs:
                y.append(self.outputs[label]) # encode label into categorical ouptut classes
            elif label in self.label_map:
                y.append(self.outputs[self.label_map[label]]) # if an error ocurrs here your label conversion is wrong
            else:
                self.log(MessageType.error, "Unknown label %s. Add it to correct mapping section in config file" % label)
                exit()
        x = np.array(x, dtype='float64')
        y = np.array(y, dtype='int8')
        flow_ids = np.array(flow_ids)
        return [x, y, labels, flow_ids]

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

            X_train, y_train, _, _ = self.parse_csvdataset(train_filename)
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
                if self.unsupervised:
                    self.model.fit(X_train)
                else:
                    self.model.fit(X_train, y_train)
            except ValueError as err:
                self.log(MessageType.error, "Problem found when training model, this classifier might be a regressor:\n%s\nIf it is, use 'regressor' option in configuration file" % self.model)
                self.log(MessageType.error, err)
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
        X_test, y_test, _, flow_ids = test_data
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
        self.stats.update(y_predicted, y_test)
        return y_predicted,flow_ids


class Stats:
    '''Holds stats from predictions. Can be updated multiple times to include more stats on tests with same labels'''

    def __init__(self, node):
        self.node = node
        self.n = self.total_correct = 0
        self.confusion_matrix = np.matrix([[0 for x in range(len(node.outputs))] for x in range(len(node.outputs))])
        self.lock = threading.Lock()

    def calculate_metrics(self, tp, tn, fp, fn, total, rep_str):
        rep_str += "Overall Acc = \033[34m%4f\033[m\n" % (float(tp+tn)/total)
        if tp+fn:
            rep_str += "Recall = %4f\n" % (float(tp)/(tp+fn))
            rep_str += "Miss Rate = %4f\n" % (float(fn)/(tp+fn))
        if tn+fp:
            rep_str += "Specificity = %4f\n" % (float(tn)/(tn+fp))
            rep_str += "Fallout = %4f\n" % (float(fp)/(tn+fp))
        if tp+fp: rep_str += "Precision = %4f\n" % (float(tp)/(tp+fp))
        if tp+fp+fn: rep_str += "F1 score = %4f\n" % (float(2*tp)/(2*tp+fp+fn))
        if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn): rep_str += "Mcc = %4f\n" % (float((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        return rep_str

    def update(self, y_predicted, y_test):
        '''Update stats values with more results. Thread safe.

            Parameters
            ----------
            - y_predicted     numpy list of predict NN outputs
            - y_test          numpy list of target outputs
        '''

        with self.lock:
            self.total_correct += accuracy_score(y_test, y_predicted, normalize=False) # counts only elements not classified as [0,0..,0]
            # TODO FIXME Using np.argmax puts unclassified entries ([0,0,...,0]) as label zero, see previous code to count those
            x = np.argmax(y_test, axis=1) if len(y_test.shape) == 2 else y_test
            y = np.argmax(y_predicted, axis=1) if len(y_predicted.shape) == 2 else y_predicted
            self.confusion_matrix += np.matrix(confusion_matrix(x, y, labels=list(range(len(self.node.attack_keys)))))
            self.n = self.confusion_matrix.sum()

    def __repr__(self):
        with self.lock:
            # confusion matrix
            lmsize = max(map(len, self.node.attack_keys[:-1])) # for output formatting
            rep_str = " Real\\Pred |" + ''.join([('%' + str(lmsize) + 's ') % label for label in self.node.attack_keys]) + "\n"
            for i, label in enumerate(self.node.attack_keys):
                rep_str += "%10s |" % label
                for j in range(len(self.node.attack_keys)):
                    rep_str += (("\033[1;32m" if i == j else '') + "%" + str(lmsize) + "d\033[m ") % self.confusion_matrix[i,j]
                rep_str += "\n"

            # stats
            if len(self.node.attack_keys) == 2: # MALIGN OR BENIGN
                if np.argmax(self.node.outputs['MALIGN']) == 1:
                    tn, fp, fn, tp = np.ravel(self.confusion_matrix)
                else:
                    tp, fn, fp, tn = np.ravel(self.confusion_matrix)
                rep_str = self.calculate_metrics(tp, tn, fp, fn, self.n, rep_str)
            elif len(self.node.attack_keys) == 3: # Layer-1 three labels
                dos_dos, dos_ps, dos_bf, ps_dos, ps_ps, ps_bf, bf_dos, bf_ps, bf_bf = np.ravel(self.confusion_matrix)
                # fastdos-portscan
                tp_dos_ps = dos_dos
                tn_dos_ps = ps_ps
                fp_dos_ps = ps_dos
                fn_dos_ps = dos_ps
                total_dos_ps = dos_dos + ps_ps + ps_dos + dos_ps
                rep_str+= "\033[1;33mfastdos-portscan accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_dos_ps, tn_dos_ps, fp_dos_ps, fn_dos_ps, total_dos_ps, rep_str)
                # fastdos-bruteforce
                tp_dos_bf = dos_dos
                tn_dos_bf = bf_bf
                fp_dos_bf = bf_dos
                fn_dos_bf = dos_bf
                total_dos_bf = dos_dos + bf_bf + bf_dos + dos_bf
                rep_str+= "\033[1;33mfastdos-bruteforce accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_dos_bf, tn_dos_bf, fp_dos_bf, fn_dos_bf, total_dos_bf, rep_str)
                # portscan-bruteforce
                tp_ps_bf = ps_ps
                tn_ps_bf = bf_bf
                fp_ps_bf = bf_ps
                fn_ps_bf = ps_bf
                total_ps_bf = ps_ps + bf_bf + bf_ps + ps_bf
                rep_str+= "\033[1;33mportscan-bruteforce accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_ps_bf, tn_ps_bf, fp_ps_bf, fn_ps_bf, total_ps_bf, rep_str)

            # unidentified
            diag = sum(np.diag(self.confusion_matrix))
            if diag - self.total_correct:
                rep_str += "Unidentified flows marked as \"%s\": \033[1;33m#%d\033[m\n" % \
                    (self.node.attack_keys[0], diag - self.total_correct)

            # append error messages
            for err_msg in self.node.message_buffer:
                rep_str += '[\033[1;31mERROR\033[m]%s\n'%err_msg
        return rep_str

    def update_curses_screen(self, curses_screen, curses):
        with self.lock:
            lmsize = max(map(len, self.node.attack_keys[:-1])) # for output formatting
            curses_screen.addstr(" Real\\Pred |" + ''.join([('%' + str(lmsize) + 's ') % label for label in self.node.attack_keys]) + "\n")
            for i, label in enumerate(self.node.attack_keys):
                curses_screen.addstr("%10s |" % label)
                for j in range(len(self.node.attack_keys)):
                    curses_screen.addstr(("%" + str(lmsize) + "d ") % self.confusion_matrix[i,j])
                curses_screen.addstr('\n')

            for err_msg in self.node.message_buffer:
                curses_screen.addstr('[')
                curses_screen.addstr('ERROR', curses.color_pair(2) | curses.A_BOLD)
                curses_screen.addstr(']%s\n' + err_msg)

class MessageType:
    error = '[\033[1;31mERROR\033[m]'
    warning = '[\033[1;33mWARNING\033[m]'
    log = '[\033[1;34m LOG \033[m]'

