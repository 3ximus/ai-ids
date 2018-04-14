from __future__ import print_function
import os, pickle, hashlib
import numpy as np
from sklearn.metrics import accuracy_score

def save_model(filename, clfmodel):
    '''Save the model to disk'''
    if not os.path.isdir(os.path.dirname(filename)) and os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename))
    model_file = open(filename,'wb')
    pickle.dump(clfmodel, model_file)
    model_file.close()
    return

def load_model(filename):
    '''Load the model from disk'''
    if not os.path.isfile(filename):
        print("File %s does not exist." % filename)
        exit()
    model_file = open(filename, 'rb')
    loaded_model = pickle.load(model_file)
    model_file.close()
    return loaded_model

def parse_csvdataset(filename):
    '''Parse a dataset'''
    x_in, y_in = [], []
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])
            y_in.append(tmp[-1]) # choose result based on label
    return [x_in, y_in]


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


def train_model(train_data, saved_model_file, classifier, classifier_module=None, scaler=None, scaler_module=None, saved_scaler_file=None, use_regressor=False, verbose=False):
    '''Train a new Neural Network model from given test dataset file

        Parameters
        ----------
        - train_data          tuple with data input and data labels
        - saved_model_file    file path to save the model (including filename)
        - classifier          string to be evaluated as the classifier
        - classifier_module   string containing the classifier module if needed
        - scaler              string to be evaluated as the scaler
        - scaler_module       string containing the scaler module if needed
        - saved_scaler_file   file path to save the scaler
        - use_regressor       boolean flag, whether classifier is a regressor
    '''

    X_train, y_train = train_data

# scaler setup
    if scaler_module:
        exec('import '+ scaler_module) # import scaler module
    if scaler:
        if verbose: print("Using scaler... ")
        scaler = eval(scaler).fit(X_train)
        X_train = scaler.transform(X_train)    # normalize
        save_model(saved_scaler_file, scaler)

# classifier setup
    if classifier_module:
        exec('import '+ classifier_module) # import classifier module
    model = eval(classifier)

# train and save the model
    if verbose: print("Training... ")
    if use_regressor:
        y_train = [np.argmax(x) for x in y_train]
    try:
        model.fit(X_train, y_train)
    except ValueError as err:
        print("\n\033[1;31mERROR\033[m: Problem found when training model in L2.")
        print("This classifier might be a regressor:\n%s\nIf it is use 'regressor' option in configuration file" % model)
        print("ValueError:", err)
        exit()
    save_model(saved_model_file, model)
    return model


def predict(classifier, test_data, saved_scaler_file=None, verbose=False):
    '''Apply the given classifier model to a test dataset

        Parameters
        ----------
        - classifier          classifier model
        - test_data           X test data input
        - saved_scaler_file   file path to load the scaler, if None the scaler isn't used
        - verbose             print actions
    '''

    if saved_scaler_file and os.path.isfile(saved_scaler_file):
        if verbose: print("Loading scaler: %s" % saved_scaler_file)
        scaler = load_model(saved_scaler_file)
        test_data = scaler.transform(test_data) # normalize

    if verbose: print("Predicting... ")
    y_predicted = classifier.predict(test_data)
    return y_predicted


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


