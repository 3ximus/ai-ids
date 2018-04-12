from __future__ import print_function
import os, pickle, hashlib
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
    return base_path + '/%s-%s-%s' % (train_filename.strip('/.csv').replace('/','-'), train_file_md5.hexdigest()[:7], used_model_md5.hexdigest()[:7])


def print_stats(y_predicted, y_test, n_labels, outputs, get_class_name, test_filename, use_regressor=False):
    '''Print Classifier Statistics on a test dataset

        Parameters
        ----------
        - y_predicted     numpy list of predict NN outputs
        - y_test          numpy list of target outputs
        - n_labels          number of labels
        - outputs         categorical ouput classes (binary class array)
        - get_class_name  function that given the output index returns the output label class name
    '''
    if not use_regressor: y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)
    print('\n'+os.path.basename(test_filename))
    print("            Type  Predicted / TOTAL")
    y_predicted_lst = y_predicted.tolist()
    y_test_lst = y_test.tolist() if not use_regressor else y_test
    for i in range(n_labels):
        predict, total = y_predicted_lst.count(outputs[i]), y_test_lst.count(outputs[i])
        color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
        print("%s%16s     % 6d / %d\033[m" % (color, get_class_name(i), predict, total))
    print('    \033[1;34m->\033[m %f%% [%d/%d]' % (accuracy_score(y_test, y_predicted, normalize=True)*100, accuracy_score(y_test, y_predicted, normalize=False) , len(y_predicted)))

    # TODO REMOVE THIS CODE
    # non_desc = sum((1 for elem in y_predicted if elem.count(1) != 1))
    # if non_desc: print("Non-descriptive output count:\033[1;33m", non_desc,"\033[mtest values")


