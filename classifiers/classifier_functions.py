from __future__ import print_function
import os, pickle
from sklearn.metrics import accuracy_score, precision_score

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

def print_stats(y_predicted, y_test, labels, outputs, get_class_name, test_filename):
    '''Print Classifier Statistics on a test dataset

        Parameters
        ----------
        - y_predicted     numpy list of predict NN outputs
        - y_test          numpy list of target outputs
        - labels          number of labels
        - outputs         categorical ouput classes (binary class array)
        - get_class_name  function that given the output index returns the output label class name
    '''
    y_predicted = (y_predicted == y_predicted.max(axis=1, keepdims=True)).astype(int)
    print("Classifier Correctly Classified:", accuracy_score(y_test, y_predicted, normalize=False) , "/" , len(y_predicted))
    print("Classifier Accuracy (" + os.path.basename(test_filename) + "):" + str(accuracy_score(y_test, y_predicted, normalize=True)))
    # For the metric below use 'micro' for the precision value: tp / (tp + fp)
    #  it seems to be the same as accuracy_score...
    # macro calculates metrics for each label, and finds their unweighted mean.
    #  This does not take label imbalance into account.
    print("Classifier Precision:", precision_score(y_test.argmax(1), y_predicted.argmax(1), average='macro'),'\n')

    print("             Type  Predicted / TOTAL")
    y_predicted_lst = y_predicted.tolist()
    y_test_lst = y_test.tolist()
    for i in range(labels):
        predict, total = y_predicted_lst.count(outputs[i]), y_test_lst.count(outputs[i])
        color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
        print("%s%16s     % 6d / %d\033[m" % (color, get_class_name(i), predict, total))
    non_desc = sum((1 for elem in y_predicted_lst if elem.count(1) != 1))
    if non_desc: print("Non-descriptive output count:\033[1;33m", non_desc,"\033[mtest values")


