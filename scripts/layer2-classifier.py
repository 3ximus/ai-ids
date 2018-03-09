#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pickle, sys, argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

#USAGE: layer2-classifier training_dataset.csv testing_dataset.csv
#		layer2-classifier neural_network2.sav testing_dataset.csv

# =====================
# ARGUMENT PARSING
# =====================

op = argparse.ArgumentParser( description="Classify MALIGN / BENIGN flows")
op.add_argument('files', metavar='file', nargs='*', help='')
op.add_argument('-l', '--load', action='store_true', help="load neural network", dest='load')
args = op.parse_args()

# =====================
#     CONFIGURATION
# =====================

ATTACK_TYPES = ("DoS-Attack", "PortScan", "FTP-Patator", "SSH-Patator", "Bot", "Infiltration", "Heartbleed", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS")
LABELS = 2
CLASSIFICATIONS = {"BENIGN": 0, "MALIGN": 1}
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]
PARAM_GRID = [{
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [
    (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,),(12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)]}]

# =====================
#      FUNCTIONS
# =====================

def save_model(filename, clfmodel):
    # save the model to disk
    print("Saving neural-network...")
    model_file = open(filename,'wb')
    pickle.dump(clfmodel, model_file)
    model_file.close()
    return

def load_model(filename):
    # load the model from disk
    print("Loading neural-network...")
    model_file = open(filename, 'rb')
    loaded_model = pickle.load(model_file)
    model_file.close()
    return loaded_model

def parse_csvdataset(filename):
    x_in = []
    y_in = []
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])
            y_tmp = [0] * LABELS
            if tmp[-1] in ATTACK_TYPES: tmp[-1] = "MALIGN"
            try:
                y_in.append(OUTPUTS[CLASSIFICATIONS[tmp[-1]]]) # choose result based on label
            except IndexError:
                print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]), file=sys.stderr)
                sys.exit(1)
            except KeyError:
                print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]), file=sys.stderr)
                sys.exit(1)
    return x_in, y_in

def print_stats(y_predicted, y_test, train_label_count=None):
    print("MLP Correctly Classified:", accuracy_score(y_test, y_predicted, normalize=False) , "/" , len(y_predicted))
    print("MLP Accuracy: ", accuracy_score(y_test, y_predicted, normalize=True))
    # for the metric below use 'micro' for the precision value: tp / (tp + fp) , it seems to be the same as accuracy_score...
    print("MLP Precision:", precision_score(y_test.argmax(1), y_predicted.argmax(1), average='macro'),'\n') # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    print("# Flows             Type  Predicted / TOTAL")
    if train_label_count==None: train_label_count = [8000]*LABELS
    CLASSIFICATION_KEYS = CLASSIFICATIONS.keys()
    y_predicted_lst = y_predicted.tolist()
    y_test_lst = y_test.tolist()
    for i in range(LABELS):
        predict, total = y_predicted_lst.count(OUTPUTS[i]), y_test_lst.count(OUTPUTS[i])
        color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
        print("%s% 7d %16s     % 6d / %d\033[m" % (color, train_label_count[i], CLASSIFICATION_KEYS[i], predict, total))
    non_desc = sum((1 for elem in y_predicted_lst if elem.count(1) != 1))
    if non_desc: print("Non-descriptive output count:\033[1;33m", non_desc,"\033[mtest values")

def train_new_network(filename):
    print('Reading Training Dataset...')
    X_train, y_train = parse_csvdataset(filename)
    train_label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)    # normalize
    neural_network2 = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(64), random_state=1)
    print("Training... (" + sys.argv[1] + ")")
    neural_network2.fit(X_train, y_train)
    return train_label_count, neural_network2

def predict(neural_network2, filename):
    print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    #X_test = scaler.transform(X_test)      # normalize
    print("Predicting... (" + filename + ")\n")
    y_predicted = neural_network2.predict(X_test)
    return y_test, y_predicted



if __name__ == '__main__':
    if args.load:
        neural_network2 = load_model(sys.argv[1])
    else:
        train_label_count, neural_network2 = train_new_network(sys.argv[1])
        save_model('neurals/neural_network2.sav', neural_network2)

    y_test, y_predicted = predict(neural_network2, sys.argv[2])

    if args.load:
        print_stats(y_predicted, y_test)
    else:
    	print_stats(y_predicted, y_test, train_label_count)

# train_test_split is not working as expected
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42,stratify=y_train)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
    #MultilayerPerceptron = MLPClassifier()
    #print("Searching Grid")
    #clf = GridSearchCV(MultilayerPerceptron, PARAM_GRID, cv=3, scoring='accuracy')
    
    #print("Best parameters set found on development set:")
    #print(clf)
