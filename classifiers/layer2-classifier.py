#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys, argparse, hashlib
from os import path
from classifier_functions import save_model, load_model, print_stats
from sklearn.neural_network import MLPClassifier
#new
from sklearn.ensemble import RandomForestRegressor

# USAGE: layer2-classifier training_dataset.csv testing_dataset.csv
#        layer2-classifier -l neural_network2.sav testing_dataset.csv

# =====================
#   ARGUMENT PARSING
# =====================

op = argparse.ArgumentParser( description="Classify MALIGN / BENIGN flows")
op.add_argument('files', metavar='file', nargs='+', help='train and test data files, if "-l | --load" option is given then just give the test data')
op.add_argument('-l', '--load', dest='load', nargs=1, metavar='NN_FILE', help="load neural network")
args = op.parse_args()

if not args.load and len(args.files) != 2:
    op.print_usage()
    print('train and test data must be given')
    sys.exit(1)

# =====================
#     CONFIGURATION
# =====================

LABELS = 2
ATTACK_TYPES = ("DoS-Attack", "PortScan", "FTP-Patator", "SSH-Patator", "Bot", "Infiltration", "Heartbleed", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS")
CLASSIFICATIONS = {"BENIGN": 0, "MALIGN": 1}
OUTPUTS = [[1 if j == i else 0 for j in range(LABELS)] for i in range(LABELS)]
PARAM_GRID = [{
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['lbfgs', 'sgd', 'adam'],
    'hidden_layer_sizes': [
    (16,),(32,),(64,),(128,)]}]

# =====================
#       FUNCTIONS
# =====================

def parse_csvdataset(filename):
    x_in = []
    y_in = []
    with open(filename, 'r') as fd:
        for line in fd:
            tmp = line.strip('\n').split(',')
            x_in.append(tmp[1:-1])
            if tmp[-1] in ATTACK_TYPES: tmp[-1] = "MALIGN"
            try:
                y_in.append(OUTPUTS[CLASSIFICATIONS[tmp[-1]]]) # choose result based on label
            except IndexError:
                print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]))
                sys.exit(1)
            except KeyError:
                print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]))
                sys.exit(1)
    return x_in, y_in

def train_new_network(filename):
    print('Reading Training Dataset... (' + filename + ')')
    X_train, y_train = parse_csvdataset(filename)
    label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')
    #scaler = preprocessing.StandardScaler().fit(X_train)
    #X_train = scaler.transform(X_train)    # normalize
    random_forest_classifier = RandomForestRegressor(max_depth=3, random_state=0)
    print("Training... (" + args.files[0] + ")")
    random_forest_classifier.fit(X_train, y_train)
    return label_count, random_forest_classifier

def predict(random_forest_classifier, filename):
    print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(filename)
    X_test = np.array(X_test, dtype='float64')
    y_test = np.array(y_test, dtype='float64')
    #X_test = scaler.transform(X_test)      # normalize
    print("Predicting... (" + filename + ")\n")
    y_predicted = random_forest_classifier.predict(X_test)
    return y_test, y_predicted

if __name__ == '__main__':
    digester = hashlib.md5()
    with open(args.files[0], 'rb') as tf: digester.update(tf.read())
    saved_path = 'saved_neural_networks/layer2/%s-%s' % (args.files[0].strip('/.csv').replace('/','-'), digester.hexdigest())
    if path.isfile(saved_path) and not args.load: # default if it exists
        random_forest_classifier = load_model(saved_path)
    elif args.load: # load nn if one is given
        random_forest_classifier = load_model(args.load[0])
    else: # create a new network
        label_count, random_forest_classifier = train_new_network(args.files[0])
        save_model(saved_path, random_forest_classifier)

    #print(random_forest_classifier.feature_importances_)
    y_test, y_predicted = predict(random_forest_classifier, args.files[-1])

    print_stats(y_predicted, y_test, LABELS, OUTPUTS,
                lambda i: list(CLASSIFICATIONS.keys())[i],
                args.files[-1], None if not args.load else label_count)

# train_test_split is not working as expected
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42,stratify=y_train)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
    #MultilayerPerceptron = MLPClassifier()
    #print("Searching Grid")
    #clf = GridSearchCV(MultilayerPerceptron, PARAM_GRID, cv=3, scoring='accuracy')

    #print("Best parameters set found on development set:")
    #print(clf)
