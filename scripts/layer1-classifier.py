#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import pickle, sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer

#USAGE: layer1-classifier training_dataset.csv testing_dataset.csv

# =====================
#     CONFIGURATION
# =====================

LABELS = 4 # 4, 7, 11
# Attack Mapping for output encoding, DoS-Attack is used when refering to all DoS type attacks. DDoS value is 0 for easier output
ATTACK_KEYS = ["DoS-Attack", "PortScan", "FTP-Patator", "SSH-Patator", "Bot", "Infiltration", "Heartbleed", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS"]
ATTACK_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
ATTACKS = dict(zip(ATTACK_KEYS, ATTACK_INDEX[:7] + (ATTACK_INDEX[-5:] if LABELS > 7 else [0] * 5)))
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
    filename = 'clf.sav'
    model_file = open(filename,'wb')
    pickle.dump(clfmodel, model_file)
    model_file.close()
    return

def load_model(filename):
    # load the model from disk
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
            try:
                y_in.append(OUTPUTS[ATTACKS[tmp[-1]]]) # choose result based on label
            except IndexError:
                print("ERROR: Dataset \"%s\" contains more labels than the ones allowed, \"%s\"." % (filename, tmp[-1]), file=sys.stderr)
                sys.exit(1)
            except KeyError:
                print("ERROR: Dataset \"%s\" contains unknown label \"%s\"." % (filename, tmp[-1]), file=sys.stderr)
                sys.exit(1)
    return x_in, y_in

def print_stats(y_predicted_lst, y_test_lst, train_label_count):
    print("# Flows             Type  Predicted / TOTAL")
    for i in range(LABELS) if LABELS != 11 else [x for x in range(1,LABELS)] + [0]: # put Dos attacks together
        predict, total = y_predicted_lst.count(OUTPUTS[i]), y_test_lst.count(OUTPUTS[i])
        color = '' if predict == total == 0 else '\033[1;3%dm' % (1 if predict > total else 2)
        print("%s% 7d %16s     % 6d / %d\033[m" % (color, train_label_count[i], 'DDoS' if LABELS == 11 and i == 0 else ATTACK_KEYS[i], predict, total))
    non_desc = sum((1 for elem in y_predicted_lst if elem.count(1) != 1))
    if non_desc: print("Non-descriptive output count:\033[1;33m", non_desc,"\033[mtest values")

if __name__ == '__main__':

# PARSE DATA AND GET TRAINING VALUES
    print('Reading Training Dataset...')
    X_train, y_train = parse_csvdataset(sys.argv[1])
    train_label_count = [y_train.count(OUTPUTS[i]) for i in range(LABELS)]
    X_train = np.array(X_train, dtype='float64')
    y_train = np.array(y_train, dtype='float64')
    scaler = preprocessing.StandardScaler().fit(X_train)
#X_train = scaler.transform(X_train)    # normalize

    print('Reading Test Dataset...')
    X_test, y_test = parse_csvdataset(sys.argv[2])
    X_test = np.array(X_test, dtype='float64')
#X_test = scaler.transform(X_test)      # normalize
    y_test = np.array(y_test, dtype='float64')

# train_test_split is not working as expected
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.25, random_state=42,stratify=y_train)

# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
    #MultilayerPerceptron = MLPClassifier()
    #print("Searching Grid")
    #clf = GridSearchCV(MultilayerPerceptron, PARAM_GRID, cv=3, scoring='accuracy')
    
    #print("Best parameters set found on development set:")
    #print(clf)

# DEFINE MODEL
    clf = MLPClassifier(activation='logistic', solver='adam', alpha=1e-5, hidden_layer_sizes=(64), random_state=1) # adam porque o dataset tem milhares de exemplos

# TRAIN MODEL
    print("Training... (" + sys.argv[1] + ")")
    clf.fit(X_train, y_train)

# PREDICT VALUES BASED ON THE GIVEN INPUT
    print("Predicting... (" + sys.argv[2] + ")\n")
    y_predicted = clf.predict(X_test)

# SAVE MODEL, LOAD MODEL
    #save_model('clf.sav', clf)
    #clfmodel = load_model('clf.sav')
    #y_predicted = clfmodel.predict(X_test)
    #result = clfmodel.score(X_test, y_test)
    #print("MLP Accuracy (Pickle): " + str(result))

# PRINT RESULTS
    print("MLP Correctly Classified:", accuracy_score(y_test, y_predicted, normalize=False) , "/" , len(y_predicted))
    print("MLP Accuracy: ", accuracy_score(y_test, y_predicted, normalize=True))
    # for the metric bellow use 'micro' for the precision value: tp / (tp + fp) , it seems to be the same as accuracy_score...
    print("MLP Precision:", precision_score(y_test.argmax(1), y_predicted.argmax(1), average='macro')) # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    print()

# LOOK AT PREDICTED VALUES AND PRINT STATS
    print_stats(y_predicted.tolist(), y_test.tolist(), train_label_count)
