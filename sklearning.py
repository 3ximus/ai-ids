import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
#from keras.models import Sequential
#from keras.layers import Activation,Dense

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

def print_anomaly_info(X_test,y_test,y_predicted,anomaly_index_lst):
    # Print the expected/true value (based on the NN training and labeled data) vs the NN's predicted value for
    # the (hopefully, or else it's useless) new input data. Please use the anomaly_edit() function for this purpose.
    for idx in anomaly_index_lst:
        print(X_test[idx])
        print(y_test[idx])
        print(y_predicted[idx])
    return

def anomaly_edit(X_test, y_test):
    # Mess with the test data used for cross-validation and tune the MLP parameters for maximum performance before predicting.
    i=0
    anomaly_index_lst = []
    while(i<len(X_test)):
        if(y_test[i]==0.0):
            #X_test[i][0]=1
            #X_test[i][1]=1
            #X_test[i][2]=1
            #X_test[i][3]=1
            #X_test[i][4]=1
            #X_test[i][5]=1
            #X_test[i][6]=1
            #X_test[i][7]=1
            #X_test[i][8]=1
            #X_test[i][9]=1
            #X_test[i][10]=0
            #print(str(X_test[i]) +":"+str(y_test[i]))
            anomaly_index_lst.append(i)
        i+=1
    return X_test, anomaly_index_lst

def get_specific_input_count(_input,data_arr):
    i=0
    for input_arr in data_arr:
        if(input_arr.tolist()==_input):
            i+=1
    return str(i)

def parse_pcapdataset(filename):
    dataset_file = open(filename,"r")
    data = []
    for pkt in dataset_file:
        binary_input = []
        for bit in pkt:
            if(bit!="\n"):
                binary_input.append(int(bit))
        data.append(binary_input)
    dataset_file.close()
    return data

def parse_csvdataset(filename):
    dataset_file = open(filename,"r")
    data = []
    for pkt in dataset_file:
        binary_input = []
        for i,bit in enumerate(pkt):
            if(i!=0 and bit!="\n"):
                binary_input.append(int(bit))
        data.append([binary_input,pkt[0]])
    dataset_file.close()
    return data

param_grid = [
    {
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'solver' : ['lbfgs', 'sgd', 'adam'],
        'hidden_layer_sizes': [
        (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        ]
    }
]

#input_lst = parse_pcapdataset("pcap/pcapdataset.txt")
testpcap_input = parse_pcapdataset("pcap/pcapdataset.txt")
data = parse_csvdataset("csv/csvdataset.txt")
input_lst = []
result_lst = []
for elem in data:
    input_lst.append(elem[0])
    result_lst.append(elem[1])

# Known Anomaly
'''
anomaly_input = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
n_E = 500
for _ in range(n_E):
    # (FIN,SYN,RST,PSH,ACK,URG,ECE,CWR|RESERVED,DONTFRAG,MOREFRAG|...)
    # (F,S,R,P,A,U,E,C|R,DF,MF|...)
    input_lst.append(anomaly_input)
'''

input_lst_X = np.array(input_lst)
input_lst_y = np.array(result_lst)
#input_lst_y[-n_E:] = 0

X_train, X_test, y_train, y_test = train_test_split(input_lst_X, input_lst_y, test_size=0.25, random_state=42,stratify=input_lst_y)
'''
print("-------------------Unaltered Training/Testing Input Info-------------------")
print(str(anomaly_input) + ":")
print("X_train count: " + get_specific_input_count(anomaly_input,X_train))
print("X_test count: " + get_specific_input_count(anomaly_input,X_test))
print("---------------------------------------------------------------------------")
'''
# FIND THE BEST PARAMETERS BASED ON TRAINING INPUT USING A GRID_SEARCH
#MultilayerPerceptron = MLPClassifier()
#clf = GridSearchCV(MultilayerPerceptron, param_grid, cv=3, scoring='accuracy')
#clf.fit(X_train,y_train)

#print("Best parameters set found on development set:")
#print(clf.best_params_)

# DEFINE MODEL
#clf = MLPClassifier(activation='identity', solver='lbfgs', hidden_layer_sizes=(1)) <-- parametros encontrados pela grid search, mas so ira servir quando os datasets reais estiverem prontos
clf = MLPClassifier(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(16), random_state=1) # adam porque o dataset tem milhares de exemplos
#print(clf)
# TRAIN MODEL
print("Training")
clf.fit(X_train, y_train)

# CHANGE ANOMALY ON TEST DATA ONLY
#X_test, anomaly_index_lst = anomaly_edit(X_test, y_test)

# PREDICT VALUES BASED ON THE GIVEN INPUT
print("Predicting")

y_predicted = clf.predict(np.array(testpcap_input))
y_test = np.zeros(len(testpcap_input))
#print(y_predicted)

# PRINT DATA FOR DEBUGGING NN
#print_anomaly_info(X_test,y_test,y_predicted,anomaly_index_lst)
#print(list(y_test).count(0))
#print(list(y_predicted).count(0))

# SAVE MODEL, LOAD MODEL
save_model('clf.sav', clf)
#clfmodel = load_model('clf.sav')
#y_predicted = clfmodel.predict(X_test)
#result = clfmodel.score(X_test, y_test)
#print("MLP Accuracy (Pickle): " + str(result))

# PRINT RESULTS
#if(len(y_predicted)==len(y_test)):
    #print("MLP Correctly Classified: " + str(accuracy_score(y_test, y_predicted,normalize=False)) + "/" + str(len(y_predicted)))
    #print("MLP Accuracy (sklearn): " + str(accuracy_score(y_test, y_predicted,normalize=True)))

#print(accuracy_score(y_test, y_predicted,normalize=True))
print(y_predicted.tolist().count('1'))