import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle

file_name = 'CategoryXCategory.p' #change this appropriately

loaded_data = pickle.load(open(file_name,"rb"))

dict_trainx = loaded_data[0]
dict_trainy = loaded_data[1]
dict_testx = loaded_data[2]
dict_testy = loaded_data[3]

clist = np.logspace(-4,2,100)

training_amt = 8 #num of presentations used for training
testing_amt = 2 #num of presentations used for testing
toSelect = 5 #num of btc's selected
tEx = 10 #number of features per BTC vector

avgacc = 0
for pair in dict_trainx:
    trainx = dict_trainx[pair]
    trainy = dict_trainy[pair]
    testx = dict_testx[pair]
    testy = dict_testy[pair]

    bst_acc = 0
    bst_c = 0

    for c in clist:
        avg_acc = 0
        for fold in range(4):
            fold_sz = int(trainx.shape[0]/4)
            valid_x = trainx[(fold_sz*fold):((fold_sz)*(fold+1))]
            valid_y = trainy[(fold_sz*fold):((fold_sz)*(fold+1))]
            tr_x = np.concatenate((trainx[:(fold_sz*fold)],trainx[((fold+1)*fold_sz):]), axis = 0)
            tr_y = np.concatenate((trainy[:(fold_sz*fold)],trainy[((fold+1)*fold_sz):]), axis = 0)

            scaler = StandardScaler()
            tr_x = scaler.fit_transform(tr_x)
            valid_x = scaler.transform(valid_x)
            tr_y = np.ravel(tr_y)
            valid_y = np.ravel(valid_y)

            classifier = LinearSVC(C = c)
            classifier.fit(tr_x,tr_y)
            avg_acc += (classifier.score(valid_x, valid_y))/4.0
        if(avg_acc > bst_acc):
            bst_acc = avg_acc
            bst_c = c
    clf = LinearSVC(C=bst_c)
    scaler = StandardScaler()
    trainx = scaler.fit_transform(train_x)
    testx = scaler.transform(testx)
    trainy = np.ravel(trainy)
    testy = np.ravel(testy)
    clf.fit(trainx, trainy)
    myscore = clf.score(testx, testy)
    avgacc+=myscore

    print("For " + str(pair[0]) + " and " + str(pair[1]) + " we picked C = " + str(bst_c))
    print("Has accuracy " + str(myscore))
    print("=========")
