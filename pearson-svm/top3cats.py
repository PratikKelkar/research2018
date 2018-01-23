import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle

#clist = np.logspace(-4,2,100)

training_amt = 8 #num of presentations used for training
testing_amt = 2 #num of presentations used for testing
toSelect = 5 #num of btc's selected
tEx = 10 #number of features per BTC vector

avgacc = 0

bst_acc = 0
bst_c = 1


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

    
clf = LinearSVC(C=bst_c, multi_class='ovr')
scaler = StandardScaler()
trainx = scaler.fit_transform(trainx)
testx = scaler.transform(testx)
trainy = np.ravel(trainy)
testy = np.ravel(testy)
clf.fit(trainx, trainy)
confidences = clf.decision_function(testx)
answers = np.zeros((testy.shape[0],3))
got = 0
for sample in range(testy.shape[0]):
    jj = np.argsort(confidences[sample])
    for i in range(3):
        answers[sample,i] = jj[-(i+1)]
    if(answers[sample,0] == testy[sample] or answers[sample,1] == testy[sample]
       or answers[sample,2] == testy[sample]):
        got+=1
print("Score: " + str(got / (testy.shape[0])))
    
#myscore = clf.score(testx, testy)
#avgacc+=myscore

#print("For " + str(pair[0]) + " and " + str(pair[1]) + " we picked C = " + str(bst_c))
#print("Has accuracy " + str(myscore))
#print("=========")

print(myscore)
