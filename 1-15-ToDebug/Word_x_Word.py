import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle

best_feature_vectors = np.load("features_train.npy")
test_feature_vectors = np.load("features_test.npy")

training_amt = 8 #num of presentations used for training
testing_amt = 2 #num of presentations used for testing
toSelect = 5 #num of btc's selected
tEx = 10 #number of features per BTC vector

save_trainx = {}
save_trainy = {}
save_testx = {}
save_testy = {}

for word1 in range(63):
    for word2 in range(word1+1,63):
        trainx = np.zeros( (training_amt * 2, toSelect * tEx) )
        trainy = np.zeros( (training_amt * 2))
        testx = np.zeros( (testing_amt * 2, toSelect * tEx))
        testy = np.zeros( (testing_amt * 2))

        ptr = 0

        for pres in range(training_amt):
            trainx[ptr] = best_feature_vectors[word1][pres]
            trainy[ptr] = 0
            ptr+=1

            trainx[ptr] = best_feature_vectors[word2][pres]
            trainy[ptr] = 1
            ptr+=1
        ptr = 0
        for pres in range(testing_amt):
            testx[ptr] = test_feature_vectors[word1][pres]
            testy[ptr] = 0
            ptr+=1

            testx[ptr] = test_feature_vectors[word2][pres]
            testy[ptr] = 1
            ptr+=1

        save_trainx[(cat1,cat2)] = trainx
        save_trainy[(cat1,cat2)] = trainy
        save_testx[(cat1,cat2)] = testx
        save_testy[(cat1,cat2)] = testy

pickle.dump( (save_trainx, save_trainy, save_testx, save_testy), open("WordXWord.p","wb"))
