import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle


lengths = np.load("category_lengths.npy")
best_feature_vectors = np.load("features_train.npy")
test_feature_vectors = np.load("features_test.npy")


category_info = np.load("words_in_categories.npy") #category_info[cat][ptr] returns the number of the word(0...62) of the ptr'th word in the category cat


training_amt = 8 #num of presentations used for training
testing_amt = 2 #num of presentations used for testing
toSelect = 5 #num of btc's selected
tEx = 10 #number of features per BTC vector
save_trainx = {}
save_trainy = {}
save_testx = {}
save_testy = {}
for cat1 in range(12):
    for cat2 in range(cat1+1,12):
        tot_words = int(lengths[cat1][0]) + int(lengths[cat2][0])

        trainx = np.zeros( (0, toSelect * tEx))
        trainy = np.zeros( (training_amt * tot_words) )
        testx = np.zeros( (0, toSelect * tEx))
        testy = np.zeros( (testing_amt * tot_words))

        ytraincnt = 0
        ytestcnt = 0
       
        for pres in range(training_amt):
            for cat1_word in category_info[cat1]:
                if cat1_word != -1:
                    trainx = np.concatenate((trainx, np.reshape(best_feature_vectors[cat1_word][pres],(1,toSelect*tEx))), axis=0)
                    trainy[ytraincnt] = 0
                    ytraincnt+=1
                   
                    if(pres<testing_amt):
                        testx = np.concatenate((testx, np.reshape(test_feature_vectors[cat1_word][pres],(1,toSelect*tEx))), axis=0)
                        testy[ytestcnt] = 0
                        ytestcnt+=1
 
            for cat2_word in category_info[cat2]:
                if cat2_word != -1:
                    trainx = np.concatenate((trainx, np.reshape(best_feature_vectors[cat2_word][pres],(1,toSelect*tEx))), axis=0)
                    trainy[ytraincnt] = 1
                    ytraincnt+=1
                   
                    if(pres<testing_amt):
                        testx = np.concatenate((testx, np.reshape(test_feature_vectors[cat2_word][pres],(1,toSelect*tEx))), axis=0)
                        testy[ytestcnt] = 1
                        ytestcnt+=1
        
        save_trainx[(cat1,cat2)] = trainx
        save_trainy[(cat1,cat2)] = trainy
        save_testx[(cat1,cat2)] = testx
        save_testy[(cat1,cat2)] = testy
pickle.dump( (save_trainx, save_trainy, save_testx, save_testy), open("CategoryXCategory.p","wb"))
            
