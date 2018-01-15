import numpy as np
from scipy.stats import pearsonr
from heapq import heappush, heappop, heappushpop
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import itertools
import pickle

all_data = np.load("all_data.npy") #holds all the data from channels
category_info = np.load("words_in_categories.npy") #category_info[cat][ptr] returns the number of the word(0...62) of the ptr'th word in the category cat
lengths = np.load("category_lengths.npy") #lengths[cat] is the number of words in category cat


total_words = 63 

tStart = 0 #start time
tEnd = 650 #end time
tWidth = 100 #width of time slice
tIncr = 50 #increment in start time
tEx = 10 #number of examples to downsample to

training_amt = 8 #8 examples for training, 2 for testing
testing_amt = 10 - training_amt

np.random.seed(63)

TrainingData = np.zeros((total_words,5,training_amt,256,650))#gives the pertinent data from all_data for the two categories
TestingData = np.zeros( (total_words,5,testing_amt,256,650)) #^
wordptr = -1 #the index of the current word, iterates from 0...total_words

categs = np.load('categorization.npy')

for i in range(63):
    wordptr+=1

    excl = [-1]*10 #excl[j] = the j'th presentation number which should be saved for testing (e.g. excl[0] = 0 means the first presentation of the wordptr'th word should be saved for testing). Ignore -1's.
    
    for pres in range(testing_amt):
        while(1): #this loop repeatedly generates a random presentation until one which hasn't been reserved for testing has been found, and then breaks it
            nxtrand = np.random.randint(0,10)
            if(excl[nxtrand]==-1):
                excl[nxtrand]=nxtrand
                break
    for bandnum in range(5):
        ptr2 = 0 #points to which presentation(0...9) of wordptr'th word we are currently copying to TrainingData
        for pres in range(10):
            if(excl[pres]!=-1): #if reserved for testing, don't include in training data
                continue
           
            TrainingData[wordptr][bandnum][ptr2]=all_data[bandnum][i][pres] #sets the channel x time matrix for TrainingData[bandnum][wordptr][ptr2]
            ptr2+=1 #move to next presentation

    for bandnum in range(5): #this loop is same as above, except now we only want the testing presentations
        ptr2=0
        for pres in range(10):
            if(excl[pres]==-1):
                continue
            TestingData[wordptr][bandnum][ptr2] = all_data[bandnum][i][excl[pres]]
            ptr2+=1

toSelect = 5 #number of top features to select

best_feature_vectors = np.zeros((total_words, training_amt,toSelect * tEx))
test_feature_vectors = np.zeros((total_words, testing_amt, toSelect * tEx))
timeSequences = np.zeros((total_words,5,12,training_amt,256,tEx))

fixedc = int(tWidth/tEx)
ptrr = 0
for t in range(tStart, tEnd-tWidth+1, tIncr):
    ptrppp = 0
    for tEStart in range(t,t+tWidth-tEx+1,tEx):
        timeSequences[:,:,ptrr,:,:,ptrppp] = np.average(TrainingData[:,:,:,:,tEStart:tEStart+fixedc], axis = 4)
        ptrppp+=1
    ptrr+=1
print(str(timeSequences.shape))

for wordnum in range(total_words):
    SHheap = [] #heap of BTC + featurevector information used to select top 400
    
    for band_num in range(5): #frequency bands
        ptrr=0
        for t in range(tStart, tEnd-tWidth+1, tIncr): #various starts of time slice
            for channel in range(256): #eeg channels

                #pairwise correlations
                avg_p = 0
                avg_p2 = 0
                #print(str(wordnum) + " " + str(band_num) + " " + str(ptrr) + " " + str(channel))
                for i in range(training_amt-1):
                    for j in range(i+1,training_amt):
                        #if(wordnum == 1):
                       #     print(str(pearsonr(timeSequences[wordnum][band_num][ptrr][channel][i],timeSequences[wordnum][band_num][ptrr][channel][j])))
                        avg_p += pearsonr(timeSequences[wordnum][band_num][ptrr][i][channel],timeSequences[wordnum][band_num][ptrr][j][channel])[0]

                '''
                for word2 in range(total_words):
                    if(wordnum==word2):
                        continue
                    avg_p2 += pearsonr(AverageWord[wordnum][band_num][ptrr][channel], AverageWord[word2][band_num][ptrr][channel])[0]
                '''
                avg_p /= training_amt*(training_amt-1)/2 #want to maximize
                #avg_p2 /= (total_words-1) #want to minimize
                #ranking_measure = (avg_p - avg_p2)/2 #want to maximize
                if(len(SHheap)<400):
                    heappush(SHheap, (avg_p,band_num,t,channel, timeSequences[wordnum,band_num,ptrr,:,channel]))
                else:
                    heappushpop(SHheap, (avg_p,band_num,t,channel, timeSequences[wordnum,band_num,ptrr,:,channel]))
            ptrr+=1
    #pick top 5
    
    #f.write("Word " + str(wordnum) +"\n")
    print("Word " + str(wordnum))

    
    current_matrix = np.zeros( (training_amt,0))
    test_matrix = np.zeros( (testing_amt,0))
    
    for i in range(400):
        (avg_p,band_num,t,channel, timeSequenc) = heappop(SHheap)
        if(i>=400-toSelect):
            #this is da guy
            #f.write(str(400-i) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p) + "\n")
            print(str(400-i) + ". " + str(band_num) + "   " + str(t) + "   " + str(channel) + "   " + str(avg_p))
            current_matrix = np.hstack( (current_matrix,timeSequenc))

            #construct testing matrix
            tmpo = np.zeros( (testing_amt,tEx))
            for itero in range(testing_amt):
                pp = 0
                for tEStart in range(t,t+tWidth-tEx+1,tEx):
                    tmpo[itero][pp] = np.average(TestingData[wordnum,band_num,itero,channel,tEStart:tEStart+int(tWidth/tEx)])
                    pp+=1
            test_matrix = np.hstack( (test_matrix,tmpo) )
            
    best_feature_vectors[wordnum] = current_matrix
    test_feature_vectors[wordnum] = test_matrix
np.save("features_train.npy",best_feature_vectors)
np.save("features_test.npy",test_feature_vectors)
