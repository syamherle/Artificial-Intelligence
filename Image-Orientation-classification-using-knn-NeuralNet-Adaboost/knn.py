import numpy as np
import collections
import operator
import os
import sys


def apply_knn(train_data, test_data):
    print "Training..."
    k=9
    X = np.array([line.strip().split() for line in open(train_data)])
    item = np.array([line.strip().split() for line in open(test_data)])
    trainimages = X[:, 2:]
    trainlabels = X[:, 1]
    testimages = item[:, 2:]
    testlabels = item[:, 1]
    predictlabel=[]
    for i in range(len(testimages)):
        temp_arr=np.sqrt(np.sum((trainimages.astype(float) - testimages[i].astype(float))**2,axis=1))
        lst = np.array([[trainlabels[j],temp_arr[j]] for j in range(len(temp_arr))])
        predictlabel.append(collections.Counter(lst[lst[:, 1].argsort()][:9, 0]).most_common(1)[0][0])

    get_accuracy(np.array(predictlabel),testlabels)

def get_accuracy(a,b):
    print 'The testing accuracy is :', np.mean(a == b) * 100.00












