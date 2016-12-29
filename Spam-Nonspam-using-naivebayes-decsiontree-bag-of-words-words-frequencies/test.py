import time


import pickle
import numpy as np
from numpy import *

import pandas as pd
import os
import collections, re
import csv
import operator
import math
from itertools import islice
import dtree as dt

class Naivebayestest:
    def __init__(self, email_path, model_file):
        self.email_path=email_path
        self.email_files = [f for f in os.listdir(email_path)]
        self.model_file=model_file
        self.classified_labels=[]

    def get_files(self):
        self.emails= []
        self.labels=[]
        stop_words=['the', 'that', 'to', 'as', 'there', 'has', 'and', 'or', 'is', 'not', 'a', 'of', 'but', 'in', 'by', 'on', 'are',
         'it', 'if']
        for e in self.email_files:
            for e_file in os.listdir(self.email_path+e):
                with open(self.email_path + e +'/'+ e_file, 'r') as email:
                    text = email.read().lower().split()
                    for w in text:
                        if w in stop_words:
                            text.remove(w)
                    self.labels.append([self.email_files.index(e)])
                    self.emails.append(text)

        return self.emails,self.labels


    def get_model(self):
        f = open(self.model_file, "r")
        self.predict_dict = pickle.load(f)
        self.words_nonspam = pickle.load(f)
        self.words_spam = pickle.load(f)
        self.prioir_prob = pickle.load(f)
        self.nonspam_freq_word=pickle.load(f)
        self.spam_freq_word = pickle.load(f)
        f.close()


    def get_bag_words(self):
        self.all_word_vector = []
        for email in self.emails:
            temp_list = [0]*len(self.predict_dict)
            for w in email:
                if w in self.predict_dict:
                    temp_list[self.predict_dict[w]]=1
            temp_list.append(self.labels[self.emails.index(email)][0])
            self.all_word_vector.append(temp_list)
        self.all_word_vector=np.array(self.all_word_vector,object)
        ele = self.all_word_vector[:, :-1]
        return ele


    def predict_spam_prob(self):
        cs=np.array(self.words_spam)
        self.ele=self.get_bag_words()
        self.ele *= cs
        del cs
        self.ele = self.ele.astype(np.float)
        self.ele = np.log(self.ele)

        for e in self.ele:
            e[e == -inf] = 0
        self.ele = np.sum(self.ele, axis=1)



    def predict_nonspam_prob(self):
        self.ele_copy = self.get_bag_words()

        cd = np.array(self.words_nonspam)
        self.ele_copy *= cd
        del cd
        self.ele_copy = self.ele_copy.astype(np.float)
        self.ele_copy = np.log(self.ele_copy)
        for e in self.ele_copy:
            e[e == -inf] = 0

        self.ele_copy = np.sum(self.ele_copy, axis=1)

    def classify_document(self):
        prob_spam_cond = float(self.prioir_prob[2])/float(self.prioir_prob[0])
        prob_nonspam_cond = float(self.prioir_prob[1]) / float(self.prioir_prob[0])
        self.ele += math.log(prob_spam_cond)
        self.ele_copy += math.log(prob_nonspam_cond)
        self.prob_array = np.empty((self.ele.shape[0],0),int,order='C')
        self.ele_copy = self.ele_copy.T
        self.ele = self.ele.T
        self.prob_array = np.c_[self.prob_array, self.ele_copy,self.ele]
        classified_mail = self.prob_array.argmin(axis=1)
        self.prob_array = np.c_[self.prob_array,classified_mail,self.labels]
        self.prob_array= self.prob_array.astype(np.float)



    def find_accuracy(self):
        nf=self.prob_array
        classifies_spam_correct,classifies_nonspam_correct,spam_count_test,non_spam_count_test=0,0,0,0
        classifies_spam_correct_freq, classifies_nonspam_correct_freq, spam_count_test_freq, non_spam_count_test_freq = 0, 0, 0, 0
        nonspam_predicted_spam,spam_predicted_nonspam=0,0
        for v in nf:
            if int(v[-2]) == int(v[-1]) == 1:
                classifies_spam_correct=classifies_spam_correct+1
            if v[-1] == 1:
                spam_count_test = spam_count_test+1
            if int(v[-2]) == int(v[-1]) == 0:
                classifies_nonspam_correct=classifies_nonspam_correct+1
            if v[-1] == 0:
                non_spam_count_test = non_spam_count_test+1

            if int(v[-2]) ==1 and int(v[-1]) == 0:
                nonspam_predicted_spam= nonspam_predicted_spam+1
            if int(v[-2]) ==0 and int(v[-1]) ==1:
                spam_predicted_nonspam= spam_predicted_nonspam+1

        nonspam_predicted_spam_freq,spam_predicted_nonspam_freq =0,0
        for i,v in enumerate(self.classified_labels):
            if int(v) == int(self.labels[i][0]) == 1:
                classifies_spam_correct_freq=classifies_spam_correct_freq+1
            if int(self.labels[i][0]) == 1:
                spam_count_test_freq = spam_count_test_freq+1

            if int(v) == int(self.labels[i][0]) == 0:
                classifies_nonspam_correct_freq=classifies_nonspam_correct_freq+1
            if int(self.labels[i][0]) == 0:
                non_spam_count_test_freq = non_spam_count_test_freq+1

            if int(v) ==1 and int(self.labels[i][0]) == 0:
                nonspam_predicted_spam_freq= nonspam_predicted_spam_freq+1
            if int(v) ==0 and int(self.labels[i][0]) ==1:
                spam_predicted_nonspam_freq= spam_predicted_nonspam_freq+1




        classifier_accuracy_bag_words = float(float(classifies_spam_correct +classifies_nonspam_correct) / float(spam_count_test+non_spam_count_test))

        classifier_accuracy_freq = float(
            float(classifies_spam_correct_freq + classifies_nonspam_correct_freq) / float(spam_count_test_freq + non_spam_count_test_freq))

        print 'Accuracy using bag of words'
        print classifier_accuracy_bag_words*100



        print 'Accuracy using word frequency'
        print classifier_accuracy_freq * 100

        print 'Confusion Matrix using bag of words'
        print 'Spam   Nonspam'
        print str(classifies_spam_correct)+' '+str(classifies_nonspam_correct)
        print str(spam_predicted_nonspam)+' '+str(nonspam_predicted_spam)

        print 'Confusion Matrix using frequency of words'
        print 'Spam   Nonspam'
        print str(classifies_spam_correct) +' ' + str(classifies_nonspam_correct)
        print str(spam_predicted_nonspam_freq) + ' ' + str(nonspam_predicted_spam_freq)

    def freq_email_classification(self):
        self.all_word_vector = []
        self.classified_labels=[0]* len(self.labels)
        for email in self.emails:
            temp_list = {}
            for w in email:
                if w in temp_list:
                    temp_list[w] = temp_list[w]+1
                else:
                    temp_list[w] =  1

            non_spam_word,spam_val_word=0,0
            for k,v in temp_list.iteritems():
                if k in self.nonspam_freq_word :
                    non_spam_word += v * self.nonspam_freq_word[k]
                if k in self.spam_freq_word:
                    spam_val_word +=  v * self.spam_freq_word[k]
            if non_spam_word < spam_val_word:
                self.classified_labels[self.emails.index(email)]=0
            else:
                self.classified_labels[self.emails.index(email)]=1


def get_dtree(model_file):
    f = open(model_file, "r")
    dt_tree = pickle.load(f)
    f.close
    return dt_tree
def dt_caller_tester(email_path,model_file):

    naive=Naivebayestest(email_path,model_file)
    emails,labels=naive.get_files()

    tree =get_dtree(model_file)
    classified_label=[]
    for idx,email in enumerate(emails):

        labels[idx].append(dt.classify(email,tree))
    print_dt_accuracy(labels)


def print_dt_accuracy(labels):
    classifies_spam_correct_freq, classifies_nonspam_correct_freq, spam_count_test_freq, non_spam_count_test_freq = 0, 0, 0, 0
    nonspam_predicted_spam_freq,spam_predicted_nonspam_freq=0,0
    for i, v in enumerate(labels):
        if int(labels[i][1]) == int(labels[i][0]) == 1:
            classifies_spam_correct_freq = classifies_spam_correct_freq + 1
        if int(labels[i][0]) == 1:
            spam_count_test_freq = spam_count_test_freq + 1

        if int(labels[i][1]) == int(labels[i][0]) == 0:
            classifies_nonspam_correct_freq = classifies_nonspam_correct_freq + 1
        if int(labels[i][0]) == 0:
            non_spam_count_test_freq = non_spam_count_test_freq + 1

        if int(labels[i][1]) == 1 and int(labels[i][0]) == 0:
            nonspam_predicted_spam_freq = nonspam_predicted_spam_freq + 1
        if int(labels[i][1]) == 0 and int(labels[i][0]) == 1:
            spam_predicted_nonspam_freq = spam_predicted_nonspam_freq + 1

    classifier_accuracy_freq = float(
        float(classifies_spam_correct_freq + classifies_nonspam_correct_freq) / float(
            spam_count_test_freq + non_spam_count_test_freq))

    print 'Descion Tree Accuracy'
    print classifier_accuracy_freq*100

    print 'Confusion Matrix using frequency of words'
    print 'Spam   Nonspam'
    print str(classifies_spam_correct_freq) + ' ' + str(classifies_nonspam_correct_freq)
    print str(spam_predicted_nonspam_freq) + ' ' + str(nonspam_predicted_spam_freq)
