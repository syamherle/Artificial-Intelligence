import time

t0 = time.time()
import pickle
import numpy as np
import os
import math
from stopwords import *
import dtree as dt
import operator

class Naivebayestrain:
    def __init__(self,email_path,model_file):
        self.email_path = email_path
        self.model_file = model_file
        self.email_files = [f for f in os.listdir(email_path)]
        self.emails=[]
        self.labels=[]
        self.all_words={}
        self.nonspam_freq_word={}
        self.spam_freq_word = {}

    def get_files(self):
        
        for e in self.email_files:
            for e_file in os.listdir(self.email_path+e):
                with open(self.email_path + e +'/'+ e_file, 'r') as email:
                    text = email.read().lower().split()
                    for w in text:
                        if w in list_stop:
                            text.remove(w)
                    for w in text:
                        if self.email_files.index(e) == 1:
                            if w not in self.spam_freq_word:
                                self.spam_freq_word[w] = 1
                            else:
                                self.spam_freq_word[w] = self.spam_freq_word[w] + 1
                        else:
                            if w not in self.nonspam_freq_word:
                                self.nonspam_freq_word[w] = 1
                            else:
                                self.nonspam_freq_word[w] = self.nonspam_freq_word[w] + 1
                    self.labels.append(self.email_files.index(e))
                    self.emails.append(text)
        return self.emails,self.labels,self.nonspam_freq_word,self.spam_freq_word

    def get_all_words(self):
        for email in self.emails:
            for w in email:
                if w not in self.all_words:
                    self.all_words[w]=len(self.all_words)



    def get_bag_words(self):
        self.all_word_vector = []
        for email in self.emails:
            temp_list = [0]*len(self.all_words)
            for w in email:
                if w in self.all_words:
                    temp_list[self.all_words[w]] = 1
            temp_list.append(self.labels[self.emails.index(email)])
            self.all_word_vector.append(temp_list)
        self.all_word_vector = np.array(self.all_word_vector)
        return self.all_word_vector



    def trainClassifier(self):
        self.spam_msg_count = self.labels.count(1)
        self.nonspam_msg_count = self.labels.count(0)
        self.nonspam_words = [ float(np.sum(self.all_word_vector[self.all_word_vector[:,-1] == 0,i])+1) / float(float(self.nonspam_msg_count) + 2) for i in range(len(self.all_word_vector.T)-1)]
        self.spam_words = [float(np.sum(self.all_word_vector[self.all_word_vector[:, -1] == 1, i])+1) / float(float(self.spam_msg_count) + 2) for i in range(len(self.all_word_vector.T) - 1)]


    def write_model(self):
        header = [self.spam_msg_count+self.nonspam_msg_count,self.nonspam_msg_count,self.spam_msg_count]
        nonspam_word_sum=sum(self.nonspam_freq_word.values())
        spam_word_sum = sum(self.spam_freq_word.values())
        for k,v in self.nonspam_freq_word.iteritems():
            self.nonspam_freq_word[k] = math.log(float(v)/float(nonspam_word_sum))
        for k,v in self.spam_freq_word.iteritems():
            self.spam_freq_word[k] = math.log(float(v)/float(spam_word_sum))
        f=open(self.model_file,"w")
        pickle.dump(self.all_words,f)
        pickle.dump(self.spam_words,f)
        pickle.dump(self.nonspam_words,f)
        pickle.dump(header, f)
        pickle.dump(self.nonspam_freq_word,f)
        pickle.dump(self.spam_freq_word,f)
        f.close()
        print 'The most frequent non spam words in naive bayes is'
        for w in sorted(self.nonspam_freq_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:20]:
            print(w[0])
        print 'The most frequent spam words in naive bayes is'
        for w in sorted(self.spam_freq_word.iteritems(), key=operator.itemgetter(1), reverse=True)[:20]:
            print(w[0])

def dt_caller(email_path,model_file):
    naive = Naivebayestrain(email_path,model_file)
    mails,labels,nonspam_dict,spam_dict=naive.get_files()
    for mail in mails:
        mail.append(labels[mails.index(mail)])
    # naive.get_all_words()
    # data=naive.get_bag_words()
    top_words=[]
    for w in sorted(nonspam_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:550]:
        top_words.append(w[0])
    for w in sorted(spam_dict.iteritems(), key=operator.itemgetter(1), reverse=True)[:550]:
        top_words.append(w[0])
    top_words = list(set(top_words))
    tree= dt.DecisionTreeClassifier(mails,top_words,model_file)
    print 'Tree of decision tree is'
    print tree