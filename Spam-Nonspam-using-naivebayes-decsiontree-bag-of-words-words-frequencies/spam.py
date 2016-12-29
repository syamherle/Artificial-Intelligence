######################################################################################################################################################
#The Auuracy of naive bayes using bag of words is 98.5%
#Th Accuracy of the naive bayes using frequency of words is 95.5%
#The accuracy of descion tree using bag of words is 67%
#The Naive bayes is better classifier than the descion tree given the dataset. As the data has many feature the decision tree comuputation time
# is more than the naive bayes and best split of feature is hard to find given there are many features
#
#
#For bag of words model a word vector was created of all the words across spam and nonspam email and each email was converted into vector
#The conditional probalaity and prioir probability are calculated and stored in the file name given along with run command
#
#For testing the model from the file is used to calculate the probability the mail falls under spam or nonspam
#
#For Naive bayes using frequency of words , all the words and thier frequency is calculated in each mail (spam and non spam) and used as conditional probality and
#prioir as the total number of spam mails and non spam mails and the model is stored in the file given along with the run command and used for testing
#
##########################################################################################################################################################
import sys
import time
from test import *
from train import *

def main(argv):
    if argv[1].lower() == 'bayes':
        if argv[0].lower() == 'train':
            naive = Naivebayestrain(argv[2].lower(),argv[3].lower())
            naive.get_files()
            naive.get_all_words()
            naive.get_bag_words()
            naive.trainClassifier()
            naive.write_model()

        if argv[0].lower() == 'test':
            naivetest=Naivebayestest(argv[2].lower(),argv[3].lower())
            naivetest.get_files()
            naivetest.get_model()
            naivetest.predict_spam_prob()
            naivetest.predict_nonspam_prob()
            naivetest.classify_document()
            naivetest.freq_email_classification()
            naivetest.find_accuracy()

    if argv[1].lower() == 'dt':
        if argv[0].lower() == 'train':

            dt_caller(argv[2].lower(),argv[3].lower())

        if argv[0].lower() == 'test':
            dt_caller_tester(argv[2].lower(),argv[3].lower())



if __name__ == '__main__':

    main(sys.argv[1:])


