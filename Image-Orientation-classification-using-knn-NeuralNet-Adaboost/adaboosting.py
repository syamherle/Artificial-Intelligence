

# In this program we have used adboost classifeir with n decsion stump. There are four classifiers
# and for each stump we are randomly selecting the 100 pair of feature and select one from
# them and assign to the specfic classifier. At the end classifier with the
# maximum value of alpha taken as the label for the test data
import numpy as np
import random
import operator
import time
import itertools



# Class Adaboost is created
class Adaboosting:
    # Initialization function of the class handles the data and global variable and it act as a constructor
    def __init__(self,train_filename,test_filename,stump_size):
        self.test_label=[]
        self.rand_point=[]
        self.stump_size = int(stump_size)
        self.train_data = np.array([line.strip().split() for line in open(train_filename)])
        self.train_dimension= self.train_data.shape[1]
        self.train_len = self.train_data.shape[0]
        self.test_data=np.array([line.strip().split() for line in open(test_filename)])
        self.test_idx = np.zeros(len(self.test_data), dtype='uint8')
        self.initialize_test_lbl()
        self.weight_col= np.ones((self.train_len,1))
        self.weight_col /= self.train_len
        self.num_classifier = 4
        self.classifier_list = []

        self.train_adaboost()
        self.evaluate()
    # We are creating a seperate list for the test labels
    def initialize_test_lbl(self):
        for i in range(len(self.test_data)):
            self.test_idx[i] = 0 if self.test_data[i,1] == '0' else 1 if self.test_data[i,1] == '90' else 2 if self.test_data[i,1] == '180' else 3
            self.test_label.append(self.test_data[i,0])
    #  Training of ada boost classifier starts here
    # Stump for each classifeier is created here and the alpha value is calculated over here
    def train_adaboost(self):
        print 'Training.....'
        for classifier in range(self.num_classifier):
            # self.train_data[:,-1] = 1.0/self.train_len

            self.weight_col = np.ones((self.train_len, 1))
            self.weight_col /= self.train_len


            descion_stup_alpha = []
            decision_stump_split = []
            decison_stump_accuracy = []
            for j in range(self.stump_size):
                self.rand_point=self.get_rand_feature()

                best_split,error,stump_lab=self.get_most_accurate_feature(classifier,j)
                # error =1.0-accuracy
                if classifier ==0:
                    classified_array = [0 if self.train_data[t][1] =='0' and stump_lab[t] == 1 or self.train_data[t][1] != '0' and stump_lab[t] == -1 else -1 for t in range(len(self.train_data))]
                elif classifier ==1:
                    classified_array = [
                        0 if self.train_data[t][1] == '90' and stump_lab[t] == 1 or self.train_data[t][1] != '90' and stump_lab[t] == -1 else -1 for t in range(len(self.train_data))]
                elif classifier == 2 :
                    classified_array = [
                        0 if self.train_data[t][1] == '180' and stump_lab[t] == 1 or self.train_data[t][1] != '180' and stump_lab[t] == -1 else -1 for t in range(len(self.train_data))]
                else:
                    classified_array = [
                        0 if self.train_data[t][1] == '270' and stump_lab[t] == 1 or self.train_data[t][1] != '270' and stump_lab[t] == -1 else -1 for t in range(len(self.train_data))]

                # Calculation of the alpha value and weight update are done here
                # Misclassified datapoint are given higher weight so that next stump will be biased towards it
                alpha= 0.5*np.log((1-error+1)/(error+1.0))

                w=np.zeros(self.train_len)

                for i in range(self.train_len):
                    if classified_array[i] == 0: w[i] = self.weight_col[i] * np.exp(-alpha)
                    else:  w[i] = self.weight_col[i] * np.exp(+alpha)

                self.weight_col = w/w.sum()

                descion_stup_alpha.append(alpha)

                decision_stump_split.append(best_split)

            decision_stump=[decision_stump_split,descion_stup_alpha]


            self.classifier_list.append(decision_stump)



    # 100 random pairs are generated which acts is used for the decision stump
    def get_rand_feature(self):
        # print random.sample(list(itertools.combinations(range(2,194),2)),100)
        return random.sample(list(itertools.combinations(range(2,194),2)),100)

    # Most accurate pair is selected and returned to train function
    def get_most_accurate_feature(self,classifier,j):
        accuracy_dict ={}
        for j in range(len(self.rand_point)):
            c1,c2=self.rand_point[j][0],self.rand_point[j][1]
            stump_lab = self.get_split_leaf(c1, c2)
            accuracy_dict[c1, c2]=self.get_accuracy_of_split(classifier,stump_lab)
        sorted_dict = sorted(accuracy_dict.items(), key=operator.itemgetter(1))
        return sorted_dict[0][0],sorted_dict[0][1],stump_lab
    # Attributes are split here and the data point is classified +1 if it belongs to the classifier class or -1
    # the assigning is done based on feature 1> feature 2
    def get_split_leaf(self,col_one,col_two):
        stump_lab=[]

        for i in range(self.train_len):
            if int(self.train_data[i,col_one]) > int(self.train_data[i,col_two]):
                stump_lab.append(1)
            else:
                stump_lab.append(-1)
        return stump_lab

    # Error of the classification of the split are calculated here and the error is nothing but respective weight of the data points.
    def get_accuracy_of_split(self,classifier,stump_lab):
        weight =0.0
        count =0.0
        if classifier == 0:
            for i in range(self.train_len):
                if self.train_data[i,1] == '0' and stump_lab[i] == -1 or self.train_data[i,1] != '0' and stump_lab[i] == 1:
                    weight = weight+self.weight_col[i]
                    count = count+1.0
        if classifier == 1:
            for i in range(self.train_len):
                if self.train_data[i,1] == '90' and stump_lab[i] == -1 or self.train_data[i,1] != '90' and stump_lab[i] == 1:
                    weight = weight+self.weight_col[i]
                    count = count+1.0
        if classifier == 2:
            for i in range(self.train_len):
                if self.train_data[i,1] == '180' and stump_lab[i] == -1 or self.train_data[i,1] != '180' and stump_lab[i] == 1:
                    weight = weight+self.weight_col[i]
                    count = count+1.0
        if classifier == 3:
            for i in range(self.train_len):
                if self.train_data[i,1] == '270' and stump_lab[i] == -1 or self.train_data[i,1] != '270' and stump_lab[i] == 1:
                    weight = weight+self.weight_col[i]
                    count = count+1.0
        return weight

    # Evauation of the test data is done here
    def evaluate(self):
        print 'Training completed.'
        print 'Testing.......'
        final_val=[]
        for i in range(len(self.test_data)):
            final_val.append(self.evaluate_test_data(i))

        print 'Testing Completed'

        print 'The testing accuracy is : ',np.mean(final_val == self.test_idx) * 100.0,'%'
        labels=[0,1,2,3]
        length = len(labels)
        confusion_mat=[[0]* length for x in range(length)]
        length = len(final_val)
        for i in range(length):
            r = labels.index(final_val[i])
            c= labels.index(self.test_idx[i])
            confusion_mat[r][c] +=1

        print 'The Confusion matrix will be '
        for item in confusion_mat:
            x=map(str,item)
            for j in range(len(x)):
                print x[j],""*(5-len(x[j])),
            print
        self.write_output(final_val)


    # Voting of the best classifer for the test data is done here
    def evaluate_test_data(self,i):
        val_classifier=[]
        for j in range(len(self.classifier_list)):
            classify_test_stump=[]
            classifier_val=0.0
            for k in range(len(self.classifier_list[j][0])):
                c1,c2 = self.classifier_list[j][0][k][0],self.classifier_list[j][0][k][1]
                classify_test_stump.append(self.split_test(c1,c2,i))
            for l in range(len(classify_test_stump)):
                if classify_test_stump[l] == 1:
                    classifier_val += classify_test_stump[l] * self.classifier_list[j][1][l]
            classifier_val /= np.sum(self.classifier_list[j][1])
            val_classifier.append(classifier_val)
        return np.argmax(val_classifier)

    # Split value -1/+1 for class is returned here
    def split_test(self,col_one,col_two,i):

        if self.test_data[i, col_one].astype(int) > self.test_data[i, col_two].astype(int):
            return 1
        else:
            return -1
    #Our model prediction is wriiten to adaboost_output.txts
    def write_output(self,predicted):
        for i in range(len(predicted)):
            if predicted[i] == 1: predicted[i] = 90
            if predicted[i] == 2: predicted[i] = 180
            if predicted[i] == 3: predicted[i] = 270
        with open('adaboost_output.txt', 'w') as proc_seqf:
            for a, am in zip(self.test_label, predicted):
                proc_seqf.write("{}\t{}\n".format(a, am))



