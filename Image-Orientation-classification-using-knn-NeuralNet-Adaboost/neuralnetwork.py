##################################################################################################################################
# In this problem I have used neurral network using a softmax classifier which gives output as a probability, and
# I have also used regularization stwpsize to keep track of the loss of data . here in this problem a step-size 0.1 is used and
# a regularization step-size of 0.001 is used, but these step size are decremented if there is jump of loss value from previous
# Since this is single layer neural network we have used one hidden layer and the input neuron is user defined value
######################################################################################################################################

import numpy as np
import random
import math
import time
from random import randrange



class Neuralnetwork:
    def __init__(self,train_filename,test_filename,neuron):
        # self.data = np.array([line.strip().split() for line in open('../'+filename)])
        self.hidden_neuron = int(neuron)
        self.test_label = []
        self.test_data=[]
        input_data = [line.strip().split() for line in open(train_filename)]

        test_input_data = [line.strip().split() for line in open(test_filename)]

        self.train_data = np.zeros((len(input_data),len(input_data[0][2:])))
        self.train_idx = np.zeros(len(input_data), dtype='uint8')

        self.test_data = np.zeros((len(test_input_data), len(test_input_data[0][2:])))
        self.test_idx = np.zeros(len(test_input_data), dtype='uint8')

        # check for speed change if necessary
        self.analyse_input(input_data,'train')
        self.analyse_input(test_input_data,'test')
        # self.analyse_input(test_filename)


        self.input_neuron = len(self.train_data[0])
        # need to change this
        self.output_neuron = 4
        # Have to change this


        self.weight_one,self.weight_two = 0.01 * np.random.randn(self.input_neuron,self.hidden_neuron),0.01 * np.random.randn(self.hidden_neuron,self.output_neuron)

        self.bias_one,self.bias_two = np.zeros((1,self.hidden_neuron)),np.zeros((1,self.output_neuron))

        #give step size
        self.step_size = 1e-0
        #regularization strength
        self.reg_size = 1e-3
        # self.initialise_weights()
        self.apply_gradient()

        self.evaluate_net()
    # here the input value are preprocessed
    def analyse_input(self,input_data,option):
        if option == 'train':
            for i in range(len(input_data)):
                self.train_data[i]=map((lambda x: float(x)/255.0), input_data[i][2:])

                self.train_idx[i] = 0 if input_data[i][1] == '0' else 1 if input_data[i][1] == '90' else 2 if input_data[i][1] == '180' else 3
        else:
            for i in range(len(input_data)):
                self.test_data[i]=map((lambda x: float(x)/255.0), input_data[i][2:])
                self.test_idx[i] = 0 if input_data[i][1] == '0' else 1 if input_data[i][1] == '90' else 2 if input_data[i][1] == '180' else 3
                self.test_label.append(input_data[i][0])

    def initialise_weights(self):
        for i in range(self.hidden_neuron):
            for j in range(self.input_neuron):
                self.weight_one[i,j] = random.uniform(-100,100)
        for i in range(self.hidden_neuron):
            self.bias_one[i] = random.uniform(-0.1,.1)

        for i in range(self.output_neuron):
            for j in range(self.hidden_neuron):
                self.weight_two[i,j] = random.uniform(-100,100)
        for i in range(self.output_neuron):
            self.bias_two[i] = random.uniform(-.1,1)


    def feed_forward(self,data):
        hidden_layer = np.maximum(0, np.dot(data, self.weight_one) + self.bias_one)
        scores = np.dot(hidden_layer, self.weight_two) + self.bias_two

        return hidden_layer,scores

    # Here gradient decent is done, the error from the output network is backpropagated to the earliest hidden layer
    # her we have updated weight in traditional gradient decent manner
    #In our neural net there are 192 neurons in input layer and 4 output neurons in output layer.
    def apply_gradient(self):
        print 'Training.....'
        self.train_data -=np.mean(self.train_data)

        self.train_data /= np.std(self.train_data)

        n_example=len(self.train_data)
        previos_loss=0.0
        #Number of epochs are specified here
        for epochs in xrange(500):
            #data is feeded into the neural network
            hidden_layer,scores = self.feed_forward(self.train_data)
            #softmax function is applied here
            exp_scores = np.exp(scores)
            output_prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            loss=self.get_loss(output_prob,n_example)

            output_prob[range(n_example),self.train_idx] -= 1
            output_prob /= n_example
            #below are the steps used for backpropagatin the error from the output layer to early hidden layer
            delta_weight_two = np.dot(hidden_layer.T, output_prob)
            delta_bias_two = np.sum(output_prob, axis=0, keepdims=True)

            dhidden = np.dot(output_prob, self.weight_two.T)
            dhidden[hidden_layer <= 0] = 0

            delta_weight_one = np.dot(self.train_data.T, dhidden)
            delta_bias_one = np.sum(dhidden, axis=0, keepdims=True)

            delta_weight_two += self.reg_size * self.weight_two
            delta_weight_one += self.reg_size * self.weight_one
            #the weights are updated here
            self.weight_one += -self.step_size * delta_weight_one
            self.bias_one += -self.step_size * delta_bias_one
            self.weight_two += -self.step_size * delta_weight_two
            self.bias_two += -self.step_size * delta_bias_two
            if previos_loss < loss :
                self.step_size = self.step_size/10.0
                self.reg_size = self.reg_size/1000.0
            previos_loss =loss



    # function to track the loss of the data
    def get_loss(self,out_probs,n):
        corect_logprobs = - np.log(out_probs[range(n), self.train_idx])
        data_loss = np.sum(corect_logprobs) / n
        reg_loss = 0.5 * self.reg_size * np.sum(self.weight_one * self.weight_one) + 0.5 * self.reg_size * np.sum(
            self.weight_two * self.weight_two)
        loss = data_loss + reg_loss
        return loss

    #function to evaluate the test data using the neural net we have created
    def evaluate_net(self):
        print'Training completed'
        print'Testing.......'
        hidden_layer, scores = self.feed_forward(self.test_data)
        predicted_class = np.argmax(scores, axis=1)

        print 'The testing accuracy is :',np.mean(predicted_class == self.test_idx)*100.00
        labels = [0, 1, 2, 3]
        length = len(labels)
        confusion_mat = [[0] * length for x in range(length)]
        length = len(predicted_class)
        for i in range(length):
            r = labels.index(predicted_class[i])
            c = labels.index(self.test_idx[i])
            confusion_mat[r][c] += 1

        print 'The Confusion matrix will be '
        for item in confusion_mat:
            x = map(str, item)
            for j in range(len(x)):
                print x[j], "" * (5 - len(x[j])),
            print
        self.write_output(predicted_class)


    #output is written on an external file nnet_output.txt the output is the predication class for the image
    def write_output(self,predicted):
        for i in range(len(predicted)):
            if predicted[i] == 1: predicted[i] = 90
            if predicted[i] == 2: predicted[i] = 180
            if predicted[i] == 3: predicted[i] = 270
        with open('nnet_output.txt', 'w') as proc_seqf:
            for a, am in zip(self.test_label, predicted):
                proc_seqf.write("{}\t{}\n".format(a, am))

    # cost function to find the error value in the ouput
    def cost_function(self,actuals,expected):
        return actuals-expected

    def sigmoid(self,z):
        return np.array(1.0/(1.0+np.exp(-z)))
    def sigmoid_diff(self,z):
        return self.sigmoid(z)*self.sigmoid(1-z)



