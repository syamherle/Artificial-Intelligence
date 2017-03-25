#Follow steps to run this model

Download and unzip data.zip to data 


Load all the files on server (unix- due to large volume of data)


run spam.py using following command

./spam mode technique dataset-directory model-file

where mode is either test or train, technique is either bayes or dt, dataset-directory is a directory containing 
two subdirectories named spam and notspam, each ﬁlled with documents of the corresponding type, 
and model-ﬁle is the ﬁlename of your trained model. In training mode, dataset-directory will be the training dataset 
and the program should write the trained model to model-ﬁle, and in testing mode dataset-directory
will be the testing dataset and your program should read its trained model from model-ﬁle
