# Steps to run the program


Clone repository

Upload following files on server (data volume is large):
    
    adaboosting.py
    
    knn.py
    
    neuralnetwork.py
    
    orient.py
    
    test-data.txt
    
    train-data.txt
    

Run the program with with following command for adboost classfier
  
    
    python orient.py train-data.txt test-data.txt adaboost stump_count
    
    where stump_count is decision stump (integer)

Run the program with with following command for knn classfier
  
    
    python orient.py train-data.txt test-data.txt nearest
  
Run the program with with following command for Neural Network classfier
  
    
    python orient.py train-data.txt test-data.txt  nnet hidden_count
    
    hidden_count is the number of neurons in the hidden layer
  
End
    
    
    
