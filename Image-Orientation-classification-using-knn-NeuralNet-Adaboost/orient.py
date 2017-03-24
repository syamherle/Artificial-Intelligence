import sys
import time
from neuralnetwork import *
from adaboosting import *
from knn import *

def main(argv):

    if argv[2].lower() == 'nearest':
        apply_knn(argv[0], argv[1])

    elif argv[2].lower() == 'nnet':
        Neuralnetwork(argv[0], argv[1],argv[3])

    elif argv[2].lower() == 'adaboost':
        Adaboosting(argv[0], argv[1],argv[3])

    elif argv[2].lower() == 'best':
        Neuralnetwork(argv[0], argv[1], 200)
    else:
        print'Please give valid input'


if __name__ == '__main__':
    t0 = time.time()
    main(sys.argv[1:])
    t1 = time.time()
    print 'Total time taken : ', float(t1 - t0) / 60.00, ' minutes'