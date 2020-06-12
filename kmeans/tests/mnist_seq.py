"""
/***********************************************/
/*                                             */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*        DO  NOT  EDIT  THIS  FILE            */
/*                                             */
/***********************************************/
"""
import os
import sys
import numpy as np

currDir = os.path.dirname(os.path.abspath(__file__))
mnistPath = os.path.join(currDir, '../mnist/mnist_encoded/encoded_train_ae.npy')
mnist_seq = os.path.join(currDir,'mnist_seq.txt')

if not os.path.isfile(mnistPath):
    print("No such MNIST file!!! please check :", mnistPath)
    sys.exit(-1)

mnist = np.load(mnistPath)
print(mnist)

