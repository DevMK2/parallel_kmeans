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
mnistPath = os.path.join(currDir, '../../mnist/mnist_encoded/encoded_train_ae.npy')
mnist_seq = os.path.join(currDir,'mnist_seq.txt')

if os.path.isfile(mnist_seq):
    sys.exit(0)

if not os.path.isfile(mnistPath):
    print("No such MNIST file!!! please check :", mnistPath)
    sys.exit(-1)

mnist = np.load(mnistPath)

dataSize = len(mnist)
featSize = len(mnist[0])

f = open(mnist_seq, 'w')

print("Making mnist_seq for test, please wait ...")
for i in range(dataSize):
    print("Making mnist_seq for test...", str(i),"/",str(dataSize))
    for j in range(featSize):
        f.write(str(mnist[i][j])+'\n')

f.close()
