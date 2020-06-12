"""
This source com from Y.LAB

[MNIST로 알아보는 비지도 학습 - 클러스터링과 차원축소의 적용]
https://yamalab.tistory.com/118
"""
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# configure
encoding_dim = 32
input_img = Input(shape=(784,))

# layers
encoded = Dense(256, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Models
autoencoder = Model(input_img, decoded) # autoencoder

encoder = Model(input_img, encoded) # encoder

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test))


# encoding result
encoded_imgs = encoder.predict(x_train)

import os
currdir = os.path.dirname(os.path.realpath(__file__))
np.save(os.path.join(currdir,'../mnist_encoded/train'),x_train)
np.save(os.path.join(currdir,'../mnist_encoded/encoded_train_deep'),encoded_imgs)
