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
encoding_dim = 200
input_img = Input(shape=(784,))


# layers
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

# Models
autoencoder = Model(input_img, decoded) # autoencoder

encoder = Model(input_img, encoded) # encoder

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input)) # decoder

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def recall(y_true, y_pred):
    y_true_yn = K.round(K.clip(y_true, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))

    count_true_positive = K.sum(y_true_yn * y_pred_yn)
    count_true_positive_false_negative = K.sum(y_true_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall


# train autoencoder
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=[rmse, recall])
autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test))


# encoding result
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)

import os
currdir = os.path.dirname(os.path.realpath(__file__))
np.save(os.path.join(currdir,'../mnist_encoded/train'),x_train)
np.save(os.path.join(currdir,'../mnist_encoded/encoded_train_ae'),encoded_imgs)
