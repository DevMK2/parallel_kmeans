from sklearn.cluster import KMeans
import numpy as np
import visuallize as viz
import os
currdir = os.path.dirname(os.path.realpath(__file__))
x_train = np.load(os.path.join(currdir, '../mnist_encoded/train.npy'))
encoded_img = np.load(os.path.join(currdir, '../mnist_encoded/encoded_train_ae.npy'))

kmeans = KMeans(n_clusters=10)
kmeans.fit(encoded_img)
viz.img(x_train, kmeans.labels_)
