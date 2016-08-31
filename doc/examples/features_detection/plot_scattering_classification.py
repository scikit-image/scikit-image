"""
===========================================
Scattering features for supervised learning
===========================================

This example shows how to use the scattering features for image classification.
They have shown to outperform any other 'non-learned'
image representations for image classification tasks [2]_.
Structurally, they are very similar to deep learning representations, with fixed filters, which are Wavelet filters.
Since the scattering transform is based on the Discrete Wavelet transform (DWT), its stable to deformations.
It is also invariant to small translations. For more details on its mathematical properties and exact definition,
 see [1]_ and [2]_.

Here, we show how to use the scattering vectors obtained from MNIST and CIFAR10 databases for classification. These are
both very challenging databases widely used in research to compare the quality of image representations and
classification methods.

# Image Classification:

The first task in supervised image classification  is to collect a set of images tagged with a class. In case of the MNIST
database we have pictures of digits, and the class is the corresponding number. The most common practice in image
classification is to compute, for every image its feature, which in our case is the scattering vector. Once we have the complete
database, we 'train' a classifier, meaning, we find the best parameters that allows to correctly classify the images.
A widely used classifier is Support Vector Machine (SVM), which searches the high dimensional plane
that allows to better separate the different classes. Here we show how reproduce the state of the art results for non-learned features
obtained with the scattering transform, together with SVM (with Gaussian kernels) in the following papers:

-*MNIST database*: The results were first presented in [1]_ Table 4.
-*CIFAR10* database: This is a very challenging database with 10 classes and 32x32 images.


..[1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.IEEE TPAMI, 2012.
..[2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering for Object Classification'. CVPR 2015
"""

import time as time

from keras.datasets import mnist
import numpy as np
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
from skimage.features.scattering import scattering


def load_images_mnist(px=32):

    #load database
    (X_train_sm, y_train), (X_test_sm, y_test) = mnist.load_data()
    num_images_ta = X_train_sm.shape[0]
    num_images_te = X_test_sm.shape[0]

    #set size to a power of 2 (32) with a zero pad border
    X_train = np.zeros((num_images_ta, px, px))
    X_train[:, 3:31, 3:31] = X_train_sm

    X_test = np.zeros((num_images_te, px, px))
    X_test[:, 3:31, 3:31] = X_test_sm

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, y_train, X_test, y_test


def load_scattering(X_train, X_test, px=32, J=3, L=8, m=2,sigma_phi=0.6957, sigma_xi=0.8506):
    i = -1
    t_images = time.time()

    print(X_train.shape[0], ' images loaded in ', time.time() - t_images, ' secs')

    #Create filters
    Filters,lw = multiresolution_filter_bank_morlet2d(px, J=J, L=L, sigma_phi=sigma_phi, sigma_xi=sigma_xi)

    ### Generate Training set
    print('Compute ', X_train.shape[0], ' scatterings:')
    t_scats = time.time()

    scatterings_train, u, scat_tree = scattering(X_train, Filters, m=m)

    print(scatterings_train.shape[0], ' scat. features computed in ', time.time() - t_scats, ' secs.')

    ### Generate Testing set
    print('Testing set:')

    t_scats = time.time()
    scatterings_test, u, scat_tree = scattering(X_test, Filters, m=m)

    print(scatterings_test.shape[0], ' scat. features computed in ', time.time() - t_scats, ' secs.')

    return scatterings_train, scatterings_test


def gethomogeneus_datast(X, y, d, n):
    num_per_class = np.int(n / d)
    ytrain = np.reshape(y, (y.shape[0],))

    X_out = []
    y_out = []
    for i_d in np.arange(d):
        indx = np.where(ytrain.ravel() == i_d)

        X_out.append(X[indx[0][0:num_per_class], :])
        y_out.append(ytrain[indx[0][0:num_per_class],])

    X_out = np.concatenate(X_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    return X_out, y_out

## Classification
# Get dataset
px=32 #number of pixels

im_train, ytrain, im_test, ytest = load_images_mnist(px=px)
Xtrain,Xtest = load_scattering(im_train, im_test, px=px, J=3, L=6, m=2, sigma_phi=0.6957, sigma_xi=0.8506)

#colapse spatial components into a vector
Xtrain_1d = Xtrain.reshape((len(Xtrain),-1))
Xtest_1d = Xtest.reshape((len(Xtest),-1))

num_samples,num_features = Xtrain_1d.shape

n = 10000
Xa,ya=gethomogeneus_datast(Xtrain_1d,ytrain,10,n)

# Classification using SVM with RBF Gaussian kernels
bestC = 4.3
bestgamma =10**0.1

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer

num_items,num_features = Xtrain_1d.shape

ns = [300, 1000, 2000, 5000, 10000,20000, 40000, 60000]
score_gaussian = np.zeros((len(ns),1))
print('% Error:')
for i,n in enumerate(ns):
    gs_gaussian = SVC(kernel='rbf',C=bestC,gamma=bestgamma)
    pip_gaussian = make_pipeline(MinMaxScaler((-1,1)),Normalizer(),gs_gaussian)

    Xa,ya=gethomogeneus_datast(Xtrain_1d,ytrain,10,n)
    pip_gaussian.fit(Xa,ya)
    out=pip_gaussian.predict(Xtest_1d)
    score_gaussian[i] = 1.0-accuracy_score(ytest, out)
    print('num images for training='+str(n)+' err:'+ score_gaussian[i])



############################ CIFAR10 classification

from skimage.color.colorconv import rgb2yuv
from keras.datasets import cifar10

def DB_rgb2yuv(X):
    num_samples,c,px,px = X.shape
    Xta = X.transpose((3,2,0,1))/255
    Iyuv = rgb2yuv(Xta).transpose((2,3,1,0)).copy()

    #stack the color channels as 3 images
    Iyuv.shape = (num_samples*3,px,px)
    return Iyuv


def load_images_cifar():
    # the data, shuffled and split between train and test sets
    (X_train_sm, y_train), (X_test_sm, y_test) = cifar10.load_data()

    #need to change to YUV
    X_train = DB_rgb2yuv(X_train_sm.astype('float32'))
    X_test = DB_rgb2yuv(X_test_sm.astype('float32'))

    return X_train, y_train, X_test, y_test


im_train, ytrain, im_test, ytest = load_images_cifar()
Xtrain, Xtest = load_scattering(im_train, im_test, px=px, J=3, L=8, m=1, sigma_phi=0.8, sigma_xi=0.8)

epsilon = 1e-6
Xtrain = np.log(np.abs(Xtrain)+epsilon)
Xtest = np.log(np.abs(Xtest)+epsilon)

# putting color channels together
num_files, scat_coefs, spatial, spatial = Xtrain.shape
Xtrain.shape = (num_files / 3, 3 * scat_coefs, spatial, spatial)
#putting color channels together
num_files,scat_coefs,spatial,spatial = Xtest.shape
Xtest.shape = (num_files/3,3*scat_coefs,spatial,spatial)

#collapse spatial coefficients
Xtrain_1d = Xtrain.reshape((len(Xtrain),-1))
Xtest_1d = Xtest.reshape((len(Xtest),-1))

########### Train for CIFAR10
from sklearn.preprocessing import StandardScaler

n = 40000 #number of samples for training
Xa,ya=gethomogeneus_datast(Xtrain_1d, ytrain, 10, n)

bestC = 4.4
bestgamma = 10**-3.55
print('C=',bestC)
print('gamma=',bestgamma)
start_t = time.time()
gs_gaussian = SVC(kernel='rbf',C=bestC,gamma=bestgamma)
pip_gaussian = make_pipeline(MinMaxScaler((-1,1)), StandardScaler(), gs_gaussian)

pip_gaussian.fit(Xa, ya)
out=pip_gaussian.predict(Xtest_1d)
score = accuracy_score(ytest, out) # should be 73.6% success fr m=1
print('% correctly classified:', score)
