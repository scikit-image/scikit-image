"""
===========================================
Scattering features for supervised learning
===========================================

This example shows how to use the scattering features for image classification.
Scattering features have shown to outperform any other 'non-learned'
image representations for image classification tasks [2]_.
Structurally, they are very similar to deep learning representations, with fixed filters, in this implementation,
to Morlet filters.

Since the scattering transform is based on the Discrete Wavelet transform (DWT), its stable to deformations.
It is also invariant to small translations. For more details on its mathematical properties and exact definition,
see [1]_ and [2]_.

Here, we show how to use the scattering vectors obtained from MNIST and CIFAR10 databases for classification. These two
databases are widely used in research to compare the quality of image representations and
classification methods.

**Supervised Learning: Image Classification**

The first task in supervised image classification is to collect a set of images tagged with a class. In case of the MNIST
database we have pictures of digits, and the class is the corresponding number. Then, for every image
we compute a feature vector, which in our case is the scattering vector. Once we have the complete
database, we 'train' a classifier, meaning, we find the best parameters that allows to correctly classify the images.

A widely used classifier is Support Vector Machine (SVM), which searches the high dimensional plane
that allows to better separate the different classes [3]_.
Here, we show how to reproduce the results obtained with the scattering transform, using with SVM (with Gaussian kernels)
first presented in the following papers:

- *MNIST database*: The results were first presented in [1]_ Table 4. Using this implementation, we were able to obtain the following error results:
            +-------------------------+---------+
            |num images training set  | % error |
            +-------------------------+---------+
            |                    300  |  5.92   |
            +-------------------------+---------+
            |                  1000   |  2.7    |
            +-------------------------+---------+
            |                  2000   |  2      |
            +-------------------------+---------+
            |                  5000   |  1.32   |
            +-------------------------+---------+
            |                 10000   |  0.96   |
            +-------------------------+---------+
            |                 20000   |  0.82   |
            +-------------------------+---------+
            |                 40000   |  0.56   |
            +-------------------------+---------+
            |                 60000   |  0.51   |
            +-------------------------+---------+

- *CIFAR10* database: This is a very challenging database with 10 classes and 32x32 color images. The code presented here can reproduce the results obtained in Table 1 (Trans., order 1) of [2]_, more specifically it can obtain 73.6% success for m=1 and 40000 images in the training set.

**Warning: For integration reasons, the code uses very few images in the datasets. To obtain full performance, you need
to change the variable num_images to 10000 at least**

.. [1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.IEEE TPAMI, 2012.
.. [2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering for Object Classification'. CVPR 2015
.. [3] https://en.wikipedia.org/wiki/Support_vector_machine
"""


#######################
# MNIST Classification

#Modules, MNIST database and functions for scattering feature generation
import time as time
from keras.datasets import mnist
import numpy as np
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
from skimage.feature.scattering import scattering
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, Normalizer,StandardScaler

def load_images_mnist(px=32, num_images=500000):
    (X_train_sm, y_train), (X_test_sm, y_test) = mnist.load_data()
    X_train_sm = X_train_sm[0:num_images, :, :]
    y_train = y_train[0:num_images]
    X_test_sm = X_test_sm[0:num_images, :, :]
    y_test = y_train[0:num_images]

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
    #Create filters
    Filters,lw = multiresolution_filter_bank_morlet2d(px, J=J, L=L, sigma_phi=sigma_phi, sigma_xi=sigma_xi)

    ### Generate Training set
    scatterings_train, u, scat_tree = scattering(X_train, Filters, m=m)

    ### Generate Testing set
    t_scats = time.time()
    scatterings_test, u, scat_tree = scattering(X_test, Filters, m=m)
    time_per_image = (time.time() - t_scats) / scatterings_test.shape[0]
    print('Scattering coefficients computed as ', str(time_per_image), ' secs. per image')

    return scatterings_train, scatterings_test

def gethomogeneus_datast(X, y, d, n):
    num_per_class = np.int(np.amin([n, X.shape[0]]) / d)
    ytrain = np.reshape(y, (y.shape[0],))

    X_out = []
    y_out = []
    for i_d in np.arange(d):
        indx = np.where(ytrain.ravel() == i_d)

        X_out.append(X[indx[0][0:num_per_class], :])
        y_out.append(ytrain[indx[0][0:num_per_class], ])

    X_out = np.concatenate(X_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    return X_out, y_out


#######################
# Classification of MNIST Database following the results presented in Bruna et al. paper.


# Get dataset
num_classes = 10
px=32  # number of pixels
num_images = 50  # for full data set set to 50000
n = np.amin([10000, num_images])

print('Warning: For exact error performance, we need at least num_images=10000.')
im_train, ytrain, im_test, ytest = load_images_mnist(px=px, num_images=num_images)
Xtrain, Xtest = load_scattering(im_train, im_test, px=px, J=3, L=6, m=2, sigma_phi=0.6957, sigma_xi=0.8506)

Xtrain_1d = Xtrain.reshape((len(Xtrain), -1)) #colapse spatial components
Xtest_1d = Xtest.reshape((len(Xtest), -1))


# Classification using SVM with RBF Gaussian kernels
bestC = 4.3 # parameters obtained by crossvalidation with 10000 images
bestgamma = 10**0.1

ns = [300, 1000, 2000, 5000, 10000, 20000, 40000, 60000]
score_gaussian = np.zeros((len(ns), 1))

print('% Error with ')
for i,n in enumerate(ns):
    # resample the dataset to have a uniform distribution on the classes
    Xa, ya = gethomogeneus_datast(Xtrain_1d, ytrain, num_classes, n)
    #define pipeline for SVM classif.
    gs_gaussian = SVC(kernel='rbf', C=bestC, gamma=bestgamma)
    pip_gaussian = make_pipeline(MinMaxScaler((-1, 1)), Normalizer(), gs_gaussian)
    pip_gaussian.fit(Xa, ya)
    out = pip_gaussian.predict(Xtest_1d)
    score_gaussian[i] = 1.0-accuracy_score(ytest, out)
    #print('num training images:' + str(n) + ' err:' + str(score_gaussian[i]*100) + '%')



############################
#  CIFAR10 classification

################
# Load CIFAR10
from skimage.color.colorconv import rgb2yuv
from keras.datasets import cifar10

def DB_rgb2yuv(X):
    num_samples, c, px, px = X.shape
    Xta = X.transpose((3, 2, 0, 1)).astype('float32')/255
    Iyuv = rgb2yuv(Xta).transpose((2, 3, 1, 0)).copy()

    # stack the color channels as 3 images
    Iyuv.shape = (num_samples*3, px, px)
    return Iyuv


def load_images_cifar(num_images):
    # the data, shuffled and split between train and test sets
    (X_train_sm, y_train), (X_test_sm, y_test) = cifar10.load_data()

    # need to change to YUV
    X_train = DB_rgb2yuv(X_train_sm[0:num_images, :, :, :])
    X_test = DB_rgb2yuv(X_test_sm[0:num_images, :, :, :])

    return X_train, y_train[0:num_images], X_test,  y_test[0:num_images]





############################
# Load data, training and testing
num_images = 50 # For accurate error result we need to train with 10000 images (not 10!)
im_train, ytrain, im_test, ytest = load_images_cifar(num_images)
#We are computing just the first order scattering transform
Xtrain, Xtest = load_scattering(im_train, im_test, px=px, J=3, L=8, m=1, sigma_phi=0.8, sigma_xi=0.8)
# note that it works better in the log domain!
epsilon = 1e-6
Xtrain = np.log(np.abs(Xtrain)+epsilon)
Xtest = np.log(np.abs(Xtest)+epsilon)

# putting color channels together
num_files, scat_coefs, spatial, spatial = Xtrain.shape
Xtrain.shape = (num_files / 3, 3 * scat_coefs, spatial, spatial)
Xtrain_1d = Xtrain.reshape((len(Xtrain), -1))
#Testing
num_files, scat_coefs, spatial, spatial = Xtest.shape
Xtest.shape = (num_files/3, 3*scat_coefs, spatial, spatial)
Xtest_1d = Xtest.reshape((len(Xtest), -1))

# Train model for CIFAR10
num_classes = 10
n = 10000  # number of samples for training
Xa, ya = gethomogeneus_datast(Xtrain_1d, ytrain, num_classes, n)

#parameters obtain with cross-validation
bestC = 4.4
bestgamma = 10**-3.55
#Define SVM pipeline
gs_gaussian = SVC(kernel='rbf',C=bestC,gamma=bestgamma)
pip_gaussian = make_pipeline(MinMaxScaler((-1,1)), StandardScaler(), gs_gaussian)

pip_gaussian.fit(Xa, ya)
out = pip_gaussian.predict(Xtest_1d)
accuracy = accuracy_score(ytest, out)
#print('Training set: ' + str(num_images) + ' % correctly classified:', str(accuracy*100)+'%')
#print('should be should be 73.6% success for m=1 and 40000 images')
