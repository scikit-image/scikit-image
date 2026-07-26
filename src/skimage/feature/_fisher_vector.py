"""

fisher_vector.py - Implementation of the Fisher vector encoding algorithm

This module contains the source code for Fisher vector computation. The
computation is separated into two distinct steps, which are called separately
by the user, namely:

learn_gmm: Used to estimate the GMM for all vectors/descriptors computed for
           all examples in the dataset (e.g. estimated using all the SIFT
           vectors computed for all images in the dataset, or at least a subset
           of this).

fisher_vector: Used to compute the Fisher vector representation for a
               single set of descriptors/vector (e.g. the SIFT
               descriptors for a single image in your dataset, or
               perhaps a test image).

Reference: Perronnin, F. and Dance, C. Fisher kernels on Visual Vocabularies
           for Image Categorization, IEEE Conference on Computer Vision and
           Pattern Recognition, 2007

Origin Author: Dan Oneata (Author of the original implementation for the Fisher
vector computation using scikit-learn and NumPy. Subsequently ported to
scikit-image (here) by other authors.)

"""

from _skimage2.feature._fisher_vector import (
    DescriptorException as DescriptorException,
    FisherVectorException as FisherVectorException,
    fisher_vector as fisher_vector,
    learn_gmm as learn_gmm,
)  # noqa: F401

__all__ = [
    'DescriptorException',
    'FisherVectorException',
    'fisher_vector',
    'learn_gmm',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
