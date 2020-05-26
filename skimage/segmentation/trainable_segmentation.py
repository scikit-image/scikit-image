from itertools import combinations_with_replacement
import itertools
import numpy as np
from skimage import filters, feature
from skimage import img_as_float32


def _singlescale_basic_features(img, sigma, intensity=True, edges=True,
                                texture=True):
    """Features for a single value of the Gaussian blurring parameter ``sigma``
    """
    features = []
    img_blur = filters.gaussian(img, sigma)
    if intensity:
        features.append(img_blur)
    if edges:
        features.append(filters.sobel(img_blur))
    if texture:
        H_elems = [
            np.gradient(np.gradient(img_blur)[ax0], axis=ax1)
            for ax0, ax1 in combinations_with_replacement(range(img.ndim), 2)
        ]
        eigvals = feature.hessian_matrix_eigvals(H_elems)
        for eigval_mat in eigvals:
            features.append(eigval_mat)
    return features


def _mutiscale_basic_features_singlechannel(
    img, intensity=True, edges=True, texture=True, sigma_min=0.5, sigma_max=16
):
    """Features for a single channel image. ``img`` can be 2d or 3d.
    """
    # computations are faster as float32
    img = img_as_float32(img)
    sigmas = np.logspace(
        np.log2(sigma_min),
        np.log2(sigma_max),
        num=int(np.log2(sigma_max) - np.log2(sigma_min) + 1),
        base=2,
        endpoint=True,
    )
    n_sigmas = len(sigmas)
    all_results = [
        _singlescale_basic_features(
            img, sigma, intensity=intensity, edges=edges, texture=texture
        )
        for sigma in sigmas
    ]
    return list(itertools.chain.from_iterable(all_results))


def multiscale_basic_features(
    img,
    multichannel=True,
    intensity=True,
    edges=True,
    texture=True,
    sigma_min=0.5,
    sigma_max=16,
):
    """Features for a single- or multi-channel image.
    """
    if img.ndim >= 3 and multichannel:
        all_results = (
            _mutiscale_basic_features_singlechannel(
                img[..., dim],
                intensity=intensity,
                edges=edges,
                texture=texture,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
            for dim in range(img.shape[-1])
        )
        features = list(itertools.chain.from_iterable(all_results))
    else:
        features = _mutiscale_basic_features_singlechannel(
            img,
            intensity=intensity,
            edges=edges,
            texture=texture,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )
    return np.array(features)


def fit_segmenter(image, labels, clf, features_func=None, downsample=10):
    """
    Segmentation using labeled parts of the image and a classifier.

    Parameters
    ----------
    image : ndarray
        Input image, which can be grayscale or multichannel, and must have a
        number of dimensions compatible with ``features_func``.
    labels : ndarray
        Labeled array of shape compatible with ``image`` (same shape for a
        single-channel image). Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        (m_features, *labels.shape). If None,
        :func:`skimage.segmentation.multiscale_basic_features` is used.
    downsample : int, optional
        downsample the number of training points. Use downsample > 1 if you
        built the training set by brushing through large areas but not all
        points are needed to train efficiently the classifier. The training
        time increases with the number of training points.

    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier trained on
        ``labels``.
    clf : classifier object
        classifier trained on ``labels``

    """
    if features_func is None:
        features_func = multiscale_basic_features
    features = features_func(image)
    training_data = features[:, labels > 0].T
    training_labels = labels[labels > 0].ravel()
    clf.fit(training_data[::downsample], training_labels[::downsample])
    data = features[:, labels == 0].T
    predicted_labels = clf.predict(data)
    output = np.copy(labels)
    output[labels == 0] = predicted_labels
    return output, clf


def predict_segmenter(image, clf, features_func=None):
    """
    Segmentation of images using a pretrained classifier.
    """
    if features_func is None:
        features_func = multiscale_basic_features
        # print a warning here?
    features = features_func(image)
    sh = features.shape
    features = features.reshape((sh[0], np.prod(sh[1:]))).T
    predicted_labels = clf.predict(features)
    output = predicted_labels.reshape(sh[1:])
    return output
