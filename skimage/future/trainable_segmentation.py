import numpy as np
from skimage.feature import multiscale_basic_features

try:
    from sklearn.exceptions import NotFittedError

    has_sklearn = True
except ImportError:
    has_sklearn = False

    class NotFittedError(Exception):
        pass


class TrainableSegmenter(object):
    """
    Estimator for classifying pixels.

    Parameters
    ----------
    clf : classifier object, optional
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        ``(m_features, *labels.shape)``. If None,
        :func:`skimage.segmentation.multiscale_basic_features` is used.

    Methods
    -------
    compute_features
    fit
    predict
    """

    def __init__(self, clf=None, features_func=None):
        if clf is None:
            try:
                from sklearn.ensemble import RandomForestClassifier

                self.clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            except ImportError:
                raise ImportError(
                    "Please install scikit-learn or pass a classifier instance"
                    "to TrainableSegmenter."
                )
        else:
            self.clf = clf
        self.features_func = features_func
        self.features = None

    def compute_features(self, image):
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        self.features = self.features_func(image)

    def fit(self, labels, image=None):
        """
        Train classifier using partially labeled (annotated) image.

        Parameters
        ----------
        labels : ndarray of ints
            Labeled array of shape compatible with ``image`` (same shape for a
            single-channel image). Labels >= 1 correspond to the training set and
            label 0 to unlabeled pixels to be segmented.
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.
        """
        if self.features is None:
            self.compute_features(image)
        output, clf = fit_segmenter(labels, self.features, self.clf)
        self.segmented_image = output

    def predict(self, image):
        """
        Segment new image using trained internal classifier.

        Parameters
        ----------
        image : ndarray
            Input image, which can be grayscale or multichannel, and must have a
            number of dimensions compatible with ``self.features_func``.

        Raises
        ------
        NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
        """
        if self.features_func is None:
            self.features_func = multiscale_basic_features
        features = self.features_func(image)
        return predict_segmenter(features, self.clf)


def fit_segmenter(labels, features, clf):
    """
    Segmentation using labeled parts of the image and a classifier.

    Parameters
    ----------
    labels : ndarray of ints
        Image of labels. Labels >= 1 correspond to the training set and
        label 0 to unlabeled pixels to be segmented.
    features : ndarray
        Array of features, with the first dimension corresponding to the number
        of features, and the other dimensions correspond to ``labels.shape``.
    clf : classifier object
        classifier object, exposing a ``fit`` and a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier.

    Returns
    -------
    clf : classifier object
        classifier trained on ``labels``

    Raises
    ------
    NotFittedError if ``self.clf`` has not been fitted yet (use ``self.fit``).
    """
    training_data = features[labels > 0]
    training_labels = labels[labels > 0].ravel()
    clf.fit(training_data, training_labels)
    return clf


def predict_segmenter(features, clf):
    """
    Segmentation of images using a pretrained classifier.

    Parameters
    ----------
    features : ndarray
        Array of features, with the last dimension corresponding to the number
        of features, and the other dimensions are compatible with the shape of
        the image to segment, or a flattened image.
    clf : classifier object
        trained classifier object, exposing a ``predict`` method as in
        scikit-learn's API, for example an instance of
        ``RandomForestClassifier`` or ``LogisticRegression`` classifier. The
        classifier must be already trained, for example with
        :func:`skimage.segmentation.fit_segmenter`.
    features_func : function, optional
        function computing features on all pixels of the image, to be passed
        to the classifier. The output should be of shape
        ``(*labels.shape, n_features)``. If None,
        :func:`skimage.segmentation.multiscale_basic_features` is used.

    Returns
    -------
    output : ndarray
        Labeled array, built from the prediction of the classifier.
    """
    sh = features.shape
    if features.ndim > 2:
        features = features.reshape((np.prod(sh[:-1]), sh[-1]))

    try:
        predicted_labels = clf.predict(features)
    except NotFittedError:
        raise NotFittedError(
            "You must train the classifier `clf` first"
            "for example with the `fit_segmenter` function."
        )
    except ValueError as err:
        if err.args and 'x must consist of vectors of length' in err.args[0]:
            raise ValueError(
                err.args[0] + '\n' +
                "Maybe you did not use the same type of features for training the classifier."
                )
    output = predicted_labels.reshape(sh[:-1])
    return output
