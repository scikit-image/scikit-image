import numpy as np
from scipy import optimize


class BaseModel(object):

    def __init__(self):
        self._params = None


class LineModel(BaseModel):

    '''Total least squares estimator for 2D lines.

    Lines are parameterized using polar coordinates as functional model:

        dist = x * cos(theta) + y * sin(theta)

    This parameterization is able to model vertical lines in contrast to the
    standard line model `y = a*x + b`.

    This estimator minimizes the squared distances from all points to the line:

        min{ sum((dist - x_i * cos(theta) + y_i * sin(theta))**2) }

    The `_params` attribute contains the parameters in the following order:

        dist, theta

    '''

    def estimate(self, data):
        '''Estimate line model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        '''

        X0 = data.mean(axis=0)

        if data.shape[0] == 2: # well determined
            theta = np.arctan2(data[1, 1] - data[0, 1], data[1, 0] - data[0, 0])
        elif data.shape[0] > 2: # over-determined
            data = data - X0
            # first principal component
            _, _, v = np.linalg.svd(data)
            theta = np.arctan2(v[0, 1], v[0, 0])
        else: # under-determined
            raise ValueError('At least 2 input points needed.')

        # angle perpendicular to line angle
        theta = (theta + np.pi / 2) % np.pi
        # line always passes through mean
        dist = X0[0] * np.cos(theta) + X0[1] * np.sin(theta)

        self._params = (dist, theta)

    def residuals(self, data):
        '''Determine residuals of data to model.

        For each point the shortest distance to the line is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        '''

        dist, theta = self._params

        x = data[:, 0]
        y = data[:, 1]

        return dist - (x * np.cos(theta) + y * np.sin(theta))

    @classmethod
    def is_degenerate(cls, data):
        '''Check whether set of points is degenerate.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        Returns
        -------
        flag : bool
            Flag indicating if data is degenerate.

        '''

        return data.shape[0] < 2

    def predict_x(self, y):
        '''Predict x-coordinates using the estimated model.

        Parameters
        ----------
        y : array
            y-coordinates.

        Returns
        -------
        x : array
            Predicted x-coordinates.

        '''

        dist, theta = self._params
        return (dist - y * np.cos(theta)) / np.cos(theta)

    def predict_y(self, x):
        '''Predict y-coordinates using the estimated model.

        Parameters
        ----------
        x : array
            x-coordinates.

        Returns
        -------
        y : array
            Predicted y-coordinates.

        '''

        dist, theta = self._params
        return (dist - x * np.cos(theta)) / np.sin(theta)


class CircleModel(BaseModel):

    '''Total least squares estimator for 2D circles.

    The functional model of the circle is:

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle:

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    The `_params` attribute contains the parameters in the following order:

        xc, yc, r

    '''

    def estimate(self, data):
        '''Estimate line model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        '''

        x = data[:, 0]
        y = data[:, 1]
        # pre-allocate for all iterations
        A = np.empty((3, data.shape[0]), dtype=np.double)
        # same for all iterations
        A[2, :] = -1

        def dist(xc, yc):
            return np.sqrt((x - xc)**2 + (y - yc)**2)

        def fun(params):
            xc, yc, r = params
            return dist(xc, yc) - r

        def Dfun(params):
            xc, yc, r = params
            d = dist(xc, yc)
            A[0, :] = -(x - xc) / d
            A[1, :] = -(y - yc) / d
            #A[2, :] = -1
            return A

        xc0 = x.mean()
        yc0 = y.mean()
        r0 = dist(xc0, yc0).mean()
        params0 = (xc0, yc0, r0)
        params, _ = optimize.leastsq(fun, params0, Dfun=Dfun, col_deriv=True)

        self._params = params

    def residuals(self, data):
        '''Determine residuals of data to model.

        For each point the shortest distance to the line is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        '''

        xc, yc, r = self._params

        x = data[:, 0]
        y = data[:, 1]

        return r - np.sqrt((x - xc)**2 + (y - yc)**2)

    @classmethod
    def is_degenerate(cls, data):
        '''Check whether set of points is degenerate.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        Returns
        -------
        flag : bool
            Flag indicating if data is degenerate.

        '''

        return data.shape[0] < 2

    def predict_xy(self, theta):
        '''Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        theta : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.

        Returns
        -------
        x : array
            Predicted x-coordinates.
        y : array
            Predicted y-coordinates.

        '''

        xc, yc, r = self._params

        x = xc + r * np.cos(theta)
        y = yc + r * np.sin(theta)

        return x, y


def ransac(data, model_class, min_samples, residual_threshold,
           max_trials=1000):
    '''Fits a model to data with the RANSAC (random sample consensus) algorithm.

    Parameters
    ----------
    data : (N, D) array
        Data set to which the model is fitted, where N is the number of data
        points and D the dimensionality of the data.
    model_class : object
        Object with the following methods implemented:

         * `estimate(data)`
         * `residuals(data)`
         * `is_degenerate(data)`

    min_samples : int
        The minimum number of data points to fit a model.
    residual_threshold : float
        Maximum distance for a data point to be classified as an inlier.
    max_trials : int, optional
        Maximum number of iterations for random sample selection.

    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N,) array
        Indices of inliers.

    '''

    best_model = None
    best_inlier_num = 0
    best_inliers = None
    data_idxs = np.arange(data.shape[0])

    for _ in range(max_trials):

        # choose random sample
        sample = data[np.random.randint(0, data.shape[0], min_samples)]

        # check if random sample is degenerate
        if model_class.is_degenerate(sample):
            continue

        # create new instance of model class for current sample
        sample_model = model_class()
        sample_model.estimate(sample)
        sample_model_residuals = sample_model.residuals(data)
        # consensus set / inliers
        sample_model_inliers = data_idxs[np.abs(sample_model_residuals)
                                         < residual_threshold]

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = sample_model_inliers.shape[0]
        if sample_inlier_num > best_inlier_num:
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inliers = sample_model_inliers

    # estimate final model using all inliers
    if best_inliers is not None:
        best_model.estimate(data[best_inliers])

    return best_model, best_inliers
