import math
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

    A minimum number of 2 points is required to solve for the parameters.

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
        dist = X0[0] * math.cos(theta) + X0[1] * math.sin(theta)

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

        return dist - (x * math.cos(theta) + y * math.sin(theta))

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

    def predict_x(self, y, params=None):
        '''Predict x-coordinates using the estimated model.

        Parameters
        ----------
        y : array
            y-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        x : array
            Predicted x-coordinates.

        '''

        if params is None:
            params = self._params
        dist, theta = params
        return (dist - y * math.cos(theta)) / math.cos(theta)

    def predict_y(self, x, params=None):
        '''Predict y-coordinates using the estimated model.

        Parameters
        ----------
        x : array
            x-coordinates.
        params : (2, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        y : array
            Predicted y-coordinates.

        '''

        if params is None:
            params = self._params
        dist, theta = params
        return (dist - x * math.cos(theta)) / math.sin(theta)


class CircleModel(BaseModel):

    '''Total least squares estimator for 2D circles.

    The functional model of the circle is:

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle:

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    The `_params` attribute contains the parameters in the following order:

        xc, yc, r

    A minimum number of 3 points is required to solve for the parameters.

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
        # pre-allocate jacobian for all iterations
        A = np.zeros((3, data.shape[0]), dtype=np.double)
        # same for all iterations: r
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

        For each point the shortest distance to the circle is returned.

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

        return data.shape[0] < 3

    def predict_xy(self, t, params=None):
        '''Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        x : array
            Predicted x-coordinates.
        y : array
            Predicted y-coordinates.

        '''
        if params is None:
            params = self._params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return x, y


class EllipseModel(BaseModel):

    '''Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is:

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where xt, yt is the closest point on the ellipse to x, y. Thus d is the
    shortest distance from the point to the ellipse.

    This estimator minimizes the squared distances from all points to the
    ellipse:

        min{ sum(d_i**2) } = min{ sum((x_i - xt)**2 + (y_i - yt)**2) }

    Thus you have `2 * N` equations (x_i, y_i) for `N + 5` unknowns (t_i, xc,
    yc, a, b, theta), which gives you an effective redundancy of `N - 5`.

    The `_params` attribute contains the parameters in the following order:

        xc, yc, a, b, theta

    A minimum number of 5 points is required to solve for the parameters.

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

        N = data.shape[0]

        # pre-allocate jacobian for all iterations
        A = np.zeros((N + 5, 2 * N), dtype=np.double)
        # same for all iterations: xc, yc
        A[0, :N] = -1
        A[1, N:] = -1

        diag_idxs = np.diag_indices(N)

        def fun(params):
            xt, yt = self.predict_xy(params[5:], params[:5])
            fx = x - xt
            fy = y - yt
            return np.append(fx, fy)

        def Dfun(params):
            xc, yc, a, b, theta = params[:5]
            t = params[5:]

            ct = np.cos(t)
            st = np.sin(t)
            ctheta = math.cos(theta)
            stheta = math.sin(theta)

            # derivatives for fx, fy in the following order:
            #       xc, yc, a, b, theta, t_i

            # fx
            A[2, :N] = - ctheta * ct
            A[3, :N] = stheta * st
            A[4, :N] = a * stheta * ct + b * ctheta * st
            A[5:, :N][diag_idxs] = a * ctheta * st + b * stheta * ct
            # fy
            A[2, N:] = - stheta * ct
            A[3, N:] = - ctheta * st
            A[4, N:] = - a * ctheta * ct + b * stheta * st
            A[5:, N:][diag_idxs] = a * stheta * st - b * ctheta * ct

            return A

        # initial guess of parameters using a circle model
        params0 = np.empty((N + 5, ), dtype=np.double)
        xc0 = x.mean()
        yc0 = y.mean()
        r0 = np.sqrt((x - xc0)**2 + (y - yc0)**2).mean()
        params0[:5] = (xc0, yc0, r0, 0, 0)
        params0[5:] = np.arctan2(y - yc0, x - xc0)

        params, _ = optimize.leastsq(fun, params0, Dfun=Dfun, col_deriv=True)

        self._params = params[:5]

    def residuals(self, data):
        '''Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with `(x, y)` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        '''

        xc, yc, a, b, theta = self._params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt)**2 + (yi - yt)**2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N, ), dtype=np.double)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

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

        return data.shape[0] < 5

    def predict_xy(self, t, params=None):
        '''Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        x : array
            Predicted x-coordinates.
        y : array
            Predicted y-coordinates.

        '''

        if params is None:
            params = self._params
        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return x, y


def ransac(data, model_class, min_samples, residual_threshold,
           max_trials=100):
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

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> a = 5
    >>> b = 10
    >>> xc = 20
    >>> yc = 30
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> np.random.seed(seed=1234)
    >>> data += np.random.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    >>> model._params
    array([  4.85808595e+02,   4.51492793e+02,   1.15018491e+03,
             5.52428289e+00,   7.32420126e-01])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 5, 3, max_trials=50)
    >>> # ransac_model._params, inliers

    Should give the correct result estimated without the fauly data:

        [ 20.12762373, 29.73563061, 4.81499637, 10.4743584, 0.05217117])
        [ 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
         38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])

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
