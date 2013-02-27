import numpy as np


class BaseModel(object):

    def __init__(self):
        self._params = None


class LineModel(BaseModel):

    '''Total least squares estimator for 2D lines.

    Lines are parameterized using polar coordinates:

        dist = x * cos(theta) + y * sin(theta)

    This parameterization is able to model vertical lines in contrast to the
    standard line model `y = a*x + b`.

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
            theta = np.arctan2(data[1,1] - data[0,1], data[1,0] - data[0,0])
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

        self._params = np.array([dist, theta])

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
        data_dists = (data[:, 0] * np.cos(theta) + data[:, 1] * np.sin(theta))
        return np.abs(dist - data_dists)

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
