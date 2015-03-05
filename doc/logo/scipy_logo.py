"""
Code used to trace Scipy logo.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import data

from skimage.measure import points_in_poly

class SymmetricAnchorPoint(object):
    """Anchor point in a parametric curve with symmetric handles

    Parameters
    ----------
    pt : length-2 sequence
        (x, y) coordinates of anchor point
    theta : float
        angle of control handle
    length : float
        half-length of symmetric control handle. Each control point is `length`
        distance away from the anchor point.
    use_degrees : bool
        If True, convert input `theta` from degrees to radians.
    """

    def __init__(self, pt, theta, length, use_degrees=False):
        self.pt = pt
        if use_degrees:
            theta = theta * np.pi / 180
        self.theta = theta
        self.length = length

    def control_points(self):
        """Return control points for symmetric handles

        The first point is in the direction of theta and the second is directly
        opposite. For example, if `theta = 0`, then the first `p1` will be
        directly to the right of the anchor point, and `p2` will be directly
        to the left.
        """
        theta = self.theta
        offset = self.length * np.array([np.cos(theta), np.sin(theta)])
        p1 = self.pt + offset
        p2 = self.pt - offset
        return p1, p2

    def __repr__(self):
        v = (self.pt, self.theta * 180/np.pi, self.length)
        return 'SymmetricAnchorPoint(pt={0}, theta={1}, length={2})'.format(*v)


def curve_from_anchor_points(pts):
    """Return curve from a list of SymmetricAnchorPoints"""
    assert len(pts) > 1
    bezier_pts = []
    for anchor in pts:
        c1, c2 = anchor.control_points()
        bezier_pts.extend([c2, anchor.pt, c1])
    # clip control points from ends
    bezier_pts = bezier_pts[1:-1]
    x, y = [], []
    # every third point is an anchor point
    for i in range(0, len(bezier_pts)-1, 3):
        xi, yi = cubic_curve(*bezier_pts[i:i+4])
        x.append(xi)
        y.append(yi)
    return np.hstack(x), np.hstack(y)


def cubic_curve(p0, p1, p2, p3, npts=20):
    """Return points on a cubic Bezier curve

    Parameters
    ----------
    p0, p3 : length-2 sequences
        end points of curve
    p1, p2 : length-2 sequences
        control points of curve
    npts : int
        number of points to return (including end points)

    Returns
    -------
    x, y : arrays
        points on cubic curve
    """
    t = np.linspace(0, 1, npts)[:, np.newaxis]
    # cubic bezier curve from http://en.wikipedia.org/wiki/Bezier_curve
    b = (1-t)**3 * p0 + 3*t*(1-t)**2 * p1 + 3*t**2*(1-t) * p2 + t**3 * p3
    x, y = b.transpose()
    return x, y


class Circle(object):

    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def point_from_angle(self, angle):
        r = self.radius
        # `angle` can be a scalar or 1D array: transpose twice for best results
        pts = r * np.array((np.cos(angle), np.sin(angle))).T + self.center
        return pts.T

    def plot(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        fc = kwargs.pop('fc', 'none')
        c = plt.Circle(self.center, self.radius, fc=fc, **kwargs)
        ax.add_patch(c)


class ScipyLogo(object):
    """Object to generate scipy logo

    Parameters
    ----------
    center : length-2 array
        the Scipy logo will be centered on this point.
    radius : float
        radius of logo
    """

    CENTER = np.array((254, 246))
    RADIUS = 252.0
    THETA_START = 2.58
    THETA_END = -0.368

    def __init__(self, center=None, radius=None):
        if center is None:
            if radius is None:
                center = self.CENTER
            else:
                center = np.array((radius, radius))
        self.center = center
        if radius is None:
            radius = self.RADIUS
        self.radius = radius


        # calculate end points of curve so that it lies exactly on circle
        logo_circle = Circle(self.CENTER, self.RADIUS)
        s_start = logo_circle.point_from_angle(self.THETA_START)
        s_end = logo_circle.point_from_angle(self.THETA_END)

        self.circle = Circle(self.center, self.radius)
        # note that angles are clockwise because of inverted y-axis
        self._anchors = [SymmetricAnchorPoint(*t, use_degrees=True)
                         for t in [(s_start,    -37, 90),
                                   ((144, 312),   7, 20),
                                   ((205, 375),  52, 50),
                                   ((330, 380), -53, 60),
                                   ((290, 260),-168, 50),
                                   ((217, 245),-168, 50),
                                   ((182, 118), -50, 60),
                                   ((317, 125),  53, 60),
                                   ((385, 198),  10, 20),
                                   (s_end,      -25, 60)]]
        # normalize anchors so they have unit radius and are centered at origin
        for a in self._anchors:
            a.pt = (a.pt - self.CENTER) / self.RADIUS
            a.length = a.length / self.RADIUS

    def snake_anchors(self):
        """Return list of SymmetricAnchorPoints defining snake curve"""
        anchors = []
        for a in self._anchors:
            pt = self.radius * a.pt + self.center
            length = self.radius * a.length
            anchors.append(SymmetricAnchorPoint(pt, a.theta, length))
        return anchors

    def snake_curve(self):
        """Return x, y coordinates of snake curve"""
        return curve_from_anchor_points(self.snake_anchors())

    def plot_snake_curve(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        x, y = self.snake_curve()
        ax.plot(x, y, 'k', **kwargs)

    def plot_circle(self, **kwargs):
        self.circle.plot(**kwargs)

    def plot_image(self, **kwargs):
        ax = kwargs.pop('ax', plt.gca())
        img = io.imread('data/scipy.png')
        ax.imshow(img, **kwargs)

    def get_mask(self, shape, region):
        """
        Parameters
        ----------
        region : {'upper left', 'lower right'}
        """
        if region == 'upper left':
            theta = np.linspace(self.THETA_END, self.THETA_START - 2 * np.pi)
        elif region == 'lower right':
            theta = np.linspace(self.THETA_END, self.THETA_START)
        else:
            msg = "Expected 'upper left' or 'lower right'; got %s" % region
            raise ValueError(msg)
        xy_circle = self.circle.point_from_angle(theta).T
        x, y = self.snake_curve()
        xy_curve = np.array((x, y)).T
        xy_poly = np.vstack((xy_curve, xy_circle))

        h, w = shape[:2]
        y_img, x_img = np.mgrid[:h, :w]
        xy_points = np.column_stack((x_img.flat, y_img.flat))

        mask = points_in_poly(xy_points, xy_poly)
        return mask.reshape((h, w))


def plot_scipy_trace():
    plt.figure()
    logo = ScipyLogo()
    logo.plot_snake_curve()
    logo.plot_circle()
    logo.plot_image()
    plot_anchors(logo.snake_anchors())


def plot_anchors(anchors, color='r', alpha=0.7):
    for a in anchors:
        c = a.control_points()
        x, y = np.transpose(c)
        plt.plot(x, y, 'o-', color=color, mfc='w', mec=color, alpha=alpha)
        plt.plot(a.pt[0], a.pt[1], 'o', color=color, alpha=alpha)


def plot_snake_overlay():
    plt.figure()
    logo = ScipyLogo((670, 250), 250)
    logo.plot_snake_curve()
    logo.plot_circle()
    img = io.imread('data/snake_pixabay.jpg')
    plt.imshow(img)


if __name__ == '__main__':
    plot_scipy_trace()
    plot_snake_overlay()

    plt.show()
