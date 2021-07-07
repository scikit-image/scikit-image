import numpy as np

from ..feature.util import (FeatureDetector, DescriptorExtractor)
from ..feature import peak_local_max

from .._shared.utils import check_nD
from ..transform import rescale
from ..filters import gaussian
from scipy.ndimage.filters import maximum_filter

upsamplings = [1, 2, 4]


class _Keypoints:
    def __init__(self, mns=None, yx=None, sigma=None, val=None, theta=None):
        self.mns = mns
        self.yx = yx
        self.sigma = sigma
        self.val = val
        self.theta = theta

    def __len__(self):
        return self.sigma.shape[0]

    def filter(self, mask, theta=False):
        self.yx = self.yx[mask]
        self.mns = self.mns[mask]
        self.sigma = self.sigma[mask]
        self.val = self.val[mask]
        if theta:
            self.theta = self.theta[mask]


class SIFT(FeatureDetector, DescriptorExtractor):
    """SIFT feature detection and oriented descriptor extraction.

        Parameters
        ----------
        upsampling : int, optional
            Prior to the feature detection the image is upscaled by a factor of 1 (no
            upscaling), 2 or 4. Method: Bi-cubic interpolation.
        n_octaves : int, optional
            Maximum number of octaves. With every octave the image size is halfed and
            the sigma doubled.
        n_scales : int, optional
            Maximum number of scales in every octave.
        sigma_min : float, optional
            The blur level of the seed image. If upsampling is enabled sigma_min is scaled
            by factor 1/upsamling
        sigma_in : float, optional
            The assumed blur level of the input image.
        c_dog : float, optional
            Threshold to discard low contrast extrema in the DoG. It's final value is
            dependent on n_scales by the relation
            final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
        c_edge : float, optional
            Threshold to discard extrema that lie in edges. If H is the Hessian of an extremum,
            its "edgeness" is decribed by tr(H)²/det(H). If the edgeness is higher than
            (c_edge + 1)²/c_edge, the extremum is discarded.
        n_bins : int, optional
            Number of bins in the histogram that describes the gradient orientations around
            keypoint.
        lambda_ori : float, optional
            The window used to find the reference orientation of a keypoint has a width of
            6 * lambda_ori * sigma and is weighted by a standard deviation of 2 * lambda_ori * sigma.
        c_max : float, optional
            The threshold at which a secondary peak in the orientation histogram is accepted as
            orientation
        lambda_descr : float, optional
            The window used to define the descriptor of a keypoint has a width of
            2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by a standard deviation of
            lambda_descr * sigma.
        n_hist : int, optional
            The window used to define the descriptor of a keypoint consists of n_hist * n_hist
            histograms.
        n_ori : int, optional
            The number of bins in the histograms of the descriptor patch.


        #todo: not correct yet, complete the documentation when the code is finished
        Attributes
        ----------
        delta_min : float
            The sampling distance of the first octave. It's final value is 1/upsampling.
        deltas : (n_octaves, n_scales + 3)
            The sampling distances of all octaves.
        keypoints : (N, 2) array
            Keypoint coordinates as ``(row, col)``.


        References
        ----------
        .. [1] Ives Rey Otero, and Mauricio Delbracio
              "Anatomy of the SIFT Method"
              Image Processing On Line, 4 (2014), pp. 370–396.
              https://doi.org/10.5201/ipol.2014.82

        Examples
        --------
        >>> from skimage.feature import SIFT, match_descriptors
        >>> from skimage.data import camera
        >>> from skimage.transform import rotate
        >>> img1 = camera()
        >>> img2 = rotate(camera(), 90)
        >>> detector_extractor1 = SIFT()
        >>> detector_extractor2 = SIFT()
        >>> detector_extractor1.detect_and_extract(img1)
        >>> detector_extractor2.detect_and_extract(img2)
        >>> matches = match_descriptors(detector_extractor1.descriptors,
        ...                             detector_extractor2.descriptors)
        #todo: add results

        """

    def __init__(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5,
                 c_dog=0.04 / 3, c_edge=10, n_bins=36, lambda_ori=1.5, c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8):
        if upsampling in [1, 2, 4]:
            self.upsampling = upsampling
        else:
            raise ValueError("upsampling must be 1, 2 or 4")
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min / upsampling
        self.sigma_in = sigma_in
        self.c_dog = (2 ** (1 / n_scales) - 1) / (2 ** (1 / 3) - 1) * c_dog
        self.c_edge = c_edge
        self.n_bins = n_bins
        self.lambda_ori = lambda_ori
        self.c_max = c_max
        self.lambda_descr = lambda_descr
        self.n_hist = n_hist
        self.n_ori = n_ori

        self.delta_min = 1 / upsampling
        self.deltas = self.delta_min * np.power(2, np.arange(self.n_octaves))
        self.keypoints = None
        self.sigmas = None
        self.descriptors = None

    def _number_of_octaves(self, n, image_shape):
        sMin = 12  # minimum size of last octave
        s0 = np.min(image_shape)
        return int(np.min((n, (np.log(s0 / sMin) / np.log(2)) + self.upsampling)))

    def _create_scalespace(self, image):
        """ Blur the image and rescale it multiple times
        """
        scalespace = []
        if self.upsampling > 1:
            image = rescale(image, self.upsampling, order=3)
        sigmas = np.empty((self.n_octaves, self.n_scales + 3))  # all sigmas for the gaussian scalespace
        current_sigma = self.sigma_min

        image = gaussian(image,
                         (1 / self.delta_min) * np.sqrt(self.sigma_min ** 2 - self.sigma_in ** 2),
                         mode='reflect')  # smooth to sigma_min, assuming sigma_in

        k = 2 ** (1 / self.n_scales)  # after n_scales steps we doubled the smoothing
        for o in range(self.n_octaves):  # one octave is represented by a 3D image with depth (n_scales+x)
            delta = self.delta_min * 2 ** o
            sigmas[o, 0] = current_sigma
            octave = np.empty(image.shape + (self.n_scales + 3,))
            octave[:, :, 0] = image
            for s in range(1, self.n_scales + 3):
                octave[:, :, s] = gaussian(octave[..., s - 1],
                                           (1 / delta) * np.sqrt((current_sigma * k) ** 2 - current_sigma ** 2),
                                           mode='reflect')  # blur new scale assuming sigma of the last one
                current_sigma = current_sigma * k
                sigmas[o, s] = current_sigma
            scalespace.append(octave)
            image = octave[:, :, self.n_scales][::2, ::2]  # downscale the image by taking every second pixel
            current_sigma = sigmas[o, self.n_scales]
        self.sigmas = sigmas
        return scalespace

    def _inrange(self, a, dim):
        return (a[:, 0] > 0) & (a[:, 0] < dim[0] - 1) & (a[:, 1] > 0) & (a[:, 1] < dim[1] - 1)

    def _hessian(self, h, d, positions):
        h[:, 0, 0] = d[positions[:, 0] - 1, positions[:, 1], positions[:, 2]] \
                     + d[positions[:, 0] + 1, positions[:, 1], positions[:, 2]] \
                     - 2 * d[positions[:, 0], positions[:, 1], positions[:, 2]]
        h[:, 1, 1] = d[positions[:, 0], positions[:, 1] - 1, positions[:, 2]] \
                     + d[positions[:, 0], positions[:, 1] + 1, positions[:, 2]] \
                     - 2 * d[positions[:, 0], positions[:, 1], positions[:, 2]]

        h[:, 2, 2] = d[positions[:, 0], positions[:, 1], positions[:, 2] - 1] \
                     + d[positions[:, 0], positions[:, 1], positions[:, 2] + 1] \
                     - 2 * d[positions[:, 0], positions[:, 1], positions[:, 2]]

        h[:, 1, 0] = h[:, 0, 1] = 0.25 * (d[positions[:, 0] + 1, positions[:, 1] + 1, positions[:, 2]]
                                          - d[positions[:, 0] - 1, positions[:, 1] + 1, positions[:, 2]]
                                          - d[positions[:, 0] + 1, positions[:, 1] - 1, positions[:, 2]]
                                          + d[positions[:, 0] - 1, positions[:, 1] - 1, positions[:, 2]])

        h[:, 2, 0] = h[:, 0, 2] = 0.25 * (d[positions[:, 0] + 1, positions[:, 1], positions[:, 2] + 1]
                                          - d[positions[:, 0] + 1, positions[:, 1], positions[:, 2] - 1]
                                          + d[positions[:, 0] - 1, positions[:, 1], positions[:, 2] - 1]
                                          - d[positions[:, 0] - 1, positions[:, 1], positions[:, 2] + 1])

        h[:, 2, 1] = h[:, 1, 2] = 0.25 * (d[positions[:, 0], positions[:, 1] + 1, positions[:, 2] + 1]
                                          - d[positions[:, 0], positions[:, 1] + 1, positions[:, 2] - 1]
                                          + d[positions[:, 0], positions[:, 1] - 1, positions[:, 2] - 1]
                                          - d[positions[:, 0], positions[:, 1] - 1, positions[:, 2] + 1])

    def _find_localize_evaluate(self, dogspace, img_shape):
        """
        1) first find all extrema of a (3, 3, 3) neighborhood
        2) use second order Taylor development to refine the positions to sub-pixel precision
        3) filter out extrema that have low contrast and lie on edges or close to the image borders
        """
        extrema = []
        threshold = self.c_dog * 0.8
        for o, (octave, delta) in enumerate(zip(dogspace, self.deltas)):
            # find extrema
            maxima = peak_local_max(octave, threshold_abs=threshold)
            minima = peak_local_max(-octave, threshold_abs=threshold)
            keys = np.vstack((maxima, minima))

            # localize extrema
            dim = octave.shape
            off = np.empty_like(keys)  # offset and Jacobian
            J = np.empty_like(keys)
            H = np.empty((len(keys), 3, 3))  # Hessian
            grad = np.gradient(octave)  # take first derivative of the whole octave
            still_in = np.ones(len(keys), dtype=bool)  # mask for all extrema that still have to be tested
            for i in range(5):
                still_in = np.logical_and(still_in, self._inrange(keys, dim))
                J = np.swapaxes(np.array(
                    [ax[keys[still_in, 0], keys[still_in, 1], keys[still_in, 2]] for ax in grad]), 0,
                    1)  # Jacoby of all extrema
                self._hessian(H, octave, keys)
                # H_inv = np.linalg.inv(H)  # invert hessian
                # off = np.einsum('ijk,ik->ij', -H_inv, J)  # offset of the extremum
                off = np.linalg.solve(-H, J)  # offset of the extremum
                wrong_position_pos = np.logical_and(off > 0.6, keys + 1 < tuple(
                    [a - 1 for a in dim]))  # offset is too big and an increase wouldnt bring us out of bounds
                wrong_position_neg = np.logical_and(off < -0.6, keys - 1 > 0)
                if (not np.any(np.logical_or(wrong_position_neg, wrong_position_pos))) or i == 4:
                    break
                keys[np.where(wrong_position_pos == True)] += 1
                keys[np.where(wrong_position_neg == True)] -= 1
            finished = np.all(np.abs(off) < 0.6,
                              axis=1)  # mask for all extrema that have been localized successfully
            keys = keys[finished]
            vals = octave[keys[:, 0], keys[:, 1], keys[:, 2]]  # value of extremum in octave (needed for next function)
            J = J[finished]
            off = off[finished]
            w = vals + 0.5 * np.sum(J * off, axis=1)  # values at interpolated point
            H = H[finished, :2, :2]
            sigmaratio = self.sigmas[0, 1] / self.sigmas[0, 0]

            contrast_threshold = self.c_dog / self.n_scales
            edge_threshold = np.square(self.c_edge + 1) / self.c_edge

            # filter for contrast, edgeness and borders
            contrast_filter = np.abs(w) > contrast_threshold
            eig, _ = np.linalg.eig(H[contrast_filter])  # eigenvalues instead of trace and determinante
            trace = eig[:, 1] + eig[:, 0]
            determinant = eig[:, 1] * eig[:, 0]
            edge_respone = np.square(trace) / determinant
            edge_filter = np.abs(edge_respone) <= edge_threshold

            keys = keys[contrast_filter][edge_filter]
            off = off[contrast_filter][edge_filter]
            yx = (keys[:, 0:2] + off[:, 0:2]) * delta

            sigmas = self.sigmas[o, keys[:, 2]] * np.power(sigmaratio, off[:, 2])
            border_filter = np.all(np.logical_and((yx - sigmas[:, np.newaxis]) > 0.0,
                                                  (yx + sigmas[:, np.newaxis]) < img_shape),
                                   axis=1)

            keypoints = _Keypoints(keys[border_filter],
                                  yx[border_filter],
                                  sigmas[border_filter],
                                  w[contrast_filter][edge_filter][border_filter])
            extrema.append(keypoints)
        return extrema

    def _fit(self, h):
        return (h[0] - h[2]) / (2 * (h[0] + h[2] - 2 * h[1]))

    def _compute_orientation(self, octave_keypoints, gaussian_scalespace):
        gradientSpace = []
        for o, (keypoints, octave) in enumerate(zip(octave_keypoints, gaussian_scalespace)):
            gradientSpace.append(np.gradient(octave))
            delta = self.deltas[o]
            dim = octave.shape[0:2]
            keypoints.theta = np.zeros_like(keypoints.sigma)
            yx = keypoints.yx / delta  # convert to octaves dimensions
            sigma = keypoints.sigma / delta
            keypoint_indices = []
            keypoint_angles = []
            keypoints_valid = np.ones_like(sigma, dtype=bool)

            # dimensions of the patch
            radius = 3 * self.lambda_ori * sigma
            Min = np.maximum(0, np.add(np.subtract(yx, radius[:, np.newaxis]), 0.5)).astype(np.int)
            Max = np.minimum(yx + radius[:, np.newaxis] + 0.5, (dim[0] - 1, dim[1] - 1)).astype(np.int)

            for k in range(len(keypoints.yx)):
                if np.all(Min[k] > 0) and np.all(Max[k] > Min[k]):
                    hist = np.zeros(self.n_bins)  # orientation histogram

                    # use the patch coordinates to get the gradient and then normalize them
                    n, m = np.mgrid[Min[k, 0]:(Max[k, 0] + 1), Min[k, 1]: (Max[k, 1] + 1)]
                    gradientY = gradientSpace[o][0][n, m, keypoints.mns[k, 2]]
                    gradientX = gradientSpace[o][1][n, m, keypoints.mns[k, 2]]
                    n = np.subtract(n, yx[k, 0])
                    m = np.subtract(m, yx[k, 1])

                    magnitude = np.sqrt(np.square(gradientY) + np.square(gradientX))  # gradient magnitude
                    theta = np.mod(np.arctan2(gradientX, gradientY), 2 * np.pi)  # angles
                    kernel = np.exp(-np.divide(np.add(np.square(n), np.square(m)),
                                               2 * (self.lambda_ori * sigma[k]) ** 2))  # more weight to center values

                    # fill the histogram
                    bins = np.floor((theta / (2 * np.pi) * self.n_bins + 0.5) % self.n_bins).astype(np.int)
                    np.add.at(hist, bins, kernel * magnitude)

                    # smooth the histogram and find the maximum
                    hist = np.hstack((hist[-3:], hist, hist[:3]))  # append end and beginning to convolve circular
                    for _ in range(6):  # number of smoothings
                        hist = np.convolve(hist, np.ones(3) / 3, mode='same')
                    hist = hist[3:-3]
                    max_filter = maximum_filter(hist, [3])
                    # if an angle is in 80% percent range of the maximum, a new keypoint is created for it
                    maxima = np.where(np.logical_and(hist >= (self.c_max * np.max(hist)), max_filter == hist))

                    # save the angles
                    for c, m in enumerate(maxima[0]):
                        neigh = np.arange(m - 1, m + 2) % len(hist)
                        # use neighbors to fit a parabola, to get more accurate result
                        ori = (m + self._fit(hist[neigh]) + 0.5) * 2 * np.pi / self.n_bins
                        if ori > np.pi: ori -= 2 * np.pi
                        if c == 0:
                            keypoints.theta[k] = ori
                        else:
                            keypoint_indices.append(k)
                            keypoint_angles.append(ori)
                else:
                    keypoints_valid[k] = False
            keypoints.mns = np.vstack((keypoints.mns, keypoints.mns[keypoint_indices]))
            keypoints.yx = np.vstack((keypoints.yx, keypoints.yx[keypoint_indices]))
            keypoints.val = np.hstack((keypoints.val, keypoints.val[keypoint_indices]))
            keypoints.sigma = np.hstack((keypoints.sigma, keypoints.sigma[keypoint_indices]))
            keypoints.theta = np.hstack((keypoints.theta, keypoint_angles))
            keypoints_valid = np.hstack((keypoints_valid, np.ones((len(keypoint_indices)), dtype=bool)))
            keypoints.filter(keypoints_valid, True)
        # return the gradientspace to reuse it to find the descriptor
        return gradientSpace

    def _rotate(self, Y, X, angle, sigma):
        c = np.cos(angle)
        s = np.sin(angle)
        rY = (c * Y - s * X) / sigma
        rX = (s * Y + c * X) / sigma
        return rY, rX

    def _descriptor(self, keypoints_octave, gradientSpace):
        """Calculates the descriptor for every Keypoint
        For deeper explanation see IPOL: Anatomy of the SIFT method Alg. 12

        Parameters
        ----------
        keypoints_octave : List[3D array]
            A list of all keypoints, grouped into their octaves
        gradientSpace : List[3D array]
            The gradient of the scalespace

        Returns
        -------
        keypoint_des : array [n_keypoints, n_hist * n_hist * n_ori]
            All keypoint descriptors
        """
        nKey = sum([len(k) for k in keypoints_octave])
        keypoint_des = np.empty((nKey, self.n_hist ** 2 * self.n_ori), dtype=int)
        key_count = 0
        for o, (keypoints, gradient) in enumerate(zip(keypoints_octave, gradientSpace)):
            delta = self.deltas[o]
            dim = gradient[0].shape[0:2]
            yx = keypoints.yx / delta
            sigma = keypoints.sigma / delta
            mns = keypoints.mns

            # dimensions of the patch
            radius = self.lambda_descr * (1 + 1 / self.n_hist) * sigma
            radiusPatch = np.sqrt(2) * radius
            Min = np.array(np.maximum(0, yx - radiusPatch[:, np.newaxis] + 0.5), dtype=int)
            Max = np.array(np.minimum(yx + radiusPatch[:, np.newaxis] + 0.5, (dim[0] - 1, dim[1] - 1)), dtype=int)

            for k in range(len(Max)):
                histograms = np.zeros((self.n_hist, self.n_hist, self.n_ori))
                m, n = np.mgrid[Min[k, 0]:Max[k, 0], Min[k, 1]: Max[k, 1]]  # the patch
                y_mn = np.copy(m) - yx[k, 0]  # normalized coordinates
                x_mn = np.copy(n) - yx[k, 1]
                y_mn, x_mn = self._rotate(y_mn, x_mn, -keypoints.theta[k], 1)

                inRadius = np.maximum(np.abs(y_mn), np.abs(x_mn)) < radius[k]
                y_mn, x_mn = y_mn[inRadius], x_mn[inRadius]
                m, n = m[inRadius], n[inRadius]

                gradientY = gradientSpace[o][0][m, n, mns[k, 2]]
                gradientX = gradientSpace[o][1][m, n, mns[k, 2]]

                theta = np.mod(np.arctan2(gradientX, gradientY) - keypoints.theta[k], 2 * np.pi)
                kernel = np.exp(-(np.square(y_mn) + np.square(x_mn)) / (2 * (self.lambda_descr * sigma[k]) ** 2))
                magnitude = np.sqrt(np.square(gradientY) + np.square(gradientX)) * kernel

                hists = np.arange(1, self.n_hist + 1)  # indices of the histograms
                bins = np.arange(1, self.n_ori + 1)  # indices of the bins
                yj_xi = (hists - (1 + self.n_hist) / 2) * ((2 * self.lambda_descr * sigma[k]) / self.n_hist)
                ok = (2 * np.pi * bins) / self.n_ori

                # distances to the histograms and bins
                dist_y = np.abs(np.subtract.outer(yj_xi, y_mn))
                dist_x = np.abs(np.subtract.outer(yj_xi, x_mn))
                dist_t = np.abs(np.mod(np.subtract.outer(ok, theta), 2 * np.pi))

                # the histograms/bins that get the contribution
                near_y = dist_y <= ((self.lambda_descr * 2 * sigma[k]) / self.n_hist)
                near_x = dist_x <= ((self.lambda_descr * 2 * sigma[k]) / self.n_hist)
                near_t = np.argmin(dist_t, axis=0)
                near_t_val = np.min(dist_t, axis=0)

                # every contribution in y direction is combined with every in x direction
                # y: histogram 3 and 4, x: histogram 2 -> contribute to (3,2) and (4,2)
                comb = np.logical_and(near_x.T[:, None, :], near_y.T[:, :, None])
                positions = np.where(comb)

                # the weights/contributions are shared bilinearly between the histograms
                weights = (1 - (self.n_hist / (2 * self.lambda_descr * sigma[k])) * dist_y[positions[1], positions[0]]) \
                          * (1 - (self.n_hist / (2 * self.lambda_descr * sigma[k])) * dist_x[
                    positions[2], positions[0]]) \
                          * magnitude[positions[0]]

                # the weight is shared linearly between the 2 nearest bins
                w1 = weights * ((self.n_ori / (2 * np.pi)) * near_t_val[positions[0]])
                w2 = weights * (1 - (self.n_ori / (2 * np.pi)) * near_t_val[positions[0]])
                k_index = near_t[positions[0]]
                np.add.at(histograms, (positions[1], positions[2], k_index), w1)
                np.add.at(histograms, (positions[1], positions[2], np.mod((k_index + 1), self.n_ori)), w2)

                # convert the histograms to a 1d descriptor
                histograms = histograms.flatten()
                # saturate the descriptor
                histograms = np.minimum(histograms, 0.2 * np.linalg.norm(histograms))
                # normalize the descriptor
                descriptor = np.minimum(np.floor((512 * histograms) / np.linalg.norm(histograms)), 255).astype(np.uint8)
                keypoint_des[key_count, :] = descriptor
                key_count += 1
        return keypoint_des

    def detect(self, image):
        """Detect the keypoints in every octave.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in gaussian_scalespace]

        keypoints = self._find_localize_evaluate(dog_scalespace, image.shape)

        self._compute_orientation(keypoints, gaussian_scalespace)

        self.keypoints = keypoints

    def extract(self, image, keypoints):
        """Detect the keypoints in every octave.

        Parameters
        ----------
        image : 2D array
            Input image.
        todo: keypoints

        """
        check_nD(image, 2)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        gradientSpace = [np.gradient(octave) for octave in gaussian_scalespace]

        descriptors = self._descriptor(keypoints, gradientSpace)

        self.descriptors = descriptors

    def detect_and_extract(self, image):
        """Detect SIFT keypoints and extract oriented descriptors.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in gaussian_scalespace]

        keypoints = self._find_localize_evaluate(dog_scalespace, image.shape)

        gradientSpace = self._compute_orientation(keypoints, gaussian_scalespace)

        descriptors = self._descriptor(keypoints, gradientSpace)

        self.keypoints = np.vstack([k.yx.round().astype(int) for k in keypoints])
        self.descriptors = descriptors
