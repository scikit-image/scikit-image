import numpy as np

from ..feature.util import (FeatureDetector, DescriptorExtractor,
                            _mask_border_keypoints,
                            _prepare_grayscale_input_2D)
from ..feature import peak_local_max

from .._shared.utils import check_nD
from ..transform import rescale
from ..filters import gaussian
from scipy.ndimage.filters import maximum_filter

upsamplings = [1, 2, 4]


class Keypoints:
    def __init__(self, mns=None, yx=None, sigma=None, val=None, theta=None):
        self.mns = mns
        self.yx = yx
        self.sigma = sigma
        self.val = val
        self.theta = theta

    def __len__(self):
        return self.sigma.shape[0]

    def getKeypoint(self, i):
        return self.mns[i], self.yx[i], self.sigma[i], self.val[i]

    def setMNS(self, mns):
        self.mns = mns

    def setYX(self, yx):
        self.yx = yx

    def setSigma(self, sigma):
        self.sigma = sigma

    def setVal(self, val):
        self.val = val

    def setTheta(self, theta):
        self.theta = theta

    def setThetaByIndex(self, i, val):
        self.theta[i] = val

    def copyKeypoint(self, i, theta):
        self.mns = np.vstack((self.mns, self.mns[i]))
        self.yx = np.vstack((self.yx, self.yx[i]))
        self.sigma = np.hstack((self.sigma, self.sigma[i]))
        self.val = np.hstack((self.val, self.val[i]))
        self.theta = np.hstack((self.theta, theta))

    def filter(self, filter, theta=False):
        self.yx = self.yx[filter]
        self.mns = self.mns[filter]
        self.sigma = self.sigma[filter]
        self.val = self.val[filter]
        if theta:
            self.theta = self.theta[filter]


# TODO: n_octaves dynamically or static
class SIFT(FeatureDetector, DescriptorExtractor):
    """

    """

    def __init__(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6, sigma_in=0.5,
                 c_dog=0.04, c_edge=10, lambda_ori=1.5, lambda_descr=6, n_hist=4, n_bins=8):
        if upsampling in [1, 2, 4]:
            self.upsampling = upsampling
        else:
            raise ValueError("upsampling must be 1, 2 or 4")
        self.n_octaves = n_octaves
        self.n_scales = n_scales
        self.sigma_min = sigma_min / upsampling
        self.sigma_in = sigma_in
        self.delta_min = 1 / upsampling
        self.deltas = self.delta_min * np.power(2, np.arange(self.n_octaves))
        self.offsetMax = 0.6
        self.c_dog = c_dog / n_scales
        self.c_edge = c_edge
        self.lambda_ori = lambda_ori
        self.lambda_descr = lambda_descr
        self.n_hist = n_hist
        self.n_bins = n_bins

        self.keypoints = None
        self.sigmas = None
        self.descriptors = None

    def _numberOfOctaves(self, n, image_shape):  # todo:einbinden
        sMin = 12  # minimum size of last octave
        s0 = np.min(image_shape)
        return int(np.min((n, (np.log(s0 / sMin) / np.log(2)) + self.upsampling)))

    def _create_scalespace(self, image):
        """ Blur the image and rescale it multiple times
        """
        scalespace = []
        if self.upsampling > 1:
            image = rescale(image, self.upsampling, order=3)
            # image = rescale(image, self.upsampling)
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
                octave[:, :, s] = gaussian(octave[..., s - 1], (1 / delta) * np.sqrt(
                    (current_sigma * k) ** 2 - current_sigma ** 2),
                                           mode='reflect')  # blur new scale assuming sigma of the last one
                current_sigma = current_sigma * k
                sigmas[o, s] = current_sigma
            scalespace.append(octave)
            image = octave[:, :, self.n_scales][::2, ::2]  # downscale the image by taking every second pixel
            current_sigma = sigmas[o, self.n_scales]
        self.sigmas = sigmas
        return scalespace

    def _inrange(self, a, dim):
        return (a[:, 0] > 0) & (a[:, 0] < dim[0]-1) & (a[:, 1] > 0) & (a[:, 1] < dim[1]-1)

    def _hessian(self, H, D, positions):
        H[:, 0, 0] = D[positions[:, 0] - 1, positions[:, 1], positions[:, 2]] \
              + D[positions[:, 0] + 1, positions[:, 1], positions[:, 2]] \
              - 2 * D[positions[:, 0], positions[:, 1], positions[:, 2]]
        H[:, 1, 1] = D[positions[:, 0], positions[:, 1] -1 , positions[:, 2]] \
              + D[positions[:, 0], positions[:, 1]+1, positions[:, 2]] \
              - 2 * D[positions[:, 0], positions[:, 1], positions[:, 2]]

        H[:, 2, 2] = D[positions[:, 0], positions[:, 1], positions[:, 2] - 1 ]  \
              + D[positions[:, 0], positions[:, 1], positions[:, 2] + 1] \
              - 2 * D[positions[:, 0], positions[:, 1], positions[:, 2]]

        H[:, 1, 0] = H[:, 0, 1] = 0.25 * (D[positions[:, 0] +1, positions[:, 1] + 1, positions[:, 2]]
                      - D[positions[:, 0]-1, positions[:, 1] +1, positions[:, 2]]
                      - D[positions[:, 0]+1, positions[:, 1] -1 , positions[:, 2]]
                      + D[positions[:, 0]-1, positions[:, 1] -1, positions[:, 2]])

        H[:, 2, 0] = H[:, 0, 2] = 0.25 * (D[positions[:, 0]+1, positions[:, 1] , positions[:, 2] + 1]
                      - D[positions[:, 0]+1, positions[:, 1], positions[:, 2] - 1]
                      + D[positions[:, 0]-1, positions[:, 1], positions[:, 2] - 1]
                      - D[positions[:, 0]-1, positions[:, 1], positions[:, 2] + 1])

        H[:, 2, 1] = H[:, 1, 2] = 0.25 * (D[positions[:, 0], positions[:, 1] + 1, positions[:, 2] + 1]
                      - D[positions[:, 0], positions[:, 1] + 1, positions[:, 2] - 1]
                      + D[positions[:, 0], positions[:, 1] - 1, positions[:, 2] - 1]
                      - D[positions[:, 0], positions[:, 1] - 1, positions[:, 2] + 1])

    def _find_localize_evaluate(self, dogspace, img_shape):
        """
        Find extrema of a (3, 3, 3) Neighborhood
        """
        extrema = []
        threshold = self.c_dog * 0.8
        for o, (octave, delta) in enumerate(zip(dogspace, self.deltas)):
            #find extrema
            maxima = peak_local_max(octave, threshold_abs=threshold)
            minima = peak_local_max(-octave, threshold_abs=threshold)
            keys = np.vstack((maxima, minima))

            #localize extrema
            dim = octave.shape
            off = np.empty_like(keys) #offset and Jacobian#todo:possible
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
                H_inv = np.linalg.inv(H)  # invert hessian
                off = np.einsum('ijk,ik->ij', -H_inv, J)  # offset of the extremum
                wrong_position_pos = np.logical_and(off > self.offsetMax, keys + 1 < tuple(
                    [a - 1 for a in dim]))  # offset is too big and an increase wouldnt bring us out of bounds
                wrong_position_neg = np.logical_and(off < -self.offsetMax, keys - 1 > 0)
                if (not np.any(np.logical_or(wrong_position_neg, wrong_position_pos))) or i == 4:
                    break
                keys[np.where(wrong_position_pos == True)] += 1
                keys[np.where(wrong_position_neg == True)] -= 1
            finished = np.all(np.abs(off) < self.offsetMax,
                              axis=1)  # mask for all extrema that have been localized successfully
            keys = keys[finished]
            vals = octave[keys[:, 0], keys[:, 1], keys[:,2]]  # value of extremum in octave (needed for next function)
            J = J[finished]
            off = off[finished]
            w = vals + 0.5 * np.sum(J * off, axis=1)  # values at interpolated point
            H = H[finished, :2, :2]
            sigmaratio = self.sigmas[0, 1] / self.sigmas[0, 0]

            contrast_threshold = self.c_dog / self.n_scales
            edge_threshold = np.square(self.c_edge+1)/self.c_edge

            #filter for contrast, edgeness and borders
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

            keypoints = Keypoints(keys[border_filter],
                                  yx[border_filter],
                                  sigmas[border_filter],
                                  w[contrast_filter][edge_filter][border_filter])
            extrema.append(keypoints)
        return extrema

    def _fit(self, h):
        return ((h[0]-h[2])/(2*(h[0]+h[2]-2*h[1])))

    def _compute_orientation(self, octave_keypoints, gaussian_scalespace):
        gradientSpace = []
        for o, (keypoints, octave) in enumerate(zip(octave_keypoints, gaussian_scalespace)):
            gradientSpace.append(np.gradient(octave))
            delta = self.deltas[o]
            dim = octave.shape[0:2]
            keypoints.setTheta(np.zeros_like(keypoints.sigma))
            yx = keypoints.yx / delta # convert to octaves dimensions
            sigma = keypoints.sigma / delta
            radius = 3 * self.lambda_ori * sigma

            Min = np.array(np.maximum(0, np.add(np.subtract(yx, radius[:, np.newaxis]), 0.5)), dtype=int)
            Max = np.array(np.minimum(yx + radius[:, np.newaxis] + 0.5, (dim[0] - 1, dim[1] - 1)), dtype=int)
            keypoint_indices = []
            keypoint_angles = []
            keypoints_valid = np.ones_like(sigma, dtype=bool)
            for k in range(len(keypoints.yx)):
                if np.all(Min[k] > 0) and np.all(Max[k] > Min[k]):
                    hist = np.zeros(self.n_bins) # orientation histogram
                    YY, XX = np.mgrid[Min[k, 0]:(Max[k, 0]+1), Min[k, 1]: (Max[k, 1]+1)]
                    gradientY = gradientSpace[o][0][YY, XX, keypoints.mns[k, 2]]
                    gradientX = gradientSpace[o][1][YY, XX, keypoints.mns[k, 2]]

                    mag = np.sqrt(np.square(gradientY) + np.square(gradientX))  # gradient magnitude
                    t = np.mod(np.arctan2(gradientX, gradientY), 2 * np.pi) # angles

                    YY = np.subtract(YY, yx[k, 0])
                    XX = np.subtract(XX, yx[k, 1])
                    kernel = np.exp(-np.divide(np.add(np.square(YY), np.square(XX)), 2 * (self.lambda_ori* sigma[k])**2))  # more weight to center values
                    bins = np.array(np.floor((t / (2*np.pi)* self.n_bins + 0.5) % self.n_bins), dtype=int)
                    np.add.at(hist, bins, kernel*mag)

                    hist = np.hstack((hist[-3:], hist, hist[:3]))  # append end and beginning to convolve circular
                    for _ in range(6): # number of smoothings
                        hist = np.convolve(hist, np.ones(3) / 3, mode='same')
                    hist = hist[3:-3]

                    filter = maximum_filter(hist, [3])  # get maximum and all smaller ones in 80% reach
                    maxima = np.where(np.logical_and(hist >= (0.8 * np.max(hist)), filter == hist))
                    for c, m in enumerate(maxima[0]): # when we have the angle, we can start to calculate the descriptor
                        neigh = np.arange(m - 1, m + 2) % len(
                            hist)  # use neighbors to fit a parabola, to get more accurate result
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
        return gradientSpace

    def rotate(self, Y, X, angle, sigma):
        c = np.cos(angle)
        s = np.sin(angle)
        rY = (c * Y - s * X)/sigma
        rX = (s * Y + c * X)/sigma
        return rY, rX

    def _descriptor(self, keypoints_octave, gradientSpace):
        """
        Calculates the descriptor for every Keypoint
        For deeper explanation see IPOL: Anatomy of the SIFT method Alg. 12
        """
        counter = 0
        nKey = sum([len(k) for k in keypoints_octave])
        keypoint_coo = np.empty((0, 2), dtype=int)
        keypoint_des = np.empty((nKey , self.n_hist ** 2 * self.n_bins), dtype=int)
        key_count = 0
        for o, (keypoints, gradient) in enumerate(zip(keypoints_octave, gradientSpace)):
            keypoint_coo= np.vstack((keypoint_coo, np.array(np.round(keypoints.yx), dtype=int)))
            delta = self.deltas[counter]
            dim = gradient.shape[0:2]
            yx = np.copy(keypoints.yx) / delta
            sigma = np.copy(keypoints.sigma) / delta
            theta = keypoints.theta
            mns = keypoints.mns
            radius = self.lambda_descr * (1 + 1 / self.n_hist)
            radiusP = np.sqrt(2) * radius * sigma
            Min = np.array(np.maximum(0, yx - radiusP[:, np.newaxis] + 0.5), dtype=int)
            Max = np.array(np.minimum(yx + radiusP[:, np.newaxis] + 0.5, (dim[0] - 1, dim[1] - 1)), dtype=int)
            for k in range(len(Max)):
                histograms = np.zeros((self.n_hist, self.n_hist, self.n_bins))
                m, n = np.mgrid[Min[k, 0]:Max[k, 0], Min[k, 1]: Max[k, 1]]
                y_mn = np.copy(m) - yx[k, 0]
                x_mn = np.copy(n) - yx[k, 1]
                y_mn, x_mn = self.rotate(y_mn, x_mn, -theta[k], sigma[k])

                inR = np.maximum(np.abs(y_mn), np.abs(x_mn)) < radius
                y_mn, x_mn = y_mn[inR], x_mn[inR]
                m, n = m[inR], n[inR]
                gradientY = gradientSpace[0][
                    m, n, mns[k, 2]]  # create window with meshgrid
                gradientX = gradientSpace[1][m, n, mns[k, 2]]
                t = np.mod(np.arctan2(gradientX, gradientY) - theta[k], 2 * np.pi)
                mag = np.sqrt(np.square(gradientY) + np.square(gradientX))
                kernel = np.exp(-np.divide(np.add(np.square(y_mn), np.square(x_mn)), 2 * (self.lambda_descr * sigma[k]) ** 2))
                mag = mag * kernel


                hists = np.arange(1, self.n_hist+1)
                bins = np.arange(1, self.n_bins + 1)
                yi = (hists - (1+self.n_hist)/2) * ((2 * self.lambda_descr * sigma[k]) /(self.n_hist))

                diffy = np.abs(np.subtract.outer(yi, y_mn))
                diffx = np.abs(np.subtract.outer(yi, x_mn))
                difft = np.abs(np.mod(np.subtract.outer((2*np.pi/self.n_bins) * bins, t), 2*np.pi))

                neary = diffy <= ((self.lambda_descr * 2) / self.n_hist)
                nearx = diffx <= ((self.lambda_descr * 2) / self.n_hist)
                neart = np.argmin(difft, axis=0)
                neart_val = np.min(difft, axis=0)

                comb = np.logical_and(neary[None, :, :], nearx[:, None, :])
                positions = np.where(comb)
                weights = (1 - (self.n_hist/ (2*self.lambda_descr)) * diffy[positions[1], positions[2]]) \
                          * (1 - (self.n_hist/ (2*self.lambda_descr)) * diffx[positions[0], positions[2]]) \
                          * (1 - (self.n_bins/ (2*np.pi)) * neart_val[positions[2]]) \
                          * mag[positions[2]]
                np.add.at(histograms, (positions[0], positions[1], neart[positions[2]]), weights)
                histograms = histograms.flatten()
                histograms = np.minimum(histograms, 0.2 * np.linalg.norm(histograms))
                descriptor = np.array(np.minimum(np.floor((512 * histograms) / np.linalg.norm(histograms)), 255), dtype=np.int)
                keypoint_des[key_count, :] = descriptor
                key_count += 1
        return (keypoint_coo, keypoint_des)

    def detect(self, image):
        """Detect the keypoints in every octave.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)

        self.n_octaves = self._numberOfOctaves(self.n_octaves, image.shape)

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

        """
        check_nD(image, 2)

        self.n_octaves = self._numberOfOctaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        gradientSpace = [np.gradient(octave) for octave in gaussian_scalespace]

        descriptors = self._descriptor(keypoints, gradientSpace)

        self.keypoints = keypoints
