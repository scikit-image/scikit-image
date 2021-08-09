import math
import warnings

import numpy as np
import scipy.ndimage as ndi

from ..feature.util import (FeatureDetector, DescriptorExtractor)
from ..feature import peak_local_max
from ..util import img_as_float
from .._shared.utils import check_nD, _supported_float_type
from ..transform import rescale
from ..filters import gaussian


def _edgeness(hxx, hyy, hxy):
    """Compute edgeness (eq. 18 of Otero et. al. IPOL paper)"""
    trace = hxx + hyy
    determinant = hxx * hyy - hxy * hxy
    return (trace * trace) / determinant


def _sparse_gradient(vol, positions):
    """Gradient of a 3D volume at the provided `positions`.

    For SIFT we only need the gradient at specific positions and do not need
    the gradient at the edge positions, so can just use this simple
    implementation instead of numpy.gradient.
    """
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    g0 = vol[p0 + 1, p1, p2] - vol[p0 - 1, p1, p2]
    g0 *= 0.5
    g1 = vol[p0, p1 + 1, p2] - vol[p0, p1 - 1, p2]
    g1 *= 0.5
    g2 = vol[p0, p1, p2 + 1] - vol[p0, p1, p2 - 1]
    g2 *= 0.5
    return g0, g1, g2


def _hessian(d, positions):
    """Source: "Anatomy of the SIFT Method"  p.380 (13)"""
    p0 = positions[..., 0]
    p1 = positions[..., 1]
    p2 = positions[..., 2]
    two_d0 = 2 * d[p0, p1, p2]
    # 0 = row, 1 = col, 2 = octave
    h00 = d[p0 - 1, p1, p2] + d[p0 + 1, p1, p2] - two_d0
    h11 = d[p0, p1 - 1, p2] + d[p0, p1 + 1, p2] - two_d0
    h22 = d[p0, p1, p2 - 1] + d[p0, p1, p2 + 1] - two_d0
    h01 = 0.25 * (d[p0 + 1, p1 + 1, p2]- d[p0 - 1, p1 + 1, p2]
                  - d[p0 + 1, p1 - 1, p2] + d[p0 - 1, p1 - 1, p2])
    h02 = 0.25 * (d[p0 + 1, p1, p2 + 1] - d[p0 + 1, p1, p2 - 1]
                  + d[p0 - 1, p1, p2 - 1] - d[p0 - 1, p1, p2 + 1])
    h12 = 0.25 * (d[p0, p1 + 1, p2 + 1] - d[p0, p1 + 1, p2 - 1]
                  + d[p0, p1 - 1, p2 - 1] - d[p0, p1 - 1, p2 + 1])
    return (h00, h11, h22, h01, h02, h12)


def _offsets(grad, hess):
    """Compute position refinement offsets from gradient and Hessian."""
    h00, h11, h22, h01, h02, h12 = hess
    g0, g1, g2 = grad
    det = h00 * h11 * h22
    det -= h00 * h12 * h12
    det -= h01 * h01 * h22
    det += 2 * h01 * h02 * h12
    det -= h02 * h02 * h11
    aa = (h11*h22 - h12*h12) / det
    ab = (h02*h12 - h01*h22) / det
    ac = (h01*h12 - h02*h11) / det
    bb = (h00*h22 - h02*h02) / det
    bc = (h01*h02 - h00*h12) / det
    cc = (h00*h11 - h01*h01) / det
    offset0 = -aa * g0 - ab * g1 - ac * g2
    offset1 = -ab * g0 - bb * g1 - bc * g2
    offset2 = -ac * g0 - bc * g1 - cc * g2
    return np.stack((offset0, offset1, offset2), axis=-1)


class SIFT(FeatureDetector, DescriptorExtractor):
    """SIFT feature detection and descriptor extraction.

        Parameters
        ----------
        upsampling : int, optional
            Prior to the feature detection the image is upscaled by a factor
            of 1 (no upscaling), 2 or 4. Method: Bi-cubic interpolation.
        n_octaves : int, optional
            Maximum number of octaves. With every octave the image size is
            halved and the sigma doubled.
        n_scales : int, optional
            Maximum number of scales in every octave.
        sigma_min : float, optional
            The blur level of the seed image. If upsampling is enabled
            sigma_min is scaled by factor 1/upsampling
        sigma_in : float, optional
            The assumed blur level of the input image.
        c_dog : float, optional
            Threshold to discard low contrast extrema in the DoG. It's final
            value is dependent on n_scales by the relation:
            final_c_dog = (2^(1/n_scales)-1) / (2^(1/3)-1) * c_dog
        c_edge : float, optional
            Threshold to discard extrema that lie in edges. If H is the
            Hessian of an extremum, its "edgeness" is described by
            tr(H)²/det(H). If the edgeness is higher than
            (c_edge + 1)²/c_edge, the extremum is discarded.
        n_bins : int, optional
            Number of bins in the histogram that describes the gradient
            orientations around keypoint.
        lambda_ori : float, optional
            The window used to find the reference orientation of a keypoint
            has a width of 6 * lambda_ori * sigma and is weighted by a
            standard deviation of 2 * lambda_ori * sigma.
        c_max : float, optional
            The threshold at which a secondary peak in the orientation
            histogram is accepted as orientation
        lambda_descr : float, optional
            The window used to define the descriptor of a keypoint has a width
            of 2 * lambda_descr * sigma * (n_hist+1)/n_hist and is weighted by
            a standard deviation of lambda_descr * sigma.
        n_hist : int, optional
            The window used to define the descriptor of a keypoint consists of
            n_hist * n_hist histograms.
        n_ori : int, optional
            The number of bins in the histograms of the descriptor patch.


        Attributes
        ----------
        delta_min : float
            The sampling distance of the first octave. It's final value is
            1/upsampling.
        deltas : (n_octaves, ) array
            The sampling distances of all octaves.
        scalespace_sigmas : (n_octaves, n_scales + 3) array
            The sigma value of all scales in all octaves.
        keypoints : (N, 2) array
            Keypoint coordinates as ``(row, col)``.
        positions : (N, 2) array
            Subpixel-precision keypoint coordinates as ``(row, col)``.
        sigmas : (N, ) array
            The corresponding sigma (blur) value of a keypoint.
        sigmas : (N, ) array
            The corresponding scale of a keypoint.
        orientations : (N, ) array
            The orientations of the gradient around every keypoint.
        octaves : (N, ) array
            The corresponding octave of a keypoint.
        descriptors : (N, n_hist*n_hist*n_ori) array
            The descriptors of a keypoint.


        References
        ----------
        .. [1] D.G. Lowe
              "Object recognition from local scale-invariant features"
              https://doi.org/10.1109/ICCV.1999.790410

        .. [2] Ives Rey Otero and Mauricio Delbracio
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
        ...                             detector_extractor2.descriptors,
        ...                             max_ratio=0.6)
        >>> matches[10:15]
        array([[ 11,  11],
               [ 12, 568],
               [ 13,  13],
               [ 14, 569],
               [ 15,  15]])
        >>> detector_extractor1.keypoints[matches[10:15, 0]]
        array([[170, 241],
               [341, 287],
               [234,  13],
               [232, 378],
               [206, 307]])
        >>> detector_extractor2.keypoints[matches[10:15, 1]]
        array([[271, 170],
               [383,  95],
               [499, 234],
               [191, 260],
               [205, 206]])

        """

    def __init__(self, upsampling=2, n_octaves=8, n_scales=3, sigma_min=1.6,
                 sigma_in=0.5,
                 c_dog=0.04 / 3, c_edge=10, n_bins=36, lambda_ori=1.5,
                 c_max=0.8, lambda_descr=6, n_hist=4, n_ori=8):
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
        self.deltas = (self.delta_min
                       * np.power(2, np.arange(self.n_octaves - 1)))
        self.scalespace_sigmas = None
        self.keypoints = None
        self.positions = None
        self.sigmas = None
        self.scales = None
        self.orientations = None
        self.octaves = None
        self.descriptors = None

    def _number_of_octaves(self, n, image_shape):
        sMin = 12  # minimum size of last octave
        s0 = min(image_shape) * self.upsampling
        max_octaves = int(math.log2(s0 / sMin) + 1)
        if max_octaves < n:
            warnings.warn(
                f"Reducing n_octaves to {max_octaves} due to small image size."
            )
            n = max_octaves
        return n

    def _create_scalespace(self, image):
        """Source: "Anatomy of the SIFT Method" Alg. 1
        Construction of the scalespace by gradually blurring (scales) and
        downscaling (octaves) the image.
        """
        scalespace = []
        dtype = image.dtype
        self.deltas = self.deltas.astype(dtype)
        if self.upsampling > 1:
            image = rescale(image, self.upsampling, order=3)

        # all sigmas for the gaussian scalespace
        sigmas = np.empty((self.n_octaves,
                           self.n_scales + 3), dtype=dtype)
        current_sigma = self.sigma_min

        # smooth to sigma_min, assuming sigma_in
        image = gaussian(image,
                         (1 / self.delta_min) * math.sqrt(
                             self.sigma_min ** 2 - self.sigma_in ** 2),
                         mode='reflect')

        # after n_scales steps we doubled the smoothing
        k = 2 ** (1 / self.n_scales)
        # one octave is represented by a 3D image with depth (n_scales+x)
        for o in range(self.n_octaves):
            delta = self.delta_min * 2 ** o
            sigmas[o, 0] = current_sigma
            octave = np.empty(image.shape + (self.n_scales + 3,), dtype=dtype)
            octave[:, :, 0] = image
            for s in range(1, self.n_scales + 3):
                # blur new scale assuming sigma of the last one
                octave[:, :, s] = gaussian(octave[..., s - 1],
                                           ((1 / delta)
                                            * np.sqrt((current_sigma * k) ** 2
                                                      - current_sigma ** 2)),
                                           mode='reflect')
                current_sigma = current_sigma * k
                sigmas[o, s] = current_sigma
            scalespace.append(octave)
            # downscale the image by taking every second pixel
            image = octave[:, :, self.n_scales][::2, ::2]
            current_sigma = sigmas[o, self.n_scales]
        self.scalespace_sigmas = sigmas
        return scalespace

    def _inrange(self, a, dim):
        return ((a[:, 0] > 0) & (a[:, 0] < dim[0] - 1)
                & (a[:, 1] > 0) & (a[:, 1] < dim[1] - 1))

    def _find_localize_evaluate(self, dogspace, img_shape):
        """Source: "Anatomy of the SIFT Method" Alg. 4-9
        1) first find all extrema of a (3, 3, 3) neighborhood
        2) use second order Taylor development to refine the positions to
           sub-pixel precision
        3) filter out extrema that have low contrast and lie on edges or close
           to the image borders
        """
        extrema_pos = []
        extrema_scales = []
        extrema_sigmas = []
        threshold = self.c_dog * 0.8
        dtype = dogspace[0].dtype
        for o, (octave, delta) in enumerate(zip(dogspace, self.deltas)):
            # find extrema
            maxima = peak_local_max(octave, threshold_abs=threshold)
            minima = peak_local_max(-octave, threshold_abs=threshold)
            keys = np.concatenate((maxima, minima), axis=0)

            # localize extrema
            oshape = octave.shape
            # mask for all extrema that still have to be tested
            for i in range(5):
                if i > 0:
                    # exclude any keys that have moved out of bounds
                    keys = keys[self._inrange(keys, oshape), :]

                # Jacobian and Hessian of all extrema
                grad = _sparse_gradient(octave, keys)
                hess = _hessian(octave, keys)

                # solve for offset of the extremum
                off = _offsets(grad, hess)
                if i == 4:
                    break
                # offset is too big and an increase would not bring us out of
                # bounds
                wrong_position_pos = np.logical_and(
                    off > 0.5,
                    keys + 1 < tuple([a - 1 for a in oshape])
                )
                wrong_position_neg = np.logical_and(off < -0.5, keys - 1 > 0)
                if (not np.any(np.logical_or(wrong_position_neg,
                                             wrong_position_pos))):
                    break
                keys[wrong_position_pos] += 1
                keys[wrong_position_neg] -= 1

            # mask for all extrema that have been localized successfully
            finished = np.all(np.abs(off) < 0.5, axis=1)
            keys = keys[finished]
            off = off[finished]
            grad = [g[finished] for g in grad]

            # value of extremum in octave
            vals = octave[keys[:, 0], keys[:, 1], keys[:, 2]]
            # values at interpolated point
            w = vals
            for i in range(3):
                w += 0.5 * grad[i] * off[:, i]

            h00, h11, h01 = \
                hess[0][finished], hess[1][finished], hess[3][finished]

            sigmaratio = (self.scalespace_sigmas[0, 1]
                          / self.scalespace_sigmas[0, 0])

            # filter for contrast, edgeness and borders
            contrast_threshold = self.c_dog
            contrast_filter = np.abs(w) > contrast_threshold

            edge_threshold = np.square(self.c_edge + 1) / self.c_edge
            edge_response = _edgeness(h00[contrast_filter],
                                      h11[contrast_filter],
                                      h01[contrast_filter])
            edge_filter = np.abs(edge_response) <= edge_threshold

            keys = keys[contrast_filter][edge_filter]
            off = off[contrast_filter][edge_filter]
            yx = ((keys[:, 0:2] + off[:, 0:2]) * delta).astype(dtype)

            sigmas = self.scalespace_sigmas[o, keys[:, 2]] * np.power(
                sigmaratio, off[:, 2])
            border_filter = np.all(
                np.logical_and((yx - sigmas[:, np.newaxis]) > 0.0,
                               (yx + sigmas[:, np.newaxis]) < img_shape),
                axis=1)
            extrema_pos.append(yx[border_filter])
            extrema_scales.append(keys[border_filter, 2])
            extrema_sigmas.append(sigmas[border_filter])

        octave_indices = np.concatenate([np.full(len(p), i)
                                        for i, p in enumerate(extrema_pos)])
        extrema_pos = np.concatenate(extrema_pos)
        extrema_scales = np.concatenate(extrema_scales)
        extrema_sigmas = np.concatenate(extrema_sigmas)
        return extrema_pos, extrema_scales, extrema_sigmas, octave_indices

    def _fit(self, h):
        """Refine the position of the peak by fitting it to a parabola"""
        return (h[0] - h[2]) / (2 * (h[0] + h[2] - 2 * h[1]))

    def _compute_orientation(self, positions_oct, scales_oct, sigmas_oct,
                             octaves, gaussian_scalespace):
        """Source: "Anatomy of the SIFT Method" Alg. 11
        Calculates the orientation of the gradient around every keypoint
        """
        gradientSpace = []
        # list for keypoints that have more than one reference orientation
        keypoint_indices = []
        keypoint_angles = []
        keypoint_octave = []
        keypoints_valid = np.ones_like(sigmas_oct, dtype=bool)
        orientations = np.zeros_like(sigmas_oct, dtype=positions_oct.dtype)
        key_count = 0
        for o in range(self.n_octaves):
            in_oct = octaves == o
            positions = positions_oct[in_oct]
            scales = scales_oct[in_oct]
            sigmas = sigmas_oct[in_oct]
            octave = gaussian_scalespace[o]

            gradientSpace.append(np.gradient(octave))
            delta = self.deltas[o]
            dim = octave.shape[0:2]
            yx = positions / delta  # convert to octaves dimensions
            sigma = sigmas / delta

            # dimensions of the patch
            radius = 3 * self.lambda_ori * sigma
            Min = np.maximum(0, np.add(np.subtract(yx, radius[:, np.newaxis]),
                                       0.5)).astype(np.int)
            Max = np.minimum(yx + radius[:, np.newaxis] + 0.5,
                             (dim[0] - 1, dim[1] - 1)).astype(np.int)

            for k in range(len(yx)):
                if np.all(Min[k] > 0) and np.all(Max[k] > Min[k]):
                    hist = np.zeros(self.n_bins)  # orientation histogram

                    # use the patch coordinates to get the gradient and then
                    # normalize them
                    m, n = np.meshgrid(np.arange(Min[k, 0], Max[k, 0] + 1),
                                       np.arange(Min[k, 1], Max[k, 1] + 1),
                                       indexing='ij', sparse=True)
                    gradientY = gradientSpace[o][0][m, n, scales[k]]
                    gradientX = gradientSpace[o][1][m, n, scales[k]]
                    m = m - yx[k, 0]
                    n = n - yx[k, 1]

                    magnitude = np.sqrt(np.square(gradientY) + np.square(
                        gradientX))  # gradient magnitude
                    theta = np.mod(np.arctan2(gradientX, gradientY),
                                   2 * np.pi)  # angles
                    # more weight to center values
                    kernel = np.exp(-np.divide(np.add(np.square(n),
                                                      np.square(m)),
                                               2 * (self.lambda_ori
                                                    * sigma[k]) ** 2))

                    # fill the histogram
                    bins = np.floor((theta / (2 * np.pi) * self.n_bins + 0.5)
                                    % self.n_bins).astype(np.int)
                    np.add.at(hist, bins, kernel * magnitude)

                    # smooth the histogram and find the maximum
                    hist = np.concatenate((hist[-3:], hist, hist[:3]))
                    avg_kernel = np.full((3,), 1 / 3)
                    for _ in range(6):  # number of smoothings
                        hist = np.convolve(hist, avg_kernel, mode='same')
                    hist = hist[3:-3]
                    max_filter = ndi.maximum_filter(hist, [3])
                    # if an angle is in 80% percent range of the maximum, a
                    # new keypoint is created for it
                    maxima = np.where(np.logical_and(
                        hist >= (self.c_max * np.max(hist)),
                        max_filter == hist))

                    # save the angles
                    for c, m in enumerate(maxima[0]):
                        neigh = np.arange(m - 1, m + 2) % len(hist)
                        # use neighbors to fit a parabola, to get more accurate
                        # result
                        ori = ((m + self._fit(hist[neigh]) + 0.5)
                               * 2 * np.pi / self.n_bins)
                        if ori > np.pi:
                            ori -= 2 * np.pi
                        if c == 0:
                            orientations[key_count] = ori
                        else:
                            keypoint_indices.append(key_count)
                            keypoint_angles.append(ori)
                            keypoint_octave.append(o)
                else:
                    keypoints_valid[key_count] = False
                key_count += 1
        self.positions = np.concatenate(
            (positions_oct[keypoints_valid], positions_oct[keypoint_indices]))
        self.scales = np.concatenate(
            (scales_oct[keypoints_valid], scales_oct[keypoint_indices]))
        self.sigmas = np.concatenate(
            (sigmas_oct[keypoints_valid], sigmas_oct[keypoint_indices]))
        self.orientations = np.concatenate(
            (orientations[keypoints_valid], keypoint_angles))
        self.octaves = np.concatenate(
            (octaves[keypoints_valid], keypoint_octave))
        # return the gradientspace to reuse it to find the descriptor
        return gradientSpace

    def _rotate(self, y, x, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rY = c * y - s * x
        rX = s * y + c * x
        return rY, rX

    def _compute_descriptor(self, gradientspace):
        """Source: "Anatomy of the SIFT Method" Alg. 12
        Calculates the descriptor for every keypoint
        """
        nKey = len(self.scales)
        self.descriptors = np.empty((nKey, self.n_hist ** 2 * self.n_ori),
                                    dtype=np.uint8)
        key_count = 0
        key_numbers = np.arange(nKey)
        for o in range(self.n_octaves):
            in_oct = self.octaves == o
            if np.count_nonzero(in_oct) == 0:
                continue
            positions = self.positions[in_oct]
            scales = self.scales[in_oct]
            sigmas = self.sigmas[in_oct]
            orientations = self.orientations[in_oct]
            numbers = key_numbers[in_oct]
            gradient = gradientspace[o]

            delta = self.deltas[o]
            dim = gradient[0].shape[0:2]
            yx = positions / delta
            sigma = sigmas / delta

            # dimensions of the patch
            radius = self.lambda_descr * (1 + 1 / self.n_hist) * sigma
            radiusPatch = np.sqrt(2) * radius
            Min = np.array(
                np.maximum(0, yx - radiusPatch[:, np.newaxis] + 0.5),
                dtype=int)
            Max = np.array(
                np.minimum(yx + radiusPatch[:, np.newaxis] + 0.5,
                           (dim[0] - 1, dim[1] - 1)), dtype=int)

            # indices of the histograms
            hists = np.arange(1, self.n_hist + 1)
            # indices of the bins
            bins = np.arange(1, self.n_ori + 1)
            for k in range(len(Max)):
                histograms = np.zeros((self.n_hist, self.n_hist, self.n_ori))
                # the patch
                m, n = np.meshgrid(np.arange(Min[k, 0], Max[k, 0]),
                                   np.arange(Min[k, 1], Max[k, 1]),
                                   indexing='ij', sparse=True)
                y_mn = m - yx[k, 0]  # normalized coordinates
                x_mn = n - yx[k, 1]
                y_mn, x_mn = self._rotate(y_mn, x_mn, -orientations[k])

                inRadius = np.maximum(np.abs(y_mn), np.abs(x_mn)) < radius[k]
                y_mn, x_mn = y_mn[inRadius], x_mn[inRadius]
                m_idx, n_idx = np.where(inRadius)
                m = m[m_idx, 0]
                n = n[0, n_idx]
                gradientY = gradient[0][m, n, scales[k]]
                gradientX = gradient[1][m, n, scales[k]]

                theta = np.arctan2(gradientX, gradientY) - orientations[k]
                lam_sig = self.lambda_descr * sigma[k]
                lam_sig_ratio = 2 * lam_sig / self.n_hist
                kernel = np.exp(-(np.square(y_mn) + np.square(x_mn))
                                / (2 * lam_sig ** 2))
                magnitude = np.sqrt(np.square(gradientY)
                                    + np.square(gradientX)) * kernel

                yj_xi = (hists - (1 + self.n_hist) / 2) * lam_sig_ratio
                ok = (2 * np.pi * bins) / self.n_ori

                # distances to the histograms and bins
                dist_y = np.abs(np.subtract.outer(yj_xi, y_mn))
                dist_x = np.abs(np.subtract.outer(yj_xi, x_mn))
                dist_t = np.abs(np.mod(np.subtract.outer(ok, theta),
                                       2 * np.pi))

                # the histograms/bins that get the contribution
                near_y = dist_y <= lam_sig_ratio
                near_x = dist_x <= lam_sig_ratio
                near_t = np.argmin(dist_t, axis=0)
                near_t_val = np.min(dist_t, axis=0)

                # every contribution in y direction is combined with every in
                # x direction
                # for example y: histogram 3 and 4, x: histogram 2
                # -> contribute to (3,2) and (4,2)
                comb = np.logical_and(near_x.T[:, None, :],
                                      near_y.T[:, :, None])
                comb_pos = np.where(comb)

                # the weights/contributions are shared bilinearly between the
                # histograms
                w0 = ((1 - (1 / lam_sig_ratio)
                       * dist_y[comb_pos[1], comb_pos[0]])
                      * (1 - (1 / lam_sig_ratio)
                         * dist_x[comb_pos[2], comb_pos[0]])
                      * magnitude[comb_pos[0]])

                # the weight is shared linearly between the 2 nearest bins
                w1 = w0 * ((self.n_ori / (2 * np.pi))
                           * near_t_val[comb_pos[0]])
                w2 = w0 * (1 - (self.n_ori / (2 * np.pi))
                           * near_t_val[comb_pos[0]])
                k_index = near_t[comb_pos[0]]
                np.add.at(histograms, (comb_pos[1], comb_pos[2], k_index), w1)
                np.add.at(histograms,
                          (comb_pos[1], comb_pos[2],
                           np.mod((k_index + 1), self.n_ori)),
                          w2)

                # convert the histograms to a 1d descriptor
                histograms = histograms.flatten()
                # saturate the descriptor
                histograms = np.minimum(histograms,
                                        0.2 * np.linalg.norm(histograms))
                # normalize the descriptor
                descriptor = np.minimum(
                    np.floor((512 * histograms) / np.linalg.norm(histograms)),
                    255).astype(np.uint8)
                self.descriptors[numbers[k], :] = descriptor
                key_count += 1

    def detect(self, image):
        """Detect the keypoints.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)
        image = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in
                          gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape)
        if len(positions) == 0:
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels.")

        self._compute_orientation(positions, scales, sigmas, octaves,
                                  gaussian_scalespace)

        self.keypoints = self.positions.round().astype(np.int)

    def extract(self, image):
        """Extract the descriptors for all keypoints in the image.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)
        image = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        gradientSpace = [np.gradient(octave) for octave in
                         gaussian_scalespace]

        self._compute_descriptor(gradientSpace)

    def detect_and_extract(self, image):
        """Detect the keypoints and extract their descriptors.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
        check_nD(image, 2)
        image = img_as_float(image)
        float_dtype = _supported_float_type(image.dtype)
        image = image.astype(float_dtype, copy=False)

        self.n_octaves = self._number_of_octaves(self.n_octaves, image.shape)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in
                          gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape)
        if len(positions) == 0:
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels.")

        gradientSpace = self._compute_orientation(positions, scales, sigmas,
                                                  octaves, gaussian_scalespace)

        self._compute_descriptor(gradientSpace)

        self.keypoints = self.positions.round().astype(np.int)
