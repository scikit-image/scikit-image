import math
import warnings

import numpy as np
import scipy.ndimage as ndi

from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import (FeatureDetector, DescriptorExtractor)
from ..filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram


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
    h01 = 0.25 * (d[p0 + 1, p1 + 1, p2] - d[p0 - 1, p1 + 1, p2]
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
        self.deltas = self._deltas()
        self.scalespace_sigmas = None
        self.keypoints = None
        self.positions = None
        self.sigmas = None
        self.scales = None
        self.orientations = None
        self.octaves = None
        self.descriptors = None

    def _number_of_octaves(self, n, image_shape):
        size_min = 12  # minimum size of last octave
        s0 = min(image_shape) * self.upsampling
        max_octaves = int(math.log2(s0 / size_min) + 1)
        if max_octaves < n:
            warnings.warn(
                f"Reducing n_octaves to {max_octaves} due to small image size."
            )
            n = max_octaves
        return n

    def _deltas(self, dtype=float):
        deltas = self.delta_min * np.power(2, np.arange(self.n_octaves))
        return deltas.astype(dtype, copy=False)

    def _create_scalespace(self, image):
        """Source: "Anatomy of the SIFT Method" Alg. 1
        Construction of the scalespace by gradually blurring (scales) and
        downscaling (octaves) the image.
        """
        scalespace = []
        dtype = image.dtype
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

        # Eq. 10:  sigmas.shape = (n_octaves, n_scales + 3).
        # The three extra scales are:
        #    One for the differences needed for DoG and two auxilliary
        #    images (one at either end) for peak_local_max with exclude
        #    border = True (see Fig. 5)
        # The smoothing doubles after n_scales steps.
        tmp = np.power(2, np.arange(self.n_scales + 3) / self.n_scales)
        tmp *= self.sigma_min
        sigmas = (self.deltas[:, np.newaxis]
                  / self.deltas[0] * tmp[np.newaxis, :])
        self.scalespace_sigmas = sigmas

        # Eq. 7: Gaussian smoothing depends on difference with previous sigma
        #        gaussian_sigmas.shape = (n_octaves, n_scales + 2)
        var_diff = np.diff(sigmas * sigmas, axis=1)
        gaussian_sigmas = np.sqrt(var_diff) / self.deltas[:, np.newaxis]

        # after n_scales steps we doubled the smoothing
        k = 2 ** (1 / self.n_scales)
        # one octave is represented by a 3D image with depth (n_scales+x)
        for o in range(self.n_octaves):
            # Temporarily put scales axis first so octave[i] is C-contiguous
            # (this makes Gaussian filtering faster).
            octave = np.empty((self.n_scales + 3,) + image.shape, dtype=dtype,
                              order='C')
            octave[0] = image
            for s in range(1, self.n_scales + 3):
                # blur new scale assuming sigma of the last one
                gaussian(octave[s - 1],
                         gaussian_sigmas[o, s - 1],
                         mode='reflect', output=octave[s])
            # move scales to last axis as expected by other methods
            scalespace.append(np.moveaxis(octave, 0, -1))
            if o < self.n_octaves - 1:
                # downscale the image by taking every second pixel
                image = octave[self.n_scales][::2, ::2]
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
            keys = _local_max(np.ascontiguousarray(octave), threshold)
            if keys.size == 0:
                continue

            # localize extrema
            oshape = octave.shape
            # mask for all extrema that still have to be tested
            refinement_iterations = 5
            offset_max = 0.6
            for i in range(refinement_iterations):
                if i > 0:
                    # exclude any keys that have moved out of bounds
                    keys = keys[self._inrange(keys, oshape), :]

                # Jacobian and Hessian of all extrema
                grad = _sparse_gradient(octave, keys)
                hess = _hessian(octave, keys)

                # solve for offset of the extremum
                off = _offsets(grad, hess)
                if i == refinement_iterations - 1:
                    break
                # offset is too big and an increase would not bring us out of
                # bounds
                wrong_position_pos = np.logical_and(
                    off > offset_max,
                    keys + 1 < tuple([a - 1 for a in oshape])
                )
                wrong_position_neg = np.logical_and(off < -offset_max, keys - 1 > 0)
                if (not np.any(np.logical_or(wrong_position_neg,
                                             wrong_position_pos))):
                    break
                keys[wrong_position_pos] += 1
                keys[wrong_position_neg] -= 1

            # mask for all extrema that have been localized successfully
            finished = np.all(np.abs(off) < offset_max, axis=1)
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

        if not extrema_pos:
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels.")

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
        gradient_space = []
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

            gradient_space.append(np.gradient(octave))
            delta = self.deltas[o]
            oshape = octave.shape[0:2]
            # convert to octave's dimensions
            yx = positions / delta
            sigma = sigmas / delta

            # dimensions of the patch
            radius = 3 * self.lambda_ori * sigma
            p_min = np.maximum(0, yx - radius[:, np.newaxis] + 0.5).astype(int)
            p_max = np.minimum(yx + radius[:, np.newaxis] + 0.5,
                               (oshape[0] - 1, oshape[1] - 1)).astype(int)
            # orientation histogram
            hist = np.empty(self.n_bins, dtype=octave.dtype)
            avg_kernel = np.full((3,), 1 / 3, dtype=octave.dtype)
            for k in range(len(yx)):
                if np.all(p_min[k] > 0) and np.all(p_max[k] > p_min[k]):
                    hist[:] = 0

                    # use the patch coordinates to get the gradient and then
                    # normalize them
                    r, c = np.meshgrid(np.arange(p_min[k, 0], p_max[k, 0] + 1),
                                       np.arange(p_min[k, 1], p_max[k, 1] + 1),
                                       indexing='ij', sparse=True)
                    gradient_row = gradient_space[o][0][r, c, scales[k]]
                    gradient_col = gradient_space[o][1][r, c, scales[k]]
                    r = r - yx[k, 0]
                    c = c - yx[k, 1]

                    # gradient magnitude and angles
                    magnitude = np.sqrt(np.square(gradient_row) + np.square(
                        gradient_col))
                    theta = np.mod(np.arctan2(gradient_col, gradient_row),
                                   2 * np.pi)

                    # more weight to center values
                    kernel = np.exp(np.divide(r * r + c * c,
                                              -2 * (self.lambda_ori
                                                    * sigma[k]) ** 2))

                    # fill the histogram
                    bins = np.floor((theta / (2 * np.pi) * self.n_bins + 0.5)
                                    % self.n_bins).astype(np.int)
                    np.add.at(hist, bins, kernel * magnitude)

                    # smooth the histogram and find the maximum
                    hist = np.concatenate((hist[-3:], hist, hist[:3]))
                    for _ in range(6):  # number of smoothings
                        hist = np.convolve(hist, avg_kernel, mode='same')
                    hist = hist[3:-3]
                    max_filter = ndi.maximum_filter(hist, [3])
                    # if an angle is in 80% percent range of the maximum, a
                    # new keypoint is created for it
                    maxima = np.nonzero(np.logical_and(
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
        # return the gradient_space to reuse it to find the descriptor
        return gradient_space

    def _rotate(self, row, col, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rot_row = c * row + s * col
        rot_col = -s * row + c * col
        return rot_row, rot_col

    def _compute_descriptor(self, gradient_space):
        """Source: "Anatomy of the SIFT Method" Alg. 12
        Calculates the descriptor for every keypoint
        """
        n_key = len(self.scales)
        self.descriptors = np.empty((n_key, self.n_hist ** 2 * self.n_ori),
                                    dtype=np.uint8)
        # indices of the histograms
        hists = np.arange(1, self.n_hist + 1)
        # indices of the bins
        bins = np.arange(1, self.n_ori + 1)

        key_numbers = np.arange(n_key)
        for o in range(self.n_octaves):
            in_oct = self.octaves == o
            if not np.any(in_oct):
                continue
            positions = self.positions[in_oct]
            scales = self.scales[in_oct]
            sigmas = self.sigmas[in_oct]
            orientations = self.orientations[in_oct]
            numbers = key_numbers[in_oct]
            gradient = gradient_space[o]

            delta = self.deltas[o]
            dim = gradient[0].shape[0:2]
            center_pos = positions / delta
            sigma = sigmas / delta

            # dimensions of the patch
            radius = self.lambda_descr * (1 + 1 / self.n_hist) * sigma
            radius_patch = math.sqrt(2) * radius
            p_min = np.asarray(
                np.maximum(0, center_pos - radius_patch[:, np.newaxis] + 0.5),
                dtype=int)
            p_max = np.asarray(
                np.minimum(center_pos + radius_patch[:, np.newaxis] + 0.5,
                           (dim[0] - 1, dim[1] - 1)), dtype=int)

            for k in range(len(p_max)):
                rad_k = radius[k]
                ori = orientations[k]
                histograms = np.zeros((self.n_hist, self.n_hist, self.n_ori))
                # the patch
                r, c = np.meshgrid(np.arange(p_min[k, 0], p_max[k, 0]),
                                   np.arange(p_min[k, 1], p_max[k, 1]),
                                   indexing='ij', sparse=True)
                # normalized coordinates
                r_norm = r - center_pos[k, 0]
                c_norm = c - center_pos[k, 1]
                r_norm, c_norm = self._rotate(r_norm, c_norm, ori)

                # select coordinates and gradient values within the patch
                inside = np.maximum(np.abs(r_norm), np.abs(c_norm)) < rad_k
                r_norm, c_norm = r_norm[inside], c_norm[inside]
                r_idx, c_idx = np.nonzero(inside)
                r = r[r_idx, 0]
                c = c[0, c_idx]
                gradient_row = gradient[0][r, c, scales[k]]
                gradient_col = gradient[1][r, c, scales[k]]
                # compute the (relative) gradient orientation
                theta = np.arctan2(gradient_col, gradient_row) - ori
                lam_sig = self.lambda_descr * sigma[k]
                # Gaussian weighted kernel magnitude
                kernel = np.exp((r_norm * r_norm + c_norm * c_norm)
                                / (-2 * lam_sig ** 2))
                magnitude = np.sqrt(gradient_row * gradient_row
                                    + gradient_col * gradient_col) * kernel

                lam_sig_ratio = 2 * lam_sig / self.n_hist
                rc_bins = (hists - (1 + self.n_hist) / 2) * lam_sig_ratio
                rc_bin_spacing = lam_sig_ratio
                ori_bins = (2 * np.pi * bins) / self.n_ori

                # distances to the histograms and bins
                dist_r = np.abs(np.subtract.outer(rc_bins, r_norm))
                dist_c = np.abs(np.subtract.outer(rc_bins, c_norm))

                # the orientation histograms/bins that get the contribution
                ori_bins = ori_bins.astype(np.float64, copy=False)
                theta = theta.astype(np.float64, copy=False)
                near_t, near_t_val = _ori_distances(ori_bins, theta)

                # create the histogram
                n_patch = len(magnitude)
                _update_histogram(histograms, near_t, magnitude,
                                  near_t_val, dist_r, dist_c,
                                  n_patch, self.n_hist, self.n_ori,
                                  rc_bin_spacing)

                # convert the histograms to a 1d descriptor
                histograms = histograms.flatten()
                # saturate the descriptor
                histograms = np.minimum(histograms,
                                        0.2 * np.linalg.norm(histograms))
                # normalize the descriptor
                descriptor = (512 * histograms) / np.linalg.norm(histograms)
                # quantize the descriptor
                descriptor = np.minimum(np.floor(descriptor), 255)
                self.descriptors[numbers[k], :] = descriptor

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
        self.deltas = self._deltas(float_dtype)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in
                          gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape)

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
        self.deltas = self._deltas(float_dtype)

        gaussian_scalespace = self._create_scalespace(image)

        gradient_space = [np.gradient(octave) for octave in
                          gaussian_scalespace]

        self._compute_descriptor(gradient_space)

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
        self.deltas = self._deltas(float_dtype)

        gaussian_scalespace = self._create_scalespace(image)

        dog_scalespace = [np.diff(layer, axis=2) for layer in
                          gaussian_scalespace]

        positions, scales, sigmas, octaves = self._find_localize_evaluate(
            dog_scalespace, image.shape)
        if len(positions) == 0:
            raise RuntimeError(
                "SIFT found no features. Try passing in an image containing "
                "greater intensity contrasts between adjacent pixels.")

        gradient_space = self._compute_orientation(positions, scales, sigmas,
                                                   octaves,
                                                   gaussian_scalespace)

        self._compute_descriptor(gradient_space)

        self.keypoints = self.positions.round().astype(np.int)
