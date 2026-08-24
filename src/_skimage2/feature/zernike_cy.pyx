#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport cos, sin
from cython.parallel cimport prange, threadid
from scipy.special.cython_special cimport binom
from scipy.linalg cimport cython_lapack

cnp.import_array()

cdef class ZernikeFeatures:
    """Extract conventional normalized features and complex-valued moments.

    This class implements computation of conventional normalized Zernike features
    and normalized complex-valued Zernike moments. It uses binomial representation
    computed using gamma functions as provided by scipy, which maybe approximations
    of factorials but are computationally stable. It does not use explicit
    factorials or recursions to compute R-polynomial coefficients [1].

    Parameters
    ----------
    image : ndarray of shape (M, N)
        A binary or grayscale image of size (M, N), scaled to [0., 1.], 64-bit floats.
    degree : int
        Zernike polynomial design parameter.
    radius : float
        Radius of the circular pupil/bbox over which features are computed.
    center_coord : ndarray of shape (2,)
        Center of the circular pupil/bbox in 2D image coordinates as 64-bit floats.
    num_threads : int, optional
        Parallel and independent computation of features over multiple threads/cores.

    References
    ----------
    .. [1] Kuo Niu and Chao Tian 2022 J. Opt. 24 123001.
            https://doi.org/10.1088/2040-8986/ac9e08
    .. [2] D. Otkupman, S. Bezdidko, V. Ostashenkova E3S Web of Conferences 310, 01002 (2021).
            https://doi.org/10.1051/e3sconf/202131001002
    .. [3] Michael Reed Teague J. Opt. Soc. Am., Vol. 70, No. 8, August 1980.
            https://doi.org/10.1364/JOSA.70.000920
    .. [4] Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Zernike_polynomials
    .. [5] Pseudo Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Pseudo-Zernike_polynomials
    .. [6] Image reconstruction example using Zernike moments
            https://stackoverflow.com/a/33339289
    """
    cdef int _degree, _num_features, _num_threads
    cdef float _radius
    cdef long _num_enclosed_pixels
    cdef Py_ssize_t _num_rows, _num_cols
    cdef cnp.int64_t[::1] _azimuthals_list # modes. m or l in ref. [1]
    cdef cnp.float64_t[::1] _scaling_factors_list # (n+1)/pi in ref. [1]
    cdef cnp.float64_t[::1] _coeffs_flat # coefficients of R in ref. [1]
    cdef cnp.int32_t[::1] _powvals_flat # (n-2s) in ref. [1]
    cdef cnp.int32_t[::1] _offsets # start index of coeffs list for each n
    cdef cnp.float64_t[:,::1] _valid_image # image within pupil
    cdef cnp.uint8_t[:, ::1] _pupil_mask # binary mask. 1 for within pupil, else 0
    cdef cnp.float64_t[:,::1] _distances_rho # normalized and centered pupil
    cdef cnp.float64_t[:,::1] _azimuthal_theta # quadrant aware angles

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        int degree,
        float radius,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
        int num_threads=4,
    ):
        """C-level initialization of parameters."""
        self._degree = degree
        self._radius = radius
        self._num_threads = num_threads if num_threads > 0 else 1
        self._num_rows, self._num_cols = image.shape[0], image.shape[1]
        self._num_enclosed_pixels = 0
        self._build_normalized_grid(image, center_coord)
        self._build_paired_data()

    cdef void _build_normalized_grid(
        self,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
    ):
        """Build centered and normalized pupil grid over the image."""
        cdef cnp.ndarray[cnp.float64_t, ndim=3] var_yx # cartesian coords
        cdef cnp.ndarray[cnp.float64_t, ndim=2] distances_rho, azimuthal_theta # polar coords
        cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] pupil_mask # allow up-cast

        # creating mesh grid of xy coordinates of image size (M, N)
        center = np.reshape(center_coord, (-1, 1, 1)) # (2, 1, 1) cuz of mgrid
        var_yx = np.mgrid[:self._num_rows, :self._num_cols].astype(np.float64) # (2, M, N)
        var_yx = (var_yx - center) / self._radius # center and normalize

        distances_rho = np.sqrt(np.sum(var_yx**2.0, axis=0)) # xy to rho, (2, M, N) -> (M, N)
        pupil_mask = (distances_rho <= 1.0).astype(np.uint8)
        valid_image = np.where(pupil_mask, image, 0.0) # suppress irrelevant pixels
        distances_rho = np.where(pupil_mask, distances_rho, 0.0) # suppress irr. distances
        azimuthal_theta = np.arctan2(var_yx[0, :, :], var_yx[1, :, :]) + np.pi
        azimuthal_theta = np.where(pupil_mask, azimuthal_theta, 0.0) # quadrant aware angles in rads

        self._num_enclosed_pixels = int(np.sum(pupil_mask)) # factor (n+1)/pi -> (n+1)/pixels
        self._distances_rho = np.ascontiguousarray(distances_rho) # contiguous arrays for fast access
        self._azimuthal_theta = np.ascontiguousarray(azimuthal_theta)
        self._pupil_mask = np.ascontiguousarray(pupil_mask)
        self._valid_image = np.ascontiguousarray(valid_image)

    cdef void _build_paired_data(self):
        """Build flattened lists for precomputed values."""

        # Paired data here means for one n in {0,...,degree+1}, the degree,
        # number of azimuthals (modes m/l), scaling factor value ((n+1)/numpix),
        # number of index s ((n-l)/2), binomial coefficient values, power values (n-2s)
        # are precomputed for fast iteration and access later in the main feature loop.

        cdef list degrees_list = [] # degree
        cdef list azimuthals_list = [] # m or l
        cdef list scaling_list = [] # (n+1)/numpix
        cdef list coeffs = [] # factorials replaced by binomials
        cdef list powvals = [] # (n-2s)
        cdef list offsets = [0] # each n-l degree has s polynomials, for flat list skip s polys
        cdef int one_degree, one_azimuthal, num, num_repeats, s
        cdef double one_sf, coeff, sign
        cdef long denom_pixels = self._num_enclosed_pixels if self._num_enclosed_pixels > 0 else 1

        for one_degree in range(self._degree + 1):
            # one_sf = (one_degree + 1) / np.pi # kept for reference only
            # one_sf = (one_degree + 1) / (<double>denom_pixels * np.pi) # kept for reference only
            one_sf = (one_degree + 1) / <double>denom_pixels
            for one_azimuthal in range(one_degree + 1): # modes m or l
                num = one_degree - one_azimuthal
                if (num >= 0) and (num % 2 == 0.0): # n-l condition for non-negative degree and even
                    degrees_list.append(one_degree)
                    azimuthals_list.append(one_azimuthal)
                    scaling_list.append(one_sf)
                    num_repeats = num // 2
                    for s in range(num_repeats + 1): # polynomial index s, (n-l)/2
                        sign = -1.0 if (s % 2) else 1.0
                        coeff = sign * binom(one_degree - s, s) * binom(one_degree - 2 * s, ((one_degree - one_azimuthal) // 2) - s)
                        coeffs.append(coeff)
                        powvals.append(one_degree - 2 * s)
                    offsets.append(len(coeffs))

        self._num_features = len(degrees_list)
        self._azimuthals_list = np.array(azimuthals_list, dtype=np.int64)
        self._scaling_factors_list = np.array(scaling_list, dtype=np.float64)
        self._coeffs_flat = np.array(coeffs, dtype=np.float64)
        self._powvals_flat = np.array(powvals, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)

    cdef tuple _compute_zernike_features(self):
        """Compute Zernike moments and features."""

        # A simplified explanation of the implementation:
        # From ref. [1], we wish to implement equation 156,
        # moment = f(rho,theta) * R(rho) * exp(j*theta)
        # it can be computed with nested loops as
        # for one_moment:
            # for one_row:
                # for one_col:
                    # moment = f(rho,theta) * R(rho) * exp(j*theta)

        # However, to avoid repeated calls to factorial, power operations
        # and for efficient memory use, the loop/axis are swapped as row, col, feature
        # and using R(rho), exp(j*theta) with precomputed, indexed and recursive values.

        # First, moment, is split into real and imaginary parts.

        # f(rho,theta) is merely pixel intensity at index [rho, theta].

        # R(rho) = sum(coefficients * rho^(n - 2s)). Here, coefficients were
        # precomputed into a flattened list and indexed with s and offset by
        # previous number of moments. rho^(n-2s) is extended from rho^0 to rho^n,
        # and appropriate rho^i is indexed by powvals.

        # exp(j*theta) is split into sine and cosine parts, and computed recursively
        # using cos(A+B), sin(A+B) and previous two cos/sin values.

        # parallel over image rows, not over features

        cdef Py_ssize_t num_rows = self._num_rows
        cdef Py_ssize_t num_cols = self._num_cols
        cdef int nfeatures = self._num_features
        cdef int degree = self._degree
        cdef int nthreads = self._num_threads
        cdef Py_ssize_t one_row, one_col
        cdef int th_id, one_feat, k, m, start, end # multithread index, coeffs[start:end]
        cdef double one_rho, one_theta, one_cos, one_sin, one_pixel, one_rad_poly

        # per-thread accumulators for independent and parallel computations.
        cdef cnp.ndarray acc_real_part = np.zeros((nthreads, nfeatures), dtype=np.float64) # acc/buf for real part
        cdef cnp.ndarray acc_imag_part = np.zeros((nthreads, nfeatures), dtype=np.float64) # acc/buf for imag part
        cdef cnp.ndarray acc_dist_rho_power = np.zeros((nthreads, degree + 1), dtype=np.float64)  # acc/buf for rho^(n)
        cdef cnp.ndarray acc_cosine = np.zeros((nthreads, degree + 1), dtype=np.float64) # acc/buf for recursive cos
        cdef cnp.ndarray acc_sine = np.zeros((nthreads, degree + 1), dtype=np.float64) # acc/buf for recursive sin

        # memory views for C-like access
        cdef double[:, ::1] acc_real_view = acc_real_part
        cdef double[:, ::1] acc_imag_view = acc_imag_part
        cdef double[:, ::1] acc_dist_rho_pow_view = acc_dist_rho_power
        cdef double[:, ::1] acc_cos_view = acc_cosine
        cdef double[:, ::1] acc_sin_view = acc_sine

        for one_row in prange(num_rows, nogil=True, num_threads=nthreads): # parallel over image rows
            th_id = threadid()
            for one_col in range(num_cols): # column axis
                if self._pupil_mask[one_row, one_col] == 0: # skip outer pixels
                    continue
                one_pixel = self._valid_image[one_row, one_col]
                if one_pixel == 0.0: # skip black pixels
                    continue
                one_rho = self._distances_rho[one_row, one_col]
                one_theta = self._azimuthal_theta[one_row, one_col]
                # build rho^(n-2s) using recurrence (rho^0, rho^1...rho^n) ref. [1]
                # powvals will index these rho values -> i=n-2s -> rho^i = rho[powval[i]]
                acc_dist_rho_pow_view[th_id, 0] = 1.0 # if n-2s=0
                for k in range(1, degree + 1):
                    acc_dist_rho_pow_view[th_id, k] = acc_dist_rho_pow_view[th_id, k - 1] * one_rho
                one_cos = cos(one_theta) # cos of current angle (B)
                one_sin = sin(one_theta) # sin of current angle (B)
                acc_cos_view[th_id, 0] = 1.0 # previous angle (A), start at A=0 radian
                acc_sin_view[th_id, 0] = 0.0 # previous angle (A), start at A=0 radian
                if degree >= 1:
                    acc_cos_view[th_id, 1] = one_cos
                    acc_sin_view[th_id, 1] = one_sin
                    for m in range(2, degree + 1):
                        # cos/sin of next angle relies on the previous angle and current angle
                        # cos(A+B)=cosAcosB - sinAsinB
                        acc_cos_view[th_id, m] = acc_cos_view[th_id, m - 1] * one_cos - acc_sin_view[th_id, m - 1] * one_sin
                        # sin(A+B)=sinAcosB + cosAsinB
                        acc_sin_view[th_id, m] = acc_sin_view[th_id, m - 1] * one_cos + acc_cos_view[th_id, m - 1] * one_sin
                for one_feat in range(nfeatures): # features axis
                    m = self._azimuthals_list[one_feat]
                    start = self._offsets[one_feat]
                    end = self._offsets[one_feat + 1]
                    one_rad_poly = 0.0
                    for k in range(start, end): # equation 14 ref. [1]
                        one_rad_poly += self._coeffs_flat[k] * acc_dist_rho_pow_view[th_id, self._powvals_flat[k]]
                    # moment = f(rho,theta) * R(rho) * exp(jtheta) ref. [1]
                    acc_real_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_cos_view[th_id, m]
                    acc_imag_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_sin_view[th_id, m]

        real_part_sum = acc_real_part.sum(axis=0) # (nths, nfs) -> (nfs,)
        imag_part_sum = acc_imag_part.sum(axis=0) # (nths, nfs) -> (nfs,)
        magnitude = np.sqrt(real_part_sum * real_part_sum + imag_part_sum * imag_part_sum)
        norm_feats = (np.asarray(self._scaling_factors_list) * magnitude).astype(np.float64)
        complex_moms = (np.asarray(self._scaling_factors_list) * (real_part_sum + 1.0j * imag_part_sum)).astype(np.complex128)
        return (norm_feats, complex_moms)

    def compute_zernike_features(self) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Compute moments and features using C-level, optimized computations."""
        norm_feats, complex_moms = self._compute_zernike_features()
        return norm_feats, complex_moms


# PseudoZernikeFeatures is almost same code as ZernikeFeatures class, except for
# 6 statements described below. Look for "# diff zf" in the main code:
# method _build_paired_data has these 4 statements as different:
    # 1. if (num >= 0):
    # 2. num_repeats = num
    # 3. coeff = sign * binom(2 * one_degree + 1 - s, s) * binom(2 * one_degree + 1 - 2 * s, one_degree - one_azimuthal - s)
    # 4. powvals.append(one_degree - s)
# method _compute_zernike_features has these 2 statements as different:
    # 5. acc_real_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_cos_view[th_id, m] * one_rho
    # 6. acc_imag_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_sin_view[th_id, m] * one_rho

# PseudoZernikeFeatures is intentionally written as a separate but duplicated class
# to leverage compiler optimizations and yield fast performance. Runtime injection
# of ZF or PZF functions incurred overhead and increase in time. Additionally, pointer
# based function injection caused problems with GIL.
# In future, however these classes might get merged.

# Because both classes are similar, the explanation and comments are not repeated
# for PseudoZernikeFeatures class below.

cdef class PseudoZernikeFeatures:
    """Extract pseudo, normalized features and complex-valued moments.

    This class implements computation of normalized pseudo-Zernike features and
    normalized complex-valued pseudo-Zernike moments. It uses binomial representation
    computed using gamma functions as provided by scipy, which maybe approximations
    of factorials but are computationally stable. It does not use explicit
    factorials or recursions to compute R-polynomial coefficients [1].

    Parameters
    ----------
    image : ndarray of shape (M, N)
        A binary or grayscale image of size (M, N), scaled to [0., 1.], 64-bit floats.
    degree : int
        Zernike polynomial design parameter.
    radius : float
        Radius of the circular pupil/bbox over which features are computed.
    center_coord : ndarray of shape (2,)
        Center of the circular pupil/bbox in 2D image coordinates as 64-bit floats.
    num_threads : int, optional
        Parallel and independent computation of features over multiple threads/cores.

    References
    ----------
    .. [1] Kuo Niu and Chao Tian 2022 J. Opt. 24 123001.
            https://doi.org/10.1088/2040-8986/ac9e08
    .. [2] D. Otkupman, S. Bezdidko, V. Ostashenkova E3S Web of Conferences 310, 01002 (2021).
            https://doi.org/10.1051/e3sconf/202131001002
    .. [3] Michael Reed Teague J. Opt. Soc. Am., Vol. 70, No. 8, August 1980.
            https://doi.org/10.1364/JOSA.70.000920
    .. [4] Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Zernike_polynomials
    .. [5] Pseudo Zernike Polynomials Wikipedia
            https://en.wikipedia.org/wiki/Pseudo-Zernike_polynomials
    .. [6] Image reconstruction example using Zernike moments
            https://stackoverflow.com/a/33339289
    """
    cdef int _degree, _num_features, _num_threads
    cdef float _radius
    cdef long _num_enclosed_pixels
    cdef Py_ssize_t _num_rows, _num_cols
    cdef cnp.int64_t[::1] _azimuthals_list
    cdef cnp.float64_t[::1] _scaling_factors_list
    cdef cnp.float64_t[::1] _coeffs_flat
    cdef cnp.int32_t[::1] _powvals_flat
    cdef cnp.int32_t[::1] _offsets
    cdef cnp.float64_t[:,::1] _valid_image
    cdef cnp.uint8_t[:, ::1] _pupil_mask
    cdef cnp.float64_t[:,::1] _distances_rho
    cdef cnp.float64_t[:,::1] _azimuthal_theta

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        int degree,
        float radius,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
        int num_threads=4,
    ):
        """C-level initialization of parameters."""
        self._degree = degree
        self._radius = radius
        self._num_threads = num_threads if num_threads > 0 else 1
        self._num_rows, self._num_cols = image.shape[0], image.shape[1]
        self._num_enclosed_pixels = 0
        self._build_normalized_grid(image, center_coord)
        self._build_paired_data()

    cdef void _build_normalized_grid(
        self,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
    ):
        """Build centered and normalized pupil grid over the image."""
        cdef cnp.ndarray[cnp.float64_t, ndim=3] var_yx
        cdef cnp.ndarray[cnp.float64_t, ndim=2] distances_rho, azimuthal_theta
        cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] pupil_mask

        center = np.reshape(center_coord, (-1, 1, 1))
        var_yx = np.mgrid[:self._num_rows, :self._num_cols].astype(np.float64)
        var_yx = (var_yx - center) / self._radius

        distances_rho = np.sqrt(np.sum(var_yx**2.0, axis=0))
        pupil_mask = (distances_rho <= 1.0).astype(np.uint8)
        valid_image = np.where(pupil_mask, image, 0.0)
        distances_rho = np.where(pupil_mask, distances_rho, 0.0)
        azimuthal_theta = np.arctan2(var_yx[0, :, :], var_yx[1, :, :]) + np.pi
        azimuthal_theta = np.where(pupil_mask, azimuthal_theta, 0.0)

        self._num_enclosed_pixels = int(np.sum(pupil_mask))
        self._distances_rho = np.ascontiguousarray(distances_rho)
        self._azimuthal_theta = np.ascontiguousarray(azimuthal_theta)
        self._pupil_mask = np.ascontiguousarray(pupil_mask)
        self._valid_image = np.ascontiguousarray(valid_image)

    cdef void _build_paired_data(self):
        """Build flattened lists for precomputed values."""
        cdef list degrees_list = []
        cdef list azimuthals_list = []
        cdef list scaling_list = []
        cdef list coeffs = []
        cdef list powvals = []
        cdef list offsets = [0]
        cdef int one_degree, one_azimuthal, num, num_repeats, s
        cdef double one_sf, coeff, sign
        cdef long denom_pixels = self._num_enclosed_pixels if self._num_enclosed_pixels > 0 else 1

        for one_degree in range(self._degree + 1):
            # one_sf = (one_degree + 1) / np.pi
            # one_sf = (one_degree + 1) / (<double>denom_pixels * np.pi)
            one_sf = (one_degree + 1) / <double>denom_pixels
            for one_azimuthal in range(one_degree + 1):
                num = one_degree - one_azimuthal
                if (num >= 0): # diff zf
                    degrees_list.append(one_degree)
                    azimuthals_list.append(one_azimuthal)
                    scaling_list.append(one_sf)
                    num_repeats = num # diff zf
                    for s in range(num_repeats + 1):
                        sign = -1.0 if (s % 2) else 1.0
                        coeff = sign * binom(2 * one_degree + 1 - s, s) * binom(2 * one_degree + 1 - 2 * s, one_degree - one_azimuthal - s) # diff zf
                        coeffs.append(coeff)
                        powvals.append(one_degree - s) # diff zf
                    offsets.append(len(coeffs))

        self._num_features = len(degrees_list)
        self._azimuthals_list = np.array(azimuthals_list, dtype=np.int64)
        self._scaling_factors_list = np.array(scaling_list, dtype=np.float64)
        self._coeffs_flat = np.array(coeffs, dtype=np.float64)
        self._powvals_flat = np.array(powvals, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)

    cdef tuple _compute_zernike_features(self):
        """Compute Zernike moments and features."""
        cdef Py_ssize_t num_rows = self._num_rows
        cdef Py_ssize_t num_cols = self._num_cols
        cdef int nfeatures = self._num_features
        cdef int degree = self._degree
        cdef int nthreads = self._num_threads
        cdef Py_ssize_t one_row, one_col
        cdef int th_id, one_feat, k, m, start, end
        cdef double one_rho, one_theta, one_cos, one_sin, one_pixel, one_rad_poly

        cdef cnp.ndarray acc_real_part = np.zeros((nthreads, nfeatures), dtype=np.float64)
        cdef cnp.ndarray acc_imag_part = np.zeros((nthreads, nfeatures), dtype=np.float64)
        cdef cnp.ndarray acc_dist_rho_power = np.zeros((nthreads, degree + 1), dtype=np.float64)
        cdef cnp.ndarray acc_cosine = np.zeros((nthreads, degree + 1), dtype=np.float64)
        cdef cnp.ndarray acc_sine = np.zeros((nthreads, degree + 1), dtype=np.float64)

        cdef double[:, ::1] acc_real_view = acc_real_part
        cdef double[:, ::1] acc_imag_view = acc_imag_part
        cdef double[:, ::1] acc_dist_rho_pow_view = acc_dist_rho_power
        cdef double[:, ::1] acc_cos_view = acc_cosine
        cdef double[:, ::1] acc_sin_view = acc_sine

        for one_row in prange(num_rows, nogil=True, num_threads=nthreads):
            th_id = threadid()
            for one_col in range(num_cols):
                if self._pupil_mask[one_row, one_col] == 0:
                    continue
                one_pixel = self._valid_image[one_row, one_col]
                if one_pixel == 0.0:
                    continue
                one_rho = self._distances_rho[one_row, one_col]
                one_theta = self._azimuthal_theta[one_row, one_col]
                acc_dist_rho_pow_view[th_id, 0] = 1.0
                for k in range(1, degree + 1):
                    acc_dist_rho_pow_view[th_id, k] = acc_dist_rho_pow_view[th_id, k - 1] * one_rho
                one_cos = cos(one_theta)
                one_sin = sin(one_theta)
                acc_cos_view[th_id, 0] = 1.0
                acc_sin_view[th_id, 0] = 0.0
                if degree >= 1:
                    acc_cos_view[th_id, 1] = one_cos
                    acc_sin_view[th_id, 1] = one_sin
                    for m in range(2, degree + 1):
                        acc_cos_view[th_id, m] = acc_cos_view[th_id, m - 1] * one_cos - acc_sin_view[th_id, m - 1] * one_sin
                        acc_sin_view[th_id, m] = acc_sin_view[th_id, m - 1] * one_cos + acc_cos_view[th_id, m - 1] * one_sin
                for one_feat in range(nfeatures):
                    m = self._azimuthals_list[one_feat]
                    start = self._offsets[one_feat]
                    end = self._offsets[one_feat + 1]
                    one_rad_poly = 0.0
                    for k in range(start, end):
                        one_rad_poly += self._coeffs_flat[k] * acc_dist_rho_pow_view[th_id, self._powvals_flat[k]]
                    acc_real_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_cos_view[th_id, m] * one_rho # diff zf
                    acc_imag_view[th_id, one_feat] += one_pixel * one_rad_poly * acc_sin_view[th_id, m] * one_rho # diff zf

        real_part_sum = acc_real_part.sum(axis=0)
        imag_part_sum = acc_imag_part.sum(axis=0)
        magnitude = np.sqrt(real_part_sum * real_part_sum + imag_part_sum * imag_part_sum)
        norm_feats = (np.asarray(self._scaling_factors_list) * magnitude).astype(np.float64)
        complex_moms = (np.asarray(self._scaling_factors_list) * (real_part_sum + 1.0j * imag_part_sum)).astype(np.complex128)
        return (norm_feats, complex_moms)

    def compute_pseudo_zernike_features(self) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Compute moments and features using C-level, optimized computations."""
        norm_feats, complex_moms = self._compute_zernike_features()
        return norm_feats, complex_moms


cdef class ZernikeArbitrary:
    cdef int _degree, _num_features, _num_threads
    cdef float _radius, _obscure_radius
    cdef long _num_enclosed_pixels
    cdef Py_ssize_t _num_rows, _num_cols
    cdef cnp.int32_t[::1] _azimuthals_list
    cdef cnp.float64_t[::1] _scaling_factors_list
    cdef cnp.float64_t[::1] _coeffs_flat
    cdef cnp.int32_t[::1] _powvals_flat
    cdef cnp.int32_t[::1] _offsets
    cdef cnp.float64_t[:,::1] _valid_image
    cdef cnp.uint8_t[:, ::1] _pupil_mask
    cdef cnp.int32_t[::1] _valid_rows
    cdef cnp.int32_t[::1] _valid_cols
    cdef cnp.float64_t[:,::1] _distances_rho
    cdef cnp.float64_t[:,::1] _azimuthal_theta
    cdef cnp.complex128_t[:,::1] _complex_basis
    cdef cnp.complex128_t[:,::1] _orthonormal_basis
    cdef cnp.float64_t[::1] _weights

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        int degree,
        float radius,
        float obscure_radius,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
        int num_threads=4,
    ):
        """C-level initialization of parameters."""
        self._degree = degree
        self._radius = radius
        self._obscure_radius = obscure_radius
        self._num_threads = num_threads if num_threads > 0 else 1
        self._num_rows, self._num_cols = image.shape[0], image.shape[1]
        self._num_enclosed_pixels = 0
        self._build_normalized_grid(image, center_coord)

    cdef void _build_normalized_grid(
        self,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
    ):
        """Build centered and normalized pupil grid over the image."""
        cdef float obscure_ratio = 0.0
        cdef cnp.ndarray[cnp.float64_t, ndim=3] var_yx
        cdef cnp.ndarray[cnp.float64_t, ndim=2] distances_rho, azimuthal_theta
        cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] pupil_mask

        center = np.reshape(center_coord, (-1, 1, 1))
        var_yx = np.mgrid[:self._num_rows, :self._num_cols].astype(np.float64)
        var_yx = (var_yx - center) / self._radius
        obscure_ratio = self._obscure_radius / self._radius

        distances_rho = np.sqrt(np.sum(var_yx**2.0, axis=0))
        pupil_mask = ((distances_rho >= obscure_ratio) & (distances_rho <= 1.0)).astype(np.uint8)
        valid_coords = np.nonzero(pupil_mask)
        valid_image = np.where(pupil_mask, image, 0.0)
        distances_rho = np.where(pupil_mask, distances_rho, 0.0)
        azimuthal_theta = np.arctan2(var_yx[0, :, :], var_yx[1, :, :]) + np.pi
        azimuthal_theta = np.where(pupil_mask, azimuthal_theta, 0.0)

        self._num_enclosed_pixels = int(np.sum(pupil_mask))
        self._distances_rho = np.ascontiguousarray(distances_rho)
        self._azimuthal_theta = np.ascontiguousarray(azimuthal_theta)
        self._pupil_mask = np.ascontiguousarray(pupil_mask)
        self._valid_image = np.ascontiguousarray(valid_image)
        self._valid_rows = np.ascontiguousarray(valid_coords[0], dtype=np.int32)
        self._valid_cols = np.ascontiguousarray(valid_coords[1], dtype=np.int32)
        # self._weights = np.ascontiguousarray((1/self._num_enclosed_pixels)*np.identity(self._num_enclosed_pixels, dtype=np.float64))
        self._weights = np.ascontiguousarray((1/<double>self._num_enclosed_pixels)*np.ones((self._num_enclosed_pixels,), dtype=np.float64))

    cdef void _build_paired_data(self):
        """Build flattened lists for precomputed values."""
        cdef list degrees_list = []
        cdef list azimuthals_list = []
        cdef list scaling_list = []
        cdef list coeffs = []
        cdef list powvals = []
        cdef list offsets = [0]
        cdef int one_degree, one_azimuthal, num, num_repeats, s
        cdef double one_sf, coeff, sign
        cdef long denom_pixels = self._num_enclosed_pixels if self._num_enclosed_pixels > 0 else 1

        for one_degree in range(self._degree + 1):
            # one_sf = (one_degree + 1) / np.pi
            # one_sf = (one_degree + 1) / (<double>denom_pixels * np.pi)
            one_sf = (one_degree + 1) / <double>denom_pixels
            for one_azimuthal in range(one_degree + 1):
                num = one_degree - one_azimuthal
                if (num >= 0) and (num % 2 == 0.0):
                    degrees_list.append(one_degree)
                    azimuthals_list.append(one_azimuthal)
                    scaling_list.append(one_sf)
                    num_repeats = num // 2
                    for s in range(num_repeats + 1):
                        sign = -1.0 if (s % 2) else 1.0
                        coeff = sign * binom(one_degree - s, s) * binom(one_degree - 2 * s, ((one_degree - one_azimuthal) // 2) - s)
                        coeffs.append(coeff)
                        powvals.append(one_degree - 2 * s)
                    offsets.append(len(coeffs))

        self._num_features = len(degrees_list)
        self._azimuthals_list = np.array(azimuthals_list, dtype=np.int32)
        self._scaling_factors_list = np.array(scaling_list, dtype=np.float64)
        self._coeffs_flat = np.array(coeffs, dtype=np.float64)
        self._powvals_flat = np.array(powvals, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)

    cdef void _build_basis_polynomials(self):
        """Build complex basis polynomials for every valid pixel, in parallel."""
        cdef int nfeatures = self._num_features
        cdef int degree = self._degree
        cdef Py_ssize_t one_idx
        cdef int one_row, one_col, one_feat, k, m, start, end, th_id
        cdef double one_rho, one_theta, one_cos, one_sin, one_rad_poly

        cdef cnp.ndarray[cnp.float64_t, ndim=2] acc_real_part = np.zeros((self._num_enclosed_pixels, nfeatures), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] acc_imag_part = np.zeros((self._num_enclosed_pixels, nfeatures), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] acc_dist_rho_power = np.zeros((self._num_threads, degree + 1,), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] acc_cosine = np.zeros((self._num_threads, degree + 1,), dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=2] acc_sine = np.zeros((self._num_threads, degree + 1,), dtype=np.float64)

        cdef double[:, ::1] acc_real_view = acc_real_part
        cdef double[:, ::1] acc_imag_view = acc_imag_part
        cdef double[:, ::1] acc_dist_rho_pow_view = acc_dist_rho_power
        cdef double[:, ::1] acc_cos_view = acc_cosine
        cdef double[:, ::1] acc_sin_view = acc_sine

        for one_idx in prange(self._num_enclosed_pixels, nogil=True, num_threads=self._num_threads, schedule="static"):
            th_id = threadid()
            one_row = self._valid_rows[one_idx]
            one_col = self._valid_cols[one_idx]
            one_rho = self._distances_rho[one_row, one_col]
            one_theta = self._azimuthal_theta[one_row, one_col]
            acc_dist_rho_pow_view[th_id, 0] = 1.0
            for k in range(1, degree + 1):
                acc_dist_rho_pow_view[th_id, k] = acc_dist_rho_pow_view[th_id, k - 1] * one_rho
            one_cos = cos(one_theta)
            one_sin = sin(one_theta)
            acc_cos_view[th_id, 0] = 1.0
            acc_sin_view[th_id, 0] = 0.0
            if degree >= 1:
                acc_cos_view[th_id, 1] = one_cos
                acc_sin_view[th_id, 1] = one_sin
                for m in range(2, degree + 1):
                    acc_cos_view[th_id, m] = acc_cos_view[th_id, m - 1] * one_cos - acc_sin_view[th_id, m - 1] * one_sin
                    acc_sin_view[th_id, m] = acc_sin_view[th_id, m - 1] * one_cos + acc_cos_view[th_id, m - 1] * one_sin
            for one_feat in range(nfeatures):
                m = self._azimuthals_list[one_feat]
                start = self._offsets[one_feat]
                end = self._offsets[one_feat + 1]
                one_rad_poly = 0.0
                for k in range(start, end):
                    one_rad_poly = one_rad_poly + self._coeffs_flat[k] * acc_dist_rho_pow_view[th_id, self._powvals_flat[k]]
                acc_real_view[one_idx, one_feat] = one_rad_poly * acc_cos_view[th_id, m]
                acc_imag_view[one_idx, one_feat] = one_rad_poly * acc_sin_view[th_id, m]

        self._complex_basis = acc_real_part + 1.0j * acc_imag_part

    cdef void _cholesky_basis(self):
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] gram_matrix = np.zeros((self._num_features,self._num_features), dtype=np.complex128)
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] decomp_cholesy_L = np.zeros((self._num_features,self._num_features), dtype=np.complex128, order="F")
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] cholesy_inverse_hermitian = np.zeros((self._num_features,self._num_features), dtype=np.complex128, order="F")
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] transformation = np.zeros((self._num_features,self._num_features), dtype=np.complex128, order="F")
        cdef char uplo = b'L'
        cdef char diag = b'N'
        cdef int lda = self._num_features
        cdef int info = 0
        cdef int i, j

        gram_matrix = ((np.transpose(np.conjugate(self._complex_basis))) * self._weights) @ self._complex_basis
        decomp_cholesy_L = np.asfortranarray(gram_matrix.copy())

        cython_lapack.zpotrf(&uplo, &lda, <double complex*> decomp_cholesy_L.data, &lda, &info)
        if info != 0:
            raise ValueError(f"LAPACK Cholesky decomposition failed. {info}")
        for i in range(self._num_features):
            for j in range(i + 1, self._num_features):
                decomp_cholesy_L[i, j] = 0.0

        cholesy_inverse_hermitian = decomp_cholesy_L.copy()
        cython_lapack.ztrtri(&uplo, &diag, &lda, <double complex*> cholesy_inverse_hermitian.data, &lda, &info)
        if info != 0:
            raise ValueError(f"LAPACK Cholesky inverse failed. {info}")

        transformation = np.transpose(np.conjugate(cholesy_inverse_hermitian))
        self._orthonormal_basis = self._complex_basis @ transformation

    cdef tuple _compute_zernike_features(self):
        cdef Py_ssize_t one_pix
        cdef cnp.ndarray[cnp.float64_t, ndim=1] pixel_vector = np.zeros((self._num_enclosed_pixels,), dtype=np.float64)
        cdef cnp.ndarray[cnp.complex128_t, ndim=1] complex_moments = np.zeros((self._num_features,), dtype=np.complex128)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] features = np.zeros((self._num_features,), dtype=np.float64)

        self._build_paired_data()
        self._build_basis_polynomials()
        self._cholesky_basis()

        for one_pix in prange(self._num_enclosed_pixels, nogil=True, schedule="static", num_threads=self._num_threads):
            pixel_vector[one_pix] = self._valid_image[self._valid_rows[one_pix], self._valid_cols[one_pix]]

        complex_moments = np.transpose(np.conjugate(self._orthonormal_basis)) @ (self._weights * pixel_vector)
        features = np.absolute(complex_moments)
        return (features, complex_moments)

    def compute_zernike_features(self) -> tuple[np.typing.NDArray, np.typing.NDArray]:
        """Compute moments and features using C-level, optimized computations."""
        norm_feats, complex_moms = self._compute_zernike_features()
        return norm_feats, complex_moms
