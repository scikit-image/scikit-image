#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, atan2, fabs, cos, sin
from cython.parallel cimport prange
from scipy.special.cython_special cimport binom
from scipy.linalg cimport cython_lapack

cnp.import_array()

# constants used in PupilGrid class
cdef double PI = 3.14159265358979323846
cdef double SQRT3 = 1.7320508075688772935


class ZernikeTypeError(TypeError):
    # defined here instead of zernike.py to avoid circular imports
    pass


class ZernikeValueError(ValueError):
    # defined here instead of zernike.py to avoid circular imports
    pass


# Below is the overview and brief explanation for the implementation of Zernike
# features computation based on Cython classes and methods.
# ZernikeFeatures() is the main Cython class called by the external module zernike.py
# This class uses composition principle and relies on 4 classes to achieve its goal
    # PupilGrid() to build the desired pupil from the image.
    # ZernikeComplexBasis() computes the complex basis polynomials (Vnl) at runtime
        # for either conventional or pseudo ones over the circular pupil.
        # These are used for circular pupils directly. For othere pupils,
        # these serve as starting basis to be orthogonalized.
    # ZernikeOrthonormalization() computes the Zernike-like orthonormal basis
        # for the desired non-circular pupil. It currently uses matrix based
        # non-recursive method that first computes the Zernike/pseudo-Zernike
        # basis (Z) over the given pupil, then computes Gram-matrix M = (Z^H)*W*Z
        # for normalization purpose. Then it computes Cholesky decomposition/factorization
        # of M as M = L*(L^H) where L is decomp lower triangular matrix. This matrix is
        # used for new orthogonalized Zernike-like basis over the given non-circular
        # pupil as F = Z * (L^-1)^H.
    # ImageReconstruction() to reconstruct the image inside the given pupil
        # using Z or F basis as grayscale, and copying non-pupil intensities
        # from the original image as it is.

# pseudo-Zernike features is almost same code as conventional Zernike features, except for
# 6 statements. Look for "# diff zf" in the main code.
# pseudo-Zernike features is intentionally written as a separate but duplicated function
# to leverage compiler optimizations and yield fast performance. Runtime injection
# of ZF or PZF functions incurred overhead and increased checking/conditionals time.
# Additionally, pointer based function injection caused problems with GIL.
# In future, however these functions might get merged.


cdef class PupilGrid:
    # This class is used to generate the normalized grid points (x, y) using image
    # size and then creates the required pupil using those points.
    cdef Py_ssize_t _num_rows, _num_cols
    cdef long _num_enclosed_pixels
    cdef double _radius
    cdef double _annulus_obscure_radius
    cdef double _rectangle_semi_width, _rectangle_semi_height
    cdef double _square_semi_side
    cdef double _hexagon_side
    cdef double _ellipse_semi_major, _ellipse_semi_minor
    cdef double _center_row, _center_col

    cdef cnp.uint8_t[:,::1] _pupil_mask
    cdef cnp.int32_t[::1] _valid_rows, _valid_cols
    cdef cnp.float64_t[::1] _center_coord, _weights
    cdef cnp.float64_t[:,::1] _image, _valid_image, _polar_rho, _polar_theta

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
    ):
        """C-level initialization of parameters."""
        # ensure the arrays are C-contiguous
        image = np.ascontiguousarray(image)
        center_coord = np.ascontiguousarray(center_coord)

        self._image = image
        self._center_coord = center_coord
        self._center_row = center_coord[0]
        self._center_col = center_coord[1]

        self._num_rows, self._num_cols = image.shape[0], image.shape[1]
        self._num_enclosed_pixels = 0
        self._radius = 0.0
        self._annulus_obscure_radius = 0.0
        self._rectangle_semi_width = 0.0
        self._rectangle_semi_height = 0.0
        self._square_semi_side = 0.0
        self._hexagon_side = 0.0
        self._ellipse_semi_major = 0.0
        self._ellipse_semi_minor = 0.0
        self._pupil_mask = np.empty((self._num_rows, self._num_cols), dtype=np.uint8)
        self._polar_rho = np.empty((self._num_rows, self._num_cols), dtype=np.float64)
        self._polar_theta = np.empty((self._num_rows, self._num_cols), dtype=np.float64)
        self._valid_image = np.empty((self._num_rows, self._num_cols), dtype=np.float64)

    cdef void _update_grid(self):
        """Update the centered and normalized circular pupil grid as per the desired pupil."""
        cdef Py_ssize_t i, j, idx
        cdef long count = 0
        cdef double inv_count = 1.0
        cdef cnp.float64_t[:,::1] img = self._image
        cdef cnp.float64_t[:,::1] vimg = self._valid_image
        cdef cnp.float64_t[:,::1] rho = self._polar_rho
        cdef cnp.float64_t[:,::1] theta = self._polar_theta
        cdef cnp.uint8_t[:,::1] mask = self._pupil_mask # updated pupil mask provided by the calling pupil function

        # update valid image, count valid pixels
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                if mask[i, j]:
                    count += 1
                    vimg[i, j] = img[i, j]
                else:
                    vimg[i, j] = 0.0
                    rho[i, j] = 0.0
                    theta[i, j] = 0.0

        inv_count = 1.0 / <double>count
        self._num_enclosed_pixels = count
        self._valid_rows = np.empty(count, dtype=np.int32)
        self._valid_cols = np.empty(count, dtype=np.int32)
        self._weights = np.empty(count, dtype=np.float64)
        cdef cnp.int32_t[::1] vr = self._valid_rows
        cdef cnp.int32_t[::1] vc = self._valid_cols
        cdef cnp.float64_t[::1] wt = self._weights

        # collect valid pixel indices into a flattend array for faster processing
        # build weight matrix used for orthonorm. later in ZernikeOrthonormalization
        idx = 0
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                if mask[i, j]:
                    vr[idx] = i
                    vc[idx] = j
                    wt[idx] = inv_count
                    idx += 1

    cdef void build_circular_normalized_grid(self, double radius):
        """Build circular pupil of given radius."""
        self._radius = radius
        cdef Py_ssize_t i, j
        cdef double inv_radius = 1.0 / self._radius
        cdef double dc, dr, vrho
        cdef cnp.float64_t[:,::1] rho = self._polar_rho
        cdef cnp.float64_t[:,::1] theta = self._polar_theta
        cdef cnp.uint8_t[:,::1] mask = self._pupil_mask

        for i in prange(self._num_rows, nogil=True):
            dr = <double>i - self._center_row
            for j in range(self._num_cols):
                dc = <double>j - self._center_col
                vrho = sqrt(dc * dc + dr * dr) * inv_radius
                rho[i, j] = vrho
                theta[i, j] = atan2(dr, dc) + PI
                if vrho <= 1.0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        # update the other arrays using the mask
        self._update_grid()

    cdef void build_annular_normalized_grid(
        self, double radius, double annulus_obscure_radius
    ):
        """Build annular pupil of given outer radius and obscuration radius."""
        cdef Py_ssize_t i, j
        cdef double ratio = annulus_obscure_radius / radius
        cdef cnp.uint8_t[:,::1] mask
        cdef cnp.float64_t[:,::1] rho
        self._radius = radius
        self._annulus_obscure_radius = annulus_obscure_radius

        self.build_circular_normalized_grid(self._radius)
        mask = self._pupil_mask
        rho = self._polar_rho
        # update pupil mask where pixels within obscure section are set to 0
        for i in range(self._num_rows):
            for j in range(self._num_cols):
                if mask[i, j] and (rho[i, j] <= ratio):
                    mask[i, j] = 0
        # update the other arrays using the updated mask
        self._update_grid()

    cdef void build_elliptical_normalized_grid(self, double ellipse_semi_major, double ellipse_semi_minor):
        """Build elliptical pupil using the given major and minor axes."""
        # The major, minor axes, orientation are not decided here, and parameters are merely two variables.
        # The orientation is decided by the zernike_features() function in zernike.py.
        # It swaps the arguments when called.
        cdef Py_ssize_t i, j
        cdef double dc, dr, dc_scaled, dr_scaled, vrho
        cdef cnp.float64_t[:,::1] rho = self._polar_rho
        cdef cnp.float64_t[:,::1] theta = self._polar_theta
        cdef cnp.uint8_t[:,::1] mask = self._pupil_mask
        self._radius = ellipse_semi_major
        self._ellipse_semi_major = ellipse_semi_major
        self._ellipse_semi_minor = ellipse_semi_minor

        for i in prange(self._num_rows, nogil=True):
            dr = <double>i - self._center_row
            dr_scaled = dr / ellipse_semi_minor
            for j in range(self._num_cols):
                dc = <double>j - self._center_col
                dc_scaled = dc / ellipse_semi_major
                vrho = sqrt(dr_scaled * dr_scaled + dc_scaled * dc_scaled)
                rho[i, j] = vrho
                theta[i, j] = atan2(dr_scaled, dc_scaled) + PI
                if vrho <= 1.0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        self._update_grid()

    cdef void build_rectangular_normalized_grid(
        self,
        double rectangle_semi_width,
        double rectangle_semi_height
    ):
        """Build rectangular pupil using the given width and height."""
        cdef Py_ssize_t i, j
        cdef double inv_radius = 1.0 / self._radius
        cdef double dc, dr, vrho
        cdef cnp.float64_t[:,::1] rho = self._polar_rho
        cdef cnp.float64_t[:,::1] theta = self._polar_theta
        cdef cnp.uint8_t[:,::1] mask = self._pupil_mask
        self._rectangle_semi_width = rectangle_semi_width
        self._rectangle_semi_height = rectangle_semi_height
        self._radius = sqrt(rectangle_semi_width * rectangle_semi_width + \
                            rectangle_semi_height * rectangle_semi_height)
        inv_radius = 1.0 / self._radius
        for i in prange(self._num_rows, nogil=True):
            dr = <double>i - self._center_row
            for j in range(self._num_cols):
                dc = <double>j - self._center_col
                vrho = sqrt(dc * dc + dr * dr) * inv_radius
                rho[i, j] = vrho
                theta[i, j] = atan2(dr, dc) + PI
                if (fabs(dc) <= rectangle_semi_width) and (fabs(dr) <= rectangle_semi_height):
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        self._update_grid()

    cdef void build_square_normalized_grid(self, double square_semi_side):
        """Build square pupil using the given side length."""
        self.build_rectangular_normalized_grid(square_semi_side, square_semi_side)

    cdef void build_hexagonal_normalized_grid(self, double hexagon_side):
        """Build regular hexagonal pupil using the given side length."""
        # This is a regular hexagon where its two of six vertices lie on the
        # horizontal line. The top and bottom are flat. For regular hexagon,
        # the side length is equal to circumscribing circle i.e. s=R.
        cdef Py_ssize_t i, j
        cdef double inv_radius = 1.0 / hexagon_side
        cdef double apothem = 0.5 * SQRT3   # sqrt(3)/2
        cdef double dc, dr, dc_norm, dr_norm, vrho
        cdef cnp.float64_t[:,::1] rho = self._polar_rho
        cdef cnp.float64_t[:,::1] theta = self._polar_theta
        cdef cnp.uint8_t[:,::1] mask = self._pupil_mask
        self._radius = hexagon_side
        self._hexagon_side = hexagon_side

        for i in prange(self._num_rows, nogil=True):
            dr = <double>i - self._center_row
            dr_norm = dr * inv_radius
            for j in range(self._num_cols):
                dc = <double>j - self._center_col
                dc_norm = dc * inv_radius
                vrho = sqrt(dr_norm * dr_norm + dc_norm * dc_norm)
                rho[i, j] = vrho
                theta[i, j] = atan2(dr, dc) + PI
                if (vrho <= 1.0 and
                    fabs(dr_norm) <= apothem and
                    fabs(SQRT3 * dc_norm + dr_norm) <= SQRT3 and
                    fabs(SQRT3 * dc_norm - dr_norm) <= SQRT3):
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
        self._update_grid()


cdef class ZernikeComplexBasis:
    """Compute orthogonal complex Zernike polynomial basis for all rho, theta inside the unit-circle pupil."""
    # This class computes the Zernike basis.
    # Here, _azimuthals_lm are l (or m) integers for each n.
    # _powers are exponents for rho^(n-2s) in radial polynomials for each index s.
    # _coefficients are factorial component for each radial polynomial.
    # _scale_factors are scaling factors (n+1/pi) for complex valued polynomials.
    # _offsets that tracks number of l (or m) values for each degree n.
    # _acc_dist_rho_power collects each rho^(n-2s) i.e. [rho^0,rho^1,...,rho^n]. Indexing with (n-2s)
    cdef int _num_enclosed_pixels, _num_features, _degree
    cdef cnp.int32_t[::1] _azimuthals_lm, _powers, _offsets, _valid_rows, _valid_cols
    cdef cnp.float64_t[::1] _scale_factors, _coefficients
    cdef cnp.float64_t[:,::1] _polar_rho, _polar_theta
    cdef cnp.complex128_t[:,::1] _complex_basis
    cdef cnp.float64_t[::1] _acc_dist_rho_power, _acc_cosine, _acc_sine

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.int32_t, ndim=1] valid_rows,
        cnp.ndarray[cnp.int32_t, ndim=1] valid_cols,
        cnp.ndarray[cnp.float64_t, ndim=2] polar_rho,
        cnp.ndarray[cnp.float64_t, ndim=2] polar_theta,
        int degree,
    ):
        valid_rows = np.ascontiguousarray(valid_rows)
        valid_cols = np.ascontiguousarray(valid_cols)
        polar_rho = np.ascontiguousarray(polar_rho)
        polar_theta = np.ascontiguousarray(polar_theta)
        self._degree = degree
        self._num_features = 0
        self._num_enclosed_pixels = valid_rows.shape[0]
        self._valid_rows = valid_rows
        self._valid_cols = valid_cols
        self._polar_rho = polar_rho
        self._polar_theta = polar_theta
        self._acc_dist_rho_power = np.empty((degree + 1,), dtype=np.float64)
        self._acc_cosine = np.empty((degree + 1,), dtype=np.float64)
        self._acc_sine = np.empty((degree + 1,), dtype=np.float64)

    cdef void _build_paired_data_conv_zf(self):
        """Build flattened lists for precomputed values for conventional Zernike polynomials."""
        # Paired data here means for one n in {0,...,degree+1}, the degree,
        # number of azimuthals (modes m/l), scaling factor value ((n+1)/numpix),
        # number of index s ((n-l)/2), binomial coefficient values, power values (n-2s)
        # are precomputed for fast iteration and access later in the main feature loop.
        cdef list deg_n = [] # degree
        cdef list azi_lm = [] # m or l
        cdef list scales = [] # (n+1)/numpix
        cdef list coeffs = [] # factorials replaced by binomials
        cdef list powvals = [] # (n-2s)
        cdef list offsets = [0] # each n-l degree has s polynomials, for flat list skip s polys
        cdef int n, l, num, npolys, s
        cdef double sf, coeff, sign
        cdef double npix = <double>self._num_enclosed_pixels

        for n in range(self._degree + 1):
            # sf = (n + 1) / np.pi # kept for reference only
            # sf = (n + 1) / (<double>npix * np.pi) # kept for reference only
            sf = (n + 1) / npix
            for l in range(n + 1): # modes m or l
                num = n - l
                if (num >= 0) and (num % 2 == 0.0): # n-l condition for non-negative degree and even
                    deg_n.append(n)
                    azi_lm.append(l)
                    scales.append(sf)
                    npolys = num // 2
                    for s in range(npolys + 1): # polynomial index s, (n-l)/2
                        sign = -1.0 if (s % 2) else 1.0
                        coeff = sign * binom(n - s, s) * binom(n - 2 * s, ((n - l) // 2) - s)
                        coeffs.append(coeff)
                        powvals.append(n - 2 * s)
                    offsets.append(len(coeffs))

        self._num_features = len(deg_n)
        self._azimuthals_lm = np.array(azi_lm, dtype=np.int32)
        self._scale_factors = np.array(scales, dtype=np.float64)
        self._coefficients = np.array(coeffs, dtype=np.float64)
        self._powers = np.array(powvals, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)

    cdef void _build_paired_data_pseudo_zf(self):
        """Build flattened lists for precomputed values for pseudo-Zernike polynomials."""
        # Similar function as build_paired_data_conv_zf() but for pseudo-Zernike
        # Here "# diff" highlights steps different from conventional Zernike
        # Instead of adding if-else condition and create one function, pseudo-Zernike
        # is repeated, but implemented separately to eliminate branching and get
        # fast performance after compiling to C-level code.
        cdef list deg_n = []
        cdef list azi_lm = []
        cdef list scales = []
        cdef list coeffs = []
        cdef list powvals = []
        cdef list offsets = [0]
        cdef int n, l, num, npolys, s
        cdef double sf, coeff, sign
        cdef double npix = <double>self._num_enclosed_pixels

        for n in range(self._degree + 1):
            # sf = (n + 1) / np.pi
            # sf = (n + 1) / (<double>npix * np.pi)
            sf = (n + 1) / npix
            for l in range(n + 1):
                num = n - l
                if (num >= 0): # diff zf
                    deg_n.append(n)
                    azi_lm.append(l)
                    scales.append(sf)
                    npolys = num # diff zf
                    for s in range(npolys + 1):
                        sign = -1.0 if (s % 2) else 1.0
                        coeff = sign * binom(2 * n + 1 - s, s) * binom(2 * n + 1 - 2 * s, n - l - s) # diff zf
                        coeffs.append(coeff)
                        powvals.append(n - s) # diff zf
                    offsets.append(len(coeffs))

        self._num_features = len(deg_n)
        self._azimuthals_lm = np.array(azi_lm, dtype=np.int32)
        self._scale_factors = np.array(scales, dtype=np.float64)
        self._coefficients = np.array(coeffs, dtype=np.float64)
        self._powers = np.array(powvals, dtype=np.int32)
        self._offsets = np.array(offsets, dtype=np.int32)

    cdef void compute_conventional_zernike_basis(self):
        """Build conventional complex basis polynomials for every valid pixel."""
        # A simplified explanation of the implementation:
        # From ref. [1], we wish to implement equation 156,
        # moment = f(rho,theta) * R(rho) * exp(j*theta)
        # it can be computed with nested loops as simple represention shown below:
        # for one_moment:
            # for one_row:
                # for one_col:
                    # rho = _polar_rho[one_row, one_col]
                    # theta = _polar_theta[one_row, one_col]
                    # moment[one_moment, one_row, one_col] = f(rho,theta) * R(rho) * exp(j*theta)

        # However, for optimization, to avoid repeated calls to factorial, power operations
        # and for efficient memory use, the valid pixel indices were already extracted and
        # flattened, and the complex basis become moment[one_pixel, one_moment]. Step-by-step:

        # First, moment, is split into real and imaginary parts. Both of shape (P, F) P:pixels, F:degrees or features

        # For basis calculations, intensities are not used in computations.
        # Thus f(rho,theta) can be skipped and moment becomes complex basis only.

        # R(rho) = sum(coefficients * rho^(n - 2s)). Here, coefficients were
        # precomputed into a flattened list and indexed with s and offset by
        # previous number of basis. rho^(n-2s) is extended from rho^0 to rho^n,
        # and appropriate rho^i is indexed by powvals.

        # exp(j*theta) is split into sine and cosine parts.

        # Thus, flattening allows fast computation over each pixel
        # basis[one_pixel, one_moment]

        self._build_paired_data_conv_zf()
        cdef int nfeatures = self._num_features
        cdef int npix = self._num_enclosed_pixels
        cdef int degree = self._degree
        cdef Py_ssize_t one_pix
        cdef int one_row, one_col, one_feat, m, s, start, end
        cdef double one_rho, one_theta, one_rad_poly

        self._complex_basis = np.empty((npix, nfeatures), dtype=np.complex128)

        cdef cnp.float64_t[:,::1] rho_mv = self._polar_rho
        cdef cnp.float64_t[:,::1] theta_mv = self._polar_theta
        cdef cnp.float64_t[::1] rho_pow_mv = self._acc_dist_rho_power
        cdef cnp.float64_t[::1] cos_mv = self._acc_cosine
        cdef cnp.float64_t[::1] sin_mv = self._acc_sine
        cdef cnp.int32_t[::1] row_mv = self._valid_rows
        cdef cnp.int32_t[::1] col_mv = self._valid_cols
        cdef cnp.int32_t[::1] azi_lm = self._azimuthals_lm
        cdef cnp.int32_t[::1] offsets = self._offsets
        cdef cnp.int32_t[::1] powers = self._powers
        cdef cnp.float64_t[::1] coeffs = self._coefficients
        cdef cnp.complex128_t[:,::1] basis = self._complex_basis

        for one_pix in range(npix):
            one_row = row_mv[one_pix]
            one_col = col_mv[one_pix]
            # get one rho
            one_rho = rho_mv[one_row, one_col]
            # get one theta
            one_theta = theta_mv[one_row, one_col]
            rho_pow_mv[0] = 1.0 # rho^0
            sin_mv[0] = 0.0 # sin(0*theta)
            cos_mv[0] = 1.0 # cos(0*theta)
            for m in range(1, degree + 1):
                # build [rho^0...rho^n]
                rho_pow_mv[m] = rho_pow_mv[m - 1] * one_rho
                # for exp(j*l*theta) = cos(l*theta) + j*sin(l*theta)
                # build [cos(0)...cos(n*theta)], similarly for sin
                sin_mv[m] = sin(m*one_theta)
                cos_mv[m] = cos(m*one_theta)
            for one_feat in range(nfeatures):
                # get integer l for current degree n
                m = azi_lm[one_feat]
                # coeffs, powers are long flattened lists.
                # These long lists need to be sliced and accessed as per
                # the number of s for combination of (n-l)/2 as n, l vary.
                start = offsets[one_feat]
                end = offsets[one_feat + 1]
                one_rad_poly = 0.0
                for s in range(start, end):
                    one_rad_poly = one_rad_poly + coeffs[s] * rho_pow_mv[powers[s]]
                # basis Vnl[rho, theta] = sum(radial polynomial) * exp(j*l*theta)
                basis[one_pix, one_feat].real = one_rad_poly * cos_mv[m]
                basis[one_pix, one_feat].imag = one_rad_poly * sin_mv[m]

    cdef void compute_pseudo_zernike_basis(self):
        """Build complex basis pseudo-Zernike polynomials for every valid pixel."""
        self._build_paired_data_pseudo_zf()
        cdef int nfeatures = self._num_features
        cdef int npix = self._num_enclosed_pixels
        cdef int degree = self._degree
        cdef Py_ssize_t one_pix
        cdef int one_row, one_col, one_feat, m, s, start, end
        cdef double one_rho, one_theta, one_rad_poly

        self._complex_basis = np.empty((npix, nfeatures), dtype=np.complex128)

        cdef cnp.float64_t[:,::1] rho_mv = self._polar_rho
        cdef cnp.float64_t[:,::1] theta_mv = self._polar_theta
        cdef cnp.float64_t[::1] rho_pow_mv = self._acc_dist_rho_power
        cdef cnp.float64_t[::1] cos_mv = self._acc_cosine
        cdef cnp.float64_t[::1] sin_mv = self._acc_sine
        cdef cnp.int32_t[::1] row_mv = self._valid_rows
        cdef cnp.int32_t[::1] col_mv = self._valid_cols
        cdef cnp.int32_t[::1] azi_lm = self._azimuthals_lm
        cdef cnp.int32_t[::1] offsets = self._offsets
        cdef cnp.int32_t[::1] powers = self._powers
        cdef cnp.float64_t[::1] coeffs = self._coefficients
        cdef cnp.complex128_t[:,::1] basis = self._complex_basis

        for one_pix in range(npix):
            one_row = row_mv[one_pix]
            one_col = col_mv[one_pix]
            one_rho = rho_mv[one_row, one_col]
            one_theta = theta_mv[one_row, one_col]
            rho_pow_mv[0] = 1.0
            sin_mv[0] = 0.0
            cos_mv[0] = 1.0
            for m in range(1, degree + 1):
                rho_pow_mv[m] = rho_pow_mv[m - 1] * one_rho
                sin_mv[m] = sin(m*one_theta)
                cos_mv[m] = cos(m*one_theta)
            for one_feat in range(nfeatures):
                m = azi_lm[one_feat]
                start = offsets[one_feat]
                end = offsets[one_feat + 1]
                one_rad_poly = 0.0
                for s in range(start, end):
                    one_rad_poly = one_rad_poly + coeffs[s] * rho_pow_mv[powers[s]]
                basis[one_pix, one_feat].real = one_rad_poly * cos_mv[m] * one_rho # diff zf
                basis[one_pix, one_feat].imag = one_rad_poly * sin_mv[m] * one_rho # diff zf


cdef class ZernikeOrthonormalization:
    """Compute orthonormal complex pseudo/Zernike-like polynomial basis for arbitrary pupil."""
    # This implementation use the non-recursive matrix method, instead of the
    # Gram-Schmidt recursive method, to compute the orthonormal complex-valued
    # conventional or pseudo Zernike basis polynomials over arbitrary pupils.
    # The various steps for obtaining these basis are:
        # 1. Get Zernike basis polynomials (Z) over all points (rho, theta)
            # inside the given pupil. The shape is (P, J) where
            # P: num_enclosed_pixels, J: num_features
        # 2. Build weight matrix W = diag(w1...wP) where wi = 1/P and sum(wi) = 1
            # The shape of W is (P, P) with off-diagonal elements as 0.0.
        # 3. Compute Gram-matrix M = (Z^H)*W*Z, where H is Hermitian operation.
            # The shape of M matrix is (J, J). M is useful for normalization.
            # Off-diagonal elements, if non-zero, imply the overlap among the
            # basis polynomials for the given pupil.
            # Assuming Z are linearly independent, off-diagonal are zero, then
            # M is a positive definite matrix.
            # M^H = M is a property of M.
            # M[j, j] = 1/(nj + 1) => discrete form of normalization factor pi/(n+1)
        # 4. Cholesky decomposition and inverse computation with M = L*L^H.
            # The shape of L is (J, J). L is decomposition/factorization of M.
            # It is a lower-triangular matrix with positive diagonal elements.
            # Upper triangle can be explicitly set to 0.0
            # The inverse-Hermitian of L is (L^-1)^H, and called the transformation
            # matrix. Same shape (J, J).
            # New orthonormal basis polynomials for the given arbitrary pupil
            # are F = Z*(L^-1)^H. The shape of F is (P, J), and satisfies
            # the condition (F^H)*W*F = I
        # 5. The final step is computing complex moments, though not implemented
            # in this class, but in the main class ZernikeFeatures below.
            # These are computed for the given pupil as zm = (F^H)*W*pv.
            # The shape of zm is (J,), and of valid pixel vector is (P,).
            # The magnitude of these zm complex moments are Zernike features.
    cdef int _num_features, _num_enclosed_pixels
    cdef cnp.float64_t[::1] _weights
    cdef cnp.complex128_t[:,::1] _complex_basis
    cdef cnp.complex128_t[:,::1] _orthonormal_basis

    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=1] weights,
        cnp.ndarray[cnp.complex128_t, ndim=2] complex_basis,
        int num_features
    ):
        weights = np.ascontiguousarray(weights)
        complex_basis = np.ascontiguousarray(complex_basis)
        self._num_features = num_features
        self._num_enclosed_pixels = weights.shape[0]
        self._weights = weights
        self._complex_basis = complex_basis
        self._orthonormal_basis = np.empty_like(complex_basis)

    cdef void compute_orthonormal_basis(self):
        """Compute Zernike‑like orthonormal basis polynomials over arbitrary pupils."""
        cdef Py_ssize_t i, j
        cdef int J = self._num_features
        cdef int info = 0
        cdef char uplo = b'L'
        cdef char diag = b'N'

        cdef cnp.ndarray[cnp.complex128_t, ndim=2] gram = (np.transpose(np.conjugate(self._complex_basis)) * self._weights) @ self._complex_basis

        # F‑order arrays for LAPACK for lower-triangular cholesky decomposition of Gram matrix
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] cholesky = gram.copy(order="F")
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] cholesky_inv = np.empty((J, J), dtype=np.complex128, order='F')
        # Cholesky decomposition of Gram matrix (L * L^H = M)
        cython_lapack.zpotrf(&uplo, &J, <double complex*>cholesky.data, &J, &info)
        if info != 0:
            raise ZernikeValueError(
                f"LAPACK Cholesky decomposition failed. Code: {info}. Either number of pixels inside pupil, or degree value are too small. Change pupil size or degree value.")

        # explicitly set upper triangle to 0.0
        cholesky = cholesky.copy(order="C")
        for i in range(J):
            for j in range(i + 1, J):
                cholesky[i, j] = 0.0
                # cholesky[j, i] = 0.0

        cholesky_inv = cholesky.copy(order="F")
        cython_lapack.ztrtri(&uplo, &diag, &J, <double complex*>cholesky_inv.data, &J, &info)
        if info != 0:
            raise ZernikeValueError(f"LAPACK Choleskky inversion failed. Code: {info}. Either number of pixels inside pupil, or degree value are too small. Change pupil size or degree value.")

        # transformation matrix (L^-1)^H
        cdef cnp.ndarray[cnp.complex128_t, ndim=2] transform = np.transpose(np.conjugate(cholesky_inv))

        # orthonormal basis: F = Z @ (L^-1)^H
        self._orthonormal_basis = self._complex_basis @ transform

cdef class ImageReconstruction:
    """Reconstruct the pixel intensities inside the pupil using basis polynomials and complex moments."""
    # Given the basis polynomials F (either circular or non-circular), and
    # complex moments zm, the complex pixel values can be computed as,
    # pv = F*zm, where pv is pixel vector of shape (P,), F is (P, J), and zm is (J,).
    # The non-pupil image intensities are copied from the image. The intensities for
    # within pupil are real part, and clipped to [0, 255] grayscale range.
    cdef int _num_threads
    cdef Py_ssize_t _num_rows, _num_cols, _num_enclosed_pixels, _num_features
    cdef cnp.complex128_t[:,::1] _orthonormal_basis
    cdef cnp.complex128_t[::1] _complex_moments
    cdef cnp.float64_t[::1] _reconstructed_pixels
    cdef cnp.float64_t[:,::1] _image
    cdef cnp.uint8_t[:,::1] _reconstructed_image
    cdef cnp.int32_t[::1] _valid_rows, _valid_cols

    def __cinit__(
        self,
        *,
        int num_rows,
        int num_cols,
        int num_enclosed_pixels,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        cnp.ndarray[cnp.complex128_t, ndim=2] orthonormal_basis,
        cnp.ndarray[cnp.complex128_t, ndim=1] complex_moments,
        cnp.ndarray[cnp.int32_t, ndim=1] valid_rows,
        cnp.ndarray[cnp.int32_t, ndim=1] valid_cols,
        int num_threads=4
    ):
        # ensure contiguous inputs
        image = np.ascontiguousarray(image)
        orthonormal_basis = np.ascontiguousarray(orthonormal_basis)
        complex_moments = np.ascontiguousarray(complex_moments)
        valid_rows = np.ascontiguousarray(valid_rows)
        valid_cols = np.ascontiguousarray(valid_cols)

        self._num_rows = num_rows
        self._num_cols = num_cols
        self._num_enclosed_pixels = num_enclosed_pixels
        self._num_threads = num_threads
        self._image = image
        self._orthonormal_basis = orthonormal_basis
        self._complex_moments = complex_moments
        self._valid_rows = valid_rows
        self._valid_cols = valid_cols
        self._num_features = complex_moments.shape[0]
        self._reconstructed_pixels = np.empty(num_enclosed_pixels, dtype=np.float64)
        self._reconstructed_image = np.zeros((num_rows, num_cols), dtype=np.uint8)

    cdef void _compute_pixel_intensities(self):
        """Compute real part of the reconstructed complex image."""
        cdef double complex complex_pix = 0.0 + 0.0j
        cdef int i, j
        for i in range(self._num_enclosed_pixels):
            complex_pix = 0.0 + 0.0j
            for j in range(self._num_features):
                complex_pix = complex_pix + self._orthonormal_basis[i, j] * self._complex_moments[j]
            self._reconstructed_pixels[i] = complex_pix.real

    cdef void reconstruct_image(self):
        cdef Py_ssize_t i, j, idx, row, col
        cdef double pix

        self._compute_pixel_intensities()

        cdef cnp.float64_t[:,::1] oimg = self._image
        cdef cnp.uint8_t[:,::1] img = self._reconstructed_image
        cdef cnp.int32_t[::1] vrows = self._valid_rows
        cdef cnp.int32_t[::1] vcols = self._valid_cols
        cdef cnp.float64_t[::1] rpix = self._reconstructed_pixels

        idx = 0
        for i in prange(
            self._num_rows, nogil=True, num_threads=self._num_threads, schedule="static"
            ):
            for j in range(self._num_cols):
                row = vrows[idx]
                col = vcols[idx]
                if (i == row) and (j == col):
                    pix = rpix[idx]
                    # clamp to [0, 1] and scale to [0, 255]
                    if pix <= 0.0:
                        img[row, col] = 0
                    elif pix >= 1.0:
                        img[row, col] = 255
                    else:
                        img[row, col] = <cnp.uint8_t>(pix * 255.0)
                    idx = idx + 1
                else:
                    img[i, j] = <cnp.uint8_t>(oimg[i, j] * 255.0)


cdef class ZernikeFeatures:
    """Extract Zernike features from a given binary or grayscale image.

    This class implements computation of conventional or pseudo normalized
    Zernike features and normalized complex-valued Zernike moments. It uses
    binomial representation computed using gamma functions as provided by scipy,
    which maybe approximations of factorials but are computationally stable.
    It does not use explicit factorials or recursions to compute R-polynomial
    coefficients [1]. Hence, this implementation can compute polynomials of
    arbitrarily high degrees.

    Parameters
    ----------
    image : ndarray of shape (M, N)
        A binary or grayscale image of size ``(M, N)``. Internally scaled to ``[0., 1.]``
        scale for feature calculation and normalization. 64-bit float.
    feature_type : int
        Choice of Zernike feature type to compute.
        Strings mapped to integer in a dict in zernike.py
    degree : int
        The highest degree of polynomial to compute.
    pupil_type : int
        Choice of pupil shape to use for Zernike orthonormal basis.
        Strings mapped to integer in a dict in zernike.py
    primary_dim : double
        Pupil's primary dimension.
    secondary_dim : double
        Pupil's secondary dimension.
    center_coord : ndarray of shape (2,)
        Center of the pupil. 64-bit float.

    Returns
    -------
    ZernikeResults : dict with 4 keys
        ``"fts"`` : ndarray of shape (f,)
            Normalized Zernike features for the given input image. 64-bit floats.
        ``"cms"`` : ndarray of shape (f,)
            Normalized complex-valued Zernike moments for the given input image.  128-bit complex.
        ``"pmk"`` : ndarray of shape (M, N)
            Binary mask generated for the given pupil. Boolean.
        ``"rim"`` : ndarray of shape (M, N)
            Grayscale image reconstructed from the computed complex moments. 8-bit unit.

    Raises
    ------
    ZernikeTypeError, ZernikeValueError
        - If ``feature_type`` is not one of ``(0, 1)``.
        - If ``pupil_type`` is not one of 6 pupil shape.
        - If Cholesky decomposition and inverse computation fails for non-circular pupils.

    References
    ----------
    .. [1] Kuo Niu and Chao Tian 2022 J. Opt. 24 123001. https://doi.org/10.1088/2040-8986/ac9e08
    .. [2] Guang-ming Dai and Virendra N. Mahajan, Opt. Lett. 32, 74-76 (2007).https://doi.org/10.1364/OL.32.000074
    .. [3] Churui Li, Gongjian Guo, and Xiang Hao, J. Opt. Soc. Am. A 43, 1053-1066 (2026). https://doi.org/10.1364/JOSAA.601210
    .. [4] Dmitriy Otkupman, Sergey Bezdidko and Victoria Ostashenkova E3S Web Conf., 310 (2021) 01002. https://doi.org/10.1051/e3sconf/202131001002D
    .. [5] Michael Reed Teague, J. Opt. Soc. Am. 70, 920-930 (1980). https://doi.org/10.1364/JOSA.70.000920
    .. [6] Cosmas Mafusire and Tjaart P. J. Krüger, J. Opt. Soc. Am. A 35, 840-849 (2018). https://doi.org/10.1364/JOSAA.35.000840
    .. [7] Charles E. Campbell, J. Opt. Soc. Am. A 20, 209-217 (2003). https://doi.org/10.1364/JOSAA.20.000209
    .. [8] Zernike Polynomials Wikipedia: https://en.wikipedia.org/wiki/Zernike_polynomials
    .. [9] Pseudo Zernike Polynomials Wikipedia: https://en.wikipedia.org/wiki/Pseudo-Zernike_polynomials
    .. [10] Image reconstruction example using Zernike moments: https://stackoverflow.com/a/33339289
    .. [11] Gram matrix Wikipedia: https://en.wikipedia.org/wiki/Gram_matrix
    .. [12] Cholesky decomposition Wikipedia: https://en.wikipedia.org/wiki/Cholesky_decomposition

    """
    cdef int _feature_type, _pupil_type, _degree, _num_threads
    cdef double _primary_dim, _secondary_dim
    cdef cnp.complex128_t[::1] _complex_moments
    cdef cnp.float64_t[::1] _features
    cdef PupilGrid _pg
    cdef ZernikeComplexBasis _zcb
    cdef ZernikeOrthonormalization _zon
    cdef ImageReconstruction _imgrc
    def __cinit__(
        self,
        *,
        cnp.ndarray[cnp.float64_t, ndim=2] image,
        int feature_type,
        int pupil_type,
        int degree,
        double primary_dim,
        double secondary_dim,
        cnp.ndarray[cnp.float64_t, ndim=1] center_coord,
        int num_threads=4
    ):
        image = np.ascontiguousarray(image)
        center_coord = np.ascontiguousarray(center_coord)
        self._feature_type = feature_type
        self._pupil_type = pupil_type
        self._degree = degree
        self._primary_dim = primary_dim
        self._secondary_dim = secondary_dim
        self._num_threads = num_threads
        self._pg = PupilGrid(image=image, center_coord=center_coord)

    cdef long _compute_features(self) except -1:
        if self._pupil_type == 0:
            self._pg.build_circular_normalized_grid(radius=self._primary_dim)
        elif self._pupil_type == 1:
            self._pg.build_annular_normalized_grid(
                radius=self._primary_dim, annulus_obscure_radius=self._secondary_dim
            )
        elif self._pupil_type == 2:
            self._pg.build_elliptical_normalized_grid(
                ellipse_semi_major=self._primary_dim, ellipse_semi_minor=self._secondary_dim
            )
        elif self._pupil_type == 3:
            self._pg.build_rectangular_normalized_grid(
                rectangle_semi_width=self._primary_dim, rectangle_semi_height=self._secondary_dim
            )
        elif self._pupil_type == 4:
            self._pg.build_square_normalized_grid(square_semi_side=self._primary_dim)
        elif self._pupil_type == 5:
            self._pg.build_hexagonal_normalized_grid(hexagon_side=self._primary_dim)
        else:
            # probably will never be raised. added to avoid dangling else.
            raise ZernikeValueError("Invalid 'pupil_type'.")

        cdef Py_ssize_t one_pix
        cdef cnp.ndarray[cnp.float64_t, ndim=1] pixel_vector = np.zeros((self._pg._num_enclosed_pixels,), dtype=np.float64)
        cdef cnp.float64_t[:,::1] vim = self._pg._valid_image
        cdef cnp.int32_t[::1] vr = self._pg._valid_rows
        cdef cnp.int32_t[::1] vc = self._pg._valid_cols

        self._zcb = ZernikeComplexBasis(
            valid_rows=np.array(self._pg._valid_rows,dtype=np.int32),
            valid_cols=np.array(self._pg._valid_cols,dtype=np.int32),
            polar_rho=np.array(self._pg._polar_rho,dtype=np.float64),
            polar_theta=np.array(self._pg._polar_theta,dtype=np.float64),
            degree=self._degree,
        )
        if self._feature_type == 0:
            # self._zcb.build_paired_data_conv_zf()
            self._zcb.compute_conventional_zernike_basis()
        elif self._feature_type == 1:
            # self._zcb.build_paired_data_pseudo_zf()
            self._zcb.compute_pseudo_zernike_basis()
        else:
            # probably will never be raised. added to avoid dangling else.
            raise ZernikeValueError("Invalid 'feature_type'")

        for one_pix in prange(
            self._pg._num_enclosed_pixels,
            nogil=True,
            schedule="static",
            num_threads=self._num_threads
        ):
            pixel_vector[one_pix] = vim[vr[one_pix], vc[one_pix]]

        if self._pupil_type == 0:
            self._complex_moments = np.array(self._zcb._scale_factors) * \
                                    (np.transpose(self._zcb._complex_basis) @ \
                                    np.array(pixel_vector, dtype=np.float64))
            self._imgrc = ImageReconstruction(
                image=np.array(self._pg._image, dtype=np.float64),
                orthonormal_basis=np.array(self._zcb._complex_basis,dtype=np.complex128),
                complex_moments=np.array(self._complex_moments,dtype=np.complex128),
                valid_rows=np.array(self._pg._valid_rows,dtype=np.int32),
                valid_cols=np.array(self._pg._valid_cols,dtype=np.int32),
                num_rows=self._pg._num_rows,
                num_cols=self._pg._num_cols,
                num_enclosed_pixels=self._pg._num_enclosed_pixels,
                num_threads=self._num_threads
            )
        else:
            self._zon = ZernikeOrthonormalization(
                weights=np.array(self._pg._weights,np.float64),
                complex_basis=np.array(self._zcb._complex_basis,dtype=np.complex128),
                num_features=self._zcb._num_features
            )
            self._zon.compute_orthonormal_basis()
            self._complex_moments = np.transpose(np.conjugate(self._zon._orthonormal_basis)) @ \
                                    (np.array(self._pg._weights, dtype=np.float64) * \
                                    np.array(pixel_vector, dtype=np.float64))
            self._imgrc = ImageReconstruction(
                image=np.array(self._pg._image, dtype=np.float64),
            orthonormal_basis=np.array(self._zon._orthonormal_basis,dtype=np.complex128),
            complex_moments=np.array(self._complex_moments,dtype=np.complex128),
            valid_rows=np.array(self._pg._valid_rows,dtype=np.int32),
            valid_cols=np.array(self._pg._valid_cols,dtype=np.int32),
            num_rows=self._pg._num_rows,
            num_cols=self._pg._num_cols,
            num_enclosed_pixels=self._pg._num_enclosed_pixels,
            num_threads=self._num_threads
            )
        self._features = np.absolute(self._complex_moments)
        self._imgrc.reconstruct_image()
        return 0

    def compute_features(self):
        self._compute_features()
        return {
            "fts": np.array(self._features, copy=True),
            "cms": np.array(self._complex_moments, copy=True),
            "pmk": np.array(self._pg._pupil_mask, dtype=np.bool, copy=True),
            "rim": np.array(self._imgrc._reconstructed_image, dtype=np.uint8, copy=True)
        }
