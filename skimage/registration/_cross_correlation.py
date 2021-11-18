"""
Implementation of the masked normalized cross-correlation.

Based on the following publication:
D. Padfield. Masked object registration in the Fourier domain.
IEEE Transactions on Image Processing (2012)

and the author's original MATLAB implementation, available on this website:
http://www.dirkpadfield.com/
"""


from functools import partial

import numpy as np
import scipy.fft as fftmodule
from scipy.fft import next_fast_len

from .._shared.utils import _supported_float_type


def cross_correlation(arr1, arr2, mask1=None, mask2=None, mode="full",
                      axes=None, pad_axes=None,
                      space="real", normalization="phase",
                      upsample_factor=1, overlap_ratio=0.3):
    """Masked/unmasked cross-correlation between two arrays.

    Parameters
    ----------
    arr1 : ndarray
        First array.
    arr2 : ndarray
        Seconds array. The dimensions of `arr2` along axes that are not
        transformed should be equal to that of `arr1`.
    mask1 : ndarray
        Mask of `arr1`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m1` should have the same shape as `arr1`.
    mask2 : ndarray
        Mask of `arr2`. The mask should evaluate to `True`
        (or 1) on valid pixels. `m2` should have the same shape as `arr2`.
    mode : {'full', 'same'}, optional
        'full':
            This returns the convolution at each point of overlap. At
            the end-points of the convolution, the signals do not overlap
            completely, and boundary effects may be seen.
        'same':
            The output is the same size as `arr1`, centered with respect
            to the `‘full’` output. Boundary effects are less prominent.
    upsample_factor: float
        Arr1 and Arr2 will be upsampled by this factor providing some
    axes : tuple of ints, optional
        Axes along which to compute the cross-correlation.
    pad_axes : tuple of ints, optional
        Axes along which to pad the data with zeros.
    normalization:{"Normalized","phase", None}
        The normalization applied to the signal. The default is to calculate the normalized
        cross-correlation.(Insert math)
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images.

    Returns
    -------
    out : ndarray
        Masked normalized cross-correlation.

    Raises
    ------
    ValueError : if correlation `mode` is not valid, or array dimensions along
        non-transformation axes are not equal.

    References
    ----------
    .. [1] Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). :DOI:`10.1109/TIP.2011.2181402`
    .. [2] D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           :DOI:`10.1109/CVPR.2010.5540032`
    """
    if mode not in {'full', 'same'}:
        raise ValueError(f"Correlation mode '{mode}' is not valid.")
    if normalization not in {'phase', 'normalized', None}:
        raise ValueError(f"Correlation normalization '{normalization}' is not valid.")
    if pad_axes is None:
        pad_axes = ()

    if mask1 is not None or mask2 is not None:
        cross_correlation = _masked_cross_correlation(arr1, arr2,
                                                      mask1, mask2,
                                                      mode, axes, pad_axes,
                                                      space, normalization,
                                                      upsample_factor=upsample_factor)
    else:
        cross_correlation = _cross_correlation(arr1, arr2,
                                               mask1, mask2,
                                               mode, axes, pad_axes,
                                               space, normalization,
                                               upsample_factor=upsample_factor)
        return cross_correlation

def _ifft_upsample(x, axes, upsample_factor=1, ):
    shifted = np.fft.fftshift(x)
    padding = np.array(np.round((upsample_factor - 1) * np.array(np.shape(x)) / 2), dtype=int)
    shifted = np.pad(shifted, padding)
    unshifted = np.fft.ifftshift(shifted)
    transformed = fftmodule.ifftn(unshifted, axes=axes)
    return transformed.real


def get_fft_ifft(arr1_shape, arr2_shape, axes, pad_axes, mode="same", upsample_factor=1):
    dim = len(arr1_shape)

    # Array dimensions along non-transformation axes should be equal.
    if axes is not None:
        transformed_axes = set((dim + a if a < 0 else a for a in axes))
        non_transformed_axes = set(range(dim)) - transformed_axes
        for axis in non_transformed_axes:
            if arr1_shape[axis] != arr2_shape[axis]:
                raise ValueError(
                    f'Array shapes along non-transformation axes should be '
                    f'equal, but dimensions along axis {axis} are not.')
    else:
        axes = range(dim)

    # Determine final size along transformation axes
    # Note that it might be faster to compute Fourier transform in a slightly
    # larger shape (`fast_shape`). Then, after all fourier transforms are done,
    # we slice back to`final_shape` using `final_slice`.
    final_shape = list(arr1_shape)
    for axis in axes:
        if axis in pad_axes:
            final_shape[axis] = arr1_shape[axis] + \
                                arr2_shape[axis] - 1
    final_shape = tuple(final_shape)
    if upsample_factor is not 1:
        final_shape = tuple([int(upsample_factor * sz) for sz in final_shape])
    if mode is "full":
        final_slice = tuple([slice(0, int(sz)) for sz in final_shape])

    else:
        final_slice = tuple([slice(int(sz // 4), int(sz - sz // 4)) for sz in final_shape])
    # Extent transform axes to the next fast length (i.e. multiple of 3, 5, or
    # 7)
    fast_shape = tuple(next_fast_len(final_shape[ax])
                       if ax in pad_axes
                       else final_shape[ax] for ax in axes)

    # We use the new scipy.fft because they allow leaving the transform axes
    # unchanged which was not possible with scipy.fftpack's
    # fftn/ifftn in older versions of SciPy.
    # E.g. arr shape (2, 3, 7), transform along axes (0, 1) with shape (4, 4)
    # results in arr_fft shape (4, 4, 7)
    fft = partial(fftmodule.fftn, s=fast_shape, axes=axes)
    if upsample_factor is 1:
        _ifft = partial(fftmodule.ifftn, s=fast_shape, axes=axes)

        # assume complex data is already in Fourier space
        def ifft(x):
            return _ifft(x).real
    else:
        ifft = partial(_ifft_upsample, axes=axes, upsample_factor=upsample_factor)

    return fft, ifft, final_slice, final_shape

def _cross_correlation(arr1, arr2,
                       mode="full",
                       axes=None, pad_axes=None,
                       normalization="phase",
                       upsample_factor=1):
    fft, ifft, final_slice, final_shape = get_fft_ifft(arr1_shape=arr1.shape,
                                                       arr2_shape=arr2.shape,
                                                       axes=axes, mode=mode,
                                                       pad_axes=pad_axes,
                                                       upsample_factor=upsample_factor)
    float_dtype = _supported_float_type([arr1.dtype, arr2.dtype])
    eps = np.finfo(float_dtype).eps
    arr1_fft = fft(arr1)
    rotated_arr2 = _flip(arr2, axes=axes)
    rotated_arr2_fft = fft(rotated_arr2)

    if normalization is None:
        return ifft(arr1_fft*rotated_arr2_fft)
    elif normalization is "phase":
        product = arr1_fft * rotated_arr2_fft
        product /= np.maximum(np.abs(product), 100 * eps)
    elif normalization is "normalized":
        numerator = ifft(arr1_fft*rotated_arr2_fft)




def _masked_cross_correlation(arr1, arr2, mask1=None,
                              mask2=None, mode="full",
                              axes=None, pad_axes=None,
                              space="real", normalization="phase",
                              upsample_factor=1,
                              overlap_ratio=0.3):
    fft, ifft, final_slice, final_shape = get_fft_ifft(arr1_shape=arr1.shape,
                                                       arr2_shape=arr2.shape,
                                                       axes=axes, mode=mode,
                                                       pad_axes=pad_axes,
                                                       upsample_factor=upsample_factor)
    float_dtype = _supported_float_type([arr1.dtype, arr2.dtype])
    eps = np.finfo(float_dtype).eps

    # set masked values eqaul to zero
    arr1[np.logical_not(mask1)] = 0.0
    arr2[np.logical_not(mask2)] = 0.0

    # N-dimensional analog to rotation by 180deg is flip over all relevant axes.
    # See [1] for discussion.
    rotated_arr2 = _flip(arr2, axes=axes)
    rotated_mask2 = _flip(mask2, axes=axes)

    arr1_fft = fft(arr1)
    mask1_fft = fft(mask1)

    rotated_arr2_fft = fft(rotated_arr2)
    rotated_mask2_fft = fft(rotated_mask2)

    # Calculate overlap of masks at every point in the convolution.
    # Locations with high overlap should not be taken into account.
    number_overlap_masked_px = ifft(rotated_mask2_fft * mask1_fft)
    number_overlap_masked_px[:] = np.round(number_overlap_masked_px)
    number_overlap_masked_px[:] = np.fmax(number_overlap_masked_px, eps)

    numerator = ifft(rotated_arr2_fft * arr1_fft)

    if normalization is None:
        # Normalize by the number of pixels which contribute to each
        # point in the correlation
        out = numerator * (number_overlap_masked_px /
                           np.max(number_overlap_masked_px))

        out = out[final_slice]
        number_overlap_masked_px = number_overlap_masked_px[final_slice]

    elif normalization is "phase":
        # Normalize
        product = rotated_arr2_fft * arr1_fft
        eps = np.finfo(product.real.dtype).eps
        product /= np.maximum(np.abs(product), 100 * eps)
        out = ifft(product) * (number_overlap_masked_px /
                               np.max(number_overlap_masked_px))
        out = out[final_slice]
        number_overlap_masked_px = number_overlap_masked_px[final_slice]
    elif normalization is "normalized":
        masked_correlated_fixed_fft = ifft(rotated_mask2_fft * arr1_fft)
        masked_correlated_rotated_moving_fft = ifft(
            mask1_fft * rotated_arr2_fft)

        numerator -= masked_correlated_fixed_fft * \
                     masked_correlated_rotated_moving_fft / number_overlap_masked_px

        arr1_squared_fft = fft(np.square(arr1))
        arr1_denom = ifft(rotated_mask2_fft * arr1_squared_fft)
        arr1_denom -= np.square(masked_correlated_fixed_fft) / \
                      number_overlap_masked_px
        arr1_denom[:] = np.fmax(arr1_denom, 0.0)

        rotated_arr2_squared_fft = fft(np.square(rotated_arr2))
        arr2_denom = ifft(mask1_fft * rotated_arr2_squared_fft)
        arr2_denom -= np.square(masked_correlated_rotated_moving_fft) / \
                      number_overlap_masked_px
        arr2_denom[:] = np.fmax(arr2_denom, 0.0)

        denom = np.sqrt(arr1_denom * arr2_denom)

        # Slice back to expected convolution shape.
        numerator = numerator[final_slice]
        denom = denom[final_slice]
        number_overlap_masked_px = number_overlap_masked_px[final_slice]

        if mode == 'same':
            if upsample_factor is not 1:
                new_shape = [int(round(sz * upsample_factor)) for sz in arr1.shape]
            else:
                new_shape = arr1.shape
            _centering = partial(_centered,
                                 newshape=new_shape, axes=axes)
            denom = _centering(denom)
            numerator = _centering(numerator)
            number_overlap_masked_px = _centering(number_overlap_masked_px)

        # Pixels where `denom` is very small will introduce large
        # numbers after division. To get around this problem,
        # we zero-out problematic pixels.
        tol = 1e3 * eps * np.max(np.abs(denom), axis=axes, keepdims=True)
        nonzero_indices = denom > tol

        # explicitly set out dtype for compatibility with SciPy < 1.4, where
        # fftmodule will be numpy.fft which always uses float64 dtype.
        out = np.zeros_like(denom, dtype=float_dtype)
        out[nonzero_indices] = numerator[nonzero_indices] / denom[nonzero_indices]
        np.clip(out, a_min=-1, a_max=1, out=out)
    else:
        return

    # Apply overlap ratio threshold
    number_px_threshold = overlap_ratio * np.max(number_overlap_masked_px,
                                                 axis=axes, keepdims=True)
    out[number_overlap_masked_px < number_px_threshold] = 0.0

    return out


def _centered(arr, newshape, axes):
    """ Return the center `newshape` portion of `arr`, leaving axes not
    in `axes` untouched. """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)

    slices = [slice(None, None)] * arr.ndim

    for ax in axes:
        startind = (currshape[ax] - newshape[ax]) // 2
        endind = startind + newshape[ax]
        slices[ax] = slice(startind, endind)

    return arr[tuple(slices)]


def _flip(arr, axes=None):
    """ Reverse array over many axes. Generalization of arr[::-1] for many
    dimensions. If `axes` is `None`, flip along all axes. """
    if axes is None:
        reverse = [slice(None, None, -1)] * arr.ndim
    else:
        reverse = [slice(None, None, None)] * arr.ndim
        for axis in axes:
            reverse[axis] = slice(None, None, -1)

    return arr[tuple(reverse)]
