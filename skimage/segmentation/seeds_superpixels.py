"""
Implementation of the SEEDS segmentation algorithm [1]_ for scikit-image.

Written by Geoffrey French 2015 and released under BSD license.

References
----------

.. [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool..
       "SEEDS: Superpixels extracted via energy-driven sampling."
       Proceedings of the European Conference on Computer Vision, pages 13-26,
       2012.
"""
import math
import numpy as np
from scipy.cluster.vq import kmeans2
from .._shared.utils import assert_nD
from ..util import img_as_float

from ..segmentation import _seeds


HIST_SIZE_RGB = 50
HIST_SIZE_GREYSCALE = 25


def _upscale_labels_by_2(labels_2d, h, w):
    """
    Upscale a label image by a factor of 2.

    Parameters
    ----------
    labels_2d : the label image to upscale
    h : target height; if `h > labels_2d.shape[0]*2`, then the last row will
    be repeated as necessary
    w : target width; if `w > labels_2d.shape[1]*2`, then the last column will
    be repeated as necessary

    Returns
    -------
    Upscaled label image of shape `(h,w)`.
    """
    h2 = labels_2d.shape[0] * 2
    w2 = labels_2d.shape[1] * 2
    rem_r = h & 1
    rem_c = w & 1

    rep = np.repeat(np.repeat(labels_2d, 2, 1), 2, 0)
    if rem_r == 0 and rem_c == 0:
        return rep
    else:
        up_labels_2d = np.zeros((h,w), dtype=np.int32)
        up_labels_2d[:h2, :w2] = rep

        if rem_r != 0:
            up_labels_2d[-1, :] = up_labels_2d[-2, :]
        if rem_c != 0:
            up_labels_2d[:, -1] = up_labels_2d[:, -2]

    return up_labels_2d


def seeds(input_2d, hist_size=None, num_superpixels=200, n_levels=5,
          block_refine_iters=10, pixel_refine_iters=10,
          score_threshold=0.0, min_label_area_factor=0.0,
          return_steps=False):
    """
    Segment and image using the SEEDS [1]_ super-pixel segmentation algorithm.

    Parameters
    ----------
    input_2d : the image to segment as either; an RGB image as a 3D array,
    a greyscale image as a 2D array (must have ``dtype`` of ``float`` or
    ``'float64'``), or an index image (must be ``dtype`` of ``'uint8'``,
    ``'uint16'`` or ``int``)
    hist_size : the number of bins in the histogram; if the image is RGB or
    greyscale, it will be quantized to have this many colours first. If
    no value is provided, 50 will be used for RGB images and 25 for greyscale.
    num_superpixels : the number of super-pixels to aim for. Given that
    the algorithm works by repeatedly downscaling by a factor of 2,
    ``num_superpixels`` is more of a ball-park
    n_levels : the algorithm tries to divide the image into
    ``num_superpixels`` square blocks. These blocks will be recursively
    divided by 2 ``n_levels`` times. Note that blocks will not be divided
    below 2 pixels in size, so increasing ``n_levels`` further will make no
    difference in these cases.
    block_refine_iters : after dividing the image into blocks, at each
    level the algorithm re-labels blocks from one label to a neighbouring
    label if doing so will improve the quality of the segmentation.
    ``block_refine_iters`` is the number of passes over the image that are
    performed for each level.
    pixel_refine_iters : after all block-level refinements have been
    performed, the same procedure is repeated at the pixel level.
    score_threshold : blocks or pixels will only be re-labelled if doing so
    will result in an improvement score greater than ``score_threshold``.
    min_label_area_factor : at the start, the image is divided into square
    labelled regions. Re-labelling blocks or pixels will only be permitted
    if doing so does not reduce the area covered by a label below
    ```initial_label_area*min_label_area_factor```.
    return_steps: If True, will return the intermediate results of the
    algorithm; see below.

    Returns
    -------
    If ``return_steps`` is False, a 2D label image, with labels starting at 1.
    If ``return_steps`` is True, a tuple of
    ```(labels, labels_by_level, index_2d, quantized_colours)```, where
    ``labels`` is the final label image, ``labels_by_level`` is a list of
    label images generated at each step, starting at the coarsest level and
    finishing at the finest, ``index_2d`` is the index image after
    quantization if ``input_2d`` is RGB or greyscale or just ``input_2d`` if
    it was an index image, and ``quantized_colours`` are the quantized
    colours or greyscale values as a 2D array if ``input_2d`` is RGB or
    greyscale or ``None`` otherwise.

    References
    ----------

    .. [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool..
           "SEEDS: Superpixels extracted via energy-driven sampling."
           Proceedings of the European Conference on Computer Vision,
           pages 13-26, 2012.
    """
    assert n_levels >= 1
    assert_nD(input_2d, [2,3])
    
    # Get dimensions of input image
    h, w = input_2d.shape[:2]

    # Compute the number of superpixels vertically and horizontally
    n_superpixels_h = int(math.sqrt(float(num_superpixels) * w / h)+0.5)
    n_superpixels_w = n_superpixels_h * w / h

    # Compute the width and height of the blocks at the finest level
    block_finest_w_f = float(w) / n_superpixels_w / (2**(n_levels-1))
    block_finest_h_f = float(h) / n_superpixels_h / (2**(n_levels-1))

    # Reduce n_levels if necessary so that finest block sizes are > 1.0
    while block_finest_w_f <= 1.0 or block_finest_h_f <= 1.0:
        n_levels -= 1
        block_finest_w_f = float(w) / n_superpixels_w / (2**(n_levels-1))
        block_finest_h_f = float(h) / n_superpixels_h / (2**(n_levels-1))
    
    # Compute seed size
    seed_w = max(int(block_finest_w_f+0.5), 1)
    seed_h = max(int(block_finest_h_f+0.5), 1)

    # Get an index image; determine what to do.
    if len(input_2d.shape) == 3 and input_2d.shape[2] == 3:
        # Its RGB; vector quantize
        if input_2d.dtype == np.uint8 or input_2d.dtype == np.uint16:
            # Convert to float
            input_2d = img_as_float(input_2d)

        # Quantize image

        # Default to 50 histogram bins
        if hist_size is None:
            hist_size = HIST_SIZE_RGB

        quantized_colours, index_1d = kmeans2(input_2d.reshape((-1,3)),
                                              hist_size, minit='points',
                                              iter=50)
        index_2d = index_1d.reshape(input_2d.shape[:2])
    elif len(input_2d.shape) == 2:
        if input_2d.dtype == float or input_2d.dtype == np.float32:
            # Floating point grey-scale; quantize
            input_2d = input_2d[:,:,None]

            # Default to 25 histogram bins
            if hist_size is None:
                hist_size = HIST_SIZE_GREYSCALE

            quantized_colours, index_1d = kmeans2(input_2d.reshape((-1,1)),
                                                  hist_size, minit='points',
                                                  iter=50)
            index_2d = index_1d.reshape(input_2d.shape[:2])
        elif input_2d.dtype == np.uint8 or input_2d.dtype == np.uint16 or \
                input_2d.dtype == int:
            # Index image; use as is

            # Override hist_size
            hist_size = input_2d.max() + 1

            # No quantixed colours
            quantized_colours = None

            # Clip away negative values
            index_2d = np.clip(input_2d, 0, hist_size)
        else:
            raise ValueError('SEEDS can only work with 2D image arrays ' +\
                'are of type float, float64, uint8, uint16 or int, ' +\
                'not {0}'.format(input_2d.dtype))
    else:
        raise ValueError('SEEDS can only work with either RGB images ' +\
            'as 3D arrays or greyscale or index images as 2D arrays; ' +\
            'don\'t know what to do with array of ' +\
            'shape {0}'.format(input_2d.shape))

    if index_2d.dtype != np.int32:
        index_2d = index_2d.astype(np.int32)

    # Build histograms for each level
    hist_2ds_by_level = []
    metrics_by_level = []
    seed_metrics = _seeds.LevelMetrics()
    seed_metrics.initialise(h, w, seed_h, seed_w)
    # Build seed level histogram
    hist_2d = _seeds.build_seed_histogram(index_2d, hist_size, seed_metrics)
    hist_2ds_by_level.append(hist_2d)
    metrics_by_level.append(seed_metrics)
    l_metrics_prev = seed_metrics
    for i in range(1, n_levels):
        hist_2d = _seeds.downscale_histogram(hist_2d)
        hist_2ds_by_level.append(hist_2d)
        l_metrics = _seeds.LevelMetrics()
        l_metrics.downscale(l_metrics_prev)
        metrics_by_level.append(l_metrics)
        l_metrics_prev = l_metrics

    bottom_labels_shape = l_metrics.n_blocks_r, l_metrics.n_blocks_c
    n_labels = np.prod(bottom_labels_shape)
    labels = np.arange(1, n_labels+1).astype(np.int32)
    labels = labels.reshape(bottom_labels_shape)
    labels_by_level = [None] * n_levels
    labels_by_level[-1] = labels
    
    label_hists = np.array(hist_2d).reshape((n_labels, hist_size))
    label_areas = []
    for y in range(l_metrics.n_blocks_r):
        for x in range(l_metrics.n_blocks_c):
            label_areas.append(l_metrics.block_area(y, x))
    label_areas = np.array(label_areas, dtype=np.int32)
    
    block_area = l_metrics.block_area(0, 0)
    block_area_threshold = int(block_area * min_label_area_factor+0.5)


    # For each level above the pixel level
    labels_prev_level = labels_by_level[-1]
    for level in range(n_levels-2, -1, -1):
        labels = _upscale_labels_by_2(labels_prev_level,
                                      metrics_by_level[level].n_blocks_r,
                                      metrics_by_level[level].n_blocks_c)
        labels_by_level[level] = labels

        block_hists = hist_2ds_by_level[level]

        _seeds.refine_blocks(block_refine_iters, labels, label_hists,
                             label_areas, block_hists, metrics_by_level[level],
                             block_area_threshold, score_threshold)
        
        labels_prev_level = labels
        

    # Pixel level                
    pixel_labels = _seeds.upscale_labels(labels_prev_level,
                                         metrics_by_level[0])

    _seeds.refine_pixels(pixel_refine_iters, pixel_labels, label_hists,
                         label_areas, index_2d, block_area_threshold,
                         score_threshold)


    if return_steps:
        for lvl in range(len(labels_by_level)):
            met = metrics_by_level[lvl]
            labels_by_level[lvl] = np.array(
                _seeds.upscale_labels(labels_by_level[lvl], met))
        return np.array(pixel_labels), labels_by_level, \
               index_2d, quantized_colours
    else:
        return np.array(pixel_labels)