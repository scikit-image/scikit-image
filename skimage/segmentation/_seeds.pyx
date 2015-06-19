"""
Helper module for implementation of the SEEDS segmentation algorithm [1]_.

Written by Geoffrey French 2015 and released under BSD license.


References
----------

.. [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool..
       "SEEDS: Superpixels extracted via energy-driven sampling."
       Proceedings of the European Conference on Computer Vision, pages 13-26,
       2012.
"""
import cython
import numpy as np



cdef class LevelMetrics:
    """
    Describes the downscaling that was performed to get to the level described
    by `self`

    Attributes
    ----------
    h, w : the height and width of the original image
    block_h, block_w : the height and width of a block in the downsampled image
    last_block_h, last_block_w : the height and width of blocks in the last
    row and column respectively; used in cases where the `block_h` and
    `block_w` do not divide exactly into `h` and `w`
    rem_y, rem_x : the remainder of dividing `h` and `h` by `block_h` and
    `block_w` respectively
    n_blocks_y, n_blocks_x : the number of `(block_h,block_w)` blocks that
    fit into the `(h,w)` image

    """
    cdef readonly int h, w
    cdef readonly int block_h, block_w
    cdef readonly int last_block_h, last_block_w
    cdef readonly int rem_y, rem_x
    cdef readonly int n_blocks_y, n_blocks_x

    cpdef initialise(self, int h, int w, int block_h, int block_w):
        """
        Initialise, given an image and a block size

        Parameters
        ----------
        h, w : the shape of the image
        block_h, block_w : the shape of blocks
        """
        self.h = h
        self.w = w
        self.block_h = block_h
        self.block_w = block_w
        self.n_blocks_y = h // block_h
        self.n_blocks_x = w // block_w
        self.rem_y = h % block_h
        self.rem_x = w % block_w
        self.last_block_h = block_h + self.rem_y
        self.last_block_w = block_w + self.rem_x

    cpdef downscale(self, LevelMetrics metrics):
        """
        Initialise by downscaling an existing level by a factor of 2.
        """
        self.h = metrics.h
        self.w = metrics.w
        self.block_h = metrics.block_h * 2
        self.block_w = metrics.block_w * 2
        self.n_blocks_y = metrics.n_blocks_y // 2
        self.n_blocks_x = metrics.n_blocks_x // 2
        self.rem_y = metrics.h % self.block_h
        self.rem_x = metrics.w % self.block_w
        self.last_block_h = self.block_h + self.rem_y
        self.last_block_w = self.block_w + self.rem_x

    cpdef int block_area(self, int y, int x):
        """
        Compute the area of the block in original image pixels at the given
        co-ordinates

        Parameters
        ----------
        y, x: the block co-ordinates

        Returns
        -------
        The area in pixels as an integer
        """
        cdef int bw, bh
        if y < (self.n_blocks_y-1):
            bh = self.block_h
        else:
            bh = self.last_block_h
        if x < (self.n_blocks_x-1):
            bw = self.block_w
        else:
            bw = self.last_block_w
        return bw * bh



@cython.boundscheck(False)
def upscale_labels_by_2(int[:,:] labels_2d, int h, int w):
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
    cdef int h2 = labels_2d.shape[0]*2, w2 = labels_2d.shape[1]*2
    cdef int rem_y = h & 1, rem_x = w & 1

    cdef int[:,:] up_labels_2d = np.zeros((h,w), dtype=int)
    up_labels_2d[:h2:2,:w2:2] = labels_2d
    up_labels_2d[1:h2:2,:w2:2] = labels_2d
    up_labels_2d[:h2:2,1:w2:2] = labels_2d
    up_labels_2d[1:h2:2,1:w2:2] = labels_2d

    if rem_y != 0:
        up_labels_2d[-1,:w2:2] = labels_2d[-1,:]
        up_labels_2d[-1,1:w2:2] = labels_2d[-1:]
        if rem_x != 0:
            up_labels_2d[-1,-1] = labels_2d[-1,-1]
    if rem_x != 0:
        up_labels_2d[:h2:2,-1] = labels_2d[:,-1]
        up_labels_2d[1:h2:2,-1] = labels_2d[:,-1]


    return up_labels_2d


@cython.boundscheck(False)
def upscale_labels(int[:,:] labels_2d, LevelMetrics metrics):
    """
    Upscale labels image according to a `LevelMetrics` instance.
    The resulting image size will be of shape `(metrics.h, metrics.w)`.
    The size of the blocks that are scaled up is
    `(metrics.block_h, metrics.block_w)`.
    The number of blocks is `(metrics.n_blocks_y, metrics.n_blocks_x)`.
    The last row and column are repeated if extra padding is required to make
    up to `(metrics.h, metrics.w)`.

    Parameters
    ----------
    labels_2d : the labels image to upscale
    metrics : the `LevelMetrics` instance that describes the scale factor

    Returns
    -------
    Upscaled label image of shape `(metrics.h, metrics.w)`
    """
    cdef int[:,:] up_labels_2d = np.zeros((metrics.h, metrics.w), dtype=np.int32)
    cdef int x, y, i, j, block_w, block_h

    for y in range(metrics.n_blocks_y):
        if y < (metrics.n_blocks_y-1):
            block_h = metrics.block_h
        else:
            block_h = metrics.last_block_h
        for x in range(metrics.n_blocks_x):
            if x < (metrics.n_blocks_x-1):
                block_w = metrics.block_w
            else:
                block_w = metrics.last_block_w
            for j in range(block_h):
                for i in range(block_w):
                    up_labels_2d[y*metrics.block_h+j,
                                 x*metrics.block_w+i] = labels_2d[y,x]

    return up_labels_2d


@cython.boundscheck(False)
def build_seed_histogram(int[:,:] index_2d, int hist_size, LevelMetrics metrics):
    """
    Build the base level histogram from the the provided index image.
    Generates a histogram for each block of shape
    `(metrics.block_h, metrics.block_w)`, of which the image is divided into
    `(metrics.n_blocks_y, metrics.n_blocks_x)` blocks. Last rows and columns
    of pixels from the image are included in their nearest respective blocks
    if the image size does not divide exactly.

    Parameters
    ----------
    index_2d : an index image giving the index of each pixel in the source
    image
    hist_size : histogram size (number of 'bars')
    metrics : the `LevelMetrics` instance that describes the scale factor

    Returns
    -------
    The histogram as an array of shape
    `(metrics.n_blocks_y, metrics.n_blocks_x, hist_size)`
    """
    cdef int[:,:,:] hist_2d = np.zeros((metrics.n_blocks_y, metrics.n_blocks_x,
                                        hist_size), dtype=np.int32)
    cdef int x, y, i, j, block_h, block_w, n
    for y in range(metrics.n_blocks_y):
        if y < (metrics.n_blocks_y-1):
            block_h = metrics.block_h
        else:
            block_h = metrics.last_block_h
        for x in range(metrics.n_blocks_x):
            if x < (metrics.n_blocks_x-1):
                block_w = metrics.block_w
            else:
                block_w = metrics.last_block_w
            for j in range(block_h):
                for i in range(block_w):
                    n = index_2d[y*metrics.block_h+j,x*metrics.block_w+i]
                    hist_2d[y,x,n] += 1
    return hist_2d

@cython.boundscheck(False)
def downscale_histogram(int[:,:,:] hist_2d):
    """
    Downscale an image histogram by a factor of 2. If the height and/or width
    do not divide by 2 exactly, the last row/column in the resulting image
    will be the result of joining 3 rows/columns from `hist_2d`.

    Parameters
    ----------
    hist_2d : the histogram to downscale, as an array of shape
    `(height, width, n_bins)`.

    Returns
    -------
    The downscaled histogram as an array of shape
    `(height/2, width/2, n_bins)`.
    """
    cdef int h2 = hist_2d.shape[0]//2, w2 = hist_2d.shape[1]//2
    cdef int hist_size = hist_2d.shape[2]
    cdef int[:,:,:] ds_hist_2d = np.zeros((h2, w2, hist_size), dtype=np.int32)
    cdef int x, y, i
    cdef int y_rem = hist_2d.shape[0] & 1
    cdef int x_rem = hist_2d.shape[1] & 1
    for y in range(h2):
        for x in range(w2):
            for i in range(hist_size):
                ds_hist_2d[y,x,i] += hist_2d[y*2,x*2,i]
                ds_hist_2d[y,x,i] += hist_2d[y*2,x*2+1,i]
                ds_hist_2d[y,x,i] += hist_2d[y*2+1,x*2,i]
                ds_hist_2d[y,x,i] += hist_2d[y*2+1,x*2+1,i]
                if x == (w2-1) and x_rem != 0:
                    ds_hist_2d[y,x,i] += hist_2d[y*2,x*2+2,i]
                    ds_hist_2d[y,x,i] += hist_2d[y*2+1,x*2+2,i]
                    if y_rem != 0:
                        ds_hist_2d[y,x,i] += hist_2d[y*2+2,x*2+2,i]
                if y == (h2-1) and y_rem != 0:
                    ds_hist_2d[y,x,i] += hist_2d[y*2+2,x*2,i]
                    ds_hist_2d[y,x,i] += hist_2d[y*2+2,x*2+1,i]

    return ds_hist_2d

@cython.boundscheck(False)
@cython.cdivision(True)
cdef double relabel_score(int[:] block_hist, int block_area,
                          int label_0, int[:] l0_hist, int l0_area,
                          int label_1, int[:] l1_hist, int l1_area):
    """
    Compute the improvement obtained by changing the label of a block from
    `label_0` to `label_1`.

    The score is obtained by first removing the contribution of the block
    from label 0's histogram. The intersection of the histogram of the block
    with the histograms of label 0 and label ` are computed and the difference
    returned as the score.

    Parameters
    ----------
    block_hist : the histogram of the block that is to have its label changed
    block_area : the area of the block
    label_0 : the current label of the block
    l0_hist : the histogram of the pixels labelled `label_0`
    l0_area : the number of pixels labelled `label_0`
    label_1 : the proposed label of the block
    l1_hist : the histogram of the pixels labelled `label_1`
    l1_area : the number of pixels labelled `label_1`

    Returns
    -------
    A score that measures the improvement, or 0.0 if `label_0 == label_1`.
    """
    cdef double l0_scale = 1.0, l1_scale = 1.0, b_scale = 1.0
    cdef double l0n = 0.0, l1n = 0.0, bn = 0.0, l0int=0.0, l1int=0.0;
    cdef int i

    if label_0 != label_1:
        l0_scale = 1.0 / (l0_area - block_area)
        l1_scale = 1.0 / l1_area
        b_scale = 1.0 / block_area

        # For each bin
        for i in range(block_hist.shape[0]):
            # Normalised histogram of label 0
            l0n = (l0_hist[i] - block_hist[0]) * l0_scale
            # Normalised histogram of label `
            l1n = l1_hist[i] * l1_scale
            # Normalised histogram of block
            bn = block_hist[i] * b_scale
            # Intersections
            l0int += min(l0n, bn)
            l1int += min(l1n, bn)

        # Score = change in intersection
        return l1int - l0int

    return 0.0


@cython.boundscheck(False)
@cython.cdivision(True)
cdef double pixel_relabel_score(int pixel_index_value,
                                int label_0, int[:] l0_hist, int l0_area,
                                int label_1, int[:] l1_hist, int l1_area):
    """
    Compute the improvement obtained by changing the label of a pixel from
    `label_0` to `label_1`.

    The score is obtained by first removing the contribution of the pixel
    from label 0's histogram. The intersection of the histogram of the pixel
    with the histograms of label 0 and label ` are computed and the difference
    returned as the score.

    Parameters
    ----------
    pixel_index_value : the index (histogram bin index) of the pixel
    label_0 : the current label of the pixel
    l0_hist : the histogram of the pixels labelled `label_0`
    l0_area : the number of pixels labelled `label_0`
    label_1 : the proposed label of the pixel
    l1_hist : the histogram of the pixels labelled `label_1`
    l1_area : the number of pixels labelled `label_1`

    Returns
    -------
    A score that measures the improvement, or 0.0 if `label_0 == label_1`.
    """
    cdef double l0_scale = 1.0, l1_scale = 1.0, l0n = 0.0, l1n = 0.0, l0int=0.0, l1int=0.0;

    if label_0 != label_1:
        l0_scale = 1.0 / (l0_area - 1)
        l1_scale = 1.0 / l1_area

        l0n = (l0_hist[pixel_index_value] - 1) * l0_scale
        l1n = l1_hist[pixel_index_value] * l1_scale
        l0int += min(l0n, 1)
        l1int += min(l1n, 1)

        return l1int - l0int

    return 0.0



@cython.boundscheck(False)
cdef bint _relabel_check(int a0, int a1, int b0, int b1, int c0, int c1):
    """
    Check if re-labelling a block or pixel will break a labelled region
    into disjoint sub-regions.

    An explanation; assume that we are relabelling a pixel to match that
    of its neighbour above.
    The pixel that is to be relabelled is represented by `b1`.
    The pixel behind/below it is `b0`.
    The pairs `a0`/`a1` and `c0`/`c1` represent pixels either side, e.g.
    `a0` represents the pixel to the left and behind, while `a1` represents
    the pixel to the right. `c0`/`c1` represent pixels to the right.

    Parameters
    ----------
    a0, a1, b0, b1, c0, c1 : the labels of pixels/blocks in a neighbourhood
    surrounding the pixel/block that is to be relabelled

    Returns
    -------
    A boolean indicating if the proposed relabelling would cause a break.
    """
    if b0 != b1 and a1 == b1 and c1 == b1:
        # Breaking run a1-b1-c1
        return False
    if a0 != b1 and a1 == b1 and b0 == b1:
        # Breaking run a1-b1-b0
        return False
    if c0 != b1 and c1 == b1 and b0 == b1:
        # Breaking run c1-b1-b0
        return False
    return True


@cython.boundscheck(False)
cdef bint relabel_check_above(int[:,:] labels_2d, int y, int x, int h, int w):
    """
    Check if re-labelling a pixel/block to match the label of its
    neighbour above would break a labelled region into disjoint sub-regions.

    Parameters
    ----------
    labels_2d : label image as a 2D array
    y : the y-coordinate of the pixel/block to be re-labelled
    x : the x-coordinate of the pixel/block to be re-labelled
    h : the height of the label image
    w : the width of the label image

    Returns
    -------
    A boolean indicating if the proposed relabelling would cause a break.
    """
    cdef int a0, a1, b0, b1, c0, c1
    a0 = labels_2d[y+1,x-1]   if y<(h-1) and x>0   else -1
    a1 = labels_2d[y,x-1]   if x>0   else -1
    b0 = labels_2d[y+1,x]   if y<(h-1)   else -1
    b1 = labels_2d[y,x]
    c0 = labels_2d[y+1,x+1]   if y<(h-1) and x<(w-1)   else -1
    c1 = labels_2d[y,x+1]   if x>0 and x<(w-1)   else -1
    return _relabel_check(a0, a1, b0, b1, c0, c1)

@cython.boundscheck(False)
cdef bint relabel_check_below(int[:,:] labels_2d, int y, int x, int h, int w):
    """
    Check if re-labelling a pixel/block to match the label of its
    neighbour below would break a labelled region into disjoint sub-regions.

    Parameters
    ----------
    labels_2d : label image as a 2D array
    y : the y-coordinate of the pixel/block to be re-labelled
    x : the x-coordinate of the pixel/block to be re-labelled
    h : the height of the label image
    w : the width of the label image

    Returns
    -------
    A boolean indicating if the proposed relabelling would cause a break.
    """
    cdef int a0, a1, b0, b1, c0, c1
    a0 = labels_2d[y-1,x-1]   if y>0 and x>0   else -1
    a1 = labels_2d[y,x-1]   if x>0   else -1
    b0 = labels_2d[y-1,x]   if y>0   else -1
    b1 = labels_2d[y,x]
    c0 = labels_2d[y-1,x+1]   if y>0 and x<(w-1)   else -1
    c1 = labels_2d[y,x+1]   if x>0 and x<(w-1)   else -1
    return _relabel_check(a0, a1, b0, b1, c0, c1)

@cython.boundscheck(False)
cdef bint relabel_check_right(int[:,:] labels_2d, int y, int x, int h, int w):
    """
    Check if re-labelling a pixel/block to match the label of its
    neighbour to the right would break a labelled region into disjoint
    sub-regions.

    Parameters
    ----------
    labels_2d : label image as a 2D array
    y : the y-coordinate of the pixel/block to be re-labelled
    x : the x-coordinate of the pixel/block to be re-labelled
    h : the height of the label image
    w : the width of the label image

    Returns
    -------
    A boolean indicating if the proposed relabelling would cause a break.
    """
    cdef int a0, a1, b0, b1, c0, c1
    a0 = labels_2d[y-1,x-1]   if y>0 and x>0   else -1
    a1 = labels_2d[y-1,x]   if y>0   else -1
    b0 = labels_2d[y,x-1]   if x>0   else -1
    b1 = labels_2d[y,x]
    c0 = labels_2d[y+1,x-1]   if y<(h-1) and x>0   else -1
    c1 = labels_2d[y+1,x]   if y<(h-1)   else -1
    return _relabel_check(a0, a1, b0, b1, c0, c1)

@cython.boundscheck(False)
cdef bint relabel_check_left(int[:,:] labels_2d, int y, int x, int h, int w):
    """
    Check if re-labelling a pixel/block to match the label of its
    neighbour to the left would break a labelled region into disjoint
    sub-regions.

    Parameters
    ----------
    labels_2d : label image as a 2D array
    y : the y-coordinate of the pixel/block to be re-labelled
    x : the x-coordinate of the pixel/block to be re-labelled
    h : the height of the label image
    w : the width of the label image

    Returns
    -------
    A boolean indicating if the proposed relabelling would cause a break.
    """
    cdef int a0, a1, b0, b1, c0, c1
    a0 = labels_2d[y-1,x+1]   if y>0 and x<(w-1)   else -1
    a1 = labels_2d[y-1,x]   if y>0   else -1
    b0 = labels_2d[y,x+1]   if x<(w-1)   else -1
    b1 = labels_2d[y,x]
    c0 = labels_2d[y+1,x+1]   if y<(h-1) and x<(w-1)   else -1
    c1 = labels_2d[y+1,x]   if y<(h-1)   else -1
    return _relabel_check(a0, a1, b0, b1, c0, c1)



@cython.boundscheck(False)
def refine_blocks(int refine_iters, int[:,:] lab_2d, int[:,:] label_hists,
                  int[:] label_areas, int[:,:,:] block_hists,
                  LevelMetrics metrics, int label_area_threshold,
                  double score_threshold):
    """
    Refine SEEDS segmentation by one block level.

    Iteratively re-labels blocks in order to maximise the score.

    Parameters
    ----------
    refine_iters : the number of iterations to perform over the image
    lab_2d : label image as a 2D array that is to be refined, pixels/blocks
    with label 0 are considered un-labelled; labels start at 1
    label_hists : label histograms as a 2D array indexed by
    `label_hists[label_index, histogram_bin]`. Note, since labels in `lab_2d`
    start at 1, subtract 1 to get a label index
    label_areas : label areas as an array, indexed by label index; as with
    `label_hists`, subtract ` from `lab_2d` to get a label index
    block_hists : block histograms as a 3D array, indexed by
    `block_hists[block_y, block_x, histogram_bin]`.
    metrics : a `LevelMetrics` instance describing the downscaling operation
    that was performed to reduce the contents of the previous level above
    label_area_threshold : will not allow label areas to drop below this value
    score_threshold : only re-label if doing so gives an improvement score
    of above this threshold

    Returns
    -------
    None
    """
    cdef int l = -1, best_label=-1
    cdef int h = lab_2d.shape[0], w = lab_2d.shape[1], block_area
    cdef double best_score = 0.0
    cdef int[:] block_hist
    cdef int r, y, x

    # For each refinement iteration
    for r in range(refine_iters):
        for y in range(lab_2d.shape[0]):
            for x in range(lab_2d.shape[1]):
                l = lab_2d[y,x]-1
                # Compute the area of the block that we are proposing to
                # re-label
                block_area = metrics.block_area(y, x)
                # Ensure that re-labelling the block will not reduce the area
                # of the region labelled `l` below the threshold
                if label_areas[l] > (label_area_threshold+block_area):
                    # Get the block histogram
                    block_hist = block_hists[y,x,:]
                    best_score = 0.0
                    best_label = -1

                    # For each direction in turn, first check if the
                    # re-labelling would break a labelled region into disjoint
                    # sub-regions
                    # If not, then compute the score of the proposed
                    # re-labelling
                    # If its better than the best so far, update the best
                    if y > 0 and relabel_check_above(lab_2d, y, x, h, w):
                        l_above = lab_2d[y-1,x]-1
                        score_above = relabel_score(block_hist, block_area,
                                l, label_hists[l], label_areas[l],
                                l_above, label_hists[l_above],
                                label_areas[l_above])
                        if score_above > best_score:
                            best_score = score_above
                            best_label = l_above
                    if y < (h-1) and relabel_check_below(lab_2d, y, x, h, w):
                        l_below = lab_2d[y+1,x]-1
                        score_below = relabel_score(block_hist, block_area,
                                l, label_hists[l], label_areas[l],
                                l_below, label_hists[l_below],
                                label_areas[l_below])
                        if score_below > best_score:
                            best_score = score_below
                            best_label = l_below
                    if x > 0 and relabel_check_left(lab_2d, y, x, h, w):
                        l_left = lab_2d[y,x-1]-1
                        score_left = relabel_score(block_hist, block_area,
                                l, label_hists[l], label_areas[l],
                                l_left, label_hists[l_left],
                                label_areas[l_left])
                        if score_left > best_score:
                            best_score = score_left
                            best_label = l_left
                    if x < (w-1) and relabel_check_right(lab_2d, y, x, h, w):
                        l_right = lab_2d[y,x+1]-1
                        score_right = relabel_score(block_hist, block_area,
                                l, label_hists[l], label_areas[l],
                                l_right, label_hists[l_right],
                                label_areas[l_right])
                        if score_right > best_score:
                            best_score = score_right
                            best_label = l_right

                    # If a potential re-labelling was found and if it meets
                    # the score threshold
                    if best_label != -1 and best_score > score_threshold:
                        # Re-label the block
                        lab_2d[y,x] = best_label+1
                        # Move the contribution of the block from the source
                        # label histogram and area to that of the target label
                        for i in xrange(block_hist.shape[0]):
                            label_hists[l,i] -= block_hist[i]
                            label_hists[best_label,i] += block_hist[i]
                        label_areas[l] -= block_area
                        label_areas[best_label] += block_area


@cython.boundscheck(False)
def refine_pixels(int refine_iters, int[:,:] lab_2d, int[:,:] label_hists,
                  int[:] label_areas, int[:,:] pixel_index_2d,
                  int label_area_threshold, double score_threshold):
    """
    Refine SEEDS segmentation to the pixel level

    Iteratively re-labels pixels in order to maximise the score.

    Parameters
    ----------
    refine_iters : the number of iterations to perform over the image
    lab_2d : label image as a 2D array that is to be refined, pixels/blocks
    with label 0 are considered un-labelled; labels start at 1
    label_hists : label histograms as a 2D array indexed by
    `label_hists[label_index, histogram_bin]`. Note, since labels in `lab_2d`
    start at 1, subtract 1 to get a label index
    label_areas : label areas as an array, indexed by label index; as with
    `label_hists`, subtract ` from `lab_2d` to get a label index
    pixel_index_2d : a pixel index image as a 2D array; indices correspond to
    histogram bins
    label_area_threshold : will not allow label areas to drop below this value
    score_threshold : only re-label if doing so gives an improvement score
    of above this threshold

    Returns
    -------
    None
    """
    cdef int l = -1, best_label=-1, h = lab_2d.shape[0], w = lab_2d.shape[1]
    cdef int pixel_index_value = -1
    cdef double best_score = 0.0
    cdef int r, y, x

    # For each refinement iteration
    for r in range(refine_iters):
        for y in range(lab_2d.shape[0]):
            for x in range(lab_2d.shape[1]):
                l = lab_2d[y,x]-1
                # Ensure that re-labelling the pixel will not reduce the area
                # of the region labelled `l` below the threshold
                if label_areas[l] > (label_area_threshold+1):
                    pixel_index_value = pixel_index_2d[y,x]
                    best_score = 0.0
                    best_label = -1

                    # For each direction in turn, first check if the
                    # re-labelling would break a labelled region into disjoint
                    # sub-regions
                    # If not, then compute the score of the proposed
                    # re-labelling
                    # If its better than the best so far, update the best
                    if y > 0 and relabel_check_above(lab_2d, y, x, h, w):
                        l_above = lab_2d[y-1,x]-1
                        score_above = pixel_relabel_score(pixel_index_value,
                                l, label_hists[l], label_areas[l],
                                l_above, label_hists[l_above],
                                label_areas[l_above])
                        if score_above > best_score:
                            best_score = score_above
                            best_label = l_above
                    if y < (h-1) and relabel_check_below(lab_2d, y, x, h, w):
                        l_below = lab_2d[y+1,x]-1
                        score_below = pixel_relabel_score(pixel_index_value,
                                l, label_hists[l], label_areas[l],
                                l_below, label_hists[l_below],
                                label_areas[l_below])
                        if score_below > best_score:
                            best_score = score_below
                            best_label = l_below
                    if x > 0 and relabel_check_left(lab_2d, y, x, h, w):
                        l_left = lab_2d[y,x-1]-1
                        score_left = pixel_relabel_score(pixel_index_value,
                                l, label_hists[l], label_areas[l],
                                l_left, label_hists[l_left],
                                label_areas[l_left])
                        if score_left > best_score:
                            best_score = score_left
                            best_label = l_left
                    if x < (w-1) and relabel_check_right(lab_2d, y, x, h, w):
                        l_right = lab_2d[y,x+1]-1
                        score_right = pixel_relabel_score(pixel_index_value,
                                l, label_hists[l], label_areas[l],
                                l_right, label_hists[l_right],
                                label_areas[l_right])
                        if score_right > best_score:
                            best_score = score_right
                            best_label = l_right

                    # If a potential re-labelling was found and if it meets
                    # the score threshold
                    if best_label != -1 and best_score > score_threshold:
                        # Re-label the pixel
                        lab_2d[y,x] = best_label+1
                        # Move the contribution of the block from the source
                        # label histogram and area to that of the target label
                        label_hists[l,pixel_index_value] -= 1
                        label_hists[best_label,pixel_index_value] += 1
                        label_areas[l] -= 1
                        label_areas[best_label] += 1
