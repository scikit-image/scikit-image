"""watershed.py - watershed algorithm

This module implements a watershed algorithm that apportions pixels into
marked basins. The algorithm uses a priority queue to hold the pixels
with the metric for the priority queue being pixel value, then the time
of entry into the queue - this settles ties in favor of the closest marker.

Some ideas taken from
Soille, "Automated Basin Delineation from Digital Elevation Models Using
Mathematical Morphology", Signal Processing 20 (1990) 171-182

The most important insight in the paper is that entry time onto the queue
solves two problems: a pixel should be assigned to the neighbor with the
largest gradient or, if there is no gradient, pixels on a plateau should
be split between markers on opposite sides.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__ = "$Revision$"

from _heapq import heapify, heappush, heappop
import numpy as np
import scipy.ndimage
from rankorder import rank_order

import _watershed

def __get_strides_for_shape(shape):
    """Get the amount to multiply at each coord when converting to flat"""
    lshape = list(shape)
    lshape.reverse()
    stride = [1]
    for i in range(len(lshape) - 1):
        stride.append(lshape[i] * stride[i])
    stride.reverse()
    return np.array(stride)

def __heapify_markers(markers,image):
    """Create a priority queue heap with the markers on it"""
    stride = __get_strides_for_shape(image.shape)
    coords = np.argwhere(markers != 0)
    ncoords= coords.shape[0]
    if ncoords > 0:
        pixels = image[markers != 0]
        age    = np.array(range(ncoords))
        offset = np.zeros(coords.shape[0], int)
        for i in range(image.ndim):
            offset = offset + stride[i] * coords[:,i]
        pq = np.column_stack((pixels, age, offset, coords))
        ordering = np.lexsort((age, pixels)) # pixels = top priority, age=second
        pq = pq[ordering,:]
    else:
        pq = np.zeros((0, markers.ndim+3), int)
    return (pq, ncoords)
    
def watershed(image, markers, connectivity=8, mask=None):
    """Return a matrix labeled using the watershed algorithm
    
    image - a two-dimensional matrix where the lowest value points are
            labeled first.
    markers - a two-dimensional matrix marking the basins with the values
              to be assigned in the label matrix. Zero means not a marker.
    connectivity - either 4 for four-connected or 8 (default) for eight-
                   connected
    mask    - don't label points in the mask
    """
    if connectivity not in (4, 8):
        raise ValueError("Connectivity was %d: it should be either four or eight"%(connectivity))
    
    image = np.array(image)
    markers = np.array(markers)
    labels = markers.copy()
    max_x  = markers.shape[0]
    max_y  = markers.shape[1]
    if connectivity == 4:
        connect_increments = ((1, 0), (0, 1), (-1, 0), (0, -1))
    else:
        connect_increments = ((1, 0), (1, 1), (0, 1), (-1, 1),
                              (-1, 0), (-1, -1), (0, -1), (1, -1))
    pq,age = __heapify_markers(markers, image)
    pq = pq.tolist()
    #
    # The second step pops a value off of the queue, then labels and pushes
    # the neighbors
    #
    while len(pq):
        pix_value, pix_age, ignore,pix_x,pix_y = heappop(pq)
        pix_label = labels[pix_x,pix_y]
        for xi,yi in connect_increments:
            x = pix_x + xi
            y = pix_y + yi
            if x < 0 or y < 0 or x >= max_x or y >= max_y:
                continue
            if labels[x,y]:
                continue
            if mask != None and not mask[x,y]:
                continue
            # label the pixel
            labels[x,y] = pix_label
            # put the pixel onto the queue
            heappush(pq, (image[x,y], age, 0, x, y))
            age += 1
    return labels

        
def fast_watershed(image, markers, connectivity=None, offset=None, mask=None):
    """Return a matrix labeled using the watershed algorithm
    
    image - an array where the lowest value points are
            labeled first.
    markers - an array marking the basins with the values
              to be assigned in the label matrix. Zero means not a marker.
              This array should be of an integer type.
    connectivity - an array whose non-zero elements indicate neighbors
                   for connection.
                   Following the scipy convention, default is a one-connected
                   array of the dimension of the image.
    offset  - offset of the connectivity (one offset per dimension)
    mask    - don't label points in the mask
    
    Returns a labeled matrix of the same type and shape as markers
    
    This implementation converts all arguments to specific, lowest common
    denominator types, then passes these to a C algorithm that operates
    as above.
    """
    
    if connectivity == None:
        c_connectivity = scipy.ndimage.generate_binary_structure(image.ndim, 1)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image.ndim:
            raise ValueError,"Connectivity dimension must be same as image"
    if offset == None:
        if any([x%2==0 for x in c_connectivity.shape]):
            raise ValueError,"Connectivity array must have an unambiguous center"
        #
        # offset to center of connectivity array
        #
        offset = np.array(c_connectivity.shape) / 2

    # pad the image, markers, and mask so that we can use the mask to keep from running off the edges
    pads = offset

    def pad(im):
        new_im = np.zeros([i + 2*p for i, p in zip(im.shape, pads)], im.dtype)
        new_im[[slice(p, -p, None) for p in pads]] = im
        return new_im

    if mask is not None:
        mask = pad(mask)
    else:
        mask = pad(np.ones(image.shape, bool))
    image = pad(image)
    markers = pad(markers)

    c_image = rank_order(image)[0].astype(np.int32)
    c_markers = np.ascontiguousarray(markers, dtype=np.int32)
    if c_markers.ndim!=c_image.ndim:
        raise ValueError,\
            "markers (ndim=%d) must have same # of dimensions "\
            "as image (ndim=%d)"%(c_markers.ndim, cimage.ndim)
    if not all([x==y for x, y in zip(c_markers.shape, c_image.shape)]):
        raise ValueError("image and markers must have the same shape")
    if mask != None:
        c_mask = np.ascontiguousarray(mask,dtype=bool)
        if c_mask.ndim != c_markers.ndim:
            raise ValueError, "mask must have same # of dimensions as image"
        if not all([x == y for x, y in zip(c_markers.shape, c_mask.shape)]):
            raise ValueError, "mask must have same shape as image"
        c_markers[np.logical_not(mask)]=0
    else:
        c_mask = None
    c_output = c_markers.copy()

    #
    # We pass a connectivity array that pre-calculates the stride for each
    # neighbor.
    #
    # The result of this bit of code is an array with one row per
    # point to be considered. The first column is the pre-computed stride
    # and the second through last are the x,y...whatever offsets
    # (to do bounds checking).
    c = []
    image_stride = __get_strides_for_shape(image.shape)
    for i in range(np.product(c_connectivity.shape)):
        multiplier = 1
        offs = []
        indexes = []
        ignore = True
        for j in range(len(c_connectivity.shape)):
            elems = c_image.shape[j]
            idx   = (i / multiplier) % c_connectivity.shape[j]
            off   = idx - offset[j]
            if off:
                ignore = False
            offs.append(off)
            indexes.append(idx)
            multiplier *= c_connectivity.shape[j]
        if (not ignore) and c_connectivity.__getitem__(tuple(indexes)):
            stride = np.dot(image_stride, np.array(offs))
            offs.insert(0, stride)
            c.append(offs)
    c = np.array(c, np.int32)

    pq,age = __heapify_markers(c_markers, c_image)
    pq = np.ascontiguousarray(pq, dtype=np.int32)
    if np.product(pq.shape) > 0:
        # If nothing is labeled, the output is empty and we don't have to
        # do anything
        c_output = c_output.flatten()
        if c_mask == None:
            c_mask = np.ones(c_image.shape,np.int8).flatten()
        else:
            c_mask = c_mask.astype(np.int8).flatten()
        _watershed.watershed(c_image.flatten(),
                             pq, age, c, 
                             c_image.ndim, 
                             c_mask,
                             np.array(c_image.shape,np.int32),
                             c_output)
    c_output = c_output.reshape(c_image.shape)[[slice(1, -1, None)] *
                                                image.ndim]
    try:
        return c_output.astype(markers.dtype)
    except:
        return c_output
