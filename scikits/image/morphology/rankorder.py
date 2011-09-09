"""rankorder.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

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
import numpy

def rank_order(image):
    """Return an image of the same shape where each pixel has the
    rank-order value of the corresponding pixel in the image.
    The returned image's elements are of type numpy.uint32 which
    simplifies processing in C code.
    """
    flat_image = image.ravel()
    sort_order = flat_image.argsort().astype(numpy.uint32)
    flat_image = flat_image[sort_order]
    sort_rank  = numpy.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    numpy.cumsum(is_different, out=sort_rank[1:])
    original_values = numpy.zeros((sort_rank[-1]+1,),image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different] 
    int_image = numpy.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return (int_image.reshape(image.shape), original_values)
    
    
