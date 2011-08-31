"""outline - given a label matrix, return a matrix of the outlines of the labeled objects

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy

def outline(labels):
    """Given a label matrix, return a matrix of the outlines of the labeled objects
    
    If a pixel is not zero and has at least one neighbor with a different
    value, then it is part of the outline.
    """
    
    output = numpy.zeros(labels.shape, labels.dtype)
    lr_different = labels[1:,:]!=labels[:-1,:]
    ud_different = labels[:,1:]!=labels[:,:-1]
    d1_different = labels[1:,1:]!=labels[:-1,:-1]
    d2_different = labels[1:,:-1]!=labels[:-1,1:]
    different = numpy.zeros(labels.shape, bool)
    different[1:,:][lr_different]  = True
    different[:-1,:][lr_different] = True
    different[:,1:][ud_different]  = True
    different[:,:-1][ud_different] = True
    different[1:,1:][d1_different] = True
    different[:-1,:-1][d1_different] = True
    different[1:,:-1][d2_different] = True
    different[:-1,1:][d2_different] = True
    #
    # Labels on edges need outlines
    #
    different[0,:] = True
    different[:,0] = True
    different[-1,:] = True
    different[:,-1] = True
    
    output[different] = labels[different]
    return output
    
