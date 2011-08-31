""" index.py - indexing tricks

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision: 11052 $"

import numpy as np

class Indexes(object):
    '''The Indexes class stores indexes for manipulating subsets on behalf of a parent set
    
    The idea here is that you have a parent set of "things", for instance
    some pixels or objects. Each of these might have, conceptually, an N-d
    array of sub-objects where each array might have different dimensions.
    This class holds indexes that help out.
    
    For instance, create 300 random objects, each of which has
    an array of sub-objects of size 1x1 to 10x20. Create weights for each
    axis for the sub-objects, take the cross-product of the axis weights
    and then sum them (you'll do something more useful, I hope):
    
    i_count = np.random.randint(1,10, size=300)
    
    j_count = np.random.randint(1,20, size=300)
    
    i_indexes = Indexes([i_count])
    
    j_indexes = Indexes([j_count])
    
    indexes = Indexes([i_count, j_count])
    
    i_weights = np.random.uniform(size=i_indexes.length)
    
    j_weights = np.random.uniform(size=j_indexes.length)
    
    weights = (i_weights[i_indexes.fwd_idx[indexes.rev_idx] + indexes.idx[0]] *
    
               j_weights[j_indexes.fwd_idx[indexes.rev_idx] + indexes.idx[1]])
               
    sums_of_weights = np.bincount(indexes.rev_idx, weights)
    '''
    
    def __init__(self, counts):
        '''Constructor
        
        counts - an NxM array of dimensions of sub-arrays
                 N is the number of dimensions of the sub-object array
                 M is the number of objects.
        '''
        counts = np.atleast_2d(counts).astype(int)
        self.__counts = counts.copy()
        if np.sum(np.prod(counts,0)) == 0:
            self.__length = 0
            self.__fwd_idx = np.zeros(counts.shape[1], int)
            self.__rev_idx = np.zeros(0, int)
            self.__idx = np.zeros((len(counts), 0), int)
            return
        cs = np.cumsum(np.prod(counts, 0))
        self.__length = cs[-1]
        self.__fwd_idx = np.hstack(([0], cs[:-1]))
        self.__rev_idx = np.zeros(self.__length,int)
        non_empty_indices = \
            np.arange(counts.shape[1]).astype(int)[np.prod(counts, 0) > 0]
        if len(non_empty_indices) > 0:
            if len(non_empty_indices) > 1:
                distance_to_next = non_empty_indices[1:] - non_empty_indices[:-1]
                self.__rev_idx[self.__fwd_idx[non_empty_indices[1:]]] = distance_to_next
                self.__rev_idx = np.cumsum(self.__rev_idx)
            self.__idx = []
            indexes = np.arange(self.length) - self.__fwd_idx[self.__rev_idx]
            for i, count in enumerate(counts[:-1]):
                modulos = np.prod(counts[(i+1):,:], 0)
                self.__idx.append((indexes / modulos[self.__rev_idx]).astype(int))
                indexes = indexes % modulos[self.__rev_idx]
            self.__idx.append(indexes)
            self.__idx = np.array(self.__idx)
            
    @property
    def length(self):
        '''The number of elements in all sub-objects
        
        Use this number to create an array that holds a value for each
        sub-object.
        '''
        return self.__length
    
    @property
    def fwd_idx(self):
        '''The index to the first sub object per super-object
        
        Use the fwd_idx as part of the address of the sub-object.
        '''
        return self.__fwd_idx
    
    @property
    def rev_idx(self):
        '''The index of the super-object per sub-object'''
        return self.__rev_idx
    
    @property
    def idx(self):
        '''For each sub-object, its indexes relative to the super-object array
        
        This lets you find the axis coordinates of any place in a sub-object
        array. For instance, if you have 2-d arrays of sub-objects,
        index.idx[0],index.idx[1] gives the coordinates of each sub-object
        in its array.
        '''
        return self.__idx

    @property
    def counts(self):
        '''The dimensions for each object along each of the axes
        
        The same values are stored here as are in the counts
        passed into the constructor.
        '''
        return self.__counts

                    
