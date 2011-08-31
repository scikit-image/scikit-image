""" testindex.py - indexing tricks tests

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import numpy as np
import unittest
from numpy.testing import assert_array_equal, run_module_suite

from scikits.image.morphology import Indexes

class TestIndexes:
  
    def test_00_00_oh_so_empty(self):
        ind = Indexes([[]])
        assert ind.length == 0
        assert len(ind.fwd_idx) == 0
        assert len(ind.rev_idx) == 0
        assert tuple(ind.idx.shape) == (1,0)
        assert_array_equal([[]], ind.counts)
        
    def test_00_01_all_are_empty(self):
        counts = [[0]]
        ind = Indexes(counts)
        assert ind.length == 0
        assert len(ind.fwd_idx) == 1
        assert ind.fwd_idx[0] == 0
        assert len(ind.rev_idx) == 0
        assert tuple(ind.idx.shape) == (1,0)
        assert_array_equal(counts, ind.counts)
        
    def test_00_02_other_ways_to_be_empty(self):
        ind = Indexes([(0,1),(1,0)])
        assert ind.length == 0
        assert len(ind.fwd_idx) == 2
        assert np.all(ind.fwd_idx == 0)
        assert len(ind.rev_idx) == 0
        assert tuple(ind.idx.shape) == (2,0)
        
    def test_01_01_one_object_1_subobject(self):
        ind = Indexes([[1]])
        assert ind.length == 1
        assert len(ind.fwd_idx) == 1
        assert ind.fwd_idx[0] == 0
        assert len(ind.rev_idx) == 1
        assert ind.rev_idx == 0
        assert tuple(ind.idx.shape) == (1,1)
        assert ind.idx[0,0] == 0
        
    def test_01_02_one_object_1x1_subobject(self):
        ind = Indexes([[1],[1]])
        assert ind.length == 1
        assert len(ind.fwd_idx) == 1
        assert ind.fwd_idx[0] == 0
        assert len(ind.rev_idx) == 1
        assert ind.rev_idx == 0
        assert tuple(ind.idx.shape) == (2,1)
        assert ind.idx[0,0] == 0
        assert ind.idx[1,0] == 0
    
    def test_01_03_one_object_NxM(self):
        counts = np.array([[4],[3]])
        hits = np.zeros(counts[:,0], int)
        ind = Indexes(counts)
        assert ind.length == np.prod(counts)
        assert len(ind.fwd_idx) == 1
        assert ind.fwd_idx[0] == 0
        assert len(ind.rev_idx) == ind.length
        np.testing.assert_array_equal(ind.rev_idx, 0)
        assert tuple(ind.idx.shape) == (2, ind.length)
        hits[ind.idx[0], ind.idx[1]] = np.arange(ind.length)+1
        assert len(np.unique(hits.ravel())) == ind.length
        
    def test_02_01_two_objects_NxM(self):
        counts = [[4,2],[3,6]]
        c0 = counts[0][0] * counts[1][0]
        ind = Indexes(counts)
        assert ind.length == np.sum(np.prod(counts,0))
        assert len(ind.fwd_idx) == 2
        assert ind.fwd_idx[0] == 0
        assert ind.fwd_idx[1] == c0
        assert len(ind.rev_idx) == ind.length
        assert_array_equal(ind.rev_idx[:c0], 0)
        assert_array_equal(ind.rev_idx[c0:], 1)
        start = 0
        for i, count in enumerate(ind.counts.transpose()):
            hits = np.zeros(count)
            n = np.prod(count)
            hits[ind.idx[0,start:(start + n)],
                 ind.idx[1,start:(start + n)]] = np.arange(np.prod(count))
            assert len(np.unique(hits.ravel())) == n
            start += n
            
    def test_02_02_multiple_objects_and_one_is_0x0(self):
        counts = np.array([[4,2,0,3],[3,6,4,5]])
        ind = Indexes(counts)
        assert ind.length == np.sum(np.prod(counts,0))
        assert len(ind.fwd_idx) == counts.shape[1]
        assert_array_equal(ind.fwd_idx, [0,12,24,24])
        assert len(ind.rev_idx) == ind.length
        start = 0
        for i, count in enumerate(ind.counts.transpose()):
            if np.prod(count) == 0:
                continue
            hits = np.zeros(count)
            n = np.prod(count)
            hits[ind.idx[0,start:(start + n)],
                 ind.idx[1,start:(start + n)]] = np.arange(np.prod(count))
            assert len(np.unique(hits.ravel())) == n
            start += n
            
    def test_02_03_one_at_end(self):
        ind = Indexes(np.array([[0,0,1]]))
        pass
        
        
if __name__ == "__main__":
    run_module_suite()
