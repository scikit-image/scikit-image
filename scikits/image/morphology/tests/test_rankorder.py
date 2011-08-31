"""test_rankorder.py - test rankorder.py

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

import unittest
import numpy
from scikits.image.morphology.rankorder import rank_order

class TestRankOrder(unittest.TestCase):
    def test_00_zeros(self):
        """Test rank_order on a matrix of all zeros"""
        x = numpy.zeros((5,5))
        output = rank_order(x)[0]
        self.assertTrue(numpy.all(output==0))
        self.assertTrue(output.dtype.type == numpy.uint32)
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[0],5)
        self.assertEqual(x.shape[1],5)
    
    def test_01_3D(self):
        x = numpy.zeros((5,5,5))
        output = rank_order(x)[0]
        self.assertTrue(numpy.all(output==0))
        self.assertEqual(x.ndim, 3)
        self.assertEqual(x.shape[0],5)
        self.assertEqual(x.shape[1],5)
        self.assertEqual(x.shape[2],5)
    
    def test_02_two_values(self):
        x = numpy.zeros((5,10))
        x[3,5] = 2
        x[4,7] = 2
        output,orig = rank_order(x)
        self.assertEqual(output[3,5],1)
        self.assertEqual(output[4,7],1)
        self.assertEqual(len(orig),2)
        self.assertEqual(orig[0],0)
        self.assertEqual(orig[1],2)
        self.assertEqual(numpy.sum(output==0),48)
    
    def test_03_three_values(self):
        x = numpy.zeros((5,10))
        x[3,5] = 4
        x[4,7] = 4
        x[0,9] = 3
        output,orig = rank_order(x)
        self.assertEqual(output[0,9],1)
        self.assertEqual(output[3,5],2)
        self.assertEqual(output[4,7],2)
        self.assertEqual(len(orig),3)
        self.assertEqual(orig[0],0)
        self.assertEqual(orig[1],3)
        self.assertEqual(orig[2],4)
        self.assertEqual(numpy.sum(output==0),47)
                
