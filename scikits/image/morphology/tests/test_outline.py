"""test_outline - test the outline function

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
import unittest

import scikits.image.morphology.outline as OL

class TestOutline(unittest.TestCase):
    def test_00_00_zeros(self):
        x = numpy.zeros((10,10),int)
        result = OL.outline(x)
        self.assertTrue(numpy.all(x==result))
    
    def test_01_01_single(self):
        x = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0]])
        e = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,0,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0]])
        result = OL.outline(x)
        self.assertTrue(numpy.all(result==e))
    
    def test_01_02_two_disjoint(self):
        x = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,0,0,0,0,0]])
        e = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,0,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,0,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,0,0,0,0,0]])
        result = OL.outline(x)
        self.assertTrue(numpy.all(result==e))

    def test_01_03_touching(self):
        x = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,0,0,0,0,0]])
        e = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,0,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,2,0,2,0,0],
                         [ 0,0,2,2,2,0,0],
                         [ 0,0,0,0,0,0,0]])
        result = OL.outline(x)
        self.assertTrue(numpy.all(result==e))
    
    def test_02_04_edge(self):
        x = numpy.array([[ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0]])
        e = numpy.array([[ 0,0,1,1,1,0,0],
                         [ 0,0,1,0,1,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,0,0,0,0,0,0]])
        result = OL.outline(x)
        self.assertTrue(numpy.all(result==e))
    
    def test_02_05_diagonal(self):
        x = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,1,1,1,1,0,0],
                         [ 0,1,1,1,1,0,0],
                         [ 0,0,1,1,0,0,0]])
        e = numpy.array([[ 0,0,0,0,0,0,0],
                         [ 0,0,1,1,1,0,0],
                         [ 0,1,1,0,1,0,0],
                         [ 0,1,1,1,1,0,0],
                         [ 0,0,1,1,0,0,0]])
        result = OL.outline(x)
        self.assertTrue(numpy.all(result==e))
        
