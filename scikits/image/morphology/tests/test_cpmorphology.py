"""test_cpmorphology - test the functions in cellprofiler.cpmath.cpmorphology

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

import base64
import unittest
import numpy as np
import scipy.ndimage as scind
import scipy.misc
import scipy.io.matlab

import scikits.image.morphology.cpmorphology as morph
from scikits.image.morphology.cpmorphology import fixup_scipy_ndimage_result as fix
#from cellprofiler.cpmath.filter import permutations

class TestFillLabeledHoles(unittest.TestCase):
    def test_01_00_zeros(self):
        """A label matrix of all zeros has no hole"""
        image = np.zeros((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==0))
    
    def test_01_01_ones(self):
        """Regression test - an image of all ones"""
        image = np.ones((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==1))

    def test_02_object_without_holes(self):
        """The label matrix of a single object without holes has no hole"""
        image = np.zeros((10,10),dtype=int)
        image[3:6,3:6] = 1
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==image))
    
    def test_03_object_with_hole(self):
        image = np.zeros((20,20),dtype=int)
        image[5:15,5:15] = 1
        image[8:12,8:12] = 0
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output[8:12,8:12] == 1))
        output[8:12,8:12] = 0 # unfill the hole again
        self.assertTrue(np.all(output==image))
    
    def test_04_holes_on_edges_are_not_holes(self):
        image = np.zeros((40,40),dtype=int)
        objects = (((15,25),(0,10),(18,22),(0,3)),
                   ((0,10),(15,25),(0,3),(18,22)),
                   ((15,25),(30,39),(18,22),(36,39)),
                   ((30,39),(15,25),(36,39),(18,22)))
        for idx,x in zip(range(1,len(objects)+1),objects):
            image[x[0][0]:x[0][1],x[1][0]:x[1][1]] = idx
            image[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 0
        output = morph.fill_labeled_holes(image)
        for x in objects:
            self.assertTrue(np.all(output[x[2][0]:x[2][1],x[3][0]:x[3][1]]==0))
            output[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 1
            self.assertTrue(np.all(output[x[0][0]:x[0][1],x[1][0]:x[1][1]]!=0))
            
    def test_05_lots_of_objects_with_holes(self):
        image = np.ones((1020,1020),bool)
        for i in range(0,51):
            image[i*20:i*20+10,:] = ~image[i*20:i*20+10,:]
            image[:,i*20:i*20+10] = ~ image[:,i*20:i*20+10]
        image = scind.binary_erosion(image, iterations = 2)
        erosion = scind.binary_erosion(image, iterations = 2)
        image = image & ~ erosion
        labeled_image,nobjects = scind.label(image)
        output = morph.fill_labeled_holes(labeled_image)
        self.assertTrue(np.all(output[erosion] > 0))
    
    def test_06_regression_diamond(self):
        """Check filling the center of a diamond"""
        image = np.zeros((5,5),int)
        image[1,2]=1
        image[2,1]=1
        image[2,3]=1
        image[3,2]=1
        output = morph.fill_labeled_holes(image)
        where = np.argwhere(image != output)
        self.assertEqual(len(where),1)
        self.assertEqual(where[0][0],2)
        self.assertEqual(where[0][1],2)
    
    def test_07_regression_nearby_holes(self):
        """Check filling an object with three holes"""
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,0,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        expec = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        output = morph.fill_labeled_holes(image)
        self.assertTrue(np.all(output==expec))
        
    def test_08_fill_small_holes(self):
        """Check filling only the small holes"""
        image = np.zeros((10,20), int)
        image[1:-1,1:-1] = 1
        image[3:8,4:7] = 0     # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = 0  # A hole with area 2*7 = 14 is filled
        
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        self.assertTrue(np.all(output == expected))
        
    def test_09_fill_binary_image(self):
        """Make sure that we can fill a binary image too"""
        image = np.zeros((10,20), bool)
        image[1:-1, 1:-1] = True
        image[3:8, 4:7] = False # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = False # A hole with area 2*7 = 14 is filled
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        self.assertEqual(image.dtype.kind, output.dtype.kind)
        self.assertTrue(np.all(output == expected))
        
    def test_10_fill_bullseye(self):
        i,j = np.mgrid[-50:50, -50:50]
        bullseye = i * i + j * j < 2000
        bullseye[i * i + j * j < 1000 ] = False
        bullseye[i * i + j * j < 500 ] = True
        bullseye[i * i + j * j < 250 ] = False
        bullseye[i * i + j * j < 100 ] = True
        labels, count = scind.label(bullseye)
        result = morph.fill_labeled_holes(labels)
        self.assertTrue(np.all(result[result != 0] == bullseye[6, 43]))
        
    def test_11_dont_fill_if_touches_2(self):
        labels = np.array([
            [ 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 1, 1, 0, 0, 2, 2, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0 ]])
        result = morph.fill_labeled_holes(labels)
        self

class TestAdjacent(unittest.TestCase):
    def test_00_00_zeros(self):
        result = morph.adjacent(np.zeros((10,10), int))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_one(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==False))
        
    def test_01_02_not_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        image[6:8,3:8] = 2
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==False))

    def test_01_03_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:8,3:5] = 1
        image[2:8,5:8] = 2
        expected = np.zeros((10,10), bool)
        expected[2:8,4:6] = True
        result = morph.adjacent(image)
        self.assertTrue(np.all(result==expected))
        
    def test_02_01_127_objects(self):
        '''Test that adjacency works for int8 and 127 labels
        
        Regression test of img-1099. Adjacent sets the background to the
        maximum value of the labels matrix + 1. For 127 and int8, it wraps
        around and uses -127.
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        result = morph.adjacent(labels)
        self.assertTrue(np.all(result == False))
        
class TestStrelDisk(unittest.TestCase):
    """Test cellprofiler.cpmath.cpmorphology.strel_disk"""
    
    def test_01_radius2(self):
        """Test strel_disk with a radius of 2"""
        x = morph.strel_disk(2)
        self.assertTrue(x.shape[0], 5)
        self.assertTrue(x.shape[1], 5)
        y = [0,0,1,0,0,
             0,1,1,1,0,
             1,1,1,1,1,
             0,1,1,1,0,
             0,0,1,0,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        self.assertTrue(np.all(x==ya))
    
    def test_02_radius2_point_5(self):
        """Test strel_disk with a radius of 2.5"""
        x = morph.strel_disk(2.5)
        self.assertTrue(x.shape[0], 5)
        self.assertTrue(x.shape[1], 5)
        y = [0,1,1,1,0,
             1,1,1,1,1,
             1,1,1,1,1,
             1,1,1,1,1,
             0,1,1,1,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        self.assertTrue(np.all(x==ya))

class TestBinaryShrink(unittest.TestCase):
    def test_01_zeros(self):
        """Shrink an empty array to itself"""
        input = np.zeros((10,10),dtype=bool)
        result = morph.binary_shrink(input,1)
        self.assertTrue(np.all(input==result))
    
    def test_02_cross(self):
        """Shrink a cross to a single point"""
        input = np.zeros((9,9),dtype=bool)
        input[4,:]=True
        input[:,4]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_03_x(self):
        input = np.zeros((9,9),dtype=bool)
        x,y = np.mgrid[-4:5,-4:5]
        input[x==y]=True
        input[x==-y]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_04_block(self):
        """A block should shrink to a point"""
        input = np.zeros((9,9), dtype=bool)
        input[3:6,3:6]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where)==1)
        self.assertTrue(input[where[0][0],where[0][1]])
    
    def test_05_hole(self):
        """A hole in a block should shrink to a ring"""
        input = np.zeros((19,19), dtype=bool)
        input[5:15,5:15]=True
        input[9,9]=False
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        self.assertTrue(len(where) > 1)
        self.assertFalse(result[9:9])

    def test_06_random_filled(self):
        """Shrink random blobs
        
        If you label a random binary image, then fill the holes,
        then shrink the result, each blob should shrink to a point
        """
        np.random.seed(0)
        input = np.random.uniform(size=(300,300)) > .8
        labels,nlabels = scind.label(input,np.ones((3,3),bool))
        filled_labels = morph.fill_labeled_holes(labels)
        input = filled_labels > 0
        result = morph.binary_shrink(input)
        my_sum = scind.sum(result.astype(int),filled_labels,np.array(range(nlabels+1),dtype=np.int32))
        my_sum = np.array(my_sum)
        self.assertTrue(np.all(my_sum[1:] == 1))
        
    def test_07_all_patterns_of_3x3(self):
        '''Run all patterns of 3x3 with a 1 in the middle
        
        All of these patterns should shrink to a single pixel since
        all are 8-connected and there are no holes
        '''
        for i in range(512):
            a = morph.pattern_of(i)
            if a[1,1]:
                result = morph.binary_shrink(a)
                self.assertEqual(np.sum(result),1)
    
    def test_08_labels(self):
        '''Run a labels matrix through shrink with two touching objects'''
        labels = np.zeros((10,10),int)
        labels[2:8,2:5] = 1
        labels[2:8,5:8] = 2
        result = morph.binary_shrink(labels)
        self.assertFalse(np.any(result[labels==0] > 0))
        my_sum = fix(scind.sum(result>0, labels, np.arange(1,3,dtype=np.int32)))
        self.assertTrue(np.all(my_sum == 1))
        
class TestCpmaximum(unittest.TestCase):
    def test_01_zeros(self):
        input = np.zeros((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(np.all(output==input))
    
    def test_01_ones(self):
        input = np.ones((10,10))
        output = morph.cpmaximum(input)
        self.assertTrue(np.all(np.abs(output-input)<=np.finfo(float).eps))

    def test_02_center_point(self):
        input = np.zeros((9,9))
        input[4,4] = 1
        expected = np.zeros((9,9))
        expected[3:6,3:6] = 1
        structure = np.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output==expected))
    
    def test_03_corner_point(self):
        input = np.zeros((9,9))
        input[0,0]=1
        expected = np.zeros((9,9))
        expected[:2,:2]=1
        structure = np.ones((3,3),dtype=bool)
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output==expected))

    def test_04_structure(self):
        input = np.zeros((9,9))
        input[0,0]=1
        input[4,4]=1
        structure = np.zeros((3,3),dtype=bool)
        structure[0,0]=1
        expected = np.zeros((9,9))
        expected[1,1]=1
        expected[5,5]=1
        output = morph.cpmaximum(input,structure,(1,1))
        self.assertTrue(np.all(output[1:,1:]==expected[1:,1:]))

    def test_05_big_structure(self):
        big_disk = morph.strel_disk(10).astype(bool)
        input = np.zeros((1001,1001))
        input[500,500] = 1
        expected = np.zeros((1001,1001))
        expected[490:551,490:551][big_disk]=1
        output = morph.cpmaximum(input,big_disk)
        self.assertTrue(np.all(output == expected))

class TestRelabel(unittest.TestCase):
    def test_00_relabel_zeros(self):
        input = np.zeros((10,10),int)
        output,count = morph.relabel(input)
        self.assertTrue(np.all(input==output))
        self.assertEqual(count, 0)
    
    def test_01_relabel_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=1
        output,count = morph.relabel(input)
        self.assertTrue(np.all(input==output))
        self.assertEqual(count,1)
    
    def test_02_relabel_two_to_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=2
        output,count = morph.relabel(input)
        self.assertTrue(np.all((output==1)[input==2]))
        self.assertTrue(np.all((input==output)[input!=2]))
        self.assertEqual(count,1)
    
    def test_03_relabel_gap(self):
        input = np.zeros((20,20),int)
        input[3:6,3:6]=1
        input[3:6,12:15]=3
        output,count = morph.relabel(input)
        self.assertTrue(np.all((output==2)[input==3]))
        self.assertTrue(np.all((input==output)[input!=3]))
        self.assertEqual(count,2)

class TestConvexHull(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure convex_hull can handle an empty array"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [])
        self.assertEqual(np.product(result.shape),0)
        self.assertEqual(np.product(counts.shape),0)
    
    def test_01_01_zeros(self):
        """Make sure convex_hull can work if a label has no points"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [1])
        self.assertEqual(np.product(result.shape),0)
        self.assertEqual(np.product(counts.shape),1)
        self.assertEqual(counts[0],0)
    
    def test_01_02_point(self):
        """Make sure convex_hull can handle the degenerate case of one point"""
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(result.shape,(1,3))
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],4)
        self.assertEqual(result[0,2],5)
        self.assertEqual(counts[0],1)
    
    def test_01_030_line(self):
        """Make sure convex_hull can handle the degenerate case of a line"""
        labels = np.zeros((10,10),int)
        labels[2:8,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,7))
        self.assertTrue(result[1,1] in (2,7))
        self.assertTrue(np.all(result[:,2]==5))
    
    def test_01_031_odd_line(self):
        """Make sure convex_hull can handle the degenerate case of a line with odd length
        
        This is a regression test: the line has a point in the center if
        it's odd and the sign of the difference of that point is zero
        which causes it to be included in the hull.
        """
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],2)
        self.assertEqual(result.shape,(2,3))
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(result[0,1] in (2,6))
        self.assertTrue(result[1,1] in (2,6))
        self.assertTrue(np.all(result[:,2]==5))
    
    def test_01_04_square(self):
        """Make sure convex_hull can handle a square which is not degenerate"""
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = np.lexsort((result[:,2], result[:,1]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,7],
                                [1,6,3],
                                [1,6,7]])
        self.assertTrue(np.all(result==expected))
    
    def test_02_01_out_of_order(self):
        """Make sure convex_hull can handle out of order indices"""
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        result,counts = morph.convex_hull(labels,[2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==1))
        
        expected = np.array([[2,5,6],[1,2,3]])
        self.assertTrue(np.all(result == expected))
    
    def test_02_02_out_of_order(self):
        """Make sure convex_hull can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:7,4:8] = 2
        result,counts = morph.convex_hull(labels, [2,1])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==(4,1)))
        self.assertEqual(result.shape,(5,3))
        order = np.lexsort((result[:,2],result[:,1],
                               np.array([0,2,1])[result[:,0]]))
        result = result[order,:]
        expected = np.array([[2,1,4],
                                [2,1,7],
                                [2,6,4],
                                [2,6,7],
                                [1,2,3]])
        self.assertTrue(np.all(result==expected))
    
    def test_02_03_two_squares(self):
        """Make sure convex_hull can handle two complex shapes"""
        labels = np.zeros((10,10),int)
        labels[1:5,3:7] = 1
        labels[6:10,1:7] = 2
        result,counts = morph.convex_hull(labels, [1,2])
        self.assertEqual(counts.shape[0],2)
        self.assertTrue(np.all(counts==(4,4)))
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,1,3],[1,1,6],[1,4,3],[1,4,6],
                                [2,6,1],[2,6,6],[2,9,1],[2,9,6]])
        self.assertTrue(np.all(result==expected))
        
    def test_03_01_concave(self):
        """Make sure convex_hull handles a square with a concavity"""
        labels = np.zeros((10,10),int)
        labels[2:8,3:9] = 1
        labels[3:7,3] = 0
        labels[2:6,4] = 0
        labels[4:5,5] = 0
        result,counts = morph.convex_hull(labels,[1])
        self.assertEqual(counts[0],4)
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,8],
                                [1,7,3],
                                [1,7,8]])
        self.assertTrue(np.all(result==expected))
        
    def test_04_01_regression(self):
        """The set of points given in this case yielded one in the interior"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        labels = np.zeros((side,side),int)
        pts = np.zeros((s*s*ct,2),int)
        index = np.array(range(pts.shape[0])).astype(float)/float(ct)
        index = index.astype(int)
        idx = 0
        for i in range(0,side,mini_side):
            for j in range(0,side,mini_side):
                idx = idx+1
                # get ct+1 unique points
                p = np.random.uniform(low=0,high=mini_side,
                                         size=(ct+1,2)).astype(int)
                while True:
                    pu = np.unique(p[:,0]+p[:,1]*mini_side)
                    if pu.shape[0] == ct+1:
                        break
                    p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                    p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                    p_size = (ct+1-pu.shape[0],2)
                    p[pu.shape[0],:] = np.random.uniform(low=0,
                                                            high=mini_side,
                                                            size=p_size)
                # Use the last point as the "center" and order
                # all of the other points according to their angles
                # to this "center"
                center = p[ct,:]
                v = p[:ct,:]-center
                angle = np.arctan2(v[:,0],v[:,1])
                order = np.lexsort((angle,))
                p = p[:ct][order]
                p[:,0] = p[:,0]+i
                p[:,1] = p[:,1]+j
                pts[(idx-1)*ct:idx*ct,:]=p
                #
                # draw lines on the labels
                #
                for k in range(ct):
                    morph.draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
        self.assertTrue(labels[5,106]==5)
        result,counts = morph.convex_hull(labels,np.array(range(100))+1)
        self.assertFalse(np.any(np.logical_and(result[:,1]==5,
                                                     result[:,2]==106)))
    
    def test_05_01_missing_labels(self):
        '''Ensure that there's an entry if a label has no corresponding points'''
        labels = np.zeros((10,10),int)
        labels[3:6,2:8] = 2
        result, counts = morph.convex_hull(labels, np.arange(2)+1)
        self.assertEqual(counts.shape[0], 2)
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[1], 4)
        
    def test_06_01_regression_373(self):
        '''Regression test of IMG-374'''
        labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        result, counts = morph.convex_hull(labels, np.array([1]))
        self.assertEqual(counts[0], 2)
        
    def test_06_02_same_point_twice(self):
        '''Regression test of convex_hull_ijv - same point twice in list'''
        
        ii = [79, 11, 65, 73, 42, 26, 46, 48, 14, 53, 73, 42, 59, 12, 59, 65,  7, 66, 84, 70]
        jj = [47, 97, 98,  0, 91, 49, 42, 85, 63, 19,  0,  9, 71, 15, 50, 98, 14, 46, 89, 47]
        h, c = morph.convex_hull_ijv(
            np.column_stack((ii, jj, np.ones(len(ii)))), [1])
        self.assertTrue(np.any((h[:,1] == 73) & (h[:,2] == 0)))

class TestConvexHullImage(unittest.TestCase):
    def test_00_00_zeros(self):
        image = np.zeros((10,13), bool)
        output = morph.convex_hull_image(image)
        self.assertTrue(np.all(output == False))
        
    def test_01_01_square(self):
        image = np.zeros((10,13), bool)
        image[2:5,3:8] = True
        output = morph.convex_hull_image(image)
        self.assertTrue(np.all(output == image))
    
    def test_01_02_concave(self):
        image = np.zeros((10,13), bool)
        image[2:5,3:8] = True
        image2 = image.copy()
        image2[4,4:7] = False
        output = morph.convex_hull_image(image2)
        self.assertTrue(np.all(output == image))
        
class TestMinimumEnclosingCircle(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure minimum_enclosing_circle can handle an empty array"""
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10),int), [])
        self.assertEqual(np.product(center.shape),0)
        self.assertEqual(np.product(radius.shape),0)
    
    def test_01_01_01_zeros(self):
        """Make sure minimum_enclosing_circle can work if a label has no points"""
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10),int), [1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(np.product(radius.shape),1)
        self.assertEqual(radius[0],0)
    
    def test_01_01_02_zeros(self):
        """Make sure minimum_enclosing_circle can work if one of two labels has no points
        
        This is a regression test of a bug
        """
        labels = np.zeros((10,10), int)
        labels[2,2:5] = 3
        labels[2,6:9] = 4
        hull_and_point_count = morph.convex_hull(labels)
        center,radius = morph.minimum_enclosing_circle(
            labels,
            hull_and_point_count=hull_and_point_count)
        self.assertEqual(center.shape,(2,2))
        self.assertEqual(np.product(radius.shape),2)
    
    def test_01_02_point(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of one point"""
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertEqual(center.shape,(1,2))
        self.assertEqual(radius.shape,(1,))
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertEqual(radius[0],0)
    
    def test_01_03_line(self):
        """Make sure minimum_enclosing_circle can handle the degenerate case of a line"""
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertEqual(radius[0],2)
    
    def test_01_04_square(self):
        """Make sure minimum_enclosing_circle can handle a square which is not degenerate"""
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        self.assertTrue(np.all(center==np.array([(4,5)])))
        self.assertAlmostEqual(radius[0],np.sqrt(8))
    
    def test_02_01_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices"""
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        center,radius = morph.minimum_enclosing_circle(labels,[2,1])
        self.assertEqual(center.shape,(2,2))
        
        expected_center = np.array(((5,6),(2,3)))
        self.assertTrue(np.all(center == expected_center))
    
    def test_02_02_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:6,4:9] = 2
        center,result = morph.minimum_enclosing_circle(labels, [2,1])
        expected_center = np.array(((3,6),(2,3)))
        self.assertTrue(np.all(center == expected_center))
    
    def test_03_01_random_polygons(self):
        """Test minimum_enclosing_circle on 250 random dodecagons"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        #
        # We keep going until we get at least 10 multi-edge cases -
        # polygons where the minimum enclosing circle intersects 3+ vertices
        #
        n_multi_edge = 0
        while n_multi_edge < 10:
            labels = np.zeros((side,side),int)
            pts = np.zeros((s*s*ct,2),int)
            index = np.array(range(pts.shape[0])).astype(float)/float(ct)
            index = index.astype(int)
            idx = 0
            for i in range(0,side,mini_side):
                for j in range(0,side,mini_side):
                    idx = idx+1
                    # get ct+1 unique points
                    p = np.random.uniform(low=0,high=mini_side,
                                             size=(ct+1,2)).astype(int)
                    while True:
                        pu = np.unique(p[:,0]+p[:,1]*mini_side)
                        if pu.shape[0] == ct+1:
                            break
                        p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                        p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                        p_size = (ct+1-pu.shape[0],2)
                        p[pu.shape[0],:] = np.random.uniform(low=0,
                                                                high=mini_side,
                                                                size=p_size)
                    # Use the last point as the "center" and order
                    # all of the other points according to their angles
                    # to this "center"
                    center = p[ct,:]
                    v = p[:ct,:]-center
                    angle = np.arctan2(v[:,0],v[:,1])
                    order = np.lexsort((angle,))
                    p = p[:ct][order]
                    p[:,0] = p[:,0]+i
                    p[:,1] = p[:,1]+j
                    pts[(idx-1)*ct:idx*ct,:]=p
                    #
                    # draw lines on the labels
                    #
                    for k in range(ct):
                        morph.draw_line(labels, p[k,:], p[(k+1)%ct,:], idx)
            center,radius = morph.minimum_enclosing_circle(labels, 
                                                           np.array(range(s**2))+1)
            epsilon = .000001
            center_per_pt = center[index]
            radius_per_pt = radius[index]
            distance_from_center = np.sqrt(np.sum((pts.astype(float)-
                                                         center_per_pt)**2,1))
            #
            # All points must be within the enclosing circle
            #
            self.assertTrue(np.all(distance_from_center - epsilon < radius_per_pt))
            pt_on_edge = np.abs(distance_from_center - radius_per_pt)<epsilon
            count_pt_on_edge = scind.sum(pt_on_edge,
                                                 index,
                                                 np.array(range(s**2),dtype=np.int32))
            count_pt_on_edge = np.array(count_pt_on_edge)
            #
            # Every dodecagon must have at least 2 points on the edge.
            #
            self.assertTrue(np.all(count_pt_on_edge>=2))
            #
            # Count the multi_edge cases
            #
            n_multi_edge += np.sum(count_pt_on_edge>=3)

class TestEllipseFromSecondMoments(unittest.TestCase):
    def assertWithinFraction(self, actual, expected, 
                             fraction=.001, message=None):
        """Assert that a "correlation" of the actual value to the expected is within the fraction
        
        actual - the value as calculated
        expected - the expected value of the variable
        fraction - the fractional difference of the two
        message - message to print on failure
        
        We divide the absolute difference by 1/2 of the sum of the variables
        to get our measurement.
        """
        measurement = abs(actual-expected)/(2*(actual+expected))
        self.assertTrue(measurement < fraction,
                        "%(actual)f != %(expected)f by the measure, abs(%(actual)f-%(expected)f)) / 2(%(actual)f + %(expected)f)"%(locals()))
        
    def test_00_00_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [])
        self.assertEqual(centers.shape,(0,2))
        self.assertEqual(eccentricity.shape[0],0)
        self.assertEqual(major_axis_length.shape[0],0)
        self.assertEqual(minor_axis_length.shape[0],0)
    
    def test_00_01_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [1])
        self.assertEqual(centers.shape,(1,2))
        self.assertEqual(eccentricity.shape[0],1)
        self.assertEqual(major_axis_length.shape[0],1)
        self.assertEqual(minor_axis_length.shape[0],1)
    
    def test_01_01_rectangle(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((10,20)),
                                              np.ones((10,20),int),
                                              [1])
        self.assertEqual(centers.shape,(1,2))
        self.assertEqual(eccentricity.shape[0],1)
        self.assertEqual(major_axis_length.shape[0],1)
        self.assertEqual(minor_axis_length.shape[0],1)
        self.assertAlmostEqual(eccentricity[0],.866,2)
        self.assertAlmostEqual(centers[0,0],4.5)
        self.assertAlmostEqual(centers[0,1],9.5)
        self.assertWithinFraction(major_axis_length[0],23.0940,.001)
        self.assertWithinFraction(minor_axis_length[0],11.5470,.001)
        self.assertAlmostEqual(theta[0],0)
    
    def test_01_02_circle(self):
        img = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        img[x*x+y*y<=2500] = 1
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((101,101)),img,[1])
        self.assertAlmostEqual(eccentricity[0],0)
        self.assertWithinFraction(major_axis_length[0],100,.001)
        self.assertWithinFraction(minor_axis_length[0],100,.001)
    
    def test_01_03_blob(self):
        '''Regression test a blob against Matlab measurements'''
        blob = np.array(
            [[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
             [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
             [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0]])
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones(blob.shape),blob,[1])
        self.assertAlmostEqual(major_axis_length[0],37.55,1)
        self.assertAlmostEqual(minor_axis_length[0],18.99,1)
        self.assertAlmostEqual(eccentricity[0],0.8627,2)
        self.assertAlmostEqual(centers[0,1],14.1689,2)
        self.assertAlmostEqual(centers[0,0],14.8691,2)
        
    def test_02_01_compactness_square(self):
        image = np.zeros((9,9), int)
        image[1:8,1:8] = 1
        compactness = morph.ellipse_from_second_moments(
            np.ones(image.shape), image, [1], True)[-1]
        i,j = np.mgrid[0:9, 0:9]
        v_i = np.var(i[image > 0])
        v_j = np.var(j[image > 0])
        v = v_i + v_j
        area = np.sum(image > 0)
        expected = 2 * np.pi * v / area
        self.assertAlmostEqual(compactness, expected)
        

class TestCalculateExtents(unittest.TestCase):
    def test_00_00_zeros(self):
        """Make sure calculate_extents doesn't throw an exception if no image"""
        extents = morph.calculate_extents(np.zeros((10,10),int), [1])
    
    def test_01_01_square(self):
        """A square should have an extent of 1"""
        labels = np.zeros((10,10),int)
        labels[1:8,2:9]=1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,1)
    
    def test_01_02_circle(self):
        """A circle should have an extent of pi/4"""
        labels = np.zeros((1001,1001),int)
        y,x = np.mgrid[-500:501,-500:501]
        labels[x*x+y*y<=250000] = 1
        extents = morph.calculate_extents(labels,[1])
        self.assertAlmostEqual(extents,np.pi/4,2)
        
    def test_01_03_two_objects(self):
        '''Make sure that calculate_extents works with more than one object
        
        Regression test of a bug: was computing area like this:
        scind.sum(labels, labels, indexes)
        which works for the object that's labeled "1", but is 2x for 2, 3x
        for 3, etc... oops.
        '''
        labels = np.zeros((10,20), int)
        labels[3:7, 2:5] = 1
        labels[3:5, 5:8] = 1
        labels[2:8, 13:17] = 2
        extents = morph.calculate_extents(labels, [1,2])
        self.assertEqual(len(extents), 2)
        self.assertAlmostEqual(extents[0], .75)
        self.assertAlmostEqual(extents[1], 1)
        
class TestMedianOfLabels(unittest.TestCase):
    def test_00_00_zeros(self):
        result = morph.median_of_labels(np.zeros((10,10)), 
                                        np.zeros((10,10), int),
                                        np.zeros(0, int))
        self.assertEqual(len(result), 0)
        
    def test_00_01_empty(self):
        result = morph.median_of_labels(np.zeros((10,10)), 
                                        np.zeros((10,10), int),
                                        [1])
        self.assertEqual(len(result), 1)
        self.assertTrue(np.isnan(result[0]))
        
    def test_01_01_one_odd(self):
        r = np.random.RandomState()
        r.seed(11)
        fill = r.uniform(size=25)
        img = np.zeros((10,10))
        labels = np.zeros((10,10), int)
        labels[3:8,3:8] = 1
        img[labels > 0] = fill
        result = morph.median_of_labels(img, labels, [ 1 ])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], np.median(fill))
        
    def test_01_02_one_even(self):
        r = np.random.RandomState()
        r.seed(12)
        fill = r.uniform(size=20)
        img = np.zeros((10,10))
        labels = np.zeros((10,10), int)
        labels[3:8,3:7] = 1
        img[labels > 0] = fill
        result = morph.median_of_labels(img, labels, [ 1 ])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], np.median(fill))
        
    def test_01_03_two(self):
        r = np.random.RandomState()
        r.seed(12)
        img = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        labels[3:8,3:7] = 1
        labels[3:8,13:18] = 2
        for i, fill in enumerate([r.uniform(size=20), r.uniform(size=25)]):
            img[labels == i+1] = fill
        result = morph.median_of_labels(img, labels, [ 1,2 ])
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0], np.median(img[labels==1]))
        self.assertAlmostEqual(result[1], np.median(img[labels==2]))
        
        
class TestCalculatePerimeters(unittest.TestCase):
    def test_00_00_zeros(self):
        """The perimeters of a zeros matrix should be all zero"""
        perimeters = morph.calculate_perimeters(np.zeros((10,10),int),[1])
        self.assertEqual(perimeters,0)
    
    def test_01_01_square(self):
        """The perimeter of a square should be the sum of the sides"""
        
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        self.assertEqual(perimeter, 4*8)
        
    def test_01_02_circle(self):
        """The perimeter of a circle should be pi * diameter"""
        labels = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        labels[x*x+y*y<=2500] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        epsilon = 20
        self.assertTrue(perimeter-np.pi*101<epsilon)
        
    def test_01_03_on_edge(self):
        """Check the perimeter of objects touching edges of matrix"""
        labels = np.zeros((10,20), int)
        labels[:4,:4] = 1 # 4x4 square = 16 pixel perimeter
        labels[-4:,-2:] = 2 # 4x2 square = 2+2+4+4 = 12
        expected = [ 16, 12]
        perimeter = morph.calculate_perimeters(labels, [1,2])
        self.assertEqual(len(perimeter), 2)
        self.assertEqual(perimeter[0], expected[0])
        self.assertEqual(perimeter[1], expected[1])

class TestCalculateConvexArea(unittest.TestCase):
    def test_00_00_degenerate_zero(self):
        """The convex area of an empty labels matrix should be zero"""
        labels = np.zeros((10,10),int)
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],0)
    
    def test_00_01_degenerate_point(self):
        """The convex area of a point should be 1"""
        labels = np.zeros((10,10),int)
        labels[4,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],1)

    def test_00_02_degenerate_line(self):
        """The convex area of a line should be its length"""
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertEqual(result[0],8)
    
    def test_01_01_square(self):
        """The convex area of a square should be its area"""
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],64)
    
    def test_01_02_cross(self):
        """The convex area of a cross should be the area of the enclosing diamond
        
        The area of a diamond is 1/2 of the area of the enclosing bounding box
        """
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        labels[4,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        self.assertEqual(result.shape[0],1)
        self.assertAlmostEqual(result[0],32)
    
    def test_02_01_degenerate_point_and_line(self):
        """Test a degenerate point and line in the same image, out of order"""
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[1:9,4] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[0],8)
        self.assertEqual(result[1],1)
    
    def test_02_02_degenerate_point_and_square(self):
        """Test a degenerate point and a square in the same image"""
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[3:8,4:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertEqual(result[1],1)
        self.assertAlmostEqual(result[0],25)
    
    def test_02_03_square_and_cross(self):
        """Test two non-degenerate figures"""
        labels = np.zeros((20,10),int)
        labels[1:9,1:9] = 1
        labels[11:19,4] = 2
        labels[14,1:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        self.assertEqual(result.shape[0],2)
        self.assertAlmostEqual(result[0],32)
        self.assertAlmostEqual(result[1],64)

class TestEulerNumber(unittest.TestCase):
    def test_00_00_even_zeros(self):
        labels = np.zeros((10,12),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_00_01_odd_zeros(self):
        labels = np.zeros((11,13),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_00_square(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
        
    def test_01_01_square_with_hole(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[3:6,3:6] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],0)
    
    def test_01_02_square_with_two_holes(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[2:4,2:8] = 0
        labels[6:8,2:8] = 0
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],-1)
    
    def test_02_01_square_touches_border(self):
        labels = np.ones((10,10),int)
        result = morph.euler_number(labels, [1])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0],1)
    
    def test_03_01_two_objects(self):
        labels = np.zeros((10,10), int)
        # First object has a hole - Euler # is zero
        labels[1:4,1:4] = 1
        labels[2,2] = 0
        # Second object has no hole - Euler # is 1
        labels[5:8,5:8] = 2
        result = morph.euler_number(labels, [1,2])
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)

class TestWhiteTophat(unittest.TestCase):
    '''Test the white_tophat function'''
    def test_01_01_zeros(self):
        '''Test white_tophat on an image of all zeros'''
        result = morph.white_tophat(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
    
    def test_01_02_ones(self):
        '''Test white_tophat on an image of all ones'''
        result = morph.white_tophat(np.ones((10,10)), 1)
        self.assertTrue(np.all(result==0))
    
    def test_01_03_edge(self):
        '''Test white_tophat on an image whose edge is zeros.
        
        The image should erode off the sides to a depth of 1
        and then should dilate, leaving the four corners as one
        '''
        image = np.zeros((10,10))
        image[1:9,1:9] = 1
        expected = np.zeros((10,10))
        expected[1,1] = 1
        expected[8,8] = 1
        expected[1,8] = 1
        expected[8,1] = 1
        result = morph.white_tophat(image, 1)
        self.assertTrue(np.all(result==expected))
    
    def test_01_04_random(self):
        '''Test that a random image has the same value as Matlab'''
        data = base64.b64decode(
            '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'
            '3XFcs7s/VgfGl78/1D/qOMjU2VXkP63mN5Mgle0/aA5hM/l+uD8EyLTvMFH'
            'tP9h31Sdu9bI/fSC4K8Ej5j+bZXP/Yy/nP5TRCc3jMcQ/wKIrWdbYoT+pfU'
            'Z3cX7mP1n5Fmd37us/WD18Z3833D8njar5DUzsP2ymbX/W5uc/djqRyDYl0'
            'z/qWvahyvrTP62fVz6zDOI/azSv9Qkc7T+YT9+enubDP/DrGHx3tqQ/UpFo'
            'm6G92z9Q1TnCXqDJP7zssnGbhN8/VS3uvkgM5j+Pa7fVeLTiP4IU97fHaN0'
            '/qXQZhZf74z+qktjZt+brP5DddNz1Dak/6LnnNKrY5j9VwV+UNtfsP0aVtu'
            'A6WNQ/30NGCX925D/RDau7ZPvvPy8SyRHWdeY/QNxHWLWRqz+CZKZA/SjvP'
            'yn4v1P92uA/3F8Q2HCP7z8kIQRry4viPxNqebuC0Oo/euy6GmB66D8GH4YS'
            'VVTgP+xuPRhNN+k/cRL7YN7+6z8zWZbhs3fqPxaMPabnEtA/Wcrye9Za5j/'
            'lMIaUW0DkPwqEGMMFQeU/wrR9+Uxy6T+iCVQ5rBvZPyYihQuyC+U/5nYnry'
            'bx0z8wY5jciPzhP65FiU6u1eQ/urfL8Cd22T+bspgPlb7tPxy38b5TF+Q/h'
            'HXUGkcy4z8OGAhjSEDhP+DaMtUZb+c/7yZx/6HW5j8cgTqj0tzEPzwgNV7W'
            'UsM/X6LwkDvS7D+sUlG39R/eP9hiY7/D9Lk/ZZu3ezL94j990CDicTvoP0A'
            '8AyZ3a4U/sKizUYL7vz8DaAIqF4bkP7iiy4PXtbE/vK/uX5FFwT/AFqyvyz'
            'icP+QgVszaNNY/oDhF1BYCuz+wmsMnDgzjP+mUWuaH1u4/qMaZmADOtz+6p'
            'G179lTTPwpMDk7G4Ow/1uiN0ob94T/9JxsTWtnlP+PqQvFHYuM/gm8o0OOR'
            '3T9toTlcJ+XgP4xYm73q/9g/RhOCsXDd2D8IRS92/xLvPzolVKKy7Oo//l6'
            '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'
            'p9kug/bbQGWCh25T+6xHr/JXfcP+rqgX9OYtg/ihr1zrPL7T9JyoXyXJXuP'
            'wIKx16+9tk/gvlhIQrk6D9EN4EtPFjOPzhGB0YNkeQ/d5iMSyq07D+M9GBT'
            'FAvoP8DMc84h6Iw/usp9Dxs62j9oRiMtIMDNPxarycPPw9I/gElN19ezjz8'
            'eYc+VolbZP6uzF/OjaeE/spbc72LE0z/08gidfcfWP8i5xhFeWus/XmZIw/'
            'oa5D/qH43w2XTpPxKSL6rs9tU/XDKVePWn5D+DJLlYc3juPyKYSm3apdQ/C'
            'tOI2/Xw1j8rnjI8Os3hPzH9zTKmHOQ/CIj2pOql2z9Uv7cwkjrAPwIuXZZB'
            'SNo/sM3K34xRwD/gXhukw4/pP0D/S9dyJrU/0J+3+jjtyz9eopXJrUHZP6C'
            'SuXQfoLU/kIN1+F0uvD8AsUI+CcTTP+hoHjXdX9g/CW3/+euO6z/uGraBnp'
            'XQP6g9nJh3tMo/YWvuUdyN6j9QR/tM7nXIPz73ZE72WOk/WG1lrOV+xj94c'
            'rE6pRC1Py6R7Mms1uk/lBEI/aI5zT8p0JE1uVfrP1Nvsp6syec/LZ5leCYI'
            '6T+g+OKGp3SkP0gAieuJaN0/gL8OEzwgxz8eQWS7ZebcP5DXZRYVCt8/DAL'
            'nAbyY2z+iBcO0DM/bPwTNnp1/4ME/sP/P0Fm7wD9EjYbNRyntP9hFZFKEYb'
            'Q/IChMYXbdoj+UbQ+TnfHPP8JYoAnjtNI/bOh6nLeizD/pWROHvSPhP4QxX'
            '1Fbs8c/7t3EzvYt6j/JsgjIuuHuPxzHOx6BWeI/JlA3reSH2D8wzARsIKe3'
            'P8DZ7fR+ddw/uPbXh74avz9FhDkPTB7mP7WEFi1qDuw/Bqtbplsw6D8wOBy'
            'jQnqgP/amk9W3IeE/K4Qdo1cq6D+4M+v8SxPHPzCE7skKPaU/5B3X6qt6zT'
            '+2NoROH5DoP0Ns+Owt5eM/nIQUHsGU5D/cQDzHwqXMP6hY8x1azbA/Gnu8V'
            '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'
            'T0No/mzkBETEK6j8=')
        expected = base64.b64decode(
            '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'
            'HxNU/TugfQWuE5z8=')
        data = np.fromstring(data)
        data.shape = (40,40)
        expected = np.fromstring(expected)
        expected.shape = (40,40)
        result = morph.white_tophat(data, 3)
        #
        # Matlab will give different results because of edge stuff
        #
        self.assertTrue(np.all(result[8:-8,8:-8] == expected[8:-8,8:-8]))
        #
        # Calculate 0:0 by hand
        #
        s = morph.strel_disk(3) > 0
        corner = np.ones((4,4))
        for i in range(4):
            for j in range(4):
                for k in range(-3,4):
                    for l in range(-3,4):
                        if s[k+3,l+3] and i+k > 0 and j+l > 0:
                            corner[i,j] = min(corner[i,j],data[i+k,j+l])
        my_max = np.max(corner[s[3:,3:]])
        my_value = data[0,0] - my_max
        self.assertEqual(my_value, result[0,0])
    
    def test_02_01_mask(self):
        '''Test white_tophat, masking the pixels that would erode'''
        image = np.zeros((10,10))
        image[1:9,1:9] = 1
        mask = image != 0
        result = morph.white_tophat(image, 1, mask)
        self.assertTrue(np.all(result==0))
    
class TestRegionalMaximum(unittest.TestCase):
    def test_00_00_zeros(self):
        '''An array of all zeros has a regional maximum of its center'''
        result = morph.regional_maximum(np.zeros((11,11)))
        self.assertEqual(np.sum(result),1)
        self.assertTrue(result[5,5])
    
    def test_00_01_zeros_with_mask(self):
        result = morph.regional_maximum(np.zeros((10,10)),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_single_maximum(self):
        '''Test that the regional maximum of a gradient from 5,5 is 5,5'''
        #
        # Create a gradient of distance from the point 5,5 in an 11x11 array
        #
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        image = 1 - i**2 - j**2
        result = morph.regional_maximum(image)
        self.assertTrue(result[5,5])
        self.assertTrue(np.all(result[image != np.max(image)]==False))
    
    def test_01_02_two_maxima(self):
        '''Test an image with two maxima'''
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        half_image = 1 - i**2 - j**2
        image = np.zeros((11,22))
        image[:,:11]=half_image
        image[:,11:]=half_image
        result = morph.regional_maximum(image)
        self.assertTrue(result[5,5])
        self.assertTrue(result[5,-6])
        self.assertTrue(np.all(result[image != np.max(image)]==False))
    
    def test_02_01_mask(self):
        '''Test that a mask eliminates one of the maxima'''
        i,j = np.mgrid[-5:6,-5:6].astype(float) / 5
        half_image = 1 - i**2 - j**2
        image = np.zeros((11,22))
        image[:,:11]=half_image
        image[:,11:]=half_image
        mask = np.ones(image.shape, bool)
        mask[4,5] = False
        result = morph.regional_maximum(image,mask)
        self.assertFalse(result[5,5])
        self.assertTrue(result[5,-6])
        self.assertTrue(np.all(result[image != np.max(image)]==False))

class TestBlackTophat(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test black tophat on an array of all zeros'''
        result = morph.black_tophat(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
        
    def test_00_01_zeros_masked(self):
        '''Test black tophat on an array that is completely masked'''
        result = morph.black_tophat(np.zeros((10,10)),1,np.zeros((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_01_01_single(self):
        '''Test black tophat of a single minimum'''
        result = morph.black_tophat(np.array([[.9,.8,.7],
                                              [.9,.5,.7],
                                              [.7,.8,.8]]),1)
        #
        # The edges should not be affected by the border
        #
        expected = np.array([[0,0,.1],[0,.3,.1],[.1,0,0]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))
    
    def test_02_01_mask(self):
        '''Test black tophat with a mask'''
        image = np.array([[.9, .8, .7],[.9,.5,.7],[.7,.8,.8]])
        mask = np.array([[1,1,0],[1,1,0],[1,0,1]],bool)
        expected = np.array([[0,.1,0],[0,.4,0],[.2,0,0]])
        result = morph.black_tophat(image, 1, mask)
        self.assertTrue(np.all(np.abs(result[mask]-expected[mask])<.0000001))

class TestClosing(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test closing on an array of all zeros'''
        result = morph.closing(np.zeros((10,10)), 1)
        self.assertTrue(np.all(result==0))
        
    def test_00_01_zeros_masked(self):
        '''Test closing on an array that is completely masked'''
        result = morph.closing(np.zeros((10,10)),1,np.zeros((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_01_01_single(self):
        '''Test closing of a single minimum'''
        result = morph.closing(np.array([[.9,.8,.7],
                                         [.9,.5,.7],
                                         [.7,.8,.8]]),1)
        #
        # The edges should not be affected by the border
        #
        expected = np.array([[.9,.8,.8],[.9,.8,.8],[.8,.8,.8]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))
    
    def test_02_01_mask(self):
        '''Test closing with a mask'''
        image = np.array([[.9, .8, .7],[.9,.5,.7],[.7,.8,.8]])
        mask = np.array([[1,1,0],[1,1,0],[1,0,1]],bool)
        expected = np.array([[.9,.9,.7],[.9,.9,.7],[.9,.8,.8]])
        result = morph.closing(image, 1, mask)
        self.assertTrue(np.all(np.abs(result[mask]-expected[mask])<.0000001))
        
    def test_03_01_8_connected(self):
        '''Test closing with an 8-connected structuring element'''
        result = morph.closing(np.array([[.9,.8,.7],
                                         [.9,.5,.7],
                                         [.7,.8,.8]]))
        expected = np.array([[.9,.8,.8],[.9,.8,.8],[.9,.8,.8]])
        self.assertTrue(np.all(np.abs(result - expected)<.00000001))

class TestBranchpoints(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test branchpoints on an array of all zeros'''
        result = morph.branchpoints(np.zeros((9,11), bool))
        self.assertTrue(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test branchpoints on an array that is completely masked'''
        result = morph.branchpoints(np.zeros((10,10),bool),
                                    np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_branchpoints_positive(self):
        '''Test branchpoints on positive cases'''
        image = np.array([[1,0,0,1,0,1,0,1,0,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,0,1,1,0,1,1,1,0,0,1]],bool)
        result = morph.branchpoints(image)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_branchpoints_negative(self):
        '''Test branchpoints on negative cases'''
        image = np.array([[1,0,0,0,1,0,0,0,1,0,1,0,1],
                          [0,1,0,0,1,0,1,1,1,0,0,1,0],
                          [0,0,1,0,1,0,0,0,0,0,0,0,0]],bool)
        result = morph.branchpoints(image)
        self.assertTrue(np.all(result==False))
        
    def test_02_01_branchpoints_masked(self):
        '''Test that masking defeats branchpoints'''
        image = np.array([[1,0,0,1,0,1,0,1,1,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,1,0,1,0,1,1,1,0,0,1]],bool)
        mask  = np.array([[0,1,1,1,1,1,1,0,0,0,1,1,0],
                          [1,1,1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,0,1,0,1,1,0,0,1,1,1]],bool)
        result = morph.branchpoints(image, mask)
        self.assertTrue(np.all(result[mask]==False))
        
class TestBridge(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test bridge on an array of all zeros'''
        result = morph.bridge(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test bridge on an array that is completely masked'''
        result = morph.bridge(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_bridge_positive(self):
        '''Test some typical positive cases of bridging'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image)
        self.assertTrue(np.all(result==expected))
    
    def test_01_02_bridge_negative(self):
        '''Test some typical negative cases of bridging'''
        image = np.array([[1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0],
                          [0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)

        expected = np.array([[1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,0],
                             [0,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1],
                             [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)
        result = morph.bridge(image)
        self.assertTrue(np.all(result==expected))

    def test_02_01_bridge_mask(self):
        '''Test that a masked pixel does not cause a bridge'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        mask = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image,mask)
        self.assertTrue(np.all(result[mask]==expected[mask]))

class TestClean(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test clean on an array of all zeros'''
        result = morph.clean(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test clean on an array that is completely masked'''
        result = morph.clean(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_clean_positive(self):
        '''Test removal of a pixel using clean'''
        image = np.array([[0,0,0],[0,1,0],[0,0,0]],bool)
        self.assertTrue(np.all(morph.clean(image) == False))
    
    def test_01_02_clean_negative(self):
        '''Test patterns that should not clean'''
        image = np.array([[1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0],
                          [0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1]],bool)
        self.assertTrue(np.all(image == morph.clean(image)))
    
    def test_02_01_clean_edge(self):
        '''Test that clean removes isolated pixels on the edge of an image'''
        
        image = np.array([[1,0,1,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,1,0,1]],bool)
        self.assertTrue(np.all(morph.clean(image) == False))
        
    def test_02_02_clean_mask(self):
        '''Test that clean removes pixels adjoining a mask'''
        image = np.array([[0,0,0],[1,1,0],[0,0,0]],bool)
        mask  = np.array([[1,1,1],[0,1,1],[1,1,1]],bool)
        result= morph.clean(image,mask)
        self.assertEqual(result[1,1], False)
    
    def test_03_01_clean_labels(self):
        '''Test clean on a labels matrix where two single-pixel objects touch'''
        
        image = np.zeros((10,10), int)
        image[2,2] = 1
        image[2,3] = 2
        image[5:8,5:8] = 3
        result = morph.clean(image)
        self.assertTrue(np.all(result[image != 3] == 0))
        self.assertTrue(np.all(result[image==3] == 3))

class TestDiag(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test diag on an array of all zeros'''
        result = morph.diag(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test diag on an array that is completely masked'''
        result = morph.diag(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_diag_positive(self):
        '''Test all cases of diag filling in a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,1,0,0,0,0,1,0,0],
                          [0,1,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.diag(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_diag_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.diag(image)))
    
    def test_02_01_diag_edge(self):
        '''Test that diag works on edges'''
        
        image = np.array([[1,0,0,0,1],
                          [0,1,0,1,0],
                          [0,0,0,0,0],
                          [0,1,0,1,0],
                          [1,0,0,0,1]],bool)
        expected = np.array([[1,1,0,1,1],
                             [1,1,0,1,1],
                             [0,0,0,0,0],
                             [1,1,0,1,1],
                             [1,1,0,1,1]],bool)
        self.assertTrue(np.all(morph.diag(image) == expected))
        image = np.array([[0,1,0,1,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,1,0,1,0]],bool)
        self.assertTrue(np.all(morph.diag(image) == expected))
        
        
    def test_02_02_diag_mask(self):
        '''Test that diag connects if one of the pixels is masked'''
        image = np.array([[0,0,0],
                          [1,0,0],
                          [1,1,0]],bool)
        mask  = np.array([[1,1,1],
                          [1,1,1],
                          [0,1,1]],bool)
        result= morph.diag(image,mask)
        self.assertEqual(result[1,1], True)
        
class TestEndpoints(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test endpoints on an array of all zeros'''
        result = morph.endpoints(np.zeros((9,11), bool))
        self.assertTrue(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test endpoints on an array that is completely masked'''
        result = morph.endpoints(np.zeros((10,10),bool),
                                 np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_positive(self):
        '''Test positive endpoint cases'''
        image = np.array([[0,0,0,1,0,1,0,0,0,0,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,0,0,0,0,0,1,0]],bool)
        result = morph.endpoints(image)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_negative(self):
        '''Test negative endpoint cases'''
        image = np.array([[0,0,1,0,0,1,0,0,0,0,0,1],
                          [0,1,0,1,0,1,0,0,1,1,0,1],
                          [1,0,0,0,1,0,0,1,0,0,1,0]],bool)
        result = morph.endpoints(image)
        self.assertTrue(np.all(result[1,:] == False))
        
    def test_02_02_mask(self):
        '''Test that masked positive pixels don't change the endpoint determination'''
        image = np.array([[0,0,1,1,0,1,0,1,0,1,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,1,0,0,0,0,1,0]],bool)
        mask  = np.array([[1,1,0,1,1,1,1,0,1,0,1],
                          [1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,0,1,1,1,1,1,1]],bool)
        result = morph.endpoints(image, mask)
        self.assertTrue(np.all(image[1,:] == result[1,:]))
    
class TestFill(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test fill on an array of all zeros'''
        result = morph.fill(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test fill on an array that is completely masked'''
        result = morph.fill(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_fill_positive(self):
        '''Test addition of a pixel using fill'''
        image = np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
        self.assertTrue(np.all(morph.fill(image)))
    
    def test_01_02_fill_negative(self):
        '''Test patterns that should not fill'''
        image = np.array([[0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1],
                          [1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1],
                          [1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0]],bool)
        self.assertTrue(np.all(image == morph.fill(image)))
    
    def test_02_01_fill_edge(self):
        '''Test that fill fills isolated pixels on an edge'''
        
        image = np.array([[0,1,0,1,0],
                          [1,1,1,1,1],
                          [0,1,1,1,0],
                          [1,1,1,1,1],
                          [0,1,0,1,0]],bool)
        self.assertTrue(np.all(morph.fill(image) == True))
        
    def test_02_02_fill_mask(self):
        '''Test that fill adds pixels if a neighbor is masked'''
        image = np.array([[1,1,1],
                          [0,0,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.fill(image,mask)
        self.assertEqual(result[1,1], True)

class TestHBreak(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test hbreak on an array of all zeros'''
        result = morph.hbreak(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test hbreak on an array that is completely masked'''
        result = morph.hbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_hbreak_positive(self):
        '''Test break of a horizontal line'''
        image = np.array([[1,1,1],
                          [0,1,0],
                          [1,1,1]],bool)
        expected = np.array([[1,1,1],
                             [0,0,0],
                             [1,1,1]],bool)
        self.assertTrue(np.all(morph.hbreak(image)==expected))
    
    def test_01_02_hbreak_negative(self):
        '''Test patterns that should not hbreak'''
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.hbreak(image)))
    
class TestVBreak(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test vbreak on an array of all zeros'''
        result = morph.vbreak(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test vbreak on an array that is completely masked'''
        result = morph.vbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_vbreak_positive(self):
        '''Test break of a vertical line'''
        image = np.array([[1,0,1],
                          [1,1,1],
                          [1,0,1]],bool)
        expected = np.array([[1,0,1],
                             [1,0,1],
                             [1,0,1]],bool)
        self.assertTrue(np.all(morph.vbreak(image)==expected))
    
    def test_01_02_vbreak_negative(self):
        '''Test patterns that should not vbreak'''
        # stolen from hbreak
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        image = image.transpose()
        self.assertTrue(np.all(image == morph.vbreak(image)))
    
class TestMajority(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test majority on an array of all zeros'''
        result = morph.majority(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test majority on an array that is completely masked'''
        result = morph.majority(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_majority(self):
        '''Test majority on a random field'''
        np.random.seed(0)
        image = np.random.uniform(size=(10,10)) > .5
        expected = scipy.ndimage.convolve(image.astype(int), np.ones((3,3)), 
                                          mode='constant', cval=0) > 4.5
        result = morph.majority(image)
        self.assertTrue(np.all(result==expected))
                                        
class TestRemove(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test remove on an array of all zeros'''
        result = morph.remove(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test remove on an array that is completely masked'''
        result = morph.remove(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_remove_positive(self):
        '''Test removing a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,1,1,0,1,1,1,0],
                          [0,1,1,1,0,1,1,1,0,1,1,1,0],
                          [0,0,1,0,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,1,1,0,1,1,1,0],
                             [0,1,0,1,0,1,0,1,0,1,0,1,0],
                             [0,0,1,0,0,0,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.remove(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_remove_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],bool)
        self.assertTrue(np.all(image == morph.remove(image)))
    
    def test_02_01_remove_edge(self):
        '''Test that remove does nothing'''
        
        image = np.array([[1,1,1,1,1],
                          [1,1,0,1,1],
                          [1,0,0,0,1],
                          [1,1,0,1,1],
                          [1,1,1,1,1]],bool)
        self.assertTrue(np.all(morph.remove(image) == image))
        
    def test_02_02_remove_mask(self):
        '''Test that a masked pixel does not cause a remove'''
        image = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.remove(image,mask)
        self.assertEqual(result[1,1], True)

class TestSkeleton(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test skeletonize on an array of all zeros'''
        result = morph.skeletonize(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test skeletonize on an array that is completely masked'''
        result = morph.skeletonize(np.zeros((10,10),bool),
                                   np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_rectangle(self):
        '''Test skeletonize on a rectangle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        #
        # The result should be four diagonals from the
        # corners, meeting in a horizontal line
        #
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_hole(self):
        '''Test skeletonize on a rectangle with a hole in the middle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        image[4,4:-4] = False
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        self.assertTrue(np.all(result == expected))
         
class TestSpur(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test spur on an array of all zeros'''
        result = morph.spur(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test spur on an array that is completely masked'''
        result = morph.spur(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_spur_positive(self):
        '''Test removing a spur pixel'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
                             [0,1,1,1,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_spur_negative(self):
        '''Test patterns that should not spur'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0],
                          [0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        l,count = scind.label(result,scind.generate_binary_structure(2, 2))
        self.assertEqual(count, 5)
        a = np.array(scind.sum(result,l,np.arange(4,dtype=np.int32)+1))
        self.assertTrue(np.all((a==1) | (a==4)))
    
    def test_02_01_spur_edge(self):
        '''Test that spurs on edges go away'''
        
        image = np.array([[1,0,0,1,0,0,1],
                          [0,1,0,1,0,1,0],
                          [0,0,1,1,1,0,0],
                          [1,1,1,1,1,1,1],
                          [0,0,1,1,1,0,0],
                          [0,1,0,1,0,1,0],
                          [1,0,0,1,0,0,1]],bool)
        expected = np.array([[0,0,0,0,0,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,1,1,1,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        self.assertTrue(np.all(result == expected))
        
    def test_02_02_spur_mask(self):
        '''Test that a masked pixel does not prevent a spur remove'''
        image = np.array([[1,0,0],
                          [1,1,0],
                          [0,0,0]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.spur(image,mask)
        self.assertEqual(result[1,1], False)

class TestThicken(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test thicken on an array of all zeros'''
        result = morph.thicken(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thicken on an array that is completely masked'''
        result = morph.thicken(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_thicken_positive(self):
        '''Test thickening positive cases'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
                             [0,1,1,1,0,0,0,1,0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,0,0],
                             [1,1,1,1,1,0,1,1,1,1,1,1,1,1,0],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],
                             [0,0,0,0,0,1,1,1,0,0,0,0,1,1,1]],bool)
        result = morph.thicken(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_thicken_negative(self):
        '''Test patterns that should not thicken'''
        image = np.array([[1,1,0,1],
                          [0,0,0,0],
                          [1,1,1,1],
                          [0,0,0,0],
                          [1,1,0,1]],bool)
        result = morph.thicken(image)
        self.assertTrue(np.all(result==image))
    
    def test_02_01_thicken_edge(self):
        '''Test thickening to the edge'''
        
        image = np.zeros((5,5),bool)
        image[1:-1,1:-1] = True
        result = morph.thicken(image)
        self.assertTrue(np.all(result))
        
class TestThin(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test thin on an array of all zeros'''
        result = morph.thin(np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thin on an array that is completely masked'''
        result = morph.thin(np.zeros((10,10),bool),np.zeros((10,10),bool))
        self.assertTrue(np.all(result==False))
    
    def test_01_01_bar(self):
        '''Test thin on a bar of width 3'''
        image = np.zeros((10,10), bool)
        image[3:6,2:8] = True
        expected = np.zeros((10,10), bool)
        expected[4,3:7] = True
        result = morph.thin(expected,iterations = None)
        self.assertTrue(np.all(result==expected))
    
    def test_02_01_random(self):
        '''A random image should preserve its Euler number'''
        np.random.seed(0)
        for i in range(20):
            image = np.random.uniform(size=(100,100)) < .1+float(i)/30.
            expected_euler_number = morph.euler_number(image)
            result = morph.thin(image)
            euler_number = morph.euler_number(result)
            if euler_number != expected_euler_number:
                from scipy.io.matlab import savemat
                savemat("c:\\temp\\euler.mat", 
                        {"orig":image, 
                         "orig_euler":np.array([expected_euler_number]),
                         "result":result,
                         "result_euler":np.array([euler_number]) },
                         False, "5", True)
            self.assertTrue(expected_euler_number == euler_number)
    
    def test_03_01_labels(self):
        '''Thin a labeled image'''
        image = np.zeros((10,10), int)
        #
        # This is two touching bars
        #
        image[3:6,2:8] = 1
        image[6:9,2:8] = 2
        expected = np.zeros((10,10),int)
        expected[4,3:7] = 1
        expected[7,3:7] = 2
        result = morph.thin(expected,iterations = None)
        self.assertTrue(np.all(result==expected))

class TestTableLookup(unittest.TestCase):
    def test_01_01_all_centers(self):
        '''Test table lookup at pixels off of the edge'''
        image = np.zeros((512*3+2,5),bool)
        for i in range(512):
            pattern = morph.pattern_of(i)
            image[i*3+1:i*3+4,1:4] = pattern
        table = np.arange(512)
        table[511] = 0 # do this to force using the normal mechanism
        index = morph.table_lookup(image, table, False, 1)
        self.assertTrue(np.all(index[2::3,2] == table))
    
    def test_01_02_all_corners(self):
        '''Test table lookup at the corners of the image'''
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for p00 in (False,True):
                for p01 in (False, True):
                    for p10 in (False, True):
                        for p11 in (False,True):
                            image = np.array([[False,False,False,False,False,False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,False,False,False,False,False]])
                            expected = morph.table_lookup(image,table,False,1)[1:-1,1:-1]
                            result = morph.table_lookup(image[1:-1,1:-1],table,False,1)
                            self.assertTrue(np.all(result==expected),
                                            "Failure case:\n%7s,%s\n%7s,%s"%
                                            (p00,p01,p10,p11))
    
    def test_01_03_all_edges(self):
        '''Test table lookup along the edges of the image'''
        image = np.zeros((32*3+2,6),bool)
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for i in range(32):
                pattern = morph.pattern_of(i)
                image[i*3+1:i*3+4,1:3] = pattern[:,:2]
                image[i*3+1:i*3+4,3:5] = pattern[:,:2]
            for im in (image,image.transpose()):
                expected = morph.table_lookup(im,table,False, 1)[1:-1,1:-1]
                result = morph.table_lookup(im[1:-1,1:-1],table,False,1)
                self.assertTrue(np.all(result==expected))
         
class TestBlock(unittest.TestCase):
    def test_01_01_one_block(self):
        labels, indexes = morph.block((10,10),(10,10))
        self.assertEqual(len(indexes),1)
        self.assertEqual(indexes[0],0)
        self.assertTrue(np.all(labels==0))
        self.assertEqual(labels.shape,(10,10))
    
    def test_01_02_six_blocks(self):
        labels, indexes = morph.block((10,15),(5,5))
        self.assertEqual(len(indexes),6)
        self.assertEqual(labels.shape, (10,15))
        i,j = np.mgrid[0:10,0:15]
        self.assertTrue(np.all(labels == (i / 5).astype(int)*3 + (j/5).astype(int)))

    def test_01_03_big_blocks(self):
        labels, indexes = morph.block((10,10),(20,20))
        self.assertEqual(len(indexes),1)
        self.assertEqual(indexes[0],0)
        self.assertTrue(np.all(labels==0))
        self.assertEqual(labels.shape,(10,10))

    def test_01_04_small_blocks(self):
        labels, indexes = morph.block((100,100),(2,4))
        self.assertEqual(len(indexes), 1250)
        i,j = np.mgrid[0:100,0:100]
        i = (i / 2).astype(int)
        j = (j / 4).astype(int)
        expected = i * 25 + j
        self.assertTrue(np.all(labels == expected))

class TestNeighbors(unittest.TestCase):
    def test_00_00_zeros(self):
        labels = np.zeros((10,10),int)
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 0)
        self.assertEqual(len(v_indexes), 0)
        self.assertEqual(len(v_neighbors), 0)
    
    def test_01_01_no_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2] = 1
        labels[7,7] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 2)
        self.assertEqual(v_counts[0], 0)
        self.assertEqual(v_counts[1], 0)
    
    def test_01_02_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2:5] = 1
        labels[3,2:5] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 2)
        self.assertEqual(v_counts[0], 1)
        self.assertEqual(v_neighbors[v_indexes[0]], 2)
        self.assertEqual(v_counts[1], 1)
        self.assertEqual(v_neighbors[v_indexes[1]], 1)
    
    def test_01_03_complex(self):
        labels = np.array([[1,1,2,2],
                           [2,2,2,3],
                           [4,3,3,3],
                           [5,6,3,3],
                           [0,7,8,9]])
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        self.assertEqual(len(v_counts), 9)
        for i, neighbors in ((1,[2]),
                             (2,[1,3,4]),
                             (3,[2,4,5,6,7,8,9]),
                             (4,[2,3,5,6]),
                             (5,[3,4,6,7]),
                             (6,[3,4,5,7,8]),
                             (7,[3,5,6,8]),
                             (8,[3,6,7,9]),
                             (9,[3,8])):
            i_neighbors = v_neighbors[v_indexes[i-1]:v_indexes[i-1]+v_counts[i-1]]
            self.assertTrue(np.all(i_neighbors == np.array(neighbors)))

class TestColor(unittest.TestCase):
    def test_01_01_color_zeros(self):
        '''Color a labels matrix of all zeros'''
        labels = np.zeros((10,10), int)
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors==0))
    
    def test_01_02_color_ones(self):
        '''color a labels matrix of all ones'''
        labels = np.ones((10,10), int)
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors==1))

    def test_01_03_color_complex(self):
        '''Create a bunch of shapes using Voroni cells and color them'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = scind.label(mask, np.ones((3,3),bool))
        distances,(i,j) = scind.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        colors = morph.color_labels(labels)
        l00 = labels[1:-2,1:-2]
        c00 = colors[1:-2,1:-2]
        for i,j in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            lij = labels[1+i:i-2,1+j:j-2]
            cij = colors[1+i:i-2,1+j:j-2]
            self.assertTrue(np.all((l00 == lij) | (c00 != cij)))
            
    def test_02_01_color_127(self):
        '''Color 127 labels stored in a int8 array
        
        Regression test of img-1099
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        colors = morph.color_labels(labels)
        self.assertTrue(np.all(colors[labels==0] == 0))
        self.assertTrue(np.all(colors[labels!=0] == 1))
            
class TestSkeletonizeLabels(unittest.TestCase):
    def test_01_01_skeletonize_complex(self):
        '''Skeletonize a complex field of shapes and check each individually'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = scind.label(mask, np.ones((3,3),bool))
        distances,(i,j) = scind.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        skel = morph.skeletonize_labels(labels)
        for i in range(1,count+1,10):
            mask = labels == i
            skel_test = morph.skeletonize(mask)
            self.assertTrue(np.all(skel[skel_test] == i))
            self.assertTrue(np.all(skel[~skel_test] != i))

class TestAssociateByDistance(unittest.TestCase):
    def test_01_01_zeros(self):
        '''Test two label matrices with nothing in them'''
        result = morph.associate_by_distance(np.zeros((10,10),int),
                                             np.zeros((10,10),int), 0)
        self.assertEqual(result.shape[0], 0)
    
    def test_01_02_one_zero(self):
        '''Test a labels matrix with objects against one without'''
        result = morph.associate_by_distance(np.ones((10,10),int),
                                             np.zeros((10,10),int), 0)
        self.assertEqual(result.shape[0], 0)
    
    def test_02_01_point_in_square(self):
        '''Test a single point in a square'''
        #
        # Point is a special case - only one point in its convex hull
        #
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,3] = 3
        l2[2,9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],3)
    
    def test_02_02_line_in_square(self):
        '''Test a line in a square'''
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,2:5] = 3
        l2[2,6:9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],3)
    
    def test_03_01_overlap(self):
        '''Test a square overlapped by four other squares'''
        
        l1 = np.zeros((20,20),int)
        l1[5:16,5:16] = 1
        l2 = np.zeros((20,20),int)
        l2[1:6,1:6] = 1
        l2[1:6,14:19] = 2
        l2[14:19,1:6] = 3
        l2[14:19,14:19] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0],4)
        self.assertTrue(np.all(result[:,0]==1))
        self.assertTrue(all([x in result[:,1] for x in range(1,5)]))
    
    def test_03_02_touching(self):
        '''Test two objects touching at one point'''
        l1 = np.zeros((10,10), int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,10), int)
        l2[5:9,5:9] = 1
        result = morph.associate_by_distance(l1, l2, 0)
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result[0,0],1)
        self.assertEqual(result[0,1],1)
    
    def test_04_01_distance_square(self):
        '''Test two squares separated by a distance'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[3:6,10:16] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        self.assertEqual(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        self.assertEqual(result.shape[0],1)
    
    def test_04_02_distance_triangle(self):
        '''Test a triangle and a square (edge to point)'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[4,10] = 1
        l2[3:6,11] = 1
        l2[2:7,12] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        self.assertEqual(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        self.assertEqual(result.shape[0],1)

class TestDistanceToEdge(unittest.TestCase):
    '''Test distance_to_edge'''
    def test_01_01_zeros(self):
        '''Test distance_to_edge with a matrix of zeros'''
        result = morph.distance_to_edge(np.zeros((10,10),int))
        self.assertTrue(np.all(result == 0))
    
    def test_01_02_square(self):
        '''Test distance_to_edge with a 3x3 square'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        self.assertTrue(np.all(result == expected))
    
    def test_01_03_touching(self):
        '''Test distance_to_edge when two objects touch each other'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        labels[6:9,3:6] = 2
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        expected[6:9,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        self.assertTrue(np.all(result == expected))

class TestGreyReconstruction(unittest.TestCase):
    '''Test grey_reconstruction'''
    def test_01_01_zeros(self):
        '''Test grey_reconstruction with image and mask of zeros'''
        self.assertTrue(np.all(morph.grey_reconstruction(np.zeros((5,7)),
                                                         np.zeros((5,7))) == 0))
    
    def test_01_02_image_equals_mask(self):
        '''Test grey_reconstruction where the image and mask are the same'''
        self.assertTrue(np.all(morph.grey_reconstruction(np.ones((7,5)),
                                                         np.ones((7,5))) == 1))
    
    def test_01_03_image_less_than_mask(self):
        '''Test grey_reconstruction where the image is uniform and less than mask'''
        image = np.ones((5,5))
        mask = np.ones((5,5)) * 2
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) == 1))
    
    def test_01_04_one_image_peak(self):
        '''Test grey_reconstruction with one peak pixel'''
        image = np.ones((5,5))
        image[2,2] = 2
        mask = np.ones((5,5)) * 3
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) == 2))
    
    def test_01_05_two_image_peaks(self):
        '''Test grey_reconstruction with two peak pixels isolated by the mask'''
        image = np.array([[1,1,1,1,1,1,1,1],
                          [1,2,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,1,1],
                          [1,1,1,1,1,1,3,1],
                          [1,1,1,1,1,1,1,1]])
        
        mask = np.array([[4,4,4,1,1,1,1,1],
                         [4,4,4,1,1,1,1,1],
                         [4,4,4,1,1,1,1,1],
                         [1,1,1,1,1,4,4,4],
                         [1,1,1,1,1,4,4,4],
                         [1,1,1,1,1,4,4,4]])

        expected = np.array([[2,2,2,1,1,1,1,1],
                             [2,2,2,1,1,1,1,1],
                             [2,2,2,1,1,1,1,1],
                             [1,1,1,1,1,3,3,3],
                             [1,1,1,1,1,3,3,3],
                             [1,1,1,1,1,3,3,3]])
        self.assertTrue(np.all(morph.grey_reconstruction(image,mask) ==
                               expected))
    
    def test_02_01_zero_image_one_mask(self):
        '''Test grey_reconstruction with an image of all zeros and a mask that's not'''
        result = morph.grey_reconstruction(np.zeros((10,10)), np.ones((10,10)))
        self.assertTrue(np.all(result == 0))
        
class TestGetLinePts(unittest.TestCase):
    def test_01_01_no_pts(self):
        '''Can we call get_line_pts with zero-length vectors?'''
        i0, j0, i1, j1 = [np.zeros((0,))] * 4
        index, count, i, j = morph.get_line_pts(i0, j0, i1, j1)
        self.assertEqual(len(index), 0)
        self.assertEqual(len(count), 0)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(j), 0)
    
    def test_01_02_horizontal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[0],[10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(i==0))
        self.assertTrue(np.all(j==np.arange(11)))
    
    def test_01_03_vertical_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[0])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==0))
        self.assertTrue(np.all(i==np.arange(11)))
    
    def test_01_04_diagonal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==np.arange(11)))
        self.assertTrue(np.all(i==np.arange(11)))
        
    def test_01_05_antidiagonal_line(self):
        index, count, i, j = morph.get_line_pts([0],[0],[10],[-10])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 11)
        self.assertTrue(np.all(j==-np.arange(11)))
        self.assertTrue(np.all(i==np.arange(11)))
        
    def test_01_06_single_point(self):
        index, count, i, j = morph.get_line_pts([0],[0],[0],[0])
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0], 0)
        self.assertEqual(len(count), 1)
        self.assertEqual(count[0], 1)
        self.assertEqual(i[0], 0)
        self.assertEqual(j[0], 0)
        
    def test_02_01_test_many(self):
        np.random.seed(0)
        n = 100
        i0,i1,j0,j1 = (np.random.uniform(size=(4,n))*100).astype(int)
        index, count, i_out, j_out = morph.get_line_pts(i0, j0, i1, j1)
        #
        # Run the Bresenham algorithm on each of the points manually
        #
        for idx in range(n):
            diff_i = abs(i1[idx]-i0[idx])
            diff_j = abs(j1[idx]-j0[idx])
            i = i0[idx]
            j = j0[idx]
            self.assertTrue(count[idx] > 0)
            self.assertEqual(i_out[index[idx]], i)
            self.assertEqual(j_out[index[idx]], j)
            step_i = (i1[idx] > i0[idx] and 1) or -1
            step_j = (j1[idx] > j0[idx] and 1) or -1
            pt_idx = 0
            if diff_j > diff_i:
                # J varies fastest, do i before j
                remainder = diff_i*2 - diff_j
                while j != j1[idx]:
                    pt_idx += 1
                    self.assertTrue(count[idx] > pt_idx)
                    if remainder >= 0:
                        i += step_i
                        remainder -= diff_j*2
                    j += step_j
                    remainder += diff_i*2
                    self.assertEqual(i_out[index[idx]+pt_idx], i)
                    self.assertEqual(j_out[index[idx]+pt_idx], j)
            else:
                remainder = diff_j*2 - diff_i
                while i != i1[idx]:
                    pt_idx += 1
                    self.assertTrue(count[idx] > pt_idx)
                    if remainder >= 0:
                        j += step_j
                        remainder -= diff_i*2
                    i += step_i
                    remainder += diff_j*2
                    self.assertEqual(j_out[index[idx]+pt_idx], j)
                    self.assertEqual(i_out[index[idx]+pt_idx], i)

class TestAllConnectedComponents(unittest.TestCase):
    def test_01_01_no_edges(self):
        result = morph.all_connected_components(np.array([], int), np.array([], int))
        self.assertEqual(len(result), 0)
        
    def test_01_02_one_component(self):
        result = morph.all_connected_components(np.array([0]), np.array([0]))
        self.assertEqual(len(result),1)
        self.assertEqual(result[0], 0)
        
    def test_01_03_two_components(self):
        result = morph.all_connected_components(np.array([0,1]), 
                                                np.array([0,1]))
        self.assertEqual(len(result),2)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[1], 1)
        
    def test_01_04_one_connection(self):
        result = morph.all_connected_components(np.array([0,1,2]),
                                                np.array([0,2,1]))
        self.assertEqual(len(result),3)
        self.assertTrue(np.all(result == np.array([0,1,1])))
        
    def test_01_05_components_can_label(self):
        #
        # all_connected_components can be used to label a matrix
        #
        np.random.seed(0)
        for d in ((10,12),(100,102)):
            mask = np.random.uniform(size=d) < .2
            mask[-1,-1] = True
            #
            # Just do 4-connectivity
            #
            labels, count = scind.label(mask)
            i,j = np.mgrid[0:d[0],0:d[1]]
            connected_top = (i > 0) & mask[i,j] & mask[i-1,j]
            idx = np.arange(np.prod(d))
            idx.shape = d
            connected_top_j = idx[connected_top] - d[1]
            
            connected_bottom = (i < d[0]-1) & mask[i,j] & mask[(i+1) % d[0],j]
            connected_bottom_j = idx[connected_bottom] + d[1]
            
            connected_left = (j > 0) & mask[i,j] & mask[i,j-1]
            connected_left_j = idx[connected_left] - 1
            
            connected_right = (j < d[1]-1) & mask[i,j] & mask[i,(j+1) % d[1]]
            connected_right_j = idx[connected_right] + 1
            
            i = np.hstack((idx[mask],
                           idx[connected_top],
                           idx[connected_bottom],
                           idx[connected_left],
                           idx[connected_right]))
            j = np.hstack((idx[mask], connected_top_j, connected_bottom_j,
                           connected_left_j, connected_right_j))
            result = morph.all_connected_components(i,j)
            self.assertEqual(len(result), np.prod(d))
            result.shape = d
            result[mask] += 1
            result[~mask] = 0
            #
            # Correlate the labels with the result
            #
            coo = scipy.sparse.coo_matrix((np.ones(np.prod(d)),
                                           (labels.flatten(),
                                            result.flatten())))
            corr = coo.toarray()
            #
            # Make sure there's either no or one hit per label association
            #
            self.assertTrue(np.all(np.sum(corr != 0,0) <= 1))
            self.assertTrue(np.all(np.sum(corr != 0,1) <= 1))
            
class TestBranchings(unittest.TestCase):
    def test_00_00_zeros(self):
        self.assertTrue(np.all(morph.branchings(np.zeros((10,11), bool)) == 0))
        
    def test_01_01_endpoint(self):
        image = np.zeros((10,11), bool)
        image[5,5:] = True
        self.assertEqual(morph.branchings(image)[5,5], 1)
        
    def test_01_02_line(self):
        image = np.zeros((10,11), bool)
        image[1:9, 5] = True
        self.assertTrue(np.all(morph.branchings(image)[2:8,5] == 2))
        
    def test_01_03_vee(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[-i == abs(j)] = True
        image[(j==0) & (i > 0)] = True
        self.assertTrue(morph.branchings(image)[5,5] == 3)
        
    def test_01_04_quadrabranch(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[abs(i) == abs(j)] = True
        self.assertTrue(morph.branchings(image)[5,5] == 4)
        
class TestLabelSkeleton(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Label a skeleton containing nothing'''
        skeleton = np.zeros((20,10), bool)
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 0)
        self.assertTrue(np.all(result == 0))
        
    def test_01_01_point(self):
        '''Label a skeleton consisting of a single point'''
        skeleton = np.zeros((20,10), bool)
        skeleton[5,5] = True
        expected = np.zeros((20,10), int)
        expected[5,5] = 1
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(result == expected))
        
    def test_01_02_line(self):
        '''Label a skeleton that's a line'''
        skeleton = np.zeros((20,10), bool)
        skeleton[5:15, 5] = True
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 1)
        self.assertTrue(np.all(result[skeleton] == 1))
        self.assertTrue(np.all(result[~skeleton] == 0))
        
    def test_01_03_branch(self):
        '''Label a skeleton that has a branchpoint'''
        skeleton = np.zeros((21,11), bool)
        i,j = np.mgrid[-10:11,-5:6]
        #
        # Looks like this:
        #  .   .
        #   . .
        #    .
        #    .
        skeleton[(i < 0) & (np.abs(i) == np.abs(j))] = True
        skeleton[(i >= 0) & (j == 0)] = True
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton] == 0))
        self.assertTrue(np.all(result[skeleton] > 0))
        self.assertEqual(result[10,5], 1)
        v1 = result[5,0]
        v2 = result[5,-1]
        v3 = result[-1, 5]
        self.assertEqual(len(np.unique((v1, v2, v3))), 3)
        self.assertTrue(np.all(result[(i < 0) & (i==j)] == v1))
        self.assertTrue(np.all(result[(i < 0) & (i==-j)] == v2))
        self.assertTrue(np.all(result[(i > 0) & (j == 0)] == v3))
        
    def test_02_01_branch_and_edge(self):
        '''A branchpoint meeting an edge at two points'''
        
        expected = np.array(((2,0,0,0,0,1),
                             (0,2,0,0,1,0),
                             (0,0,3,1,0,0),
                             (0,0,4,0,0,0),
                             (0,4,0,0,0,0),
                             (4,0,0,0,0,0)))
        skeleton = expected > 0
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton] == 0))
        self.assertEqual(len(np.unique(result)), 5)
        self.assertEqual(np.max(result), 4)
        self.assertEqual(np.min(result), 0)
        for i in range(1,5):
            self.assertEqual(len(np.unique(result[expected == i])), 1)

    def test_02_02_four_edges_meet(self):
        '''Check the odd case of four edges meeting at a square
        
        The shape is something like this:
        
        .    .
         .  .
          ..
          ..
         .  .
        .    .
        None of the points above are branchpoints - they're sort of
        half-branchpoints.
        '''
        i,j = np.mgrid[-10:10,-10:10]
        i[i<0] += 1
        j[j<0] += 1
        skeleton=np.abs(i) == np.abs(j)
        result, count = morph.label_skeleton(skeleton)
        self.assertEqual(count, 4)
        self.assertTrue(np.all(result[~skeleton]==0))
        self.assertEqual(np.max(result), 4)
        self.assertEqual(np.min(result), 0)
        self.assertEqual(len(np.unique(result)), 5)
        for im in (-1, 1):
            for jm in (-1, 1):
                self.assertEqual(len(np.unique(result[(i*im == j*jm) & 
                                                      (i*im > 0) &
                                                      (j*jm > 0)])), 1)
                
class TestPairwisePermutations(unittest.TestCase):
    def test_00_00_empty(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([]), np.array([]))
        for x in (i, j1, j2):
            self.assertEqual(len(x), 0)
            
    def test_00_01_no_permutations_of_one(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([4]), np.array([3]))
        for x in (i, j1, j2):
            self.assertEqual(len(x), 0)

    def test_01_01_two(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([4,4]), np.array([9,3]))
        for x, v in ((i, 4), (j1, 3), (j2, 9)):
            self.assertEqual(len(x), 1)
            self.assertEqual(x[0], v)
    
    def test_01_02_many(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([7,7,7,5,5,5,5,9,9,9,9,9,9]),
                                              np.array([1,3,2,4,5,8,6,1,2,3,4,5,6]))
        for x, v in (
            (i,  np.array([5,5,5,5,5,5,7,7,7,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9])),
            (j1, np.array([4,4,4,5,5,6,1,1,2,1,1,1,1,1,2,2,2,2,3,3,3,4,4,5])),
            (j2, np.array([5,6,8,6,8,8,2,3,3,2,3,4,5,6,3,4,5,6,4,5,6,5,6,6]))):
            self.assertEqual(len(x), len(v))
            self.assertTrue(np.all(x == v))
            
    def test_01_03_gaps(self):
        i,j1,j2 = morph.pairwise_permutations(np.array([1,1,1,2,3,3,3]),
                                              np.array([3,2,1,1,1,2,3]))
        for x, v in (
            (i, np.array([1,1,1,3,3,3])),
            (j1, np.array([1,1,2,1,1,2])),
            (j2, np.array([2,3,3,2,3,3]))):
            self.assertEqual(len(x), len(v))
            self.assertTrue(np.all(x == v))
                                                       
    def test_01_04_gaps(self):
        '''regression test of img-1485'''
        i = np.array([
            3585,3585,3585,3586,3586,3587,3587,3588,3588,
            3589,3589,3589,3590,3590,3591,3591,3592,3592,
            3593,3594,3594,3595,3595,3596,3596,3597,3597,
            3598,3598,3599,3599,3600,3600,3601,3601,3602,
            3602,3603,3603,3604,3604,3605,3605,3606,3606,
            3607,3607,3608,3608,3609,3609,3610,3610,3611,
            3611,3612,3612,3613,3613,3614,3614,3615,3615,
            3616,3616,3617,3617,3618,3618,3619,3619,3620,
            3620,3621,3621,3622,3622,3623,3623,3624,3624,
            3625,3625,3626,3626,3627,3627,3628,3628,3629,
            3629,3630,3630,3631,3631,3632,3632,3633,3633,
            3634,3634,3635,3635,3636,3636,3637,3637,3638,
            3638,3639,3639,3640,3640,3641,3641,3642,3642,
            3643,3643,3644,3644,3645,3645,3646,3646,3647,
            3647,3648,3648,3648,3649,3649,3650,3650,3651,
            3651,3652,3652,3653,3653,3654,3654,3655,3655,
            3656,3656,3656,3657,3657,3658,3658,3659,3659,
            3660,3660,3661,3661,3662,3662,3663,3663,3664,
            3664,3664,3665,3665,3666,3666,3667,3667,3668,
            3668,3669,3669,3670,3670,3671,3671,3672,3672,
            3673,3673,3674,3674,3675,3675,3676,3676,3677,
            3677,3678,3678,3679,3679,3680,3680,3681,3681,
            3681,3682,3682,3683,3683,3684,3684,3685,3685,
            3686,3686,3687,3687,3688,3688,3689,3689,3690,
            3690,3691,3691,3692,3692,3693,3693,3694,3694,
            3695,3695,3696,3696,3697,3697,3698,3698,3699,
            3699,3700,3700,3701,3701,3702,3702,3703,3703,
            3704,3704,3705,3705,3706,3706,3707,3707,3708,
            3708,3709,3709,3710,3710,3711,3711,3712,3712,
            3713,3713,3714,3714,3715,3715,3716,3716,3717,
            3717,3718,3718,3719,3719,3720,3720,3721,3721,
            3722,3722,3723,3723,3724,3724,3725,3725,3726,
            3726,3727,3727,3728,3728,3728,3729,3729,3730,
            3730,3731,3731,3732,3732,3732,3733,3733,3734,
            3734,3735,3735,3736,3736,3737,3737,3738,3738,
            3739,3739,3740,3740,3741,3741,3742,3742,3743,
            3743,3744,3744,3745,3745,3746,3746,3747,3747,
            3748,3748,3749,3749,3750,3750,3751,3751,3752,
            3752,3753,3753,3754,3754,3755,3755,3756,3756,
            3757,3757,3758,3758,3759,3759,3760,3760,3761,
            3761,3762,3762,3763,3763,3764,3764,3765,3765,
            3766,3766,3767,3767,3768,3768,3769,3769,3770,
            3770,3771,3771,3772,3772,3773,3773,3774,3774,
            3775,3775,3776,3776,3777,3777,3778,3778,3779,
            3779,3780,3780,3781,3781,3782,3782,3783,3783,
            3784,3784,3785,3785,3786,3786,3787,3787,3788,
            3788,3789,3789,3790,3790,3791,3791,3792,3792,
            3793,3793,3794,3794,3795,3795,3796,3796,3797,
            3797,3798,3798,3799,3799,3800,3800,3801,3801,
            3802,3802,3803,3803,3804,3804,3805,3805,3806,
            3806,3807,3807,3808,3808,3809,3809,3810,3810,
            3811,3811,3812,3812,3813,3813,3814,3814,3815,
            3815,3816,3816,3817,3817,3818,3818,3819,3819,
            3820,3820,3821,3821,3822,3822,3823,3823,3824,
            3824,3825,3825,3826,3826,3827,3827,3828,3828,
            3829,3829,3830,3830,3831,3831,3832,3832,3833,
            3833,3834,3834,3835,3835,3836,3836,3837,3837,
            3838,3838,3839,3839,3840,3840,3841,3841,3842,
            3842,3843,3843,3844,3844,3845,3845,3846,3846,
            3847,3847,3848,3848,3848,3849,3849,3850,3850,
            3851,3851,3852,3852,3853,3853,3854,3854,3855,
            3855,3856,3856,3857,3857,3858,3858,3859,3859,
            3860,3860,3861,3861,3862,3862,3863,3863,3864,
            3864,3865,3865,3866,3866,3867,3867,3868,3868,
            3869,3869,3870,3870,3871,3871,3872,3872,3873,
            3873,3874,3874,3875,3875,3876,3876,3877,3877,
            3878,3878,3879,3879,3880,3880,3881,3881,3881,
            3882,3882,3883,3883,3884,3884,3885,3885,3886,
            3886,3887,3887,3888,3888,3889,3889,3890,3890,
            3891,3891,3892,3892,3893,3893,3894,3894,3895,
            3895,3896,3896,3897,3897,3898,3898,3899,3899,
            3900,3900,3901,3901,3902,3902,3903,3903,3904,
            3904,3905,3905,3906,3906,3907,3907,3908,3908,
            3909,3909,3910,3910,3911,3911,3912,3912,3913,
            3913,3914,3914,3915,3915,3916,3916,3917,3917,
            3918,3918,3919,3919,3920,3920,3921,3921,3922,
            3922,3923,3923,3924,3924,3925,3925,3926,3926,
            3927,3927,3928,3928,3929,3929,3930,3930,3931,
            3931,3932,3932,3933,3933,3934,3934,3935,3935,
            3935,3936,3936,3937,3937,3937,3938,3938,3939,
            3939,3940,3940,3941,3941,3942,3942,3943,3943,
            3944,3944,3945,3945,3946,3946,3947,3947,3948,
            3948,3949,3949,3950,3950,3951,3951,3952,3952,
            3953,3953,3954,3954,3955,3955,3956,3956,3957,
            3957,3957,3958,3958,3959,3959,3960,3960,3961,
            3961,3962,3962,3963,3963,3964,3964,3965,3965,
            3966,3966,3967,3967,3968,3968,3969,3969,3970,
            3970,3971,3971,3972,3972,3973,3973,3974,3974,
            3975,3975,3976,3976,3977,3977,3978,3978,3979,
            3979,3980,3980,3981,3981,3982,3982,3983,3983,
            3984,3984,3985,3985,3986,3986,3987,3987,3988,
            3988,3989,3989,3990,3990,3991,3991,3992,3992,
            3993,3993,3994,3994,3995,3995,3996,3996,3997,
            3997,3998,3998,3999,3999,4000,4000,4001,4001,
            4002,4002,4003,4003,4004,4004,4005,4005,4006,
            4006,4007,4007,4008,4008,4009,4009,4010,4010,
            4011,4011,4012,4012,4013,4013,4014,4014,4015,
            4015,4016,4016,4017,4017,4018,4018,4019,4019,
            4020,4020,4021,4021,4022,4022,4023,4023,4024,
            4024,4025,4025,4026,4026,4027,4027,4028,4028,
            4028,4029,4029,4030,4030,4031,4031,4032,4032,
            4033,4033,4034,4034,4035,4035,4036,4036,4037,
            4037,4038,4038,4039,4039,4040,4040,4041,4041,
            4042,4042,4043,4043,4044,4044,4045,4045,4046,
            4046,4047,4047,4048,4048,4049,4049,4050,4050,
            4051,4051,4052,4052,4053,4053,4054,4054,4055,
            4055,4056,4056,4057,4057,4058,4058,4059,4059,
            4060,4060,4061,4061,4062,4062,4063,4063,4064,
            4064,4065,4065,4066,4066,4067,4067,4068,4068,
            4069,4069,4070,4070,4071,4071,4072,4072,4073,
            4073,4074,4074,4075,4075,4076,4076,4077,4077,
            4078,4078,4079,4079,4080,4080,4081,4081,4082,
            4082,4083,4083,4084,4084,4085,4085,4086,4086,
            4087,4087,4088,4088,4089,4089,4090,4090,4091,
            4091,4092,4092,4093,4093,4094,4094,4095,4095,
            4096,4096,4097,4097,4098,4098,4099,4099,4100,
            4100,4101,4101,4102,4102,4103,4103,4104,4104,
            4105,4105,4106,4106,4107,4107,4108,4108,4109,
            4109,4110,4110,4111,4111,4112,4112,4113,4113,
            4114,4114,4115,4115,4116,4116,4117,4117,4118,
            4118,4119,4119,4120,4120,4121,4121,4122,4122,
            4123,4123,4124,4124,4125,4125,4126,4126,4127,
            4127,4128,4128,4129,4129,4130,4130,4131,4131,
            4132,4132,4133,4133,4134,4134,4135,4135,4136,
            4136,4137,4137,4138,4138,4139,4139,4140,4140,
            4141,4141,4142,4142,4143,4143,4144,4144,4145,
            4145,4146,4146,4147,4147,4148,4148,4149,4149,
            4150,4150,4151,4151,4152,4152,4153,4153,4153,
            4154,4154,4155,4155,4156,4156,4157,4157,4158,
            4158,4159,4159,4160,4160,4161,4161,4162,4162,
            4163,4163,4164,4164,4165,4165,4165,4166,4166,
            4167,4167,4168,4168,4169,4169,4170,4170,4171,
            4171,4172,4172,4173,4173,4174,4174,4175,4175,
            4176,4176,4177,4177,4178,4179,4179,4180,4180,
            4181,4181,4182,4182,4183,4183,4184,4184,4185,
            4185,4186,4186,4187,4187,4188,4188,4189,4189,
            4190,4190,4191,4191,4192,4192,4193,4193,4194,
            4194,4195,4195,4196,4196,4197,4197,4198,4199,
            4199,4200,4200,4201,4201,4202,4202,4203,4203,
            4204,4204,4205,4205,4206,4206,4207,4207,4208,
            4208,4209,4210,4211,4211,4212,4212,4213,4213,
            4214,4214,4215,4215,4216,4216,4217,4217,4218,
            4218,4219,4219,4220,4220,4221,4221,4222,4222,
            4223,4223,4224,4224,4225,4225,4226,4226,4227,
            4227,4228,4228,4229,4229,4230,4230,4231,4231,
            4232,4232,4233,4233,4234,4234,4235,4235,4236,
            4236,4237,4237,4238,4238,4239,4239,4240,4240,
            4241,4241,4242,4242,4243,4243,4244,4244,4245,
            4245,4246,4246,4247,4247,4248,4248,4249,4249,
            4250,4250,4251,4251,4252,4252,4253,4253,4254,
            4254,4255,4255,4256,4256,4257,4257,4258,4258,
            4259,4259,4260,4260,4261,4261,4262,4262,4263,
            4263,4264,4264,4265,4265,4266,4266,4267,4267,
            4268,4268,4269,4269,4270,4270,4271,4271,4272,
            4272,4273,4273,4274,4274,4275,4275,4276,4276,
            4277,4277,4278,4278,4279,4279,4280,4280,4281,
            4281,4282,4282,4283,4283,4284,4284,4285,4285,
            4286,4286,4287,4287,4288,4288,4289,4289,4290,
            4290,4291,4291,4292,4292,4293,4293,4294,4294,
            4295,4295,4296,4296,4297,4297,4298,4298,4299,
            4299,4300,4300,4301,4301,4302,4302,4303,4303,
            4304,4304,4305,4305,4306,4306,4307,4307,4308,
            4308,4309,4309,4310,4310,4311,4311,4312,4312,
            4313,4313,4314,4314,4315,4315,4316,4316,4317,
            4317,4318,4318,4319,4319,4320,4320,4321,4321,
            4322,4322,4323,4323,4324,4324,4325,4325,4326,
            4326,4327,4327,4328,4328,4329,4329,4330,4330,
            4331,4331,4332,4332,4333,4333,4334,4334,4335,
            4335,4336,4336,4337,4337,4338,4338,4338,4339,
            4339,4340,4340,4341,4341,4342,4342,4343,4343,
            4344,4344,4345,4345,4346,4346,4347,4347,4348,
            4348,4349,4349,4350,4350,4351,4351,4352,4352,
            4353,4353,4354,4354,4355,4355,4356,4356,4357,
            4357,4358,4358,4359,4359,4360,4360,4361,4361,
            4362,4362,4363,4363,4364,4364,4365,4365,4366,
            4366,4367,4367,4368,4368,4369,4369,4370,4370,
            4371,4371,4372,4372,4373,4373,4374,4374,4375,
            4375,4376,4376,4377,4377,4378,4378,4379,4379,
            4380,4380,4381,4381,4382,4382,4383,4383,4384,
            4384,4385,4385,4386,4386,4387,4387,4388,4388,
            4389,4389,4390,4390,4391,4391,4392,4392,4393,
            4393,4394,4394,4395,4395,4396,4396,4397,4397,
            4398,4398,4399,4399,4400,4400,4401,4401,4402,
            4402,4403,4403,4404,4404,4405,4405,4406,4406,
            4407,4407,4408,4408,4409,4409,4410,4410,4411,
            4411,4412,4412,4413,4413,4414,4414,4415,4415,
            4416,4416,4417,4417,4418,4418,4419,4419,4420,
            4420,4421,4421,4422,4422,4423,4423,4424,4424,
            4425,4425,4426,4426,4427,4427,4428,4428,4429,
            4429,4430,4430,4431,4431,4432,4432,4433,4433,
            4434,4434,4435,4435,4436,4436,4437,4437,4438,
            4438,4439,4439,4440,4440,4441,4441,4442,4442,
            4443,4443,4444,4444,4445,4445,4446,4446,4447,
            4447,4448,4448,4449,4449,4450,4450,4451,4451,
            4452,4452,4453,4453,4454,4454,4455,4455,4456,
            4456,4457,4457,4458,4458,4459,4459,4460,4460,
            4461,4461,4462,4462,4463,4463,4464,4464,4465,
            4465,4466,4466,4467,4467,4468,4468,4469,4469,
            4470,4470,4471,4471,4472,4472,4473,4473,4474,
            4474,4475,4475,4476,4476,4477,4477,4478,4478,
            4479,4479,4480,4481,4481,4482,4482,4483,4483,
            4484,4484,4485,4485,4486,4486,4487,4487,4488,
            4488,4489,4489,4490,4490,4491,4492,4492,4493,
            4493,4494,4494,4495,4495,4496,4496,4497,4497,
            4498,4498,4499,4499,4500,4500,4501,4501,4502,
            4502,4503,4503,4504,4504,4505,4505,4506,4506,
            4507,4507,4508,4508,4509,4509,4510,4510,4511,
            4511,4512,4512,4513,4513,4514,4514,4514,4515,
            4515,4516,4516,4517,4517,4518,4518,4519,4519,
            4520,4520,4521,4521,4522,4522,4523,4523,4524,
            4524,4525,4525,4526,4526,4527,4527,4528,4528,
            4529,4529,4530,4530,4531,4531,4532,4532,4533,
            4533,4534,4534,4535,4535,4536,4536,4537,4537,
            4538,4538,4539,4539,4540,4540,4541,4541,4542,
            4542,4543,4543,4544,4544,4545,4545,4546,4546,
            4547,4547,4548,4548,4549,4549,4550,4550,4551,
            4551,4552,4552,4553,4553,4554,4554,4555,4555,
            4556,4556,4557,4557,4558,4558,4559,4559,4560,
            4560,4561,4561,4562,4562,4563,4563,4564,4564,
            4565,4565,4566,4566,4567,4567,4568,4568,4569,
            4569,4570,4570,4571,4571,4572,4572,4573,4573,
            4574,4574,4575,4575,4576,4576,4577,4577,4578,
            4578,4579,4579,4580,4580,4581,4581,4582,4582,
            4583,4583,4584,4584,4585,4585,4586,4586,4587,
            4587,4588,4588,4589,4589,4590,4590,4591,4591,
            4592,4592,4593,4593,4594,4594,4595,4595,4596,
            4596,4597,4597,4598,4598,4599,4599,4600,4600,
            4601,4601,4602,4602,4603,4603,4604,4604,4605,
            4605,4606,4606,4607,4607,4608,4608,4609,4609,
            4610,4610,4611,4611,4612,4612,4613,4613,4614,
            4614,4615,4615,4616,4616,4617,4617,4618,4618,
            4619,4619,4620,4620,4621,4621,4622,4622,4623,
            4623,4624,4624,4625,4625,4626,4626,4627,4627,
            4628,4628,4629,4629,4630,4630,4631,4631,4632,
            4632,4633,4633,4634,4634,4635,4635,4636,4636,
            4637,4637,4638,4638,4639,4639,4640,4640,4641,
            4641,4642,4642,4643,4643,4644,4644,4645,4645,
            4646,4646,4647,4647,4648,4648,4649,4649,4650,
            4650,4651,4651,4652,4652,4653,4653,4654,4654,
            4655,4655,4656,4656,4657,4657,4658,4658,4659,
            4659,4660,4660,4661,4661,4662,4662,4663,4663,
            4664,4664,4665,4665,4666,4666,4667,4667,4668,
            4668,4669,4669,4670,4670,4670,4671,4671,4672,
            4672,4673,4673,4674,4674,4675,4675,4676,4676,
            4677,4677,4678,4678,4679,4679,4680,4680,4681,
            4681,4682,4682,4683,4683,4684,4684,4685,4685,
            4686,4686,4687,4687,4688,4688,4689,4689,4690,
            4690,4691,4691,4692,4692,4693,4693,4694,4694,
            4695,4695,4696,4696,4697,4697,4698,4698,4699,
            4699,4700,4700,4701,4701,4702,4702,4703,4703,
            4704,4704,4705,4705,4706,4706,4707,4707,4708,
            4708,4709,4709,4710,4710,4711,4711,4712,4712,
            4713,4713,4714,4714,4715,4715,4716,4716,4717,
            4717,4718,4718,4719,4719,4720,4720,4721,4721,
            4722,4722,4723,4723,4724,4724,4725,4725,4726,
            4726,4727,4727,4728,4728,4729,4729,4730,4730,
            4731,4731,4732,4732,4733,4733,4734,4734,4735,
            4735,4736,4736,4737,4737,4738,4738,4739,4739,
            4740,4740,4741,4741,4742,4742,4743,4743,4744,
            4744,4745,4745,4746,4746,4747,4747,4748,4748,
            4749,4749,4750,4750,4751,4751,4752,4752,4753,
            4753,4754,4754,4755,4755,4756,4756,4757,4757,
            4758,4758,4759,4759,4760,4760,4761,4761,4762,
            4762,4763,4763,4764,4764,4765,4765,4766,4766,
            4767,4767,4768,4768,4769,4769,4770,4770,4771,
            4771,4772,4772,4773,4773,4774,4774,4775,4775,
            4776,4776,4777,4777,4778,4778,4779,4779,4780,
            4780,4781,4781,4782,4782,4783,4783,4784,4784,
            4785,4785,4786,4786,4787,4787,4788,4788,4789,
            4789,4790,4790,4791,4791,4792,4792,4793,4793,
            4794,4794,4795,4795,4796,4796,4797,4797,4798,
            4798,4799,4799,4800,4800,4801,4801,4802,4802,
            4803,4803,4804,4804,4805,4805,4806,4806,4807,
            4807,4808,4808,4809,4809,4810,4810,4811,4811,
            4812,4812,4813,4813,4814,4814,4815,4815,4816,
            4816,4817,4817,4818,4818,4819,4819,4820,4820,
            4821,4821,4822,4822,4823,4823,4824,4824,4825,
            4825,4826,4826,4827,4827,4828,4828,4829,4829,
            4830,4830,4831,4831,4832,4832,4833,4833,4834,
            4834,4835,4835,4836,4836,4837,4837,4838,4838,
            4839,4839,4840,4840,4841,4841,4842,4842,4843,
            4843,4844,4844,4845,4845,4846,4846,4847,4847,
            4848,4848,4849,4849,4850,4850,4851,4851,4852,
            4852,4853,4853,4854,4854,4855,4855,4856,4856,
            4857,4857,4858,4858,4859,4859,4860,4860,4861,
            4861,4862,4862,4863,4863,4864,4864,4865,4865,
            4866,4866,4867,4867,4868,4868,4869,4869,4870,
            4870,4871,4871,4872,4872,4873,4873,4874,4874,
            4875,4875,4876,4876,4877,4877,4878,4878,4879,
            4879,4880,4880,4881,4881,4882,4882,4883,4883,
            4884,4884,4885,4885,4886,4886,4887,4887,4888,
            4888,4889,4889,4890,4890,4891,4891,4892,4892,
            4893,4893,4894,4894,4895,4895,4896,4896,4897,
            4897,4898,4898,4899,4899,4900,4900,4901,4901,
            4902,4902,4903,4903,4904,4904,4905,4905,4906,
            4906,4907,4907,4908,4908,4908,4909,4909,4910,
            4910,4911,4911,4912,4912,4913,4913,4914,4914,
            4915,4915,4916,4916,4917,4917,4918,4918,4919,
            4919,4920,4920,4921,4921,4922,4922,4923,4923,
            4923,4924,4924,4925,4925,4926,4926,4927,4927,
            4928,4928,4929,4929,4930,4930,4931,4931,4932,
            4932,4933,4933,4934,4934,4935,4935,4936,4936,
            4937,4937,4938,4938,4939,4939,4940,4940,4941,
            4941,4942,4942,4943,4943,4944,4944,4945,4945,
            4946,4946,4947,4947,4948,4948,4949,4949,4950,
            4950,4951,4951,4952,4952,4953,4953,4954,4954,
            4955,4955,4956,4956,4957,4957,4958,4958,4959,
            4959,4960,4960,4961,4961,4962,4962,4963,4963,
            4964,4964,4965,4965,4966,4966,4967,4967,4968,
            4968,4969,4969,4970,4970,4971,4971,4972,4972,
            4973,4973,4974,4974,4975,4975,4976,4976,4977,
            4977,4978,4978,4979,4979,4980,4980,4981,4981,
            4982,4982,4983,4983,4984,4984,4985,4986,4986,
            4987,4987,4988,4988,4989,4990,4990,4991,4991,
            4992,4992,4993,4993,4994,4994,4995,4995,4996,
            4996,4997,4997,4998,4998,4999,4999,5000,5000,
            5001,5001,5002,5002,5003,5003,5004,5004,5005,
            5005,5006,5006,5007,5007,5008,5008,5009,5009,
            5010,5010,5011,5011,5012,5012,5013,5013,5014,
            5014,5015,5015,5016,5016,5017,5017,5018,5018,
            5019,5019,5020,5020,5021,5021,5022,5022,5023,
            5023,5024,5024,5025,5025,5026,5026,5027,5027,
            5028,5028,5029,5029,5030,5030,5031,5031,5032,
            5032,5033,5033,5034,5034,5035,5035,5036,5036,
            5037,5037,5038,5038,5039,5039,5040,5040,5041,
            5041,5042,5042,5043,5043,5044,5044,5045,5045,
            5046,5046,5047,5047,5048,5048,5049,5049,5050,
            5050,5051,5051,5052,5052,5053,5053,5054,5054,
            5055,5055,5056,5056,5057,5057,5058,5058,5059,
            5059,5060,5060,5061,5061,5062,5062,5063,5063,
            5064,5064,5065,5065,5066,5066,5067,5067,5068,
            5068,5069,5069,5070,5070,5071,5071,5072,5072,
            5073,5073,5074,5074,5075,5075,5075,5076,5076,
            5077,5077,5078,5078,5079,5079,5080,5080,5081,
            5081,5082,5082,5083,5083,5084,5084,5085,5085,
            5086,5086,5087,5088,5088,5089,5089,5090,5090,
            5091,5091,5092,5092,5093,5094,5094,5095,5095,
            5096,5096,5097,5097,5098,5098,5099,5099,5100,
            5100,5101,5101,5102,5102,5103,5103,5104,5104,
            5105,5105,5106,5106,5107,5107,5108,5108,5109,
            5109,5110,5110,5111,5111,5112,5112,5113,5113,
            5114,5114,5115,5115,5116,5116,5117,5117,5118,
            5118,5119,5119,5120,5120,5121,5121,5122,5122,
            5123,5123,5124,5124,5125,5125,5126,5126,5127,
            5127,5128,5128,5129,5129,5130,5130,5131,5131,
            5131,5132,5132,5133,5133,5134,5134,5135,5135,
            5136,5136,5137,5137,5138,5138,5139,5139,5140,
            5140,5141,5141,5142,5142,5143,5143,5144,5144,
            5145,5145,5146,5146,5147,5147,5147,5148,5148,
            5149,5149,5150,5150,5151,5151,5152,5152,5153,
            5153,5154,5154,5155,5155,5156,5156,5157,5157,
            5158,5158,5159,5159,5160,5160,5161,5161,5162,
            5162,5163,5163,5164,5164,5165,5165,5166,5166,
            5167,5167,5168,5168,5169,5169,5170,5170,5171,
            5171,5172,5172,5173,5173,5174,5174,5175,5175,
            5176,5176,5177,5177,5178,5178,5179,5179,5180,
            5180,5181,5181,5182,5182,5183,5183,5184,5184,
            5185,5185,5186,5186,5187,5187,5188,5188,5189,
            5189,5190,5190,5191,5191,5192,5192,5193,5193,
            5194,5194,5195,5195,5196,5196,5197,5197,5198,
            5198,5199,5199,5200,5200,5201,5201,5202,5202,
            5203,5203,5204,5204,5205,5205,5206,5206,5207,
            5207,5208,5208,5209,5209,5210,5210,5211,5211,
            5212,5212,5213,5213,5214,5214,5215,5215,5216,
            5216,5217,5217,5218,5218,5219,5219,5220,5220,
            5221,5221,5222,5222,5223,5223,5224,5224,5225,
            5225,5226,5226,5227,5227,5228,5228,5229,5229,
            5230,5230,5231,5231,5232,5232,5233,5233,5234,
            5234,5235,5235,5236,5236,5237,5237,5238,5238,
            5239,5239,5240,5240,5240,5241,5241,5242,5242,
            5243,5243,5244,5244,5245,5245,5246,5246,5247,
            5247,5248,5248,5249,5249,5250,5250,5251,5251,
            5252,5252,5253,5253,5254,5254,5255,5255,5256,
            5256,5257,5257,5258,5258,5259,5259,5260,5260,
            5261,5261,5262,5262,5263,5263,5264,5264,5265,
            5265,5265,5266,5266,5267,5267,5268,5268,5269,
            5269,5270,5270,5271,5271,5272,5272,5273,5273,
            5274,5274,5275,5275,5276,5276,5277,5277,5278,
            5278,5279,5279,5280,5280,5281,5281,5282,5282,
            5283,5283,5284,5284,5285,5285,5286,5286,5287,
            5287,5288,5288,5289,5289,5290,5290,5291,5291,
            5292,5292,5293,5293,5294,5294,5295,5295,5296,
            5296,5297,5297,5298,5298,5299,5299,5300,5300,
            5301,5301,5302,5302,5303,5303,5304,5304,5305,
            5305,5306,5306,5307,5307,5308,5308,5309,5309,
            5310,5310,5311,5311,5312,5312,5313,5313,5314,
            5314,5315,5315,5316,5316,5317,5317,5318,5318,
            5319,5319,5320,5320,5321,5321,5322,5322,5323,
            5323,5324,5324,5325,5325,5326,5326,5327,5327,
            5328,5328,5329,5329,5330,5330,5331,5331,5332,
            5332,5333,5333,5334,5334,5335,5335,5336,5336,
            5337,5337,5338,5338,5339,5339,5340,5340,5341,
            5341,5342,5342,5343,5343,5344,5344,5345,5345,
            5346,5346,5347,5347,5348,5348,5349,5349,5350,
            5350,5351,5351,5352,5352,5353,5353,5354,5354,
            5355,5355,5356,5356,5357,5357,5358,5358,5358,
            5359,5359,5360,5360,5361,5361,5362,5362,5363,
            5363,5364,5364,5365,5365,5366,5366,5367,5367,
            5368,5368,5369,5369,5370,5370,5371,5371,5372,
            5372,5373,5373,5374,5374,5375,5375,5376,5376,
            5377,5377,5378,5378,5379,5379,5380,5380,5381,
            5381,5382,5382,5382,5383,5383,5384,5384,5385,
            5385,5386,5386,5387,5387,5388,5388,5389,5389,
            5389,5390,5390,5391,5391,5392,5392,5393,5393,
            5394,5394,5395,5395,5396,5396,5397,5397,5397,
            5398,5398,5399,5399,5400,5400,5401,5401,5402,
            5402,5403,5403,5404,5404,5405,5405,5406,5406,
            5407,5407,5408,5408,5409,5409,5410,5410,5411,
            5411,5412,5412,5413,5413,5414,5414,5415,5415,
            5416,5416,5417,5417,5418,5418,5419,5419,5420,
            5420,5421,5421,5422,5422,5423,5423,5423,5424,
            5424,5425,5425,5426,5426,5427,5427,5428,5428,
            5429,5429,5430,5430,5431,5431,5432,5432,5433,
            5433,5434,5434,5435,5435,5436,5436,5437,5437,
            5438,5438,5439,5439,5440,5440,5441,5441,5442,
            5442,5443,5443,5444,5444,5445,5445,5446,5446,
            5447,5447,5448,5448,5449,5449,5450,5450,5451,
            5451,5452,5452,5453,5453,5454,5454,5455,5455,
            5456,5456,5457,5457,5458,5458,5459,5459,5460,
            5460,5461,5461,5462,5462,5463,5463,5464,5464,
            5465,5465,5466,5466,5467,5467,5468,5468,5469,
            5469,5469,5470,5470,5471,5471,5472,5472,5473,
            5473,5474,5474,5475,5475,5476,5476,5477,5477,
            5478,5478,5479,5479,5480,5480,5481,5481,5482,
            5482,5483,5483,5484,5484,5485,5485,5486,5486,
            5487,5487,5488,5488,5489,5489,5490,5490,5491,
            5491,5492,5492,5493,5493,5494,5494,5495,5495,
            5496,5496,5497,5497,5498,5498,5499,5499,5500,
            5500,5501,5501,5502,5502,5503,5503,5504,5504,
            5505,5505,5506,5506,5507,5507,5508,5508,5509,
            5509,5510,5510,5511,5511,5512,5512,5513,5513,
            5514,5514,5515,5515,5516,5516,5517,5517,5518,
            5518,5519,5519,5520,5520,5521,5521,5522,5522,
            5523,5523,5524,5524,5525,5525,5526,5526,5527,
            5527,5528,5528,5529,5529,5530,5530,5531,5531,
            5532,5532,5533,5533,5534,5534,5535,5535,5536,
            5536,5537,5537,5538,5538,5539,5539,5540,5540,
            5541,5541,5542,5542,5543,5543,5544,5544,5545,
            5545,5546,5546,5547,5547,5548,5548,5549,5549,
            5550,5550,5551,5551,5552,5552,5553,5553,5554,
            5554,5555,5555,5556,5556,5557,5557,5558,5558,
            5559,5559,5560,5560,5561,5561,5562,5562,5563,
            5563,5564,5564,5565,5565,5566,5566,5567,5567,
            5568,5568,5569,5569,5570,5570,5571,5571,5572,
            5572,5573,5573,5574,5574,5575,5575,5576,5576,
            5577,5577,5578,5578,5579,5579,5580,5580,5581,
            5581,5582,5582,5583,5583,5584,5584,5585,5585,
            5585,5586,5586,5587,5587,5588,5588,5589,5589,
            5590,5590,5591,5591,5592,5592,5593,5593,5594,
            5594,5595,5595,5596,5596,5597,5597,5597,5598,
            5598,5599,5599,5600,5600,5601,5601,5602,5602,
            5603,5603,5604,5604,5605,5605,5606,5606,5607,
            5607,5608,5608,5609,5609,5610,5610,5611,5611,
            5612,5612,5613,5613,5614,5614,5615,5615,5616,
            5616,5617,5617,5617,5618,5618,5619,5619,5620,
            5620,5621,5621,5622,5622,5623,5623,5624,5624,
            5625,5625,5626,5626,5627,5627,5628,5628,5629,
            5629,5630,5630,5631,5631,5632,5632,5633,5633,
            5634,5634,5635,5635,5636,5636,5637,5637,5638,
            5638,5639,5639,5640,5640,5641,5641,5642,5642,
            5643,5643,5644,5644,5645,5645,5646,5646,5647,
            5647,5648,5648,5649,5649,5650,5650,5651,5651,
            5652,5652,5653,5653,5654,5654,5655,5655,5656,
            5656,5657,5657,5658,5658,5659,5659,5660,5660,
            5661,5661,5662,5662,5663,5663,5664,5664,5665,
            5665,5666,5666,5667,5667,5668,5668,5669,5669,
            5670,5670,5671,5671,5672,5672,5673,5673,5674,
            5674,5675,5675,5676,5676,5677,5677,5678,5678,
            5679,5679,5680,5680,5681,5681,5682,5682,5683,
            5683,5684,5684,5685,5685,5686,5686,5687,5687,
            5688,5688,5689,5689,5690,5690,5691,5691,5692,
            5692,5693,5693,5694,5694,5695,5695,5696,5696,
            5697,5697,5698,5698,5699,5699,5700,5700,5701,
            5701,5702,5702,5703,5703,5704,5704,5705,5705,
            5706,5706,5707,5707,5708,5708,5709,5709,5710,
            5710,5711,5711,5712,5712,5713,5713,5714,5714,
            5715,5715,5716,5716,5717,5717,5718,5718,5719,
            5719,5720,5720,5721,5721,5722,5722,5723,5723,
            5724,5724,5725,5725,5726,5726,5727,5727,5728,
            5728,5729,5729,5730,5730,5731,5731,5732,5732,
            5733,5733,5734,5734,5735,5735,5736,5736,5737,
            5737,5738,5738,5739,5739,5740,5740,5741,5741,
            5742,5742,5743,5743,5744,5744,5745,5745,5746,
            5746,5746,5747,5747,5748,5748,5749,5749,5750,
            5750,5751,5751,5752,5752,5753,5753,5754,5754,
            5755,5755,5756,5756,5757,5757,5758,5758,5759,
            5759,5760,5760,5761,5761,5762,5762,5763,5763,
            5764,5764,5765,5765,5766,5766,5767,5767,5768,
            5768,5769,5769,5770,5770,5771,5771,5772,5772,
            5773,5773,5774,5774,5775,5775,5776,5776,5777,
            5777,5778,5778,5778,5779,5779,5780,5780,5781,
            5781,5782,5782,5783,5783,5784,5784,5785,5785,
            5786,5786,5787,5787,5788,5788,5789,5789,5790,
            5790,5791,5791,5792,5792,5793,5793,5794,5794,
            5795,5795,5796,5796,5796,5797,5797,5798,5798,
            5799,5799,5800,5800,5801,5801,5802,5802,5803,
            5803,5804,5804,5805,5805,5806,5806,5807,5807,
            5808,5808,5809,5809,5810,5810,5811,5811,5811,
            5812,5812,5813,5813,5814,5814,5815,5815,5816,
            5816,5817,5817,5818,5818,5819,5819,5820,5820,
            5821,5821,5822,5822,5823,5823,5824,5824,5825,
            5825,5826,5826,5827,5827,5828,5828,5829,5829,
            5830,5830,5831,5831,5832,5832,5833,5833,5834,
            5834,5835,5835,5836,5836,5837,5837,5838,5838,
            5839,5839,5840,5840,5841,5841,5842,5842,5843,
            5843,5844,5844,5845,5845,5846,5846,5847,5847,
            5848,5848,5849,5849,5850,5850,5851,5851,5852,
            5852,5853,5853,5854,5854,5855,5855,5856,5856,
            5857,5857,5858,5858,5859,5859,5860,5860,5861,
            5861,5862,5862,5863,5863,5864,5864,5865,5865,
            5866,5866,5867,5867,5868,5868,5869,5869,5870,
            5870,5871,5871,5872,5872,5873,5873,5874,5874,
            5875,5875,5876,5876,5877,5877,5878,5878,5879,
            5879,5880,5880,5881,5881,5882,5882,5883,5883,
            5884,5884,5885,5885,5886,5886,5887,5887,5888,
            5888,5889,5889,5890,5890,5891,5891,5891,5892,
            5892,5893,5893,5894,5894,5895,5895,5896,5896,
            5897,5897,5898,5898,5899,5899,5900,5900,5901,
            5901,5902,5902,5903,5903,5904,5904,5905,5905,
            5906,5906,5908,5908,5909,5909,5909,5910,5910,
            5911,5911,5912,5912,5913,5913,5914,5914,5915,
            5915,5916,5916,5917,5917,5918,5918,5919,5919,
            5920,5920,5921,5921,5922,5922,5923,5923,5924,
            5924,5925,5925,5926,5926,5927,5927,5928,5928,
            5929,5929,5930,5930,5931,5931,5932,5932,5933,
            5933,5934,5934,5935,5935,5936,5936,5937,5937,
            5938,5938,5939,5939,5940,5940,5941,5941,5942,
            5942,5943,5943,5944,5944,5945,5945,5946,5946,
            5947,5947,5948,5948,5949,5949,5950,5950,5951,
            5951,5952,5952,5953,5953,5954,5954,5955,5955,
            5955,5956,5956,5957,5957,5958,5958,5959,5959,
            5960,5960,5961,5961,5962,5962,5963,5963,5964,
            5964,5965,5965,5966,5966,5967,5967,5968,5968,
            5969,5969,5970,5970,5971,5971,5972,5972,5973,
            5973,5974,5974,5975,5975,5976,5976,5977,5977,
            5978,5978,5979,5979,5980,5980,5981,5981,5982,
            5982,5983,5983,5984,5984,5985,5985,5986,5986,
            5987,5987,5988,5988,5989,5989,5990,5990,5991,
            5991,5992,5992,5993,5993,5994,5994,5995,5995,
            5996,5996,5997,5997,5998,5998,5999,5999,6000,
            6000,6001,6001,6002,6002,6003,6003,6004,6004,
            6005,6005,6006,6006,6007,6007,6008,6008,6009,
            6009,6010,6010,6011,6011,6012,6012,6013,6013,
            6014,6014,6015,6015,6016,6016,6017,6017,6018,
            6018,6019,6019,6020,6020,6021,6021,6022,6022,
            6023,6023,6024,6024,6025,6025,6026,6026,6027,
            6027,6028,6028,6029,6029,6030,6030,6031,6031,
            6032,6032,6033,6033,6034,6034,6035,6035,6036,
            6036,6037,6037,6038,6038,6039,6039,6040,6040,
            6041,6041,6042,6042,6043,6043,6044,6044,6045,
            6045,6046,6046,6047,6047,6048,6048,6049,6049,
            6050,6050,6051,6051,6052,6052,6053,6053,6054,
            6054,6055,6055,6056,6056,6057,6057,6058,6058,
            6059,6059,6060,6060,6061,6061,6062,6062,6063,
            6063,6064,6064,6065,6065,6066,6066,6067,6067,
            6068,6068,6069,6069,6070,6070,6071,6071,6072,
            6072,6073,6073,6074,6074,6075,6075,6076,6076,
            6077,6077,6078,6078,6079,6079,6080,6080,6081,
            6081,6082,6082,6083,6083,6084,6084,6085,6085,
            6086,6086,6087,6087,6088,6088,6089,6089,6090,
            6090,6091,6091,6092,6092,6093,6093,6094,6094,
            6095,6095,6096,6096,6097,6097,6098,6098,6099,
            6099,6100,6100,6101,6101,6102,6102,6103,6103,
            6104,6104,6105,6105,6106,6106,6107,6107,6108,
            6108,6109,6109,6110,6110,6111,6111,6112,6112,
            6113,6113,6114,6114,6115,6115,6116,6116,6117,
            6117,6118,6118,6119,6119,6120,6120,6121,6121,
            6122,6122,6123,6123,6124,6124,6125,6125,6126,
            6126,6127,6127,6128,6128,6129,6129,6130,6130,
            6131,6131,6132,6132,6133,6133,6134,6134,6135,
            6135,6136,6136,6137,6137,6138,6138,6139,6139,
            6140,6140,6141,6141,6142,6142,6143,6143,6144,
            6144,6145,6145,6146,6146,6147,6147,6148,6148,
            6149,6149,6150,6150,6151,6151,6152,6152,6152,
            6153,6153,6154,6154,6155,6155,6156,6156,6157,
            6157,6158,6158,6159,6159,6160,6160,6161,6161,
            6162,6162,6163,6163,6164,6164,6165,6165,6166,
            6166,6167,6167,6168,6168,6169,6169,6170,6170,
            6171,6171,6172,6172,6173,6173,6174,6174,6175,
            6175,6176,6176,6177,6177,6178,6178,6179,6179,
            6180,6180,6181,6181,6182,6182,6183,6183,6184,
            6184,6185,6185,6186,6186,6187,6187,6188,6188,
            6189,6189,6190,6190,6191,6191,6192,6192,6193,
            6193,6194,6194,6195,6195,6196,6196,6197,6197,
            6198,6198,6199,6199,6200,6200,6201,6202,6202,
            6203,6203,6204,6204,6205,6205,6206,6206,6207,
            6207,6208,6208,6209,6209,6210,6210,6211,6211,
            6212,6212,6213,6213,6214,6214,6215,6215,6216,
            6217,6217,6218,6218,6219,6219,6220,6220,6221,
            6221,6222,6222,6223,6223,6224,6225,6226,6226,
            6227,6227,6228,6228,6229,6229,6230,6230,6231,
            6231,6232,6232,6233,6233,6234,6234,6235,6235,
            6236,6236,6237,6237,6238,6238,6239,6239,6240,
            6240,6241,6241,6242,6242,6243,6243,6244,6244,
            6245,6245,6246,6246,6247,6247,6248,6248,6249,
            6249,6250,6250,6251,6251,6252,6252,6253,6253,
            6254,6254,6255,6255,6256,6256,6257,6257,6258,
            6258,6259,6259,6260,6260,6261,6261,6262,6262,
            6263,6263,6264,6264,6265,6265,6266,6266,6267,
            6267,6268,6268,6269,6269,6270,6270,6271,6271,
            6272,6272,6273,6273,6274,6274,6275,6275,6276,
            6276,6277,6277,6278,6278,6279,6279,6280,6280,
            6281,6281,6282,6282,6282,6283,6283,6284,6284,
            6285,6285,6286,6286,6287,6287,6288,6288,6289,
            6289,6290,6290,6291,6291,6292,6292,6293,6293,
            6294,6294,6295,6295,6296,6296,6297,6297,6298,
            6298,6299,6299,6300,6300,6301,6301,6302,6302,
            6303,6303,6304,6304,6305,6305,6306,6306,6307,
            6307,6308,6308,6309,6309,6310,6310,6311,6311,
            6312,6312,6313,6313,6314,6314,6315,6315,6316,
            6316,6317,6317,6318,6318,6319,6319,6320,6320,
            6321,6321,6322,6322,6323,6323,6324,6324,6325,
            6325,6326,6326,6327,6327,6328,6328,6329,6329,
            6330,6330,6331,6331,6332,6332,6333,6333,6334,
            6334,6335,6335,6336,6336,6337,6337,6338,6338,
            6339,6339,6340,6340,6341,6341,6342,6342,6343,
            6343,6344,6344,6345,6345,6346,6346,6347,6347,
            6348,6348,6349,6349,6350,6350,6351,6351,6352,
            6352,6353,6353,6354,6354,6355,6355,6356,6356,
            6357,6357,6358,6358,6359,6359,6360,6360,6361,
            6361,6362,6362,6363,6363,6364,6364,6365,6365,
            6366,6366,6367,6367,6368,6368,6369,6369,6370,
            6370,6371,6371,6372,6372,6373,6373,6374,6374,
            6375,6375,6376,6376,6377,6377,6378,6378,6379,
            6379,6380,6380,6381,6381,6382,6382,6383,6383,
            6384,6384,6385,6385,6386,6386,6387,6387,6388,
            6388,6389,6389,6390,6390,6391,6391,6392,6392,
            6393,6393,6394,6394,6395,6395,6396,6396,6397,
            6397,6398,6398,6399,6399,6400,6400,6401,6401,
            6402,6402,6403,6403,6404,6404,6405,6405,6406,
            6406,6407,6407,6408,6408,6409,6409,6410,6410,
            6411,6411,6412,6412,6413,6413,6414,6414,6415,
            6415,6416,6416,6417,6417,6418,6418,6419,6419,
            6420,6420,6421,6421,6422,6422,6423,6423,6424,
            6424,6425,6425,6426,6426,6427,6427,6428,6428,
            6429,6429,6430,6430,6431,6431,6432,6432,6433,
            6433,6434,6434,6435,6435,6436,6436,6437,6437,
            6438,6438,6439,6439,6440,6440,6441,6441,6442,
            6442,6443,6443,6444,6444,6445,6445,6446,6446,
            6447,6447,6448,6448,6449,6449,6450,6450,6451,
            6451,6452,6452,6453,6453,6454,6454,6454,6455,
            6455,6456,6456,6457,6457,6458,6458,6459,6459,
            6460,6460,6461,6461,6462,6462,6463,6463,6464,
            6464,6465,6465,6466,6466,6467,6467,6468,6468,
            6469,6469,6470,6470,6471,6471,6472,6472,6473,
            6473,6474,6474,6475,6475,6476,6476,6477,6477,
            6478,6478,6479,6479,6480,6480,6481,6481,6482,
            6482,6483,6483,6484,6484,6485,6485,6486,6486,
            6487,6487,6488,6488,6489,6489,6490,6490,6491,
            6491,6492,6492,6493,6493,6494,6494,6495,6495,
            6496,6496,6497,6497,6498,6498,6499,6499,6500,
            6500,6501,6501,6502,6502,6503,6503,6504,6504,
            6505,6505,6506,6506,6507,6507,6508,6508,6509,
            6509,6510,6510,6511,6511,6512,6512,6513,6513,
            6514,6514,6515,6515,6516,6516,6517,6517,6518,
            6518,6519,6519,6520,6520,6521,6521,6522,6522,
            6523,6523,6524,6524,6525,6525,6526,6526,6527,
            6527,6528,6528,6529,6529,6530,6530,6531,6531,
            6532,6532,6533,6533,6534,6534,6535,6535,6536,
            6536,6537,6537,6538,6538,6539,6539,6540,6540,
            6541,6541,6542,6542,6543,6543,6544,6544,6545,
            6545,6546,6546,6547,6547,6548,6548,6549,6549,
            6550,6550,6551,6551,6552,6552,6553,6553,6554,
            6554,6555,6555,6556,6556,6557,6557,6558,6558,
            6559,6559,6560,6560,6561,6561,6562,6562,6563,
            6563,6564,6564,6565,6565,6566,6566,6567,6567,
            6567,6568,6568,6569,6569,6570,6570,6571,6571,
            6572,6572,6573,6573,6574,6574,6575,6575,6576,
            6576,6577,6577,6578,6578,6579,6579,6580,6580,
            6581,6581,6582,6582,6583,6583,6584,6584,6585,
            6585,6586,6586,6587,6587,6588,6588,6589,6589,
            6590,6590,6591,6591,6592,6592,6593,6593,6594,
            6594,6595,6595,6596,6596,6597,6597,6598,6598,
            6599,6599,6600,6600,6601,6601,6603,6603,6603,
            6604,6604,6605,6605,6606,6606,6607,6607,6608,
            6608,6609,6609,6610,6610,6611,6611,6612,6612,
            6613,6613,6614,6615,6615,6616,6616,6617,6617,
            6618,6619,6619,6620,6620,6621,6621,6622,6622,
            6623,6623,6624,6624,6625,6625,6626,6626,6627,
            6627,6628,6628,6629,6629,6630,6630,6631,6631,
            6632,6632,6633,6633,6634,6634,6635,6635,6636,
            6636,6637,6637,6638,6638,6639,6639,6640,6640,
            6641,6641,6642,6642,6643,6643,6644,6644,6645,
            6645,6646,6646,6647,6647,6648,6648,6649,6649,
            6650,6650,6651,6651,6652,6652,6653,6653,6654,
            6654,6655,6655,6656,6656,6657,6657,6658,6658,
            6659,6659,6660,6660,6661,6661,6662,6662,6663,
            6663,6664,6664,6665,6665,6666,6666,6667,6667,
            6668,6668,6669,6669,6670,6670,6671,6671,6672,
            6672,6673,6673,6674,6674,6675,6675,6675,6676,
            6676,6677,6677,6678,6678,6679,6679,6680,6680,
            6681,6681,6682,6682,6683,6683,6684,6684,6685,
            6685,6686,6686,6687,6687,6688,6688,6689,6689,
            6690,6690,6691,6691,6692,6692,6693,6693,6694,
            6694,6695,6695,6696,6696,6697,6697,6698,6698,
            6699,6699,6700,6700,6701,6701,6702,6702,6703,
            6703,6704,6704,6705,6705,6706,6706,6707,6707,
            6708,6708,6709,6709,6710,6710,6711,6711,6712,
            6712,6713,6713,6714,6714,6715,6715,6716,6716,
            6717,6717,6718,6718,6719,6719,6720,6720,6721,
            6721,6722,6722,6723,6723,6724,6724,6725,6725,
            6726,6726,6727,6727,6728,6728,6729,6729,6730,
            6730,6731,6731,6732,6732,6733,6733,6734,6734,
            6735,6735,6736,6736,6737,6737,6738,6738,6739,
            6739,6740,6740,6741,6741,6742,6742,6743,6743,
            6744,6744,6745,6745,6746,6746,6747,6747,6748,
            6748,6749,6749,6750,6750,6751,6751,6752,6752,
            6753,6753,6754,6754,6755,6755,6756,6756,6757,
            6757,6758,6758,6759,6759,6760,6760,6761,6761,
            6762,6762,6763,6763,6764,6764,6765,6765,6766,
            6766,6767,6767,6768,6768,6769,6769,6770,6770,
            6771,6771,6772,6772,6773,6773,6774,6774,6775,
            6775,6776,6776,6777,6777,6778,6778,6779,6779,
            6780,6780,6781,6781,6782,6782,6783,6783,6784,
            6784,6785,6785,6786,6786,6787,6787,6788,6788,
            6789,6789,6790,6790,6791,6791,6792,6792,6793,
            6793,6794,6794,6795,6795,6796,6796,6797,6797,
            6798,6798,6799,6799,6800,6800,6801,6801,6802,
            6802,6803,6803,6804,6804,6805,6805,6806,6806,
            6807,6807,6808,6808,6809,6809,6810,6810,6811,
            6811,6812,6812,6813,6813,6814,6814,6815,6815,
            6816,6816,6817,6817,6818,6818,6819,6819,6820,
            6820,6821,6822,6823,6823,6824,6824,6825,6825,
            6826,6826,6827,6827,6828,6828,6829,6829,6830,
            6830,6831,6831,6832,6832,6833,6833,6834,6834,
            6835,6835,6836,6836,6837,6837,6838,6838,6839,
            6839,6840,6840,6841,6841,6842,6842,6843,6843,
            6844,6844,6845,6845,6846,6846,6847,6847,6848,
            6848,6849,6849,6850,6850,6851,6851,6852,6852,
            6853,6853,6854,6854,6855,6855,6856,6856,6857,
            6857,6858,6858,6859,6859,6860,6860,6861,6861,
            6862,6862,6863,6863,6864,6864,6865,6865,6866,
            6866,6867,6867,6868,6868,6869,6869,6870,6870,
            6871,6871,6872,6872,6873,6873,6874,6874,6875,
            6875,6876,6876,6877,6877,6878,6878,6879,6879,
            6880,6880,6881,6881,6882,6882,6883,6883,6884,
            6884,6885,6885,6886,6886,6887,6887,6888,6888,
            6889,6889,6890,6890,6891,6891,6892,6892,6893,
            6893,6894,6894,6895,6895,6896,6896,6897,6897])
        j = np.array([
            1,1,35,1,2,3,4,4,5,
            5,5,30,6,7,7,8,8,9,
            14,10,11,11,12,13,28,4,27,
            7,17,8,14,11,32,16,37,15,
            23,14,22,18,25,20,23,17,44,
            17,114,21,26,21,33,24,32,25,
            28,27,48,23,31,25,29,35,97,
            27,30,28,72,31,62,30,42,31,
            94,32,40,34,46,33,37,47,72,
            35,141,38,68,41,48,42,54,39,
            44,40,55,37,68,40,47,42,87,
            49,54,50,62,44,60,46,52,47,
            96,57,58,48,87,49,59,51,119,
            56,63,54,79,61,63,60,74,60,
            114,78,97,98,65,82,67,71,63,
            78,66,128,66,94,68,73,73,82,
            70,80,81,73,109,75,91,71,85,
            72,89,76,84,78,120,83,95,80,
            81,92,81,85,82,93,86,90,85,
            88,90,93,87,113,91,92,90,100,
            100,105,95,112,91,103,92,107,99,
            109,93,121,94,111,96,139,97,98,
            132,98,120,100,122,106,112,107,118,
            108,138,107,117,109,158,112,126,115,
            125,113,123,113,116,114,125,117,126,
            119,161,117,143,119,161,120,145,121,
            153,121,129,122,129,122,130,125,136,
            126,159,127,130,129,133,130,134,132,
            141,132,137,135,138,142,164,138,142,
            139,151,144,146,141,147,145,152,142,
            172,143,159,143,146,145,154,149,157,
            152,155,153,178,153,156,157,158,157,
            162,158,163,159,160,171,160,180,163,
            165,165,169,161,196,197,163,173,164,
            168,166,174,167,171,164,176,165,177,
            169,182,170,183,169,181,171,186,175,
            184,172,184,178,182,174,195,174,179,
            176,200,181,194,181,191,182,188,183,
            185,184,187,186,192,186,189,195,226,
            190,196,193,194,198,233,194,213,195,
            227,196,202,197,203,201,206,205,206,
            206,208,209,217,212,219,213,214,215,
            232,216,234,218,219,217,269,217,245,
            220,222,219,228,221,230,233,292,223,
            255,224,231,225,239,222,239,222,232,
            228,250,226,227,226,242,227,242,229,
            237,231,236,228,254,233,248,230,238,
            230,241,234,235,232,274,237,238,234,
            241,235,243,240,256,246,251,237,258,
            238,252,239,253,241,252,244,253,242,
            250,245,266,246,248,247,264,245,269,
            246,251,249,263,248,262,251,255,252,
            294,253,263,256,281,257,268,254,279,
            259,272,256,291,260,265,261,303,258,
            267,262,292,262,267,263,272,264,268,
            264,265,270,281,265,278,267,294,272,
            289,274,285,273,282,273,277,274,286,
            276,282,280,287,284,372,281,307,282,
            288,285,286,285,289,286,314,287,290,
            289,301,291,293,299,292,339,293,297,
            295,301,296,313,298,308,296,309,299,
            304,300,322,299,302,303,308,301,320,
            304,307,305,328,322,331,303,361,306,
            319,304,310,307,311,308,367,312,316,
            309,323,309,324,313,319,320,378,313,
            318,317,323,318,327,321,397,316,341,
            316,330,317,326,324,327,318,319,325,
            320,357,322,332,323,333,324,334,325,
            407,330,331,327,401,328,360,352,374,
            330,337,331,366,332,338,335,351,336,
            341,332,345,339,342,339,354,351,358,
            343,352,343,348,345,366,345,390,350,
            376,356,382,353,362,349,396,351,379,
            352,374,359,360,362,369,363,371,364,
            376,357,377,357,405,358,392,365,379,
            372,417,360,384,361,412,361,367,368,
            375,369,375,362,396,370,371,373,374,
            366,380,367,413,369,403,387,400,371,
            381,388,397,372,382,380,441,375,394,
            395,381,387,376,400,401,378,405,378,
            388,379,415,380,462,385,394,386,391,
            381,391,382,508,384,439,389,398,387,
            414,388,451,391,399,392,402,392,415,
            394,425,395,406,396,403,398,427,400,
            401,414,404,407,402,451,408,446,405,
            435,409,418,410,418,411,428,406,466,
            407,430,416,417,419,428,412,423,414,
            537,421,431,415,438,417,442,418,426,
            427,443,422,432,426,439,424,454,430,
            431,425,476,425,429,426,437,427,492,
            428,432,433,445,430,464,436,442,432,
            448,440,543,438,456,438,456,441,470,
            441,488,444,448,442,495,443,492,445,
            494,445,474,447,472,464,467,446,517,
            446,473,450,452,448,462,453,471,457,
            474,452,460,454,472,454,497,458,479,
            459,473,455,465,455,469,456,461,465,
            468,460,489,460,463,462,488,471,494,
            467,528,464,515,465,475,466,476,466,
            550,467,507,471,512,472,527,473,598,
            599,474,490,489,534,477,494,476,580,
            479,517,479,503,483,495,484,496,485,
            499,490,500,491,497,488,521,489,534,
            490,523,492,510,493,544,495,505,498,
            511,496,511,497,527,501,509,499,519,
            502,544,504,505,508,509,506,556,505,
            525,508,541,510,568,510,568,511,543,
            512,522,512,564,523,564,518,533,520,
            545,515,540,517,529,522,526,523,570,
            525,555,525,541,530,542,534,560,527,
            535,531,550,532,546,546,547,528,551,
            528,538,529,643,545,549,562,575,536,
            553,537,551,538,539,533,542,533,552,
            535,582,535,557,538,558,541,572,542,
            618,543,552,548,569,545,611,546,553,
            547,590,549,556,550,562,551,563,552,
            673,553,576,556,569,561,573,562,567,
            564,581,566,571,567,571,567,578,568,
            587,573,577,569,577,571,583,575,579,
            573,593,575,608,576,590,577,585,577,
            584,578,597,578,608,579,617,581,596,
            581,615,585,632,584,602,584,594,586,
            600,588,601,585,602,589,667,592,600,
            587,623,587,610,595,604,596,604,590,
            657,594,613,594,644,603,604,596,621,
            597,606,597,607,598,629,599,600,601,
            610,601,618,602,613,609,625,608,635,
            610,631,611,634,611,619,619,620,624,
            612,620,612,614,616,647,617,627,613,
            626,621,645,617,628,622,701,618,717,
            632,633,619,639,620,624,641,621,651,
            625,630,624,640,627,635,629,643,625,
            658,626,638,634,644,629,695,637,669,
            632,654,633,660,634,635,668,642,649,
            645,646,643,672,648,665,649,650,644,
            675,645,651,647,667,647,748,652,695,
            653,691,649,666,662,663,651,664,656,
            668,659,716,654,674,654,660,661,658,
            701,658,750,663,739,665,669,665,732,
            667,723,668,705,669,737,673,685,674,
            685,675,690,676,694,677,709,678,679,
            681,736,673,717,674,689,675,821,686,
            689,683,699,687,700,688,700,685,738,
            692,721,689,698,693,703,694,704,691,
            726,691,701,696,702,703,707,694,714,
            695,710,705,719,700,726,706,720,716,
            727,702,728,703,801,708,756,705,718,
            711,764,720,736,712,727,713,722,709,
            723,710,749,738,903,722,739,714,744,
            716,754,717,721,720,730,722,827,723,
            837,726,758,727,754,728,765,741,785,
            730,759,740,747,734,756,732,760,732,
            752,735,809,735,758,742,759,736,745,
            737,745,737,760,738,784,739,827,740,
            755,741,837,741,802,746,752,753,754,
            749,829,750,791,751,781,745,793,755,
            761,747,761,748,777,748,778,757,771,
            749,775,750,764,752,776,755,769,756,
            836,763,790,759,774,760,794,765,783,
            761,828,766,779,768,772,763,809,764,
            773,765,788,801,836,770,789,768,797,
            773,781,774,799,783,784,779,789,771,
            780,771,786,772,803,773,907,774,787,
            777,785,777,778,778,864,779,822,781,
            830,793,832,783,881,784,795,785,802,
            786,891,786,803,797,803,790,791,792,
            815,811,812,787,812,799,806,796,808,
            789,822,791,804,793,806,800,825,797,
            898,798,809,798,810,799,813,814,807,
            820,801,828,802,838,802,883,806,852,
            808,905,808,944,810,876,810,850,819,
            832,821,903,829,849,815,830,815,833,
            823,834,816,834,820,825,820,862,826,
            881,821,846,822,840,852,936,825,869,
            856,915,827,857,828,859,829,957,830,
            867,832,853,835,856,838,863,839,840,
            841,874,843,899,834,861,855,869,845,
            862,847,904,836,940,837,863,840,865,
            851,877,853,854,858,882,848,923,848,
            883,865,897,860,875,849,875,852,888,
            853,902,856,889,859,882,859,1009,864,
            883,866,899,867,877,861,879,870,962,
            871,967,864,933,865,896,873,884,867,
            972,879,887,869,937,872,911,872,932,
            875,957,876,892,876,912,886,906,877,
            919,879,908,881,952,889,916,882,910,
            885,891,887,901,915,952,889,1007,895,
            910,896,924,890,926,891,898,898,948,
            904,905,896,1004,899,993,901,920,902,
            928,902,909,903,964,904,939,905,944,
            907,918,907,982,908,920,908,973,916,
            921,910,930,988,1034,912,927,917,949,
            916,943,922,940,924,925,919,983,920,
            961,931,945,923,941,923,947,924,1014,
            935,948,928,976,928,951,938,939,945,
            946,933,969,936,974,936,975,937,977,
            937,1115,939,965,940,953,950,982,944,
            1141,953,954,945,1003,948,955,949,960,
            954,966,967,968,956,992,957,992,958,
            981,962,1060,952,1006,963,986,979,1008,
            953,1043,954,966,980,968,1003,969,970,
            960,995,972,983,961,1005,961,984,975,
            1031,962,1165,978,1007,990,966,1001,967,
            1011,968,1011,991,1012,969,991,972,1022,
            973,984,973,996,974,996,974,997,975,
            997,976,1031,976,1059,985,1016,989,990,
            981,1020,994,995,982,1022,983,1005,984,
            1071,986,1016,998,1042,999,1008,1018,1008,
            1032,988,1009,988,1033,1000,1001,1002,1026,
            991,1044,992,1164,993,1081,993,1020,995,
            1029,996,1015,997,1094,998,1074,999,1062,
            1000,1024,1001,1064,1010,1026,1003,1078,1013,
            1045,1004,1045,1004,1046,1108,1127,1005,1070,
            1015,1093,1006,1060,1006,1040,1017,1042,1009,
            1076,1011,1036,1014,1068,1014,1047,1019,1090,
            1021,1039,1022,1039,1023,1094,1040,1053,1016,
            1041,1017,1062,1018,1054,1034,1097,1025,1055,
            1020,1048,1027,1037,1029,1051,1024,1043,1026,
            1055,1036,1114,1048,1092,1029,1052,1030,1119,
            1030,1084,1031,1095,1041,1053,1034,1097,1036,
            1106,1037,1103,1039,1057,1040,1170,1041,1061,
            1061,1074,1042,1075,1054,1138,1043,1063,1045,
            1065,1046,1067,1046,1068,1048,1151,1051,1086,
            1070,1071,1058,1095,1053,1072,1054,1062,1063,
            1077,1055,1116,1065,1066,1068,1079,1056,1102,
            1069,1118,1060,1153,1075,1105,1063,1180,1065,
            1143,1080,1081,1082,1086,1070,1111,1083,1093,
            1084,1140,1073,1087,1073,1096,1074,1105,1075,
            1124,1078,1098,1078,1099,1089,1090,1081,1151,
            1083,1111,1084,1122,1098,1106,1091,1109,1092,
            1102,1086,1118,1096,1113,1090,1108,1092,1182,
            1093,1112,1095,1104,1096,1121,1097,1126,1098,
            1117,1109,1128,1103,1110,1103,1129,1104,1123,
            1105,1135,1106,1142,1107,1146,1108,1144,1109,
            1145,1110,1139,1111,1132,1114,1149,1117,1146,
            1115,1134,1115,1152,1116,1181,1117,1142,1118,
            1129,1125,1126,1127,1128,1123,1133,1124,1136,
            1126,1180,1127,1148,1128,1145,1137,1138,1131,
            1189,1138,1178,1144,1147,1147,1148,1139,1182,
            1142,1154,1146,1184,1144,1173,1145,1157,1149,
            1154,1150,1161,1147,1156,1148,1163,1149,1188,
            1151,1158,1152,1174,1152,1165,1153,1166,1153,
            1159,1154,1285,1155,1161,1162,1156,1163,1156,
            1173,1159,1169,1160,1170,1159,1166,1161,1176,
            1162,1167,1163,1171,1166,1193,1170,1183,1171,
            1214,1173,1293,1174,1175,1176,1184,1176,1179,
            1180,1292,1181,1187,1181,1186,1186,1188,1182,
            1208,1185,1198,1183,1194,1183,1190,1184,1204,
            1186,1197,1189,1234,1188,1207,1189,1230,1193,
            1198,1193,1224,1194,1212,1194,1202,1195,1262,
            1195,1196,1196,1233,1200,1214,1198,1211,1203,
            1207,1203,1213,1204,1285,1209,1223,1206,1210,
            1207,1216,1211,1218,1212,1220,1212,1221,1214,
            1240,1217,1336,1217,1226,1223,1228,1219,1228,
            1220,1224,1220,1227,1223,1274,1224,1235,1226,
            1277,1227,1248,1228,1331,1229,1233,1231,1239,
            1232,1250,1230,1234,1230,1260,1234,1237,1236,
            1256,1236,1245,1238,1247,1239,1243,1240,1252,
            1241,1246,1242,1253,1244,1250,1242,1254,1246,
            1259,1247,1289,1247,1253,1249,1278,1256,1269,
            1252,1271,1250,1264,1255,1261,1258,1263,1252,
            1272,1253,1290,1256,1276,1258,1270,1264,1265,
            1262,1275,1263,1270,1270,1281,1271,1293,1264,
            1278,1268,1280,1274,1280,1269,1291,1271,1282,
            1274,1331,1277,1336,1277,1288,1278,1304,1280,
            1284,1281,1298,1286,1289,1288,1307,1287,1311,
            1285,1333,1288,1313,1289,1294,1290,1314,1290,
            1295,1292,1316,1292,1337,1293,1300,1298,1319,
            1294,1301,1294,1323,1296,1303,1297,1315,1303,
            1304,1299,1305,1300,1322,1302,1318,1308,1311,
            1299,1319,1306,1312,1300,1335,1303,1325,1304,
            1315,1308,1316,1307,1313,1310,1343,1308,1327,
            1312,1320,1314,1323,1311,1327,1320,1328,1312,
            1321,1313,1329,1314,1324,1315,1347,1316,1341,
            1318,1345,1326,1346,1320,1321,1321,1344,1322,
            1334,1325,1372,1327,1340,1328,1358,1332,1333,
            1331,1375,1343,1353,1333,1359,1335,1342,1336,
            1350,1339,1353,1337,1361,1337,1367,1341,1360,
            1341,1354,1345,1351,1347,1348,1346,1356,1356,
            1413,1343,1373,1345,1352,1346,1349,1347,1385,
            1350,1368,1350,1363,1354,1361,1355,1422,1353,
            1386,1357,1360,1354,1365,1358,1402,1356,1413,
            1358,1402,1362,1376,1360,1364,1361,1366,1363,
            1371,1364,1365,1363,1371,1364,1395,1365,1369,
            1372,1385,1375,1378,1373,1376,1374,1380,1372,
            1380,1373,1387,1377,1381,1375,1393,1376,1394,
            1381,1382,1380,1384,1381,1388,1382,1390,1386,
            1387,1385,1418,1386,1427,1387,1400,1388,1406,
            1389,1392,1390,1406,1391,1407,1393,1398,1396,
            1415,1390,1407,1392,1403,1392,1424,1399,1412,
            1394,1400,1394,1410,1401,1410,1404,1425,1405,
            1409,1402,1419,1407,1408,1411,1423,1405,1414,
            1406,1415,1407,1417,1412,1416,1410,1452,1412,
            1428,1413,1434,1415,1417,1418,1420,1417,1429,
            1418,1437,1421,1430,1422,1449,1423,1431,1422,
            1473,1423,1433,1426,1437,1440,1428,1432,1429,
            1490,1424,1463,1424,1425,1425,1455,1429,1490,
            1430,1445,1431,1451,1435,1439,1433,1445,1434,
            1446,1434,1448,1436,1439,1438,1444,1441,1444,
            1447,1442,1449,1439,1487,1440,1496,1440,1493,
            1443,1456,1445,1450,1447,1460,1447,1450,1449,
            1465,1450,1460,1452,1454,1451,1461,1451,1480,
            1452,1477,1454,1468,1457,1475,1458,1461,1454,
            1462,1459,1504,1455,1463,1455,1470,1456,1515,
            1456,1475,1458,1464,1460,1476,1461,1479,1467,
            1472,1466,1481,1468,1469,1463,1491,1471,1476,
            1465,1473,1467,1503,1468,1482,1474,1492,1473,
            1481,1477,1553,1478,1489,1475,1529,1476,1516,
            1479,1480,1477,1518,1483,1489,1484,1502,1479,
            1524,1480,1497,1485,1521,1481,1494,1486,1505,
            1487,1492,1488,1496,1487,1550,1489,1537,1490,
            1526,1492,1571,1493,1514,1493,1502,1498,1511,
            1499,1526,1500,1501,1496,1514,1501,1502,1507,
            1503,1525,1503,1534,1506,1504,1519,1504,1554,
            1506,1527,1508,1521,1509,1538,1510,1517,1505,
            1517,1505,1511,1513,1547,1506,1520,1511,1604,
            1518,1535,1520,1528,1514,1541,1515,1523,1515,
            1529,1516,1558,1516,1524,1517,1530,1518,1553,
            1519,1537,1520,1527,1521,1533,1522,1574,1525,
            1538,1528,1532,1524,1533,1525,1583,1526,1547,
            1527,1540,1529,1552,1535,1539,1536,1546,1533,
            1566,1535,1542,1537,1556,1538,1566,1540,1562,
            1540,1586,1543,1549,1544,1568,1551,1565,1546,
            1568,1554,1569,1547,1562,1549,1555,1549,1550,
            1550,1571,1553,1567,1554,1575,1555,1563,1557,
            1559,1559,1560,1556,1561,1556,1582,1558,1630,
            1558,1577,1559,1595,1575,1584,1562,1585,1565,
            1578,1573,1605,1568,1595,1569,1576,1571,1596,
            1572,1581,1572,1583,1574,1620,1574,1610,1575,
            1584,1576,1582,1576,1584,1586,1611,1578,1612,
            1579,1593,1580,1599,1581,1662,1578,1597,1581,
            1602,1604,1689,1582,1609,1588,1601,1589,1593,
            1590,1602,1585,1610,1585,1586,1591,1590,1663,
            1594,1635,1591,1606,1596,1649,1592,1601,1593,
            1626,1599,1600,1595,1605,1596,1612,1599,1627,
            1603,1613,1601,1602,1652,1607,1613,1604,1647,
            1608,1692,1625,1638,1616,1626,1627,1608,1635,
            1609,1622,1609,1628,1610,1624,1611,1621,1612,
            1667,1616,1617,1613,1643,1637,1644,1618,1632,
            1619,1620,1616,1625,1630,1669,1623,1632,1621,
            1629,1634,1637,1624,1708,1625,1674,1626,1651,
            1627,1749,1631,1643,1630,1669,1633,1634,1632,
            1666,1636,1638,1634,1640,1635,1641,1639,1647,
            1637,1640,1638,1674,1640,1687,1642,1648,1650,
            1651,1653,1676,1641,1660,1641,1698,1643,1645,
            1653,1646,1664,1644,1691,1644,1660,1648,1655,
            1652,1663,1645,1654,1647,1664,1648,1666,1649,
            1667,1651,1713,1653,1704,1658,1678,1661,1673,
            1665,1677,1660,1670,1662,1668,1688,1663,1676,
            1664,1717,1666,1702,1667,1694,1682,1688,1668,
            1731,1670,1691,1671,1692,1675,1681,1669,1681,
            1678,1686,1673,1680,1673,1724,1674,1755,1677,
            1689,1677,1684,1678,1685,1679,1701,1679,1697,
            1680,1699,1681,1714,1686,1700,1690,1721,1688,
            1744,1689,1717,1690,1707,1692,1698,1693,1701,
            1702,1759,1697,1701,1697,1706,1698,1703,1699,
            1729,1703,1706,1702,1768,1704,1736,1705,1709,
            1703,1722,1725,1749,1704,1716,1706,1732,1707,
            1709,1707,1721,1708,1723,1708,1718,1710,1759,
            1711,1724,1712,1733,1713,1725,1714,1715,1716,
            1738,1709,1720,1718,1742,1713,1725,1714,1726,
            1717,1727,1718,1728,1719,1735,1719,1739,1721,
            1745,1723,1794,1723,1765,1724,1785,1726,1734,
            1726,1730,1731,1738,1727,1752,1727,1740,1729,
            1753,1729,1771,1731,1747,1732,1758,1732,1774,
            1733,1796,1733,1743,1741,1770,1738,1751,1747,
            1757,1745,1772,1745,1758,1748,1753,1749,1756,
            1754,1780,1760,1768,1753,1771,1755,1766,1757,
            1762,1757,1767,1758,1773,1759,1770,1763,1776,
            1764,1776,1792,1792,1793,1765,1798,1765,1850,
            1772,1773,1769,1774,1768,1810,1775,1784,1770,
            1789,1771,1839,1777,1786,1774,1783,1776,1801,
            1780,1785,1780,1795,1783,1788,1784,1789,1785,
            1816,1786,1799,1786,1802,1806,1861,1790,1799,
            1791,1807,1788,1797,1789,1820,1796,1802,1792,
            1793,1827,1793,1809,1794,1811,1796,1806,1800,
            1807,1798,1865,1799,1815,1803,1813,1801,1810,
            1801,1823,1805,1815,1802,1899,1806,1861,1812,
            1866,1807,1847,1809,1849,1810,1872,1813,1822,
            1814,1822,1815,1845,1817,1859,1818,1837,1819,
            1839,1816,1920,1816,1840,1820,1834,1820,1827,
            1822,1833,1825,1835,1826,1848,1829,1838,1827,
            1838,1830,1865,1833,1856,1831,1835,1833,1868,
            1835,1875,1836,1842,1837,1856,1837,1838,1839,
            1850,1849,1857,1845,1898,1847,1908,1847,1894,
            1852,1864,1849,1882,1858,1862,1859,1896,1860,
            1864,1856,1889,1857,1918,1859,1888,1861,1869,
            1862,1863,1862,1881,1864,1867,1865,1943,1866,
            1907,1866,1953,1867,1876,1867,1871,1868,1877,
            1868,1889,1870,1873,1869,1891,1869,1880,1881,
            1966,1871,1884,1877,1884,1872,1879,1873,1874,
            1920,1944,1871,1883,1872,1878,1873,1879,1943,
            1949,1880,1886,1875,1886,1875,1970,1877,1954,
            1878,1919,1878,1879,1880,1892,1881,1966,1888,
            1909,1887,1901,1884,1942,1890,1911,1886,1970,
            1893,1907,1895,1896,1896,1932,1889,1897,1923,
            1898,1899,1891,1904,1894,1908,1894,1909,1895,
            1991,1897,1917,1898,1934,1899,1910,1901,1941,
            1905,1912,1918,1919,1904,1936,1905,1941,1913,
            1927,1914,1927,1906,1928,1916,1921,1907,1953,
            1908,1963,1909,1991,1922,1932,1922,1933,1910,
            1935,1910,1911,1926,1920,1961,1931,1958,1922,
            1960,1950,1982,1937,1951,1938,1951,1927,1928,
            2042,1930,1967,1939,1952,1941,1950,1945,1976,
            1946,2026,1942,1947,1942,1954,1943,1948,1962,
            1944,1964,1944,1973,1955,1979,1948,1985,1949,
            1985,1949,1964,1950,2025,1951,1974,1956,1967,
            1952,1958,1953,1968,1959,1963,1954,1978,1958,
            1975,1961,1969,1961,1965,1963,1977,1964,1972,
            1966,2001,1967,1983,1968,1975,1968,1976,1971,
            1979,1970,1982,1974,2048,1972,1973,1995,1973,
            1980,1974,2132,1975,2055,1976,2008,1978,2010,
            1981,1997,1983,2002,1979,2014,1986,1998,1987,
            1998,1982,2041,1983,2011,1985,2015,1989,1997,
            1990,2002,2012,2013,1992,2026,1991,2021,1993,
            2022,1995,2003,1997,2033,1998,2037,2001,2064,
            2001,2094,2002,2043,2005,2009,2006,2056,2007,
            2059,2007,2080,2008,2012,2009,2013,2008,2055,
            2009,2017,2016,2028,2011,2054,2011,2070,2012,
            2013,2014,2023,2014,2056,2018,2027,2019,2029,
            2020,2025,2030,2032,2029,2038,2022,2098,2022,
            2082,2083,2031,2037,2025,2119,2026,2035,2027,
            2045,2028,2046,2029,2075,2033,2163,2031,2073,
            2050,2101,2032,2035,2032,2036,2033,2045,2038,
            2039,2040,2041,2043,2065,2066,2117,2035,2066,
            2044,2057,2037,2050,2038,2075,2041,2059,2048,
            2053,2042,2053,2042,2054,2043,2097,2049,2113,
            2045,2100,2051,2063,2050,2074,2053,2147,2054,
            2070,2061,2072,2056,2106,2057,2107,2063,2064,
            2060,2122,2060,2134,2079,2087,2063,2078,2064,
            2094,2066,2142,2071,2098,2073,2086,2074,2086,
            2076,2120,2077,2120,2070,2172,2083,2096,2072,
            2096,2073,2085,2074,2157,2075,2102,2081,2139,
            2084,2099,2080,2115,2089,2095,2082,2123,2083,
            2110,2091,2099,2092,2100,2086,2114,2093,2126,
            2094,2104,2104,2127,2095,2116,2096,2105,2101,
            2102,2103,2124,2097,2172,2097,2140,2098,2123,
            2099,2152,2106,2107,2113,2118,2100,2183,2102,
            2158,2104,2178,2109,2130,2105,2112,2106,2168,
            2107,2118,2115,2119,2116,2125,2113,2131,2115,
            2194,2117,2141,2117,2149,2118,2189,2119,2153,
            2137,2143,2120,2144,2125,2127,2123,2130,2131,
            2135,2124,2137,2124,2138,2125,2145,2128,2179,
            2129,2162,2127,2178,2133,2134,2130,2162,2131,
            2185,2132,2146,2132,2147,2134,2148,2149,2150,
            2142,2156,2157,2174,2137,2155,2138,2238,2138,
            2139,2139,2144,2146,2161,2151,2167,2143,2153,
            2159,2144,2210,2154,2161,2146,2147,2149,2180,
            2150,2166,2152,2177,2152,2208,2155,2159,2157,
            2169,2158,2181,2158,2181,2161,2165,2170,2162,
            2184,2167,2177,2168,2173,2163,2186,2163,2183,
            2179,2207,2167,2175,2168,2215,2173,2189,2172,
            2200,2173,2216,2174,2176,2174,2187,2177,2208,
            2178,2252,2179,2213,2181,2206,2183,2217,2184,
            2196,2185,2190,2205,2187,2199,2189,2235,2190,
            2248,2197,2198,2192,2199,2192,2206,2194,2203,
            2197,2205,2201,2207,2207,2211,2205,2221,2206,
            2220,2208,2214,2209,2214,2210,2218,2211,2213,
            2210,2225,2212,2219,2211,2253,2215,2229,2218,
            2219,2213,2236,2214,2242,2215,2231,2220,2237,
            2218,2225,2219,2222,2220,2223,2223,2241,2224,
            2272,2226,2287,2227,2231,2228,2229,2234,2235,
            2223,2251,2225,2295,2230,2236,2233,2243,2231,
            2232,2232,2243,2235,2239,2236,2246,2240,2255,
            2238,2259,2238,2319,2239,2268,2239,2248,2245,
            2279,2242,2261,2242,2311,2249,2255,2248,2283,
            2251,2258,2251,2256,2291,2308,2254,2270,2257,
            2265,2255,2271,2260,2309,2263,2297,2259,2278,
            2259,2274,2266,2306,2267,2282,2276,2315,2269,
            2280,2272,2304,2265,2300,2275,2286,2270,2280,
            2272,2294,2279,2295,2282,2316,2279,2285,2281,
            2285,2286,2287,2288,2311,2280,2296,2284,2317,
            2306,2341,2282,2310,2283,2292,2283,2293,2289,
            2321,2285,2301,2291,2320,2294,2304,2295,2397,
            2302,2327,2296,2315,2296,2352,2303,2317,2300,
            2318,2300,2322,2301,2305,2301,2307,2303,2335,
            2304,2318,2308,2313,2306,2341,2312,2322,2308,
            2320,2309,2344,2310,2316,2311,2328,2314,2364,
            2315,2352,2319,2397,2316,2323,2317,2329,2318,
            2342,2319,2326,2320,2337,2323,2324,2321,2325,
            2321,2353,2322,2343,2327,2348,2323,2350,2329,
            2373,2327,2330,2329,2494,2331,2339,2333,2360,
            2334,2356,2346,2353,2337,2347,2338,2363,2339,
            2340,2345,2340,2358,2342,2371,2343,2359,2341,
            2355,2344,2374,2342,2343,2348,2349,2350,2351,
            2348,2356,2349,2384,2350,2368,2354,2379,2352,
            2373,2357,2358,2353,2377,2359,2365,2361,2366,
            2356,2367,2358,2369,2362,2383,2359,2378,2364,
            2450,2364,2449,2368,2374,2368,2375,2375,2381,
            2369,2389,2372,2387,2376,2414,2373,2415,2391,
            2456,2374,2380,2384,2375,2386,2382,2389,2377,
            2383,2377,2432,2379,2390,2379,2391,2383,2431,
            2394,2396,2388,2396,2384,2399,2385,2403,2385,
            2411,2392,2393,2389,2431,2391,2456,2395,2407,
            2396,2408,2398,2406,2407,2414,2448,2397,2404,
            2399,2400,2401,2441,2402,2408,2403,2410,2404,
            2463,2405,2427,2402,2436,2409,2413,2403,2413,
            2404,2463,2412,2423,2406,2418,2406,2407,2419,
            2408,2420,2415,2454,2416,2439,2417,2455,2419,
            2429,2424,2429,2421,2438,2414,2468,2415,2484,
            2422,2439,2419,2446,2421,2437,2426,2451,2423,
            2427,2423,2434,2428,2445,2425,2436,2433,2443,
            2427,2489,2429,2457,2436,2437,2438,2459,2431,
            2441,2432,2486,2432,2472,2440,2443,2435,2465,
            2435,2458,2437,2453,2438,2447,2439,2485,2442,
            2462,2448,2449,2450,2461,2441,2485,2443,2451,
            2445,2452,2445,2446,2446,2464,2448,2491,2449,
            2499,2450,2482,2455,2477,2451,2474,2452,2489,
            2457,2465,2453,2470,2453,2466,2460,2471,2456,
            2478,2463,2487,2474,2508,2459,2467,2459,2471,
            2468,2491,2462,2516,2469,2477,2464,2481,2465,
            2475,2468,2490,2473,2478,2471,2479,2476,2495,
            2474,2501,2486,2554,2477,2511,2478,2522,2483,
            2496,2488,2492,2484,2521,2482,2499,2482,2496,
            2483,2488,2484,2493,2485,2554,2486,2500,2488,
            2497,2489,2550,2491,2509,2493,2494,2546,2495,
            2498,2495,2527,2496,2514,2498,2547,2500,2504,
            2504,2505,2499,2510,2502,2514,2503,2520,2500,
            2507,2501,2524,2501,2508,2506,2508,2504,2515,
            2511,2528,2509,2534,2512,2531,2532,2509,2513,
            2516,2528,2518,2532,2516,2543,2522,2549,2524,
            2555,2520,2563,2528,2529,2522,2536,2524,2562,
            2550,2606,2525,2540,2530,2555,2532,2578,2542,
            2543,2550,2619,2540,2568,2544,2558,2545,2558,
            2543,2565,2551,2594,2549,2559,2549,2560,2556,
            2574,2557,2571,2554,2576,2565,2579,2561,2569,
            2555,2566,2558,2568,2563,2590,2564,2575,2562,
            2566,2562,2601,2567,2582,2563,2605,2565,2587,
            2566,2598,2570,2577,2571,2594,2568,2607,2569,
            2616,2573,2618,2571,2574,2574,2584,2575,2576,
            2602,2576,2585,2580,2620,2581,2597,2577,2588,
            2578,2589,2578,2584,2583,2595,2582,2601,2584,
            2629,2590,2595,2593,2598,2590,2611,2596,2602,
            2591,2603,2597,2616,2594,2629,2599,2630,2595,
            2612,2597,2613,2600,2613,2598,2617,2604,2617,
            2601,2722,2609,2610,2602,2638,2606,2621,2607,
            2683,2608,2651,2605,2610,2605,2644,2614,2627,
            2606,2618,2607,2653,2610,2623,2611,2644,2611,
            2624,2615,2632,2613,2636,2620,2669,2616,2647,
            2617,2722,2618,2637,2619,2621,2619,2628,2622,
            2631,2625,2633,2620,2704,2621,2637,2632,2638,
            2635,2647,2627,2643,2629,2690,2630,2648,2630,
            2631,2631,2662,2632,2645,2633,2646,2636,2709,
            2637,2658,2640,2653,2641,2648,2645,2661,2644,
            2716,2645,2649,2646,2672,2646,2652,2647,2695,
            2648,2659,2651,2660,2650,2655,2651,2721,2654,
            2660,2655,2661,2656,2702,2657,2672,2653,2696,
            2659,2662,2655,2674,2658,2670,2658,2667,2659,
            2691,2660,2679,2664,2675,2665,2675,2666,2688,
            2662,2677,2668,2674,2669,2686,2671,2707,2669,
            2704,2673,2688,2677,2684,2674,2707,2675,2702,
            2680,2698,2676,2703,2676,2681,2683,2691,2677,
            2699,2679,2721,2679,2694,2687,2796,2689,2720,
            2720,2728,2683,2696,2684,2692,2697,2700,2708,
            2726,2686,2705,2688,2718,2691,2711,2692,2700,
            2698,2703,2695,2798,2695,2709,2696,2710,2698,
            2815,2700,2712,2706,2742,2703,2779,2704,2757,
            2717,2718,2710,2760,2707,2726,2708,2793,2709,
            2830,2714,2723,2715,2724,2710,2755,2711,2789,
            2711,2747,2719,2731,2716,2762,2716,2832,2717,
            2763,2718,2751,2720,2733,2724,2729,2721,2756,
            2722,2744,2723,2727,2723,2738,2724,2735,2730,
            2822,2726,2742,2731,2809,2728,2764,2728,2739,
            2739,2745,2734,2745,2729,2765,2732,2818,2731,
            2781,2736,2759,2737,2748,2733,2764,2735,2746,
            2749,2777,2743,2813,2744,2753,2738,2771,2739,
            2773,2747,2766,2748,2762,2742,2783,2744,2770,
            2745,2758,2747,2789,2748,2761,2751,2763,2758,
            2775,2756,2767,2756,2776,2757,2769,2758,2774,
            2759,2765,2759,2837,2762,2832,2768,2867,2763,
            2807,2770,2780,2772,2773,2765,2791,2777,2782,
            2777,2783,2779,2794,2770,2786,2771,2781,2771,
            2810,2772,2810,2773,2831,2779,2815,2785,2805,
            2781,2816,2788,2792,2783,2823,2784,2824,2825,
            2784,2804,2790,2791,2789,2797,2793,2828,2795,
            2805,2791,2858,2792,2811,2797,2801,2793,2813,
            2796,2870,2796,2798,2799,2806,2800,2811,2797,
            2812,2802,2833,2803,2823,2807,2826,2808,2827,
            2809,2816,2805,2829,2806,2845,2807,2871,2809,
            2827,2810,2831,2814,2837,2811,2838,2817,2847,
            2813,2824,2815,2842,2818,2896,2819,2835,2816,
            2821,2818,2875,2820,2829,2822,2891,2838,2847,
            2822,2920,2823,2833,2825,2894,2827,2835,2828,
            2848,2829,2843,2831,2846,2832,2856,2833,2840,
            2834,2845,2836,2879,2835,2880,2839,2840,2887,
            2841,2842,2843,2849,2846,2850,2850,2851,2837,
            2855,2838,2852,2840,2882,2842,2854,2843,2897,
            2845,2902,2846,2864,2852,2855,2847,2904,2853,
            2848,2940,2848,2868,2849,2857,2850,2864,2852,
            2932,2858,2860,2855,2931,2861,2933,2858,2931,
            2862,2933,2867,2893,2863,2895,2865,2873,2873,
            2926,2866,2905,2869,2895,2871,2878,2874,2893,
            2867,2946,2888,2909,2870,2898,2870,2883,2879,
            2930,2881,2899,2873,2925,2882,2906,2875,2914,
            2875,2896,2888,2898,2883,2955,2879,2930,2886,
            2919,2882,2947,2897,2948,2883,2941,2884,2902,
            2884,2889,2890,2919,2891,2912,2900,2950,2892,
            2913,2894,2908,2888,2916,2889,2903,2891,2921,
            2893,2913,2894,2907,2895,2908,2901,2914,2896,
            2923,2897,3004,2898,2917,2899,2942,2899,2911,
            2906,2927,2902,2976,2904,2932,2904,2944,2912,
            2920,2905,2926,2905,2922,2906,2951,2908,2928,
            2915,2935,2918,2930,2936,2912,2938,2921,2925,
            2913,2950,2914,2929,2919,2942,2949,2965,2920,
            2945,2921,2938,2925,2939,2926,2939,2934,2957,
            2927,2951,2927,2940,2935,2948,2931,2953,2932,
            3001,2937,2996,2933,2949,2935,2943,2936,2961,
            2938,3002,2939,2992,2940,2978,2942,2956,2946,
            2981,2946,2972,2948,3012,2959,2961,2952,2959,
            2949,2964,2950,2957,2951,2954,2956,3029,2955,
            2962,2955,2967,2956,3019,2966,3031,2957,2966,
            2958,2962,2960,2965,2959,2971,2959,2968,2961,
            2998,2963,2968,2962,2975,2967,2976,2964,2989,
            2964,2986,2965,2986,2969,2981,2970,2988,2968,
            2985,2969,3041,2973,2977,2970,2994,2971,3007,
            2971,2985,2989,2996,2972,2990,2979,2987,2976,
            2995,2978,3066,2978,2993,2980,3014,2981,3011,
            2985,3018,2986,2997,2991,2993,2987,3009,2989,
            3062,2992,3002,2992,3030,2993,3080,2995,3006,
            2995,3112,2998,3007,2999,3028,3001,3008,2996,
            3022,3003,3011,3005,3015,3006,3010,2998,3040,
            3001,3016,3002,3052,3012,3013,3004,3013,3004,
            3014,3006,3027,3007,3018,3008,3020,3009,3043,
            3009,3026,3011,3056,3017,3032,3012,3026,3013,
            3074,3014,3015,3015,3074,3019,3029,3021,3077,
            3023,3038,3025,3033,3018,3045,3019,3051,3022,
            3077,3022,3062,3030,3031,3036,3041,3033,3043,
            3026,3120,3027,3035,3028,3131,3028,3076,3029,
            3051,3093,3114,3030,3115,3031,3036,3037,3071,
            3038,3071,3032,3046,3039,3048,3033,3042,3045,
            3143,3036,3145,3038,3085,3040,3044,3040,3082,
            3041,3117,3046,3081,3047,3059,3043,3120,3050,
            3126,3045,3082,3046,3058,3055,3059,3048,3073,
            3051,3068,3052,3098,3052,3065,3060,3092,3057,
            3079,3063,3100,3059,3092,3062,3070,3072,3108,
            3073,3135,3068,3076,3064,3097,3088,3097,3066,
            3079,3066,3083,3075,3087,3068,3093,3078,3098,
            3084,3089,3080,3084,3071,3101,3081,3086,3072,
            3111,3073,3105,3074,3142,3082,3087,3076,3113,
            3077,3088,3079,3095,3083,3089,3080,3099,3081,
            3091,3083,3095,3085,3179,3085,3086,3119,3087,
            3153,3094,3097,3088,3128,3089,3124,3096,3107,
            3092,3105,3093,3102,3095,3118,3102,3113,3097,
            3246,3099,3124,3099,3187,3104,3163,3100,3110,
            3100,3101,3101,3210,3107,3110,3105,3108,3108,
            3111,3114,3176,3116,3129,3117,3123,3110,3140,
            3111,3134,3125,3135,3112,3181,3112,3126,3121,
            3131,3114,3144,3115,3171,3115,3145,3117,3129,
            3120,3141,3127,3148,3121,3162,3124,3133,3130,
            3134,3136,3141,3126,3213,3128,3277,3129,3132,
            3137,3142,3138,3161,3139,3170,3131,3159,3134,
            3249,3135,3156,3146,3151,3142,3167,3143,3241,
            3143,3148,3145,3178,3147,3173,3147,3174,3150,
            3154,3148,3175,3151,3160,3153,3157,3154,3155,
            3156,3240,3167,3243,3153,3204,3154,3168,3163,
            3164,3156,3216,3157,3207,3157,3255,3161,3170,
            3159,3185,3159,3162,3162,3176,3165,3166,3161,
            3198,3163,3191,3173,3174,3167,3243,3170,3194,
            3177,3182,3172,3230,3173,3197,3175,3189,3176,
            3199,3187,3206,3179,3210,3179,3195,3180,3188,
            3181,3184,3181,3213,3185,3194,3182,3246,3185,
            3208,3187,3219,3191,3206,3195,3221,3191,3236,
            3196,3197,3194,3269,3195,3215,3197,3211,3199,
            3202,3203,3219,3205,3225,3209,3218,3206,3312,
            3212,3251,3218,3224,3216,3234,3216,3250,3217,
            3252,3217,3261,3220,3237,3221,3233,3218,3222,
            3223,3224,3225,3219,3235,3227,3241,3224,3244,
            3229,3235,3230,3231,3226,3239,3226,3253,3234,
            3240,3229,3259,3237,3238,3234,3250,3235,3247,
            3236,3262,3237,3271,3238,3253,3242,3249,3240,
            3290,3241,3256,3243,3265,3248,3257,3245,3258,
            3251,3252,3246,3301,3247,3260,3247,3288,3249,
            3272,3250,3254,3251,3293,3252,3280,3253,3302,
            3255,3267,3255,3256,3256,3303,3257,3258,3263,
            3258,3272,3261,3274,3264,3281,3266,3273,3261,
            3280,3265,3282,3265,3273,3269,3275,3271,3319,
            3272,3278,3273,3275,3307,3276,3307,3277,3298,
            3279,3275,3284,3277,3296,3279,3300,3283,3284,
            3278,3289,3278,3321,3279,3292,3280,3310,3290,
            3291,3281,3291,3281,3299,3284,3344,3288,3324,
            3289,3294,3290,3305,3291,3322,3296,3317,3296,
            3308,3302,3315,3299,3306,3301,3328,3301,3308,
            3306,3309,3303,3313,3303,3332,3305,3364,3305,
            3322,3306,3326,3307,3316,3308,3346,3311,3318,
            3312,3337,3312,3330,3314,3323,3315,3319,3327,
            3365,3316,3340,3316,3327,3318,3335,3321,3343,
            3321,3370,3322,3326,3323,3334,3324,3329,3324,
            3337,3326,3347,3327,3356,3334,3351,3335,3349,
            3336,3342,3329,3342,3331,3343,3332,3339,3329,
            3368,3334,3357,3335,3357,3341,3349,3352,3337,
            3353,3340,3355,3345,3374,3342,3367,3343,3369,
            3344,3360,3344,3371,3348,3361,3350,3359,3359,
            3360,3354,3366,3352,3362,3352,3358,3355,3371,
            3355,3372,3356,3373,3356,3381,3363,3376,3364,
            3370,3359,3385,3360,3378,3361,3366,3361,3408,
            3362,3375,3364,3399,3365,3381,3365,3386,3366,
            3374,3375,3390,3376,3380,3367,3380,3368,3396,
            3377,3384,3370,3384,3370,3398,3371,3404,3374,
            3400,3375,3383,3382,3388,3383,3387,3380,3389,
            3381,3395,3383,3394,3384,3392,3388,3390,3388,
            3413,3389,3391,3389,3396,3393,3400,3390,3394,
            3394,3413,3396,3397,3400,3407,3401,3435,3402,
            3408,3403,3409,3409,3422,3405,3406,3406,3407,
            3407,3411,3406,3412,3408,3419,3410,3420,3409,
            3434,3414,3416,3415,3418,3412,3462,3412,3419,
            3413,3426,3416,3420,3418,3423,3421,3422,3418,
            3430,3419,3425,3424,3435,3421,3429,3422,3434,
            3426,3426,3425,3433,3425,3431,3427,3439,3428,
            3439,3430,3438,3434,3510,3437,3472,3438,3471,
            3436,3440,3437,3446,3438,3487,3439,3453,3442,
            3444,3440,3482,3445,3455,3447,3456,3444,3451,
            3449,3451,3445,3461,3446,3497,3448,3452,3450,
            3457,3452,3456,3454,3460,3451,3458,3464,3464,
            3452,3474,3453,3498,3453,3499,3457,3525,3459,
            3467,3456,3470,3457,3458,3458,3482,3460,3471,
            3462,3469,3462,3465,3464,3478,3466,3481,3481,
            3515,3467,3468,3469,3477,3470,3478,3470,3474,
            3471,3489,3472,3505,3474,3483,3480,3485,3478,
            3503,3479,3485,3483,3488,3481,3539,3484,3488,
            3482,3490,3483,3512,3486,3494,3485,3495,3487,
            3542,3487,3489,3490,3496,3488,3493,3491,3493,
            3492,3500,3501,3511,3489,3501,3490,3540,3498,
            3499,3494,3495,3497,3505,3493,3504,3494,3500,
            3495,3511,3497,3515,3502,3508,3498,3522,3499,
            3526,3500,3507,3501,3538,3503,3506,3503,3512,
            3505,3519,3508,3509,3508,3517,3510,3513,3510,
            3530,3514,3530,3511,3538,3512,3531,3515,3559,
            3518,3536,3517,3519,3517,3545,3519,3545,3520,
            3547,3524,3528,3525,3540,3527,3533,3528,3543,
            3531,3532,3529,3534,3530,3550,3531,3549,3535,
            3548,3536,3548,3536,3562,3537,3539,3538,3568,
            3539,3563,3541,3552,3540,3558,3547,3549,3545,
            3564,3546,3551,3546,3552,3547,3553,3548,3560,
            3550,3556,3549,3561,3554,3569,3550,3565,3552,
            3566,3553,3566,3553,3561,3555,3565,3557,3570,
            3559,3563,3559,3583,3560,3569,3560,3580,3561,
            3572,3562,3581,3562,3573,3563,3582,3564,3575,
            3564,3576,3567,3570,3565,3570,3566,3571,3571,
            3572,3569,3579,3571,3577,3572,3578,3575,3584])            
        i, j1, j2 = morph.pairwise_permutations(i, j)
        pass
    
class TestIsLocalMaximum(unittest.TestCase):
    def test_00_00_empty(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(~ result))
        
    def test_01_01_one_point(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        labels[5,5] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == (labels == 1)))
        
    def test_01_02_adjacent_and_same(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5:6] = 1
        labels[5,5:6] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == (labels == 1)))
        
    def test_01_03_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5:6] = 1
        expected = (image == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_01_04_not_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,8] = .5
        labels[image > 0] = 1
        expected = (labels == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_01_05_two_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,15] = .5
        labels[5,5] = 1
        labels[5,15] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))

    def test_01_06_adjacent_different_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5] = 1
        labels[5,6] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        self.assertTrue(np.all(result == expected))
        
    def test_02_01_four_quadrants(self):
        np.random.seed(21)
        image = np.random.uniform(size=(40,60))
        i,j = np.mgrid[0:40,0:60]
        labels = 1 + (i >= 20) + (j >= 30) * 2
        i,j = np.mgrid[-3:4,-3:4]
        footprint = (i*i + j*j <=9)
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 20), (20, 40)):
            for jmin, jmax in ((0, 30), (30, 60)):
                expected[imin:imax,jmin:jmax] = scind.maximum_filter(
                    image[imin:imax, jmin:jmax], footprint = footprint)
        expected = (expected == image)
        result = morph.is_local_maximum(image, labels, footprint)
        self.assertTrue(np.all(result == expected))
        
    def test_03_01_disk_1(self):
        '''regression test of img-1194, footprint = [1]
        
        Test is_local_maximum when every point is a local maximum
        '''
        np.random.seed(31)
        image = np.random.uniform(size=(10,20))
        footprint = morph.strel_disk(.5)
        self.assertEqual(np.prod(footprint.shape), 1)
        self.assertEqual(footprint[0,0], 1)
        result = morph.is_local_maximum(image, np.ones((10,20)), footprint)
        self.assertTrue(np.all(result))
        

class TestAngularDistribution(unittest.TestCase):
    def test_00_00_angular_dist(self):
        np.random.seed(0)
        # random labels from 0 to 9
        labels = (np.random.uniform(0, 0.95, (1000, 1000)) * 10).astype(np.int)
        # filled square of 11 (NB: skipped 10)
        labels[200:300, 600:900] = 11
        angdist = morph.angular_distribution(labels)
        # 10 is an empty label
        assert np.all(angdist[9, :] == 0.0)
        # check approximation to chord ratio of filled rectangle (roughly 3.16)
        resolution = angdist.shape[1]
        angdist2 = angdist[-1, :resolution/2] + angdist[-1, resolution/2:]
        assert np.abs(3.16 - np.sqrt(angdist2.max() / angdist2.min())) < 0.05

class TestFeretDiameter(unittest.TestCase):
    def test_00_00_none(self):
        result = morph.feret_diameter(np.zeros((0,3)), np.zeros(0, int), [])
        self.assertEqual(len(result), 0)
        
    def test_00_01_point(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0]]),
            np.ones(1, int), [1])
        self.assertEqual(len(min_result), 1)
        self.assertEqual(min_result[0], 0)
        self.assertEqual(len(max_result), 1)
        self.assertEqual(max_result[0], 0)
        
    def test_01_02_line(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0], [1, 1, 1]]),
            np.array([2], int), [1])
        self.assertEqual(len(min_result), 1)
        self.assertEqual(min_result[0], 0)
        self.assertEqual(len(max_result), 1)
        self.assertEqual(max_result[0], np.sqrt(2))
        
    def test_01_03_single(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        iii = r.randint(0, 100, size=(20 * niterations))
        jjj = r.randint(0, 100, size=(20 * niterations))
        for iteration in range(100):
            ii = iii[(iteration * 20):((iteration + 1) * 20)]
            jj = jjj[(iteration * 20):((iteration + 1) * 20)]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((ii, jj, np.ones(20, int))), [1])
            min_result, max_result = morph.feret_diameter(chulls, counts, [1])
            self.assertEqual(len(min_result), 1)
            distances = np.sqrt(
                ((ii[:,np.newaxis] - ii[np.newaxis,:]) ** 2 +
                 (jj[:,np.newaxis] - jj[np.newaxis,:]) ** 2).astype(float))
            expected = np.max(distances)
            if abs(max_result - expected) > .000001:
                a0,a1 = np.argwhere(distances == expected)[0]
                self.assertAlmostEqual(
                    max_result[0], expected,
                    msg = "Expected %f, got %f, antipodes are %d,%d and %d,%d" %
                (expected, result, ii[a0], jj[a0], ii[a1], jj[a1]))
            #
            # Do a 180 degree sweep, measuring
            # the Feret diameter at each angle. Stupid but an independent test.
            #
            # Draw a line segment from the origin to a point at the given
            # angle from the horizontal axis
            #
            angles = np.pi * np.arange(20).astype(float) / 20.0
            i = -np.sin(angles)
            j = np.cos(angles)
            chull_idx, angle_idx = np.mgrid[0:counts[0],0:20]
            #
            # Compose a list of all vertices on the convex hull and all lines
            #
            v = chulls[chull_idx.ravel(),1:]
            pt1 = np.zeros((20 * counts[0], 2))
            pt2 = np.column_stack([i[angle_idx.ravel()], j[angle_idx.ravel()]])
            #
            # For angles from 90 to 180, the parallel line has to be sort of
            # at negative infinity instead of zero to keep all points on
            # the same side
            #
            pt1[angle_idx.ravel() < 10,1] -= 200
            pt2[angle_idx.ravel() < 10,1] -= 200
            pt1[angle_idx.ravel() >= 10,0] += 200
            pt2[angle_idx.ravel() >= 10,0] += 200
            distances = np.sqrt(morph.distance2_to_line(v, pt1, pt2))
            distances.shape = (counts[0], 20)
            dmin = np.min(distances, 0)
            dmax = np.max(distances, 0)
            expected_min = np.min(dmax - dmin)
            self.assertTrue(min_result[0] <= expected_min)
            
    def test_02_01_multiple_objects(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        ii = r.randint(0, 100, size=(20 * niterations))
        jj = r.randint(0, 100, size=(20 * niterations))
        vv = np.hstack([np.ones(20) * i for i in range(1,niterations+1)])
        indexes = np.arange(1, niterations+1)
        chulls, counts = morph.convex_hull_ijv(
            np.column_stack((ii, jj, vv)), indexes)
        min_result, max_result = morph.feret_diameter(chulls, counts, indexes)
        self.assertEqual(len(max_result), niterations)
        for i in range(niterations):
            #
            # Make sure values are same as single (validated) case.
            #
            iii = ii[(20*i):(20*(i+1))]
            jjj = jj[(20*i):(20*(i+1))]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((iii, jjj, np.ones(len(iii), int))), [1])
            expected_min, expected_max = morph.feret_diameter(chulls, counts, [1])
            self.assertAlmostEqual(expected_min[0], min_result[i])
            self.assertAlmostEqual(expected_max[0], max_result[i])


class TestIsObtuse(unittest.TestCase):
    def test_00_00_empty(self):
        result = morph.is_obtuse(np.zeros((0,2)),np.zeros((0,2)),np.zeros((0,2)))
        self.assertEqual(len(result), 0)
        
    def test_01_01_is_obtuse(self):
        result = morph.is_obtuse(np.array([[-1,1]]),
                                 np.array([[0,0]]),
                                 np.array([[1,0]]))
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0])
        
    def test_01_02_is_not_obtuse(self):
        result = morph.is_obtuse(np.array([[1,1]]),
                                 np.array([[0,0]]),
                                 np.array([[1,0]]))
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0])
        
    def test_01_03_many(self):
        r = np.random.RandomState()
        r.seed(13)
        p1 = np.random.uniform(size=(100,2))
        v = np.random.uniform(size=(100,2))
        p2 = np.random.uniform(size=(100,2))
        vp1 = np.sqrt(np.sum((v - p1) * (v - p1), 1))
        vp2 = np.sqrt(np.sum((v - p2) * (v - p2), 1))
        p1p2 = np.sqrt(np.sum((p1-p2) * (p1-p2), 1))
        # Law of cosines
        theta = np.arccos((vp1**2 + vp2**2 - p1p2 **2) / (2 * vp1 * vp2))
        result = morph.is_obtuse(p1, v, p2)
        is_obtuse = theta > np.pi / 2
        np.testing.assert_array_equal(result, is_obtuse)
        
class TestSingleShortestPaths(unittest.TestCase):
    def test_00_00_one_node(self):
        p, c = morph.single_shortest_paths(0, np.zeros((1,1)))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0], 0)
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0], 0)
        
    def test_01_01_two_nodes(self):
        p, c = morph.single_shortest_paths(0, np.array([[0,1],[1,0]]))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0], 0)
        self.assertEqual(p[1], 0)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 1)
        
    def test_01_02_two_nodes_backwards(self):
        p, c = morph.single_shortest_paths(1, np.array([[0,1],[1,0]]))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0], 1)
        self.assertEqual(p[1], 1)
        self.assertEqual(len(c), 2)
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 0)
        
    def test_01_03_5x5(self):
        # All paths from 0 to 4
        all_permutations = np.array([
            [ 0, 0, 0, 0, 4],
            [ 0, 0, 0, 1, 4],
            [ 0, 0, 0, 2, 4],
            [ 0, 0, 0, 3, 4],
            [ 0, 0, 1, 2, 4],
            [ 0, 0, 1, 3, 4],
            [ 0, 0, 2, 1, 4],
            [ 0, 0, 2, 3, 4],
            [ 0, 0, 3, 1, 4],
            [ 0, 0, 3, 2, 4],
            [ 0, 1, 2, 3, 4],
            [ 0, 1, 3, 2, 4],
            [ 0, 2, 1, 3, 4],
            [ 0, 2, 3, 1, 4],
            [ 0, 3, 1, 2, 4],
            [ 0, 3, 2, 1, 4]
        ])
        r = np.random.RandomState()
        r.seed(13)
        for _ in range(1000):
            c = r.uniform(size=(5,5))
            c[np.arange(5), np.arange(5)] = 0
            steps = c[all_permutations[:, :-1],
                      all_permutations[:, 1:]]
            all_costs = np.sum(steps, 1)
            best_path = all_permutations[np.argmin(all_costs)]
            best_path = list(reversed(best_path[best_path != 0][:-1]))
            best_score = np.min(all_costs)
            paths, scores = morph.single_shortest_paths(0, c)
            self.assertEqual(scores[4], best_score)
            step_count = 0
            found_path = []
            i = 4
            while step_count != 5 and paths[i] != 0:
                i = paths[i]
                found_path.append(i)
                step_count += 1
            self.assertEqual(len(found_path), len(best_path))
            self.assertTrue(all([a == b for a,b in zip(found_path, best_path)]))
            
