# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(edgeitems=10)

from skimage.measure import label, label_match

def test_relabel_single_overlap():
    filtered_r = np.array(np.logical_not(np.array(plt.imread('test1_no_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    filtered_b = np.array(np.logical_not(np.array(plt.imread('test1b_no_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    overlap = np.logical_and(filtered_b, filtered_r)
    
    label_1, label_2 = label(filtered_r, background=-1), label(filtered_b, background=-1)

    print(label_1[40:70:2,45:85:2])
    print("\n")
    print(label_2[40:70:2,45:85:2])
    print("-"*80)

    rlabel_1, rlabel_2 = label_match(label_1, label_2, relabel=True, background=0)
    
    print(rlabel_1[40:70:2,45:85:2])
    print("\n")
    print(rlabel_2[40:70:2,45:85:2])
    print("-"*80)
    
    l1o = label_1[overlap]
    rl1o = rlabel_2[overlap]

    np.testing.assert_almost_equal(rl1o[l1o == 1], 1)
    np.testing.assert_almost_equal(rl1o[l1o == 2], 2)

def test_norelabel_single_overlap():
    filtered_r = np.array(np.logical_not(np.array(plt.imread('test1_no_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    filtered_b = np.array(np.logical_not(np.array(plt.imread('test1b_no_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    overlap = np.logical_and(filtered_b, filtered_r)
    
    label_1, label_2 = label(filtered_r, background=-1), label(filtered_b, background=-1)

    print(label_1[40:70:2,45:85:2])
    print("\n")
    print(label_2[40:70:2,45:85:2])
    print("-"*80)

    rlabel_1, rlabel_2 = label_match(label_1, label_2, relabel=False, background=0)
    
    print(rlabel_1[40:70:2,45:85:2])
    print("\n")
    print(rlabel_2[40:70:2,45:85:2])
    print("-"*80)
    
    l1o = label_1[overlap]
    rl1o = rlabel_2[overlap]

    np.testing.assert_almost_equal(rl1o[l1o == 1], 2)
    np.testing.assert_almost_equal(rl1o[l1o == 2], 1)
    
def test_relabel_double_overlap_remove():
    filtered_r = np.array(np.logical_not(np.array(plt.imread('test1_no_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    filtered_b = np.array(np.logical_not(np.array(plt.imread('test2b_overlap.png')[...,0],dtype=np.int64)),dtype=np.int64)
    
    label_1, label_2 = label(filtered_r, background=-1), label(filtered_b, background=-1)

    print(label_1[40:70:2,45:85:2])
    print("\n")
    print(label_2[40:70:2,45:85:2])
    print("-"*80)

    rlabel_1, rlabel_2 = label_match(label_1, label_2, relabel=True, background=0)
    
    print(rlabel_1[40:70:2,45:85:2])
    print("\n")
    print(rlabel_2[40:70:2,45:85:2])
    print("-"*80)

    # check that the two regions labelled as 1 in the input images are still one
    np.testing.assert_almost_equal(rlabel_1[np.logical_and(label_1 == 1, label_2==1)], 1)
    # check that the region labelled 3 in image 2 is remmoved
    np.testing.assert_almost_equal(rlabel_1[np.logical_and(label_1 == 1, label_2==3)], 0)
    # check that the single overlap region 2 in both images is still there.
    np.testing.assert_almost_equal(rlabel_1[np.logical_and(label_1 == 2, label_2==2)], 2)
