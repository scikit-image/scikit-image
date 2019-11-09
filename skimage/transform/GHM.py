import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import GHMHelperFuncs as helper

# ROCKYFIX: look for ROCKYFIX, TODO, FIXME, XXX, FIX


def calc_C_T_mtx(m, n, A, B, dist, cdf):
    """ A and B are the matrices of histograms.
    """
    C = np.zeros((m, n))
    T = np.zeros((m, n), dtype=np.int64)

    if cdf:
        A = helper.calc_cdf(A)
        B = helper.calc_cdf(B)

    # initialize C
    for j in range(n):
        C[0,j] = helper.row_cost(A, B, 0, 0, j, dist)
    # T is already initialized to zeros.
    print("Finished initializing C and T.")
    
    # fill out rest of C and T
    if cdf:
        for i in range(1, m):
            for j in range(n):
                index_min_cost_of_prev_row = np.argmin(C[i-1, :j + 1])
                min_cost_of_prev_row = C[i-1, index_min_cost_of_prev_row]
                cost_of_setting_indicator_in_col_j = helper.row_cost(A, B, i, j, j, dist)
                C[i,j] = cost_of_setting_indicator_in_col_j + min_cost_of_prev_row
                T[i,j] = index_min_cost_of_prev_row
    else:
        for i in range(1, m):
            for j in range(n):
                # TODO merge the two lines below?
                prev_row_costs = [helper.row_cost(A, B, i, jj+1, j, dist) for jj in range(0, j+1)]
                costs = [helper.row_cost(A, B, i , 0, j, dist)] + [C[i-1, jj] + prev_row_costs[jj] for jj in range(0,j+1)]
                argmin = np.argmin(costs)
                C[i,j] = costs[argmin]
                T[i,j] = argmin - 1 # j = -1, 0 ... n-2, n-1 = length n+1. but it's 0, 1, ..., n
    print("Finished finding C and T.")
    return C, T
            
# TODO add assertion that the M, C, and T matrices follow rules listed in paper
def find_mapping(A, B, index_to_pix_A, index_to_pix_B, dist='L1', cdf=False):
    """ Find function that maps values (bins) in A's range to values (bins) in B's range.
        See paper for some details.
    """
    print("*********")
    print("Starting find_mapping")
    
    (n, k_1) = A.shape
    (m, k_2) = B.shape
    assert k_1 == k_2, "The number of columns (histograms) in A and B must match."
    k = k_1
    
    # print(m)
    # print(n)
    # print(k)
    
    C, T = calc_C_T_mtx(m, n, A, B, dist, cdf)

    M = np.zeros((m, n), dtype=np.int64)
    
    # find path in M from T
    j = n-1
    jj = n-1
    for i in range(m-1, 0, -1): # starting from bottom row, going to all but top and looking at each row in T and the row above it
        # print(i)
        # print(j)
        jj = T[i,j]
        M[i, jj+1:j+1] = 1
        j = jj
    M[0, 0:jj+1] = 1 # row 0
    
    mapping = {}
    for j in range(n):
        col = M[:,j]
        i = np.nonzero(col)[0][0] # only one non-zero element (that element being 1) per column
        # Mapping from index j to index i. Convert into a mapping from old pixel value to new pixel value.
        pix_A = index_to_pix_A[j]
        new_pix_A = index_to_pix_B[i]
        mapping[pix_A] = new_pix_A
    print("Mapping:")
    print(mapping)
    print("Done finding mapping")
    return mapping, C, T, M


# TODO decide what file formats we accept. Currently we do jpg. See https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.imread.html See "Notes".

#GHM and cdfGHM
def GHM(imgA, imgB, cdf=True, num_histograms_per_dim=1, dist='L1'):
    """ imgA:                   image being processed
        imgB:                   template image
        cdf:                    if True, use cdf GHM; otherwise, use pdf GHM
        num_histograms_per_dim: how many separate subsections the image should be split into per dimension (i.e. num_histograms_per_dim squared is the number of histograms generated for a 2D image)
        dist:                   distance type (default is L1 squared); distance must be an additive distance
    """
    A, pix_to_index_A, index_to_pix_A = helper.create_matrix(imgA, num_histograms_per_dim)
    B, pix_to_index_B, index_to_pix_B = helper.create_matrix(imgB, num_histograms_per_dim)
    mapping, C, T, M = find_mapping(A, B, index_to_pix_A, index_to_pix_B, dist, cdf=cdf)
    matched_imgA = helper.convert(imgA, mapping)
    return matched_imgA
