import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
from GHMHelperFuncs import *

def calc_C_T_mtx(m, n, A, B, dist, cdf):
    """
    A and B are the matrices of PDFs/histograms.
    """
    C = np.zeros((m, n))
    T = np.zeros((m, n), dtype=np.int64)

    if cdf:
        A = calc_cdf(A)
        B = calc_cdf(B)

    # initialize C
    for j in range(n):
        C[0, j] = row_cost(A, B, 0, 0, j, dist)
    # T is already initialized to zeros.
    print("Finished initializing C and T.")
    
    # fill out rest of C and T
    for i in range(1, m):
        for j in range(n):
    if cdf:          
        index_min_cost_of_prev_row = np.argmin(C[i-1, :j + 1])
        min_cost_of_prev_row = C[i-1, index_min_cost_of_prev_row]
        cost_of_setting_indicator_in_col_j = row_cost(A, B, i, j, j, dist)
        C[i,j] = cost_of_setting_indicator_in_col_j + min_cost_of_prev_row
        T[i, j] = index_min_cost_of_prev_row 
    else:
        prev_row_costs = [row_cost(A, B, i, jj+1, j, dist) for jj in range(0, j+1)]
        costs = [C[i-1, jj] + prev_row_costs[jj] for jj in range(0,j+1)]
        costs = [row_cost(A, B, i , 0, j, dist)] + costs
        C[i, j] = min(costs)
        T[i, j] = np.argmin(costs)
    print("Finished finding C and T.")
    return C, T
            
# TODO add assertion that the M, C, and T matrices follow rules listed in paper
def find_mapping(A, B, index_to_pix_A, index_to_pix_B, dist='L1', cdf=False):
    """
    Find function that maps values (bins) in A's range to values (bins) in B's range.
    See paper for some details.
    """
    print("*********")
    print("Starting find_mapping")
    
    (n, k_1) = A.shape
    (m, k_2) = B.shape
    assert k_1 == k_2, "The number of columns (histograms) in A and B must match."
    k = k_1
    
    print(m)
    print(n)
    print(k)
    
    C, T = calc_C_T_mtx(m, n, A, B, dist, cdf)

    M = np.zeros((m, n), dtype=np.int64)
    
    # find path in M from T
    j = n-1
    for i in range(m-1, 0, -1): # starting from bottom row, going to all but top and looking at each row in T and the row above it
        jj = T[i,j]
        M[i, jj+1:j+1] = 1
        j = jj
    M[0,0:jj+1] = 1 # row 0
    
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



#GHM and cdfGHM
def GHM(imgA, imgB, dist='L1'):
    A = create_matrix(imgA)
    B = create_matrix(imgB)
    mapping, C, T, M = find_mapping(A, B, index_to_pix_A, index_to_pix_B, dist, cdf=False)

    matched_imgA = convert(imgA, mapping)
    return matched_imgA


def cdfGHM(imgA, imgB, num_histograms_per_dim=1, dist ='L1'):
    A, pix_to_index_A, index_to_pix_A = create_matrix(imgA, num_histograms_per_dim)
    B, pix_to_index_B, index_to_pix_B = create_matrix(imgB, num_histograms_per_dim)
    mapping, C, T, M = find_mapping(A, B, index_to_pix_A, index_to_pix_B, dist, cdf=True)
    matched_imgA = convert(imgA, mapping)
    return matched_imgA

