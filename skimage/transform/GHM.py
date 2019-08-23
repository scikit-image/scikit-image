import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt
import GHMHelperFuncs

def calc_C_T_mtx(m, n, A, B, dist, cdf):
    if cdf:
        C = np.zeros((m, n))
        T = np.zeros((m, n), dtype=np.int64)
        
        A_cdf = calc_cdf(A)
        B_cdf = calc_cdf(B)

        # T is already initialized to zeros.

        # initialize C
        for j in range(n):
            C[0, j] = row_cost(A_cdf, B_cdf, 0, 0, j, dist)
        print("Finished initializing C and T. in CDF")


        # fill out rest of C and T
        for i in range(1, m):
    #         print("i:", i)
            for j in range(n):
    #             print("j:", j)
                index_min_cost_of_prev_row = np.argmin(C[i-1, :j + 1])
                min_cost_of_prev_row = C[i-1, index_min_cost_of_prev_row]
                cost_of_setting_indicator_in_col_j = row_cost(A_cdf, B_cdf, i, j, j, dist)
                C[i,j] = cost_of_setting_indicator_in_col_j + min_cost_of_prev_row
                T[i, j] = index_min_cost_of_prev_row
                
    else:
        C = np.zeros((m, n))
        T = np.zeros((m, n), dtype=np.int64)

        # T is already initialized to zeros.

        # initialize C
        for j in range(n):
            C[0, j] = row_cost(A, B, 0, 0, j, dist)
        print("Finished initializing C and T.")

        # fill out rest of C and T
        for i in range(1, m):
    #         print("i:", i)
            for j in range(n):
    #             print("j:", j)
                prev_row_costs = [row_cost(A, B, i, jj+1, j, dist) for jj in range(0, j+1)]
                costs = [C[i-1, jj] + prev_row_costs[jj] for jj in range(0,j+1)]
                costs = [row_cost(A, B, i , 0, j, dist)] + costs
                C[i, j] = min(costs)
                T[i, j] = np.argmin(costs)
        
    return C, T
            

def find_mapping(A, B, dist='L1', cdf=False):
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

    print("Finished finding C and T.")
    
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
        A_inds = np.nonzero(col)
        mapping[j] = A_inds[0][0] # only one non-zero element (1) per column
    print("Mapping:")
    print(mapping)
    print("Done finding mapping")
    return mapping, C, T, M


#GHM and cdfGHM
def GHM(imgA, imgB, dist='L1'):
    A = create_matrix(imgA)
    B = create_matrix(imgB)
    mapping, C, T, M = find_mapping(A, B, dist, cdf=False)
    matched_imgA = convert(imgA, mapping)
    return matched_imgA


def cdfGHM(imgA, imgB, dist ='L1'):
    A = create_matrix(imgA)
    B = create_matrix(imgB)
    mapping, C, T, M = find_mapping(A, B, dist, cdf=True)
    matched_imgA = convert(imgA, mapping)
    return matched_imgA
