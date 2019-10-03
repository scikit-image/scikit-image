import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import matplotlib.pyplot as plt

#GENERAL IMAGES HELPER FUNCTIONS
def is_grayscale(img):
    if len(img.shape) == 2:
        return True
    c1 = img[:,:,0]
    c2 = img[:,:,1]
    c3 = img[:,:,2]
    return np.array_equal(c1,c2) and np.array_equal(c2,c3)

def plot_hist(img):
    plt.hist(img.ravel(), range=(0,256), bins=256)
    plt.show()

def show_img(img):
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.show()

def is_from_0_to_255(img):
    (h, w) = img.shape
    for row in range(h):
        for col in range(w):
            if (img[row, col] > 255 or img[row, col] < 0):
                return False
    return True

def read_and_check_img(filepath):
    if filepath[-4:] != ".jpg":
        raise Exception("Expect a .jpg image. Received something else.")
    print("Reading image " + filepath)
    img = plt.imread(filepath)
    assert is_grayscale(img), "Image must be grayscale."
    if len(img.shape) != 2:
        img = img[:,:,0]
    print("Done reading "+ filepath)
    # print(img)
    # show_img(img)
    # plot_hist(img)
    return img

#GHM HELPER FUNCTIONS
def distance(value1, value2, metric='L1'):
    """
    Any distance must be 'additive' (see paper).
    """
    if metric == 'L1':
        return abs(value2-value1)
    elif metric == 'L2':
        return (value2-value1)**2
    else:
        raise Exception("Currently only accepts L1 squared and L2 squared distance metrics.")

# used in find_mapping
def row_cost(A, B, i, jj, j, dist='L1'):
    """
    See paper for rowCost function. Note that j and k are inclusive.
    """
    dist_sum = 0
    (n, k) = A.shape
    for col in range(k):
        dist_sum += distance(sum(A[jj:j+1, col]), B[i, col], dist)
    return dist_sum

def pix_index_mapping(img):
    """
    Returns both pix_to_index and index_to_pix.
    pix_to_index is a dictionary that maps each pixel value in the image to its index in the to-be-created histograms.
    index_to_pix is an array that maps indices in a histogram to pixel values.

    Note that pix_to_index and index_to_pix are strictly monotonically increasing
    (i.e. higher pix implies higher index and higher index implies higher pix).
    """
    pix_to_index = {}
    index_to_pix = []
    pixes = set(img.flat)
    index_to_pix = list(pixes)
    index_to_pix.sort()
    pix_to_index = {}
    pix_to_index = dict(zip(index_to_pix, range(0, len(index_to_pix))))
    return pix_to_index, index_to_pix

def calc_histogram(img, pix_to_index):
    """
    Calculate the normalized histogram.
    The histogram is a numpy array, where each index corresponds to a pixel value.
    The histogram is normalized so that the sum of all entries equals 1.
    """
    histogram = [0]*len(pix_to_index)
    add_to_histogram = np.vectorize(lambda pix : histogram[pix_to_index[pix]] += 1)
    add_to_histogram(img)
    histogram = np.array([histogram])
    histogram = histogram / img.size
    return histogram.T

def calc_cdf(mtx):
    n = mtx.shape[0]
    H = np.tril(np.ones((n, n)))
    return np.matmul(H, mtx)


def create_matrix(img, num_histograms_per_dim=1):
    """
    """
    (height, width) = img.shape
    matrix = None
    box_height = height // num_histograms_per_dim
    box_width = width // num_histograms_per_dim

    pix_to_index, index_to_pix = pix_index_mapping(img)

    for i in range(num_histograms_per_dim-1):
        for j in range(num_histograms_per_dim-1):
            top = i*box_height
            bottom = (i+1)*box_height
            left = j*box_width
            right = (j+1)*box_width
            sub_img = img[top:bottom, left:right] # TODO: here and other places in this function - maybe don't create whole sub-image, this pass in indices to calc_histogram?
            show_img(sub_img)
            column = calc_histogram(sub_img, pix_to_index)
            if matrix is None:
                matrix = column
            else:
                matrix = np.hstack((matrix, column))
    # last boxes:
    # on the bottom:
    i = num_histograms_per_dim - 1
    top = i*box_height
    for j in range(num_histograms_per_dim-1):
        left = j*box_width
        right = (j+1)*box_width
        sub_img = img[top:, left:right]
        show_img(sub_img)
        column = calc_histogram(sub_img, pix_to_index)
        if matrix is None:
            matrix = column
        else:
            matrix = np.hstack((matrix, column))
    # on the right:
    j = num_histograms_per_dim - 1
    left = j*box_width
    for i in range(num_histograms_per_dim-1):
        top = i*box_height
        bottom = (i+1)*box_height
        sub_img = img[top:bottom, left:]
        show_img(sub_img)
        column = calc_histogram(sub_img, pix_to_index)
        if matrix is None:
            matrix = column
        else:
            matrix = np.hstack((matrix, column))
    # bottom-right box
    i = num_histograms_per_dim - 1
    j = num_histograms_per_dim - 1
    top = i*box_height
    left = j*box_width
    sub_img = img[top:, left:]
    show_img(sub_img)
    column = calc_histogram(sub_img, pix_to_index)
    if matrix is None:
        matrix = column
    else:
        matrix = np.hstack((matrix, column))
    return matrix, pix_to_index, index_to_pix

def convert(imgA, mapping):
    """
    This function takes imgA, and a mapping from pixel values in imgA to another set of pixel values.
    This function returns a modified version of imgA according to the mapping.
    """
    converter = np.vectorize(lambda pix : mapping[pix])
    return converter(imgA)









########################## 3D functions ##########################

# def create_matrix3D(img, num_histograms_per_dim=1):
#     """Creates the matrices using histograms as columns (see paper)."""
#     (height, width, depth) = img.shape
#     box_height = height // num_histograms_per_dim
#     box_width = width // num_histograms_per_dim
#     box_depth = depth // num_histograms_per_dim
    
#     pix_to_index, index_to_pix = pix_index_mapping(img)
    
#     matrix = None
#     for i in range(num_histograms_per_dim-1):
#         for j in range(num_histograms_per_dim-1):
#             for k in range(num_histograms_per_dim-1):
#                 top = i*box_height
#                 bottom = (i+1)*box_height
#                 left = j*box_width
#                 right = (j+1)*box_width
#                 front = k*box_depth
#                 back = (k+1)*box_depth
#                 sub_img = img[top:bottom, left:right, front:back]
#                 show_slices(sub_img)
#                 column = calc_histogram(sub_img, pix_to_index)
#                 if matrix is None:
#                     matrix = column
#                 else:
#                     matrix = np.hstack((matrix, column))
    
#     # last boxes:
#     for k in range(num_histograms_per_dim-1): # for each slice except last
#         front = k*box_depth
#         back = (k+1)*box_depth
#         # on the bottom:
#         i = num_histograms_per_dim - 1
#         top = i*box_height
#         for j in range(num_histograms_per_dim-1):
#             left = j*box_width
#             right = (j+1)*box_width
#             sub_img = img[top:, left:right, front:back]
#             show_slices(sub_img)
#             column = calc_histogram(sub_img, pix_to_index)
#             if matrix is None:
#                 matrix = column
#             else:
#                 matrix = np.hstack((matrix, column))
#         # on the right:
#         j = num_histograms_per_dim - 1
#         left = j*box_width
#         for i in range(num_histograms_per_dim-1):
#             top = i*box_height
#             bottom = (i+1)*box_height
#             sub_img = img[top:bottom, left:, front:back]
#             show_slices(sub_img)
#             column = calc_histogram(sub_img, pix_to_index)
#             if matrix is None:
#                 matrix = column
#             else:
#                 matrix = np.hstack((matrix, column))
#         # bottom-right box
#         i = num_histograms_per_dim - 1
#         j = num_histograms_per_dim - 1
#         top = i*box_height
#         left = j*box_width
#         sub_img = img[top:, left:, front:back]
#         show_slices(sub_img)
#         column = calc_histogram(sub_img, pix_to_index)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))
    
#     # very back slice
#     k = num_histograms_per_dim - 1
#     front = k*box_depth
#     # on the bottom:
#     i = num_histograms_per_dim - 1
#     top = i*box_height
#     for j in range(num_histograms_per_dim-1):
#         left = j*box_width
#         right = (j+1)*box_width
#         sub_img = img[top:, left:right, front:]
#         show_slices(sub_img, pix_to_index)
#         column = calc_histogram(sub_img)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))
#     # on the right:
#     j = num_histograms_per_dim - 1
#     left = j*box_width
#     for i in range(num_histograms_per_dim-1):
#         top = i*box_height
#         bottom = (i+1)*box_height
#         sub_img = img[top:bottom, left:, front:]
#         show_slices(sub_img)
#         column = calc_histogram(sub_img, pix_to_index)
#         if matrix is None:
#             matrix = column
#         else:
#             matrix = np.hstack((matrix, column))
#     # bottom-right box
#     i = num_histograms_per_dim - 1
#     j = num_histograms_per_dim - 1
#     top = i*box_height
#     left = j*box_width
#     sub_img = img[top:, left:, front:]
#     show_slices(sub_img)
#     column = calc_histogram(sub_img, pix_to_index)
#     if matrix is None:
#         matrix = column
#     else:
#         matrix = np.hstack((matrix, column))
#     return matrix, pix_to_index, index_to_pix

