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

def read_and_check_img(filepath):
    if filepath[-4:] != ".jpg":
        raise Exception("Expect a .jpg image. Received something else.")
    img = plt.imread(filepath)
    print(img.shape)
    print(is_grayscale(img))
    if not is_grayscale(img):
        raise Exception("Image is not a grayscale image (meaning not all 3 channels of jpg are the same.)")
    if len(img.shape) != 2:
        img = img[:,:,0]
    show_img(img)
    plot_hist(img)
    return img

#GHM HELPER FUNCTIONS
# any distance must be 'additive' (see paper)
def distance(value1, value2, metric='L1'):
    if metric == 'L1':
        return abs(value2-value1)
    elif metric == 'L2':
        return (value2-value1)**2
    else:
        raise Exception("Currently only accepts L1 squared and L2 squared distance metrics.")

# used in find_mapping
def row_cost(A, B, i, jj, j, dist='L1'):
    """See paper for rowCost function. Note that j and k are inclusive."""
    dist_sum = 0
    (n, k) = A.shape
    for col in range(k):
        dist_sum += distance(sum(A[jj:j+1, col]), B[i, col], dist)
    return dist_sum

# used in create_matrix
def calc_histogram(img):
#     vals, indices, counts = np.unique(img.ravel(), return_inverse=True, return_counts=True)
#     counts = counts / img.size    
#     return np.array([counts]).T, vals
    histogram = [0 for _ in range(256)]
    (h,w) = img.shape
    for i in range(h):
        for j in range(w):
            histogram[img[i,j]] += 1
    histogram = np.array([histogram])
    histogram = histogram / img.size
    return histogram.T

def calc_cdf(mtx):
    n = mtx.shape[0]
    H = np.tril(np.ones((n, n)))
    return np.matmul(H, mtx)

def create_matrix(img, num_histograms_per_dim=1):
    """For now, it'll just take a histogram of the whole images"""
    (height, width) = img.shape
    matrix = None
    box_height = height // num_histograms_per_dim
    box_width = width // num_histograms_per_dim
    for i in range(num_histograms_per_dim-1):
        for j in range(num_histograms_per_dim-1):
            top = i*box_height
            bottom = (i+1)*box_height
            left = j*box_width
            right = (j+1)*box_width
            sub_img = img[top:bottom, left:right]
            show_img(sub_img)
            column = calc_histogram(sub_img)
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
        column = calc_histogram(sub_img)
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
        column = calc_histogram(sub_img)
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
    column = calc_histogram(sub_img)
    if matrix is None:
        matrix = column
    else:
        matrix = np.hstack((matrix, column))
    
#     matrix = calc_histogram(img)
    return matrix

def convert(imgA, mapping):
    (height, width) = imgA.shape
    matched_imgA = np.zeros(imgA.shape)
    for i in range(height):
        for j in range(width):
            matched_imgA[i,j] = mapping[imgA[i,j]]
    return matched_imgA