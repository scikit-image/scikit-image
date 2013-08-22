import numpy as np
from scipy import pi, arctan2, cos, sin


def gradient(image, same_size=False):
    """ Computes the Gradients of the image

    Parameters
    ----------
    image: image of shape (imy, imx)
    same_size: boolean, optional, default is True
        If True, boundaries are duplicated so that the gradients
        has the same size as the original image.
        Otherwise, the gradients will have shape (imy-1, imx-1)
        
    Returns
    -------
    (Gradient X, Gradient Y), two numpy array with the same shape as image
        (if same_size=True)
    """
    sx, sy = image.shape
    if same_size:
        gx = np.zeros(image.shape)
        gx[:, 1:-1] = -image[:, :-2] + image[:, 2:]
        gx[:, 0] = -image[:, 0] + image[:, 1]
        gx[:, -1] = -image[:, -2] + image[:, -1]
    
        gy = np.zeros(image.shape)
        gy[1:-1, :] = -image[:-2, :] + image[2:, :]
        gy[0, :] = -image[0, :] + image[1, :]
        gy[-1, :] = -image[-2, :] + image[-1, :]
    
    else:
        gx = np.zeros((sx-2, sy-2))
        gx[:, :] = -image[1:-1, :-2] + image[1:-1, 2:]

        gy = np.zeros((sx-2, sy-2))
        gy[:, :] = -image[:-2, 1:-1] + image[2:, 1:-1]
    
    return gx, gy




def magnitude_orientation(gx, gy, max_angle=360):
    """ Computes the magnitude and orientation matrices from the gradients gx gy

    Parameters
    ----------
    gx: gradient x of the image
    gy: gradient y of the image
    max_angle: {180, 360}
    
    Returns 
    -------
    (magnitude, orientation)
    
    Warning
    -------
    The orientation is in degree, NOT radian!!
    """
    if max_angle!=180 and max_angle!=360:
        print "WARNING: in magnitude_orientation, max_angle is %f." % max_angle
        print "         Should be 180 or 360."
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = (arctan2(gy, gx) * 180 / pi) % max_angle
            
    return magnitude, orientation



def build_histogram(magnitude, orientation, cell_size=(8, 8), cells_per_block=(2, 2), max_angle=180,
        nbins=9, visualise=False, flatten=True):
    """ Builds the HOG using the given magnitude and orientation matrices

    Compute a Histogram of Oriented Gradients (HOG) by

    1. computing gradient histograms
    2. normalising across blocks
    3. flattening into a feature vector (if flatten=True)

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True

    Returns
    -------
    hog : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    
    sx, sy = magnitude.shape
    csx, csy = cell_size
    bx, by = cells_per_block
    
    # Consider only the right part of the image
    # (the rest doesn't fill a whole cell, just drop it)
    sx -= sx % csx
    sy -= sy % csy
    n_cells_x = sx//csx
    n_cells_y = sy//csy
    magnitude = magnitude[:sx, :sy]
    orientation = orientation[:sx, :sy]
    
    dx = csx//2
    dy = csy//2
    
    # For each cell, compute the distance to the bin center
    distance = 1 - np.sqrt(
	np.arange(dx)[:, np.newaxis]**2 + np.arange(dy)[np.newaxis, :]**2 + 1) / (max(csx, csy) + 1)
    distance = np.flipud(np.fliplr(distance))
    coefs = np.zeros((csx, csx))
    coefs[:dx, :dy] = distance
    coefs[:dx, -dy:] = 1 - distance
    coefs[-dx:, :dy] = 1 - distance
    coefs[-dx:, -dy:] = 1 - distance
    coefs = np.tile(coefs, (n_cells_x - 1, n_cells_y - 1))
    
    b_step = max_angle/nbins
    b0 = orientation // b_step
    b0[np.where(b0>=nbins)]=0
    b1 = b0 + 1
    b1[np.where(b1>=nbins)]=0
    b = np.abs(orientation % b_step - b_step/2) / b_step
    
    #linear interpolation to the bins
    # Coefficients corresponding to the bin interpolation
    temp_coefs = np.zeros((sx, sy, nbins))
    for i in range(nbins):
        temp_coefs[:, :, i] += np.where(b0==i, b, 0)
        temp_coefs[:, :, i] += np.where(b1==i, (1-b), 0)
    
    temp = np.zeros((sx, sy, nbins))
    #bilinear interpolation to the cells
    #(border of length dx, dy) (they are only interpolated to themselves)    
    temp[:dx, :, :] += temp_coefs[:dx, :, :]*magnitude[:dx, :, np.newaxis]
    temp[-dx:, :, :] += temp_coefs[-dx:, :, :]*magnitude[-dx:, :, np.newaxis]
    temp[:, :dy, :] += temp_coefs[:, :dy, :]*magnitude[:, :dy, np.newaxis]
    temp[:, -dy:, :] += temp_coefs[:, -dy:, :]*magnitude[:, -dy:, np.newaxis]

    # hist(x0, y0)
    temp[:-csx, :-csy, :] += temp_coefs[dx:-dx, dy:-dy, :]*(magnitude[dx:-dx, dy:-dy]*coefs)[:, :, np.newaxis]
    coefs = np.flipud(coefs)
    # hist(x1, y0)
    temp[csx:, :-csy, :] += temp_coefs[dx:-dx, dy:-dy, :]*(magnitude[dx:-dx, dy:-dy]*coefs)[:, :, np.newaxis]
    coefs = np.fliplr(coefs)
    # hist(x1, y1)
    temp[csx:, csy:, :] += temp_coefs[dx:-dx, dy:-dy, :]*(magnitude[dx:-dx, dy:-dy]*coefs)[:, :, np.newaxis]
    coefs = np.flipud(coefs)
    # hist(x0, y1)
    temp[:-csx, csy:, :] += temp_coefs[dx:-dx, dy:-dy, :]*(magnitude[dx:-dx, dy:-dy]*coefs)[:, :, np.newaxis]
    
    # Compute the histogram: sum over the cells
    orientation_histogram = temp.reshape((n_cells_x, csx, n_cells_y, csy, nbins)).sum(axis=3).sum(axis=1)
    

    if visualise:
        from skimage import draw

        radius = min(csx, csy) // 2 - 1
        hog_image = np.zeros((sx, sy), dtype=float)
        for x in range(n_cells_x):
            for y in range(n_cells_y):
                for o in range(nbins):
                    centre = tuple([x * csx + csx // 2, y * csy + csy // 2])
                    dy = radius * cos(float(o) / nbins * np.pi)
                    dx = radius * sin(float(o) / nbins * np.pi)
                    rr, cc = draw.line(int(centre[0] - dy),
                                       int(centre[1] - dx),
                                       int(centre[0] + dy),
                                       int(centre[1] + dx))
                    hog_image[rr, cc] += orientation_histogram[x, y, o]
    
    n_blocksx = (n_cells_x - bx) + 1
    n_blocksy = (n_cells_y - by) + 1
    normalised_blocks = np.zeros((n_blocksx, n_blocksy, bx, by, nbins))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[x:x + bx, y:y + by, :]
            eps = 1e-5
            normalised_blocks[x, y, :] = block / np.sqrt(block.sum()**2 + eps)

    if flatten:
        orientation_histogram = orientation_histogram.reshape(orientation_histogram.shape[0], -1)

    if visualise:
        return orientation_histogram, hog_image
    else:
        return orientation_histogram



def histogram_from_gradient(gx, gy, cell_size=(8, 8), max_angle=180, nbins=9):
    """ Computes the histogram of oriented gradient using the gradients gx and gy
    """
    magnitude, orientation = magnitude_orientation(gx, gy, max_angle)
    return build_histogram(magnitude, orientation, cell_size=cell_size, max_angle=max_angle, nbins=nbins)



def hog(image, cell_size=(8, 8), cells_per_blocks=(2, 2), max_angle=180, nbins=9, visualise=False):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

    1. computing the gradient image in x and y
    2. computing gradient histograms
    3. normalising across blocks
    4. flattening into a feature vector if flatten=True

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    nbins : int, optional, default is 9
        Number of orientation bins.
    cell_size : 2 tuple (int, int), optional, default is (8, 8)
        Size (in pixels) of a cell.
    cells_per_block : 2 tuple (int,int), optional, default is (2, 2)
        Number of cells in each block.
    visualise : bool, optional, default is False
        Also return an image of the HOG.
    flatten: bool, optional, default is True

    Returns
    -------
    hog : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
    Human Detection, IEEE Computer Society Conference on Computer
    Vision and Pattern Recognition 2005 San Diego, CA, USA

    """
    gx, gy = gradient(image)
    magnitude, orientation = magnitude_orientation(gx, gy, max_angle)
    return build_histogram(magnitude, orientation, cell_size=cell_size,
         max_angle=max_angle, nbins=nbins, visualise=visualise)

