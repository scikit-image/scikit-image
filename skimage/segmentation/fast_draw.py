import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg as cg_solver

from ..transform import resize
from ..measure import label as find_label
from ..morphology import binary_dilation, disk
from ..segmentation import random_walker
from ..color import rgb2grey
from .. import img_as_float

from .._shared.utils import warn

class FastDRaW():
    """A fast segmentation algorithm based on the random walker algorithm.
        
    FastDRaW implemented for 2D images as described in [1].
    The algorithm performs in a two-step segmetnation. In the first step, a
    random walker segmentation is performed on a small (down-sampled)
    version of the image to obtain a coarse segmentation contour. In the
    second step, the result is refined by applying a second random walker
    segmentation over a narrow strip around the coarse contour.
    
    Parameters
    ----------
    image : array_like
        Image to be segmented. If `image` is multi-channel it will
        be converted to gray-level image before segmentation.
    beta : float
        Penalization coefficient for the random walker motion
        (the greater `beta`, the more difficult the diffusion).
    downsampled_size : int, default 100
        The size of the down-sampled image. Should be smaller than the
        size of the original image. Recommended values between 100 and 200.
    tol : float
        tolerance to achieve when solving the linear system, in
        cg' and 'cg_mg' modes.
    return_full_prob : bool, default False
        If True, the probability that a pixel belongs to each of the labels
        will be returned, instead of boolean array.
    
    See also
    --------
    skimage.segmentation.random_walker: random walker segmentation
        The original random walker algorithm.
        
    References
    ----------
    [1] H.-E. Gueziri, L. Lakhdar, M. J. McGuffin and C. Laporte,
    "FastDRaW - Fast Delineation by Random Walker: application to large
    images", MICCAI Workshop on Interactive Medical Image Computing (IMIC),
    Athens, Greece, (2016).
    
    Examples
    --------
    >>> from skimage.data import coins
    >>> import matplotlib.pyplot as plt
    >>> image = coins()
    >>> labels = np.zeros_like(image)
    >>> labels[[129, 199], [155, 155]] = 1 # label some pixels as foreground
    >>> labels[[162, 224], [131, 184]] = 2 # label some pixels as background
    >>> fastdraw = FastDRaW(image, beta=100, downsampled_size=100)
    >>> segm = fastdraw.update(labels)
    >>> plt.imshow(image,'gray')
    >>> plt.imshow(segm, alpha=0.7)
    """
    
    def __init__(self, image, beta=300, downsampled_size=100,
                 tol=1.e-3, return_full_prob=False):
        
        assert (beta > 0), 'beta should be positive.'
        self.beta = beta
        self.return_full_prob = return_full_prob
        self.image = rgb2grey(image)
        ## It is important to normalize the data between [0,1] so that beta
        ## makes sens
        self.image = img_as_float(self.image)
        
        if downsampled_size > min(self.image.shape):
            warn('The size of the downsampled image if larger than the '
                 'original image. The computation time could be affected.')
        
        ## Compute the graph's Laplacian for both the full resolution `L` 
        ## and the down-sampled `ds_L` images
        self.L = self._buildGraph(self.image)
        ratio = float(self.image.shape[0])/self.image.shape[1]
        self.dim = (int(downsampled_size*ratio), downsampled_size)
        self.ds_image = resize(self.image, self.dim, order=0,
                                 preserve_range=True)
        self.ds_L = self._buildGraph(self.ds_image)
        
        ## Initialize the ROI to zero
        self.ds_maskROI = np.zeros_like(self.ds_image, dtype=np.bool)
        ## `full_to_ds_ratio` is used to convert labels from full resolution
        ## image to down-sampled image
        self.full_to_ds_ratio = (float(self.dim[0])/self.image.shape[0],
                                 float(self.dim[1])/self.image.shape[1])

    def _buildGraph(self, image):
        """Builds the graph: vertices, edges and weights from the `image` 
        and returns the graph's Laplacian `L`.
        """
        # Building the graph: vertices, edges and weights
        nsize = reduce(lambda x, y : x*y, image.shape,1)
        vertices = np.arange(nsize).reshape(image.shape)
        edges_right = np.vstack((vertices[:, :-1].ravel(),
                                 vertices[:, 1:].ravel()))
        edges_down = np.vstack((vertices[:-1].ravel(),
                                vertices[1:].ravel()))
        edges = np.hstack((edges_right, edges_down))
        
        gr_right = np.abs(image[:, :-1] - image[:, 1:]).ravel()
        gr_down = np.abs(image[:-1] - image[1:]).ravel()
        weights = np.exp(-self.beta * np.r_[gr_right, gr_down]**2)+1e-6
        
        # Compute the graph's Laplacian L
        pixel_nb = edges.max() + 1
        diag = np.arange(pixel_nb)
        i_indices = np.hstack((edges[0], edges[1]))
        j_indices = np.hstack((edges[1], edges[0]))
        data = np.hstack((-weights, -weights))
        lap = coo_matrix((data, (i_indices, j_indices)),
                                shape=(pixel_nb, pixel_nb))
        connect = - np.ravel(lap.sum(axis=1))
        lap = coo_matrix((np.hstack((data, connect)),
                         (np.hstack((i_indices, diag)),
                         np.hstack((j_indices, diag)))),
                         shape=(pixel_nb, pixel_nb))
        L = lap.tocsr()
        
        return L
        
    def _check_parameters(self, labels, target_label):
        if target_label not in np.unique(labels):
            warn('The target label '+str(target_label)+ \
                 ' does not match any label')
            return 1
        if (labels != 0).all():
            warn('The segmentation is computed on the unlabeled area '
                 '(labels == 0). No zero valued areas in labels were '
                 'found. Returning provided labels.')
            return -1
        return 1
                
    def _compute_relevance_map(self, labels):
        """Computes the relevance map from labels and initialize 
        down-sampled label image `ds_labels`.
        
        The relevance map assumes that the object boundary is more likely to
        be located somwhere between different label categories, i.e. two labels
        of the same category generate a low energy, where two labels of 
        different categories generate high energy, therefore precluding 
        redundent label information. The relevance map is computed using the 
        sum of the distance transforms for each label category."""
        
        self.ds_labels = np.zeros(self.dim)
        ds_relevance_map = 0
        for i in np.unique(labels):
            if i != 0:
                # 2.1- Compute the coarse label image
                y,x = np.where(labels == i)
                self.ds_labels[np.int32(y*self.full_to_ds_ratio[0]),
                          np.int32(x*self.full_to_ds_ratio[1])] = i
                # 2.2- Compute the energy map
                M = np.ones_like(self.ds_labels)
                M[self.ds_labels == i] = 0
                distance_map = distance_transform_edt(M)
                ds_relevance_map +=  distance_map
        
        # 2.3- Normalize the energy map and compute the ROI
        ds_relevance_map = ds_relevance_map / ds_relevance_map.max()
        return ds_relevance_map
        
    def _coarse_random_walker(self, target_label):
        """Performs a coarse random walker segmentation on the down-sampled
        image.
        
        Parameters
        ----------
        target_label : int
            The label category to comput the segmentation for. `labels` should
            contain at least one pixel with value `target_label`
        
        Returns
        -------
        ds_probability : ndarray of the same size as `ds_image`
            Array of the probability between [0.0, 1.0], of each pixel
            to belong to `target_label`
        """
        unlabeled = np.ravel_multi_index(np.where((self.ds_labels == 0) & \
                        (self.ds_maskROI)), self.ds_labels.shape)
        labeled = np.ravel_multi_index(np.where((self.ds_labels > 0) & \
                        (self.ds_maskROI)), self.ds_labels.shape)
        # 3.1- Preparing the right handside of the equation BT xs
        B = self.ds_L[unlabeled][:, labeled]
        mask = self.ds_labels.flatten()[labeled] == target_label
        fs = csr_matrix(mask)
        fs = fs.transpose()
        rhs = B * fs
        # 3.2- Preparing the left handside of the equation Lu
        Lu = self.ds_L[unlabeled][:, unlabeled]
        # 3.3- Solve the linear equation Lu xu = -BT xs
        xu = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]

        ds_probability = np.zeros_like(self.ds_labels, dtype=np.float32)
        ds_probability[(self.ds_labels == 0) & (self.ds_maskROI)] = xu
        ds_probability[(self.ds_labels == target_label) & (self.ds_maskROI)] = 1
        
        return ds_probability
        
    def _refinement_random_walker(self, target_label, labels):
        """Performs a random walker segmentation over a small region 
        `self.maskROI` around the coarse contour on the full resolution image. 
        
        Requires `target_label` and `labels`
        
        Returns
        -------
        probability : ndarray of the same size as `image`
            Array of the probability between [0.0, 1.0], of each pixel
            to belong to `target_label`
        """
        labeledImage = find_label(self.maskROI, background=True)
        ds_added_labels = self.ds_labels
        # for pixels outside the refinement region (ring), if their connected
        # region contains `target_label` pixels, than all pixels of the region 
        # should be labeled as `target_label`.
        # TODO : this code could be optimized
        for area in np.unique(labeledImage):
            if area != -1:
                if target_label in self.ds_labels[labeledImage == area]:
                    ds_added_labels[labeledImage == area] = target_label
        
        added_labels = resize(ds_added_labels, labels.shape, order=0,
                              preserve_range=True)

        self.maskROI = resize(self.maskROI, labels.shape, order=0,
                              preserve_range=True)
        self.maskROI = self.maskROI.astype(np.bool)
        
        # Extract labelled and unlabelled vertices
        m_unlabeled = (added_labels == 0) & (self.maskROI)
        m_foreground = (added_labels == target_label)
        
        unlabeled = np.ravel_multi_index(np.where(m_unlabeled), labels.shape)
        labeled = np.ravel_multi_index(np.where((m_foreground) | \
                                 (added_labels > 0)), labels.shape)

        # Preparing the right handside of the equation BT xs
        B = self.L[unlabeled][:, labeled]
        mask = (added_labels[added_labels > 0]).flatten() == target_label
        fs = csr_matrix(mask).transpose()
        rhs = B * fs
        
        # Preparing the left handside of the equation Lu
        Lu = self.L[unlabeled][:, unlabeled]
        
        # Solve the linear equation Lu xu = -BT xs
        xu = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]
        
        probability = np.zeros_like(labels, dtype=np.float32)
        probability[m_unlabeled] = xu
        probability[m_foreground] = 1
        
        return probability
        
    def update(self, labels, target_label=1, k=1):
        """Updates the segmentation according to `labels` using the
        FastDRaW algorithm.
        The segmentation is computed in two stages. (1) coputes a coarse 
        segmentation on a down-sampled version of the image, (2) refines the 
        segmentation on the original image.
        
        Parameters
        ----------
        labels : array of ints, of same shape as `image`
            Array of seed markers labeled with different positive integers
            (each label category is represented with an integer value). 
            Zero-labeled pixels represent unlabeled pixels.
        target_label : int
            The label category to comput the segmentation for. `labels` should
            contain at least one pixel with value `target_label`
        k : float
            Control the size of the region of interest (ROI). Large positive
            value of `k` allows a larger ROI.
        
        Returns
        -------
        output : ndarray
            * If `return_full_prob` is False, array of bools of same shape as
              `image`, in which pixels have been labeled True if they belong
              to `target_label`, and False otherwise.
            * If `return_full_prob` is True, array of floats of same shape as
              `image`. in witch each pixel is assigned the probability to
              belong to `target_label`.
          """
        ## 1- Checking if inputs are valide
        _err = self._check_parameters(labels, target_label)
        if _err == -1:
            segm = labels == target_label
            if self.return_full_prob:
                return segm.astype(np.float)
            else:
                return segm
        
        ## 2- Create down-sampled (coarse) image size 
        ## and compute the energy map
        ds_relevance_map = self._compute_relevance_map(labels)
        
        # Threshold the energy map and append new region to the existing ROI
        threshold = ds_relevance_map.mean() + k*ds_relevance_map.std()
        self.ds_maskROI = self.ds_maskROI | (ds_relevance_map <= threshold)
    
        ## 3- Performe a corse RW segmentation on the down-sampled image
        ds_probability = self._coarse_random_walker(target_label) 
        
        # Compute the corse segmentation result 
        mask = ds_probability >= 0.5
        mask = (binary_dilation(mask, disk(1)) - mask).astype(np.bool)
        # Compute the refinement region around the corse result
        self.maskROI = binary_dilation(mask, disk(3))
#        relevance_map = distance_transform_edt(mask != 1)
#        self.maskROI = (relevance_map <= 3)
            
        ## 4- Performe a fine RW segmentation on the full resolution image
        ##    only on the refinement region
        probability = self._refinement_random_walker(target_label, labels)
        
        # 5- threshold the probability map above 0.5
        if self.return_full_prob:
            return  probability
        else:
            segm = (probability >= 0.5)
            return segm
     
