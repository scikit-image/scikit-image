#!/usr/bin/python
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import cg as cg_solver

from ..transform import resize as skresize
from ..measure import label as find_label
from ..morphology import dilation, disk
from ..segmentation import random_walker
from ..color import rgb2grey

from .._shared.utils import warn

class FastDRaW():
    
    def __init__(self, image, beta=300, downsampled_size=100,
                 tol=1.e-3, return_full_prob=False):
        """FastDRaW - Fast Delineation by Random walker algorithm.
        
        FastDRaW implemented for 2D images as described in
        H.-E. Gueziri et al. "FastDRaW - Fast Delineation by Random Walker:
        application to large images", MICCAI Workshop on Interactive Medical
        Image Computation (IMIC), Athens, Greece, (2016).
        
        Parameters
        ----------
        image : array_like
            Image to be segmented. If `image` is multi-channel it will
            be converted to gray-level image before segmentation.
        beta : float
            Penalization coefficient for the random walker motion
            (the greater `beta`, the more difficult the diffusion).
        downsampled_size : integer, default 100
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
            
        Examples
        --------
        >>> from skimage.data import coins
        >>> image = coins()
        >>> labels = np.zeros_like(image)
        >>> labels[[129, 199], [155, 155]] = 1
        >>> labels[[162, 224], [131, 184]] = 2
        >>> fastdraw = FastDRaW(image, beta=100, downsampled_size=100)
        >>> segm = fastdraw.update(labels)
        >>> imshow(image,'gray')
        >>> imshow(segm, alpha=0.7)
    """
    
        assert (beta > 0), 'beta should be positive.'
        self.beta = beta
        self.return_full_prob = return_full_prob
        self.image = rgb2grey(image)
        ## It is important to normalize the data between [0,1] so that beta
        ## makes sens
        self.image = self.image / 255.0
        
        if downsampled_size > min(self.image.shape):
            warn('The size of the downsampled image if larger than the '
                 'original image. The computation time could be affected.')
        
        ## Compute the graph's Laplacian for both the full resolution `L` and the 
        ## down-sampled `ds_L` images
        self.L = self._buildGraph(self.image)
        ratio = float(self.image.shape[0])/self.image.shape[1]
        self.dim = (int(downsampled_size*ratio), downsampled_size)
        self.ds_image = skresize(self.image, self.dim, order=0,
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
        nsize = reduce(lambda x,y: x*y,image.shape,1)
        vertices = np.arange(nsize).reshape(image.shape)
        edges_right = np.vstack((vertices[:, :-1].ravel(),
                                 vertices[:, 1:].ravel()))
        edges_down = np.vstack((vertices[:-1].ravel(),
                                vertices[1:].ravel()))
        edges = np.hstack((edges_right, edges_down))
        
        gr_right = np.abs(image[:, :-1] - image[:, 1:]).ravel()
        gr_down = np.abs(image[:-1] - image[1:]).ravel()
        weights = np.exp(- self.beta * np.r_[gr_right, gr_down]**2)+1e-6
        
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
        
    def update(self, labels, target_label=1):
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
        target_label : integer
            The label category to comput the segmentation for. `labels` should
            contain at least one pixel with value `target_label`
        
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
        if target_label not in np.unique(labels):
            warn('The target label '+str(target_label)+ \
                 ' does not match any label')
        if (labels != 0).all():
            warn('The segmentation is computed on the unlabeled area '
                 '(labels == 0). No zero valued areas in labels were '
                 'found. Returning provided labels.')
            segm = labels == target_label
            if self.return_full_prob:
                return segm.astype(np.float)
            else:
                return segm
        
        ## 2- Create down-sampled (coarse) image size and compute the energy map
        ds_labels = np.zeros(self.dim)
        ds_entropyMap = 0
        for i in np.unique(labels):
            if i != 0:
                # 2.1- Compute the coarse label image
                x,y = np.where(labels == i)
                ds_labels[np.int32(x*self.full_to_ds_ratio[0]),
                          np.int32(y*self.full_to_ds_ratio[1])] = i
                # 2.2- Compute the energy map
                M = np.ones_like(ds_labels)
                M[ds_labels == i] = 0
                distMap = distance_transform_edt(M)
                ds_entropyMap +=  distMap
        
        # 2.3- Normalize the energy map and compute the ROI
        ds_entropyMap = ds_entropyMap / ds_entropyMap.max()
        threshold = ds_entropyMap.mean() + ds_entropyMap.std()
        self.ds_maskROI = self.ds_maskROI | (ds_entropyMap <= threshold)
    
        ## 3- Performe a corse RW segmentation on the down-sampled image
        unlabeled = np.ravel_multi_index(np.where((ds_labels == 0) & \
                        (self.ds_maskROI)), ds_labels.shape)
        labeled = np.ravel_multi_index(np.where((ds_labels > 0) & \
                        (self.ds_maskROI)), ds_labels.shape)
        # 3.1- Preparing the right handside of the equation BT xs
        B = self.ds_L[unlabeled][:, labeled]
        mask = ds_labels.flatten()[labeled] == target_label
        fs = csr_matrix(mask)
        fs = fs.transpose()
        rhs = B * fs
        # 3.2- Preparing the left handside of the equation Lu
        Lu = self.ds_L[unlabeled][:, unlabeled]
        # 3.3- Solve the linear equation Lu xu = -BT xs
        probability = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]

        ds_proba = np.zeros_like(ds_labels, dtype=np.float32)
        ds_proba[(ds_labels == 0) & (self.ds_maskROI)] = probability
        ds_proba[(ds_labels == target_label) & (self.ds_maskROI)] = 1   
    
        # 3.4- Compute the corse segmentation result and the refinement region
        #      around the corse result
        mask = ds_proba >= 0.5
        mask = (dilation(mask, disk(1)) - mask).astype(np.bool)
        
        entropyMap = distance_transform_edt(mask != 1)
        self.maskROI = (entropyMap <= 3)
            
        ## 4- Performe a fine RW segmentation on the full resolution image
        ##    only on the refinement region
        labeledImage = find_label(self.maskROI, background=True)
        labeledImage = skresize(labeledImage, labels.shape, order=0,
                                preserve_range=True)
        labeledImage = labeledImage.astype(np.int8)
        
        self.maskROI = skresize(self.maskROI, labels.shape, order=0,
                                preserve_range=True)
        self.maskROI = self.maskROI.astype(np.bool)
        
        # 4.1- Extract labelled and unlabelled vertices
        m_unlabeled = (labels == 0) & (self.maskROI)
        m_foreground = (labels == target_label) | (labeledImage >= 1)
        
        unlabeled = np.ravel_multi_index(np.where(m_unlabeled), labels.shape)
        labeled = np.ravel_multi_index(np.where((labels > 0) | \
                                (labeledImage >= 0)), labels.shape)
        
        # 4.2- Preparing the right handside of the equation BT xs
        B = self.L[unlabeled][:, labeled]
        mask = (m_foreground).flatten()[labeled]
        fs = csr_matrix(mask).transpose()
        rhs = B * fs
        
        # 4.3- Preparing the left handside of the equation Lu
        Lu = self.L[unlabeled][:, unlabeled]
        
        # 4.4- Solve the linear equation Lu xu = -BT xs
        probability = cg_solver(Lu, -rhs.todense(), tol=1e-3, maxiter=120)[0]
        
        x0 = np.zeros_like(labels, dtype=np.float32)
        x0[m_unlabeled] = probability
        x0[m_foreground] = 1
        
        # 5- threshold the probability map above 0.5
        if self.return_full_prob:
            return  x0
        else:
            segm = (x0 >= 0.5)
            return segm
        
