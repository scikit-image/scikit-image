# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# distutils: language = c++


import numpy as np
cimport numpy as cnp
cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from skimage._shared.transform cimport integrate

from skimage._shared.interpolation cimport round, fmax, fmin

from cython.parallel import prange
from ..color import rgb2gray
from ..transform import integral_image
import xml.etree.ElementTree as ET
from ._texture cimport _multiblock_lbp
import math

cnp.import_array()

# Struct for storing a single detection.
cdef struct Detection:

    int r
    int c
    int width
    int height


# Struct for storing cluster of rectangles that represent detections.
# As the rectangles are dynamically added, the sum of row, col positions,
# width and heights are stored with the count of rectangles that belong
# to this cluster. This way,  we don't have to store all the rectangles
# information as array and the average of all detections in a cluster
# can be easily computed in a constant time.
cdef struct DetectionsCluster:

    int r_sum
    int c_sum
    int width_sum
    int height_sum
    int count


# Struct for storing multi-block binary pattern position.
# Defines the parameters of multi-block binary pattern feature.
# Read more in skimage.feature.texture.multiblock_lbp.
cdef struct MBLBP:

    Py_ssize_t r
    Py_ssize_t c
    Py_ssize_t width
    Py_ssize_t height


# Struct for storing information about trained MBLBP feature.
# Feature_id contains an index to array where the parameters of MBLBP features
# are stored using MBLBP struct. Index is used because some stages in cascade
# can have repeating features. The lut_idx contains an index to a look-up table
# which gives, depending on the computed value of a feature, an answer whether
# an object is present in the current detection window. Based on the value of
# look-up table (0 or 1) positive(right) or negative(left) weight is added to
# the overall score of a stage.
cdef struct MBLBPStump:

    Py_ssize_t feature_id
    Py_ssize_t lut_idx
    cnp.float32_t left
    cnp.float32_t right


# Struct for storing a stage of classifier which itself consists of
# MBLBPStumps. It has the index that maps to the starting stump and amount of
# stumps that belong to a stage after this index. In each stage all the stumps
# are evaluated and their output values( `left` or `right` depending on the
# input) are summed up and compared to the threshold. If the value is higher
# than the threshold, the stage is passed and Cascade classifier goes to the
# next stage. If all the stages are passed, the object is predicted to be
# present in the input image patch.
cdef struct Stage:

    Py_ssize_t first_idx
    Py_ssize_t amount
    cnp.float32_t threshold


cdef vector[Detection] _group_detections(vector[Detection] detections,
                                         cnp.float32_t intersection_score_threshold=0.5,
                                         int min_neighbour_number=4):
    """Group similar detections into a single detection and eliminate weak
    (non-overlapping) detections.

    We assume that a true detection is characterized by a high number of
    overlapping detections. Such detections are isolated and gathered into
    one cluster. The average of each cluster is returned. Averaging means
    that the row and column positions of top left corners and the width
    and height parameters of each rectangle in a cluster are used to compute
    values of average rectangle that will represent cluster.

    Parameters
    ----------
    detections : vector[Detection]
        A cluster of detections.
    min_neighbour_number : int
        Minimum amount of intersecting detections in order for detection
        to be approved by the function.
    intersection_score_threshold : cnp.float32_t
        The minimum value of value of ratio
        (intersection area) / (small rectangle ratio) in order to merge
        two rectangles into one cluster.

    Returns
    -------
    output : vector[Detection]
        The grouped detections.
    """

    cdef:
        Detection mean_detection
        vector[DetectionsCluster] clusters
        vector[int] clusters_scores
        Py_ssize_t nr_of_clusters
        Py_ssize_t current_detection_nr
        Py_ssize_t current_cluster_nr
        Py_ssize_t nr_of_detections = detections.size()
        Py_ssize_t best_cluster_nr
        bint new_cluster
        cnp.float32_t best_score
        cnp.float32_t intersection_score

    # Check if detections array is not empty.
    # Push first detection as first cluster.
    if nr_of_detections:
        clusters.push_back(cluster_from_detection(detections[0]))

    for current_detection_nr in range(1, nr_of_detections):

        best_score = intersection_score_threshold
        best_cluster_nr = 0
        new_cluster = True

        nr_of_clusters = clusters.size()

        for current_cluster_nr in range(nr_of_clusters):

            mean_detection = mean_detection_from_cluster(
                                    clusters[current_cluster_nr])

            intersection_score = rect_intersection_score(
                                        detections[current_detection_nr],
                                        mean_detection)

            if intersection_score > best_score:

                new_cluster = False
                best_cluster_nr = current_cluster_nr
                best_score = intersection_score

        if new_cluster:

            clusters.push_back(cluster_from_detection(
                                    detections[current_detection_nr]))
        else:

            clusters[best_cluster_nr] = update_cluster(
                                            clusters[best_cluster_nr],
                                            detections[current_detection_nr])

    clusters = threshold_clusters(clusters, min_neighbour_number)
    return get_mean_detections(clusters)


cdef DetectionsCluster update_cluster(DetectionsCluster cluster,
                                      Detection detection):
    """Updated the cluster by adding new detection.

    Updates the cluster by adding new detection to it. The added
    detection contributes to the mean value of the cluster.

    Parameters
    ----------
    cluster : DetectionsCluster
        A cluster of detections.
    detection : Detection
        The detection to be added to cluster.

    Returns
    -------
    updated_cluster : DetectionsCluster
        The updated cluster.
    """

    cdef DetectionsCluster updated_cluster = cluster

    updated_cluster.r_sum += detection.r
    updated_cluster.c_sum += detection.c
    updated_cluster.width_sum += detection.width
    updated_cluster.height_sum += detection.height
    updated_cluster.count += 1

    return updated_cluster


cdef Detection mean_detection_from_cluster(DetectionsCluster cluster):
    """Compute the mean detection from the cluster.

    Returns the mean detection computed from the all rectangles that
    belong to current cluster.

    Parameters
    ----------
    cluster : DetectionsCluster
        A cluster of detections.

    Returns
    -------
    mean : Detection
        The mean detection.
    """

    cdef Detection mean

    mean.r = cluster.r_sum / cluster.count
    mean.c = cluster.c_sum / cluster.count
    mean.width = cluster.width_sum / cluster.count
    mean.height = cluster.height_sum / cluster.count

    return mean


cdef DetectionsCluster cluster_from_detection(Detection detection):
    """Create a cluster from a single detection.

    Creates a cluster with count one and values that are taken from detection.

    Parameters
    ----------
    detection : Detection
        A single detection.

    Returns
    -------
    new_cluster : DetectionsCluster
        The cluster struct that was created from detection.
    """

    cdef DetectionsCluster new_cluster

    new_cluster.r_sum = detection.r
    new_cluster.c_sum = detection.c
    new_cluster.width_sum = detection.width
    new_cluster.height_sum = detection.height
    new_cluster.count = 1

    return new_cluster


cdef vector[DetectionsCluster] threshold_clusters(vector[DetectionsCluster] clusters,
                                                  int count_threshold):
    """Threshold clusters depending on the amount of rectangles in them.

    Only the clusters with the amount of rectangles greater than the threshold
    are left.

    Parameters
    ----------
    clusters : vector[DetectionsCluster]
        Array of rectangles clusters.
    count_threshold : int
        The threshold number of rectangles that is used.

    Returns
    -------
    output : vector[DetectionsCluster]
        The array of clusters that satisfy the threshold criteria.
    """

    cdef:
        Py_ssize_t clusters_amount
        Py_ssize_t current_cluster
        vector[DetectionsCluster] output

    clusters_amount = clusters.size()

    for current_cluster in range(clusters_amount):

        if clusters[current_cluster].count >= count_threshold:
            output.push_back(clusters[current_cluster])

    return output


cdef vector[Detection] get_mean_detections(vector[DetectionsCluster] clusters):
    """Computes the mean of each cluster of detections in the array.

    Each cluster is replaced with a single detection that represents
    the mean of the cluster, computed from the rectangles that belong
    to the cluster.

    Parameters
    ----------
    clusters : vector[DetectionsCluster]
        Array of rectangles clusters.

    Returns
    -------
    detections : vector[Detection]
        The array of mean detections. Each detection represent mean
        for one cluster.
    """

    cdef:
        Py_ssize_t current_cluster
        Py_ssize_t clusters_amount = clusters.size()
        vector[Detection] detections

    detections.resize(clusters_amount)

    for current_cluster in range(clusters_amount):
         detections[current_cluster] = mean_detection_from_cluster(clusters[current_cluster])

    return detections


cdef cnp.float32_t rect_intersection_area(Detection rect_a, Detection rect_b):
    """Computes the intersection area of two rectangles.


    Parameters
    ----------
    rect_a : Detection
        Struct of the first rectangle.
    rect_a : Detection
        Struct of the second rectangle.

    Returns
    -------
    result : cnp.float32_t
        The intersection score area.
    """

    cdef:
        Py_ssize_t r_a_1 = rect_a.r
        Py_ssize_t r_a_2 = rect_a.r + rect_a.height
        Py_ssize_t c_a_1 = rect_a.c
        Py_ssize_t c_a_2 = rect_a.c + rect_a.width

        Py_ssize_t r_b_1 = rect_b.r
        Py_ssize_t r_b_2 = rect_b.r + rect_b.height
        Py_ssize_t c_b_1 = rect_b.c
        Py_ssize_t c_b_2 = rect_b.c + rect_b.width

    return (fmax(0, fmin(c_a_2, c_b_2) - fmax(c_a_1, c_b_1)) *
            fmax(0, fmin(r_a_2, r_b_2) - fmax(r_a_1, r_b_1)))


cdef cnp.float32_t rect_intersection_score(Detection rect_a, Detection rect_b):
    """Computes the intersection score of two rectangles.

    The score is computed by dividing the intersection area of rectangles
    by the area of the rectangle with the smallest area.

    Parameters
    ----------
    rect_a : Detection
        Struct of the first rectangle.
    rect_a : Detection
        Struct of the second rectangle.

    Returns
    -------
    result : cnp.float32_t
        The intersection score. The number in the interval ``[0, 1]``.
        1 means rectangles fully intersect, 0 means they don't.
    """

    cdef:
        cnp.float32_t intersection_area
        cnp.float32_t union_area
        cnp.float32_t smaller_area
        cnp.float32_t area_a = rect_a.height * rect_a.width
        cnp.float32_t area_b = rect_b.height * rect_b.width

    intersection_area = rect_intersection_area(rect_a, rect_b)

    smaller_area = area_a if area_b > area_a else area_b

    return intersection_area / smaller_area


cdef class Cascade:
    """Class for cascade of classifiers that is used for object detection.

    The main idea behind cascade of classifiers is to create classifiers
    of medium accuracy and ensemble them into one strong classifier
    instead of just creating a strong one. The second advantage of cascade
    classifier is that easy examples can be classified only by evaluating
    some of the classifiers in the cascade, making the process much faster
    than the process of evaluating a one strong classifier.

    Attributes
    ----------
    eps : cnp.float32_t
        Accuracy parameter. Increasing it, makes the classifier detect less
        false positives but at the same time the false negative score increases.
    stages_number : Py_ssize_t
        Amount of stages in a cascade. Each cascade consists of stumps i.e.
        trained features.
    stumps_number : Py_ssize_t
        The overall amount of stumps in all the stages of cascade.
    features_number : Py_ssize_t
        The overall amount of different features used by cascade.
        Two stumps can use the same features but has different trained
        values.
    window_width : Py_ssize_t
        The width of a detection window that is used. Objects smaller than
        this window can't be detected.
    window_height : Py_ssize_t
        The height of a detection window.
    stages : Stage*
        A pointer to the C array that stores stages information using a
        Stage struct.
    features : MBLBP*
        A pointer to the C array that stores MBLBP features using an MBLBP
        struct.
    LUTs : cnp.uint32_t*
        A pointer to the C array with look-up tables that are used by trained
        MBLBP features (MBLBPStumps) to evaluate a particular region.

    Notes
    -----
    The cascade approach was first described by Viola and Jones [1]_, [2]_,
    although these initial publications used a set of Haar-like features. This
    implementation instead uses multi-scale block local binary pattern (MB-LBP)
    features [3]_.

    References
    ----------
    .. [1] Viola, P. and Jones, M. "Rapid object detection using a boosted
           cascade of simple features," In: Proceedings of the 2001 IEEE
           Computer Society Conference on Computer Vision and Pattern
           Recognition. CVPR 2001, pp. I-I.
           :DOI:`10.1109/CVPR.2001.990517`
    .. [2] Viola, P. and Jones, M.J, "Robust Real-Time Face Detection",
           International Journal of Computer Vision 57, 137–154 (2004).
           :DOI:`10.1023/B:VISI.0000013087.49260.fb`
    .. [3] Liao, S. et al. Learning Multi-scale Block Local Binary Patterns for
           Face Recognition. International Conference on Biometrics (ICB),
           2007, pp. 828-837. In: Lecture Notes in Computer Science, vol 4642.
           Springer, Berlin, Heidelberg.
           :DOI:`10.1007/978-3-540-74549-5_87`
    """

    cdef:
        public cnp.float32_t eps
        public Py_ssize_t stages_number
        public Py_ssize_t stumps_number
        public Py_ssize_t features_number
        public Py_ssize_t window_width
        public Py_ssize_t window_height
        Stage* stages
        MBLBPStump* stumps
        MBLBP* features
        cnp.uint32_t* LUTs

    def __dealloc__(self):

        # Free the memory that was used for c-arrays.
        free(self.stages)
        free(self.stumps)
        free(self.features)
        free(self.LUTs)

    def __init__(self, xml_file, eps=1e-5):
        """Initialize cascade classifier.

        Parameters
        ----------
        xml_file : file's path or file's object
            A file in a OpenCv format from which all the cascade classifier's
            parameters are loaded.
        eps : cnp.float32_t
            Accuracy parameter. Increasing it, makes the classifier
            detect less false positives but at the same time the false
            negative score increases.

        """

        self._load_xml(xml_file, eps)

    cdef bint classify(self, cnp.float32_t[:, ::1] int_img, Py_ssize_t row,
                       Py_ssize_t col, cnp.float32_t scale) nogil:
        """Classify the provided image patch i.e. check if the classifier
        detects an object in the given image patch.

        The function takes the original window size that is stored in the
        trained file, scales it and places in the specified part of the
        provided image, carries out classification and gives a binary result.

        Parameters
        ----------
        int_img : cnp.float32_t[:, ::1]
            Memory-view to integral image.
        row : Py_ssize_t
            Row coordinate of the rectangle in the given image to classify.
            Top left corner of window.
        col : Py_ssize_t
            Column coordinate of the rectangle in the given image to classify.
            Top left corner of window.
        scale : cnp.float32_t
            The scale by which the search window is multiplied.
            After multiplication the result is rounded to the lowest integer.

        Returns
        -------
        result : int
            The binary output that takes only 0 or 1. Gives 1 if the classifier
            detects the object in specified region and 0 otherwise.
        """

        cdef:
            cnp.float32_t stage_threshold
            cnp.float32_t stage_points
            int lbp_code
            int bit
            Py_ssize_t stage_number
            Py_ssize_t weak_classifier_number
            Py_ssize_t feature_number
            Py_ssize_t features_number
            Py_ssize_t stumps_number
            Py_ssize_t first_stump_idx
            Py_ssize_t lut_idx
            Py_ssize_t r, c, widht, height
            cnp.uint32_t[::1] current_lut
            Stage current_stage
            MBLBPStump current_stump
            MBLBP current_feature


        for stage_number in range(self.stages_number):

            current_stage = self.stages[stage_number]
            first_stump_idx = current_stage.first_idx
            stage_points = 0

            for weak_classifier_number in range(current_stage.amount):

                current_stump = self.stumps[first_stump_idx +
                                            weak_classifier_number]

                current_feature = self.features[current_stump.feature_id]

                r = <Py_ssize_t>(current_feature.r * scale)
                c = <Py_ssize_t>(current_feature.c * scale)
                width = <Py_ssize_t>(current_feature.width * scale)
                height = <Py_ssize_t>(current_feature.height * scale)


                lbp_code = _multiblock_lbp(int_img, row + r, col + c,
                                           width, height)

                lut_idx = current_stump.lut_idx

                bit = (self.LUTs[lut_idx + (lbp_code >> 5)] >> (lbp_code & 31)) & 1

                stage_points += current_stump.left if bit else current_stump.right

            if stage_points < (current_stage.threshold - self.eps):

                return False

        return True

    def _get_valid_scale_factors(self, min_size, max_size, scale_step):
        """Get the valid scale multipliers for the original window size.

        The function takes the minimal size of window and maximum size of
        window as interval and finds all the multipliers that will give the
        windows which sizes will be not less than the min_size and not bigger
        than the max_size.

        Parameters
        ----------
        min_size : typle (int, int)
            Minimum size of window for which to search the scale factor.
        max_size : typle (int, int)
            Maximum size of window for which to search the scale factor.
        scale_step : cnp.float32_t
            The scale by which the search window is multiplied
            on each iteration.

        Returns
        -------
        scale_factors : 1-D cnp.float32_ts ndarray
            The scale factors that give the window sizes that are in the
            specified interval after multiplying the search window.
        """

        current_size = np.array((self.window_height, self.window_width))
        min_size = np.array(min_size, dtype=np.float32)
        max_size = np.array(max_size, dtype=np.float32)

        row_power_max = math.log(max_size[0]/current_size[0], scale_step)
        col_power_max = math.log(max_size[1]/current_size[1], scale_step)

        row_power_min = math.log(min_size[0]/current_size[0], scale_step)
        col_power_min = math.log(min_size[1]/current_size[1], scale_step)

        mn = max(row_power_min, col_power_min, 0)
        mx = min(row_power_max, col_power_max)

        powers = np.arange(mn, mx)

        scale_factors = np.power(scale_step, powers, dtype=np.float32)

        return scale_factors

    def _get_contiguous_integral_image(self, img):
        """Get a c-contiguous array that represents the integral image.

        The function converts the input image into the integral image in
        a format that is suitable for work of internal functions of
        the cascade classifier class. The function converts the image
        to gray-scale float representation, computes the integral image
        and makes it c-contiguous.

        Parameters
        ----------
        img : 2-D or 3-D ndarray
            Ndarray that represents the input image.

        Returns
        -------
        int_img : 2-D floats ndarray
            C-contiguous integral image of the input image.
        """
        if len(img.shape) > 2:
            img = rgb2gray(img)
        int_img = integral_image(img)
        int_img = np.ascontiguousarray(int_img, dtype=np.float32)

        return int_img


    def detect_multi_scale(self, img, cnp.float32_t scale_factor,
                           cnp.float32_t step_ratio, min_size, max_size,
                           min_neighbour_number=4,
                           intersection_score_threshold=0.5):
        """Search for the object on multiple scales of input image.

        The function takes the input image, the scale factor by which the
        searching window is multiplied on each step, minimum window size
        and maximum window size that specify the interval for the search
        windows that are applied to the input image to detect objects.

        Parameters
        ----------
        img : 2-D or 3-D ndarray
            Ndarray that represents the input image.
        scale_factor : cnp.float32_t
            The scale by which searching window is multiplied on each step.
        step_ratio : cnp.float32_t
            The ratio by which the search step in multiplied on each scale
            of the image. 1 represents the exaustive search and usually is
            slow. By setting this parameter to higher values the results will
            be worse but the computation will be much faster. Usually, values
            in the interval [1, 1.5] give good results.
        min_size : typle (int, int)
            Minimum size of the search window.
        max_size : typle (int, int)
            Maximum size of the search window.
        min_neighbour_number : int
            Minimum amount of intersecting detections in order for detection
            to be approved by the function.
        intersection_score_threshold : cnp.float32_t
            The minimum value of value of ratio
            (intersection area) / (small rectangle ratio) in order to merge
            two detections into one.

        Returns
        -------
        output : list of dicts
            Dict have form {'r': int, 'c': int, 'width': int, 'height': int},
            where 'r' represents row position of top left corner of detected
            window, 'c' - col position, 'width' - width of detected window,
            'height' - height of detected window.
        """

        cdef:
            Py_ssize_t max_row
            Py_ssize_t max_col
            Py_ssize_t current_height
            Py_ssize_t current_width
            Py_ssize_t current_row
            Py_ssize_t current_col
            Py_ssize_t current_step
            Py_ssize_t number_of_scales
            Py_ssize_t img_height
            Py_ssize_t img_width
            Py_ssize_t scale_number
            Py_ssize_t window_height = self.window_height
            Py_ssize_t window_width = self.window_width
            int result
            cnp.float32_t[::1] scale_factors
            cnp.float32_t[:, ::1] int_img
            cnp.float32_t current_scale_factor
            vector[Detection] output
            Detection new_detection

        int_img = self._get_contiguous_integral_image(img)
        img_height = int_img.shape[0]
        img_width = int_img.shape[1]

        scale_factors = self._get_valid_scale_factors(min_size,
                                                      max_size, scale_factor)
        number_of_scales = scale_factors.shape[0]

        # Initialize lock to enable thread-safe writes to the array
        # in concurrent loop.
        cdef openmp.omp_lock_t mylock

        if have_openmp:
            openmp.omp_init_lock(&mylock)


        # As the amount of work between the threads is not equal we
        # use `dynamic` schedule which enables them to use computing
        # power on demand.
        for scale_number in prange(0, number_of_scales,
                                   schedule='dynamic', nogil=True):

            current_scale_factor = scale_factors[scale_number]
            current_step = <Py_ssize_t>round(current_scale_factor * step_ratio)
            current_height = <Py_ssize_t>(window_height * current_scale_factor)
            current_width = <Py_ssize_t>(window_width * current_scale_factor)
            max_row = img_height - current_height
            max_col = img_width - current_width

            # Check if scaled detection window fits in image.
            if (max_row < 0) or (max_col < 0):
                continue

            current_row = 0
            current_col = 0

            while current_row < max_row:
                while current_col < max_col:

                    result = self.classify(int_img, current_row,
                                           current_col,
                                           scale_factors[scale_number])

                    if result:

                        new_detection = Detection()
                        new_detection.r = current_row
                        new_detection.c = current_col
                        new_detection.width = current_width
                        new_detection.height = current_height

                        if have_openmp:
                            openmp.omp_set_lock(&mylock)

                        output.push_back(new_detection)

                        if have_openmp:
                            openmp.omp_unset_lock(&mylock)

                    current_col = current_col + current_step

                current_row = current_row + current_step
                current_col = 0

        if have_openmp:
            openmp.omp_destroy_lock(&mylock)

        return list(_group_detections(output, intersection_score_threshold,
                                      min_neighbour_number))

    def _load_xml(self, xml_file, eps=1e-5):
        """Load the parameters of cascade classifier into the class.

        The function takes the file with the parameters that represent
        trained cascade classifier and loads them into class for later
        use.

        Parameters
        ----------
        xml_file : filename or file object
            File that contains the cascade classifier.
        eps : cnp.float32_t
            Accuracy parameter. Increasing it, makes the classifier
            detect less false positives but at the same time the false
            negative score increases.

        """

        cdef:
            Stage* stages_carr
            MBLBPStump* stumps_carr
            MBLBP* features_carr
            cnp.uint32_t* LUTs_carr

            cnp.float32_t stage_threshold

            Py_ssize_t stage_number
            Py_ssize_t stages_number
            Py_ssize_t window_height
            Py_ssize_t window_width

            Py_ssize_t weak_classifiers_amount
            Py_ssize_t weak_classifier_number

            Py_ssize_t feature_number
            Py_ssize_t features_number
            Py_ssize_t stump_lut_idx
            Py_ssize_t stump_idx
            Py_ssize_t i

            cnp.uint32_t[::1] lut

            MBLBP new_feature
            MBLBPStump new_stump
            Stage new_stage

        tree = ET.parse(xml_file)

        # Load entities.
        features = tree.find('.//features')
        stages = tree.find('.//stages')

        # Get the respective amounts.
        stages_number = int(tree.find('.//stageNum').text)
        window_height = int(tree.find('.//height').text)
        window_width = int(tree.find('.//width').text)
        features_number = len(features)

        # Count the stumps.
        stumps_number = 0
        for stage_number in range(stages_number):
            current_stage = stages[stage_number]
            weak_classifiers_amount = int(current_stage.find('maxWeakCount').text)
            stumps_number += weak_classifiers_amount

        # Allocate memory for data.
        features_carr = <MBLBP*>malloc(features_number * sizeof(MBLBP))
        stumps_carr = <MBLBPStump*>malloc(stumps_number * sizeof(MBLBPStump))
        stages_carr = <Stage*>malloc(stages_number*sizeof(Stage))
        # Each look-up table consists of 8 u-int numbers.
        LUTs_carr = <cnp.uint32_t*>malloc(8 * stumps_number *
                                          sizeof(cnp.uint32_t))

        # Check if memory was allocated.
        if not (features_carr and stumps_carr and stages_carr and LUTs_carr):
            free(features_carr)
            free(stumps_carr)
            free(stages_carr)
            free(LUTs_carr)
            raise MemoryError("Failed to allocate memory while parsing XML.")

        # Parse and load features in memory.
        for feature_number in range(features_number):
            params = features[feature_number][0].text.split()
            # list() is for Python3 fix here
            params = list(map(lambda x: int(x), params))
            new_feature = MBLBP(params[1], params[0], params[2], params[3])
            features_carr[feature_number] = new_feature

        stump_lut_idx = 0
        stump_idx = 0

        # Parse and load stumps, stages.
        for stage_number in range(stages_number):

            current_stage = stages[stage_number]

            # Parse and load current stage.
            stage_threshold = float(current_stage.find('stageThreshold').text)
            weak_classifiers_amount = int(current_stage.find('maxWeakCount').text)
            new_stage = Stage(stump_idx, weak_classifiers_amount,
                              stage_threshold)
            stages_carr[stage_number] = new_stage

            weak_classifiers = current_stage.find('weakClassifiers')

            for weak_classifier_number in range(weak_classifiers_amount):

                current_weak_classifier = weak_classifiers[weak_classifier_number]

                # Stump's leaf values. First negative if image is probably not
                # a face. Second positive if image is probably a face.
                leaf_values = current_weak_classifier.find('leafValues').text
                # list() is for Python3 fix here
                leaf_values = list(map(lambda x: float(x), leaf_values.split()))

                # Extract the elements only starting from second.
                # First two are useless
                internal_nodes = current_weak_classifier.find('internalNodes')
                internal_nodes = internal_nodes.text.split()[2:]

                # Extract the feature number and respective parameters.
                # The MBLBP position and size.
                feature_number = int(internal_nodes[0])
                # list() is for Python3 fix here
                lut_array = list(map(lambda x: int(x), internal_nodes[1:]))
                lut = np.asarray(lut_array, dtype='uint32')

                # Copy array to the main LUT array
                for i in range(8):
                    LUTs_carr[stump_lut_idx + i] = lut[i]

                new_stump = MBLBPStump(feature_number, stump_lut_idx,
                                       leaf_values[0], leaf_values[1])
                stumps_carr[stump_idx] = new_stump

                stump_lut_idx += 8
                stump_idx += 1

        self.eps = eps
        self.window_height = window_height
        self.window_width = window_width
        self.features = features_carr
        self.stumps = stumps_carr
        self.stages = stages_carr
        self.LUTs = LUTs_carr
        self.stages_number = stages_number
        self.features_number = features_number
        self.stumps_number = stumps_number
