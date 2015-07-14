# cython: cdivision=True
# cython: boundscheck=False
# cython: nonecheck=False
# cython: wraparound=False
# distutils: language = c++

import numpy as np
cimport numpy as cnp
cimport openmp
from skimage._shared.transform cimport integrate
from libc.stdlib cimport malloc, free
from skimage._shared.interpolation cimport round, fmax, fmin
from libcpp.vector cimport vector

from cython.parallel import prange
from skimage.color import rgb2gray
from skimage.transform import integral_image
import xml.etree.ElementTree as ET
from ...feature._texture cimport _multiblock_lbp


cdef struct DetectionsCluster:

    int r_sum
    int c_sum
    int width_sum
    int height_sum
    int count

cdef struct Detection:

    int r
    int c
    int width
    int height

cdef struct MBLBP:

    Py_ssize_t r
    Py_ssize_t c
    Py_ssize_t width
    Py_ssize_t height

cdef struct MBLBPStump:

    Py_ssize_t feature_id
    Py_ssize_t lut_idx
    float left
    float right

cdef struct Stage:

    Py_ssize_t first_idx
    Py_ssize_t amount
    float threshold

cdef vector[Detection] _post_process_detections(vector[Detection] detections, int min_neighbour_amount=4):

    cdef:
        Detection mean_detection
        vector[DetectionsCluster] clusters
        vector[int] clusters_scores
        Py_ssize_t clusters_amount
        Py_ssize_t current_detection
        Py_ssize_t current_cluster
        Py_ssize_t detections_amount = detections.size()
        Py_ssize_t best_cluster_number
        bint new_cluster
        float best_score
        float intersection_score

    # Check if detections array is not empty.
    # Push first detection as first cluster.
    if detections_amount:
        clusters.push_back(cluster_from_detection(detections[0]))

    for current_detection in range(1, detections_amount):

        best_score = 0.5
        best_cluster_number = 0
        new_cluster = True

        clusters_amount = clusters.size()

        for current_cluster in range(clusters_amount):

            mean_detection = mean_detection_from_cluster(clusters[current_cluster])
            intersection_score = rect_intersection_score(detections[current_detection], mean_detection)

            if intersection_score > best_score:

                new_cluster = False
                best_cluster_number = current_cluster
                best_score = intersection_score

        if new_cluster:

            clusters.push_back(cluster_from_detection(detections[current_detection]))
        else:

            clusters[best_cluster_number] = update_cluster(clusters[best_cluster_number],
                                                           detections[current_detection])

    clusters = threshold_clusters(clusters, min_neighbour_amount)
    return get_mean_detections(clusters)

cdef DetectionsCluster update_cluster(DetectionsCluster cluster, Detection detection):

    cdef DetectionsCluster updated_cluster = cluster

    updated_cluster.r_sum += detection.r
    updated_cluster.c_sum += detection.c
    updated_cluster.width_sum += detection.width
    updated_cluster.height_sum += detection.height
    updated_cluster.count += 1

    return updated_cluster


cdef Detection mean_detection_from_cluster(DetectionsCluster cluster):

    cdef Detection mean

    mean.r = cluster.r_sum / cluster.count
    mean.c = cluster.c_sum / cluster.count
    mean.width = cluster.width_sum / cluster.count
    mean.height = cluster.height_sum / cluster.count

    return mean

cdef DetectionsCluster cluster_from_detection(Detection detection):

    cdef DetectionsCluster new_cluster

    new_cluster.r_sum = detection.r
    new_cluster.c_sum = detection.c
    new_cluster.width_sum = detection.width
    new_cluster.height_sum = detection.height
    new_cluster.count = 1

    return new_cluster

cdef vector[DetectionsCluster] threshold_clusters(vector[DetectionsCluster] clusters, int count_threshold):

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

    cdef:
        Py_ssize_t current_cluster
        Py_ssize_t clusters_amount = clusters.size()
        vector[Detection] detections

    detections.resize(clusters_amount)

    for current_cluster in range(clusters_amount):
         detections[current_cluster] = mean_detection_from_cluster(clusters[current_cluster])

    return detections


cdef float rect_intersection_area(Detection rect_a, Detection rect_b):

    cdef:
        Py_ssize_t r_a_1 = rect_a.r
        Py_ssize_t r_a_2 = rect_a.r + rect_a.height
        Py_ssize_t c_a_1 = rect_a.c
        Py_ssize_t c_a_2 = rect_a.c + rect_a.width

        Py_ssize_t r_b_1 = rect_b.r
        Py_ssize_t r_b_2 = rect_b.r + rect_b.height
        Py_ssize_t c_b_1 = rect_b.c
        Py_ssize_t c_b_2 = rect_b.c + rect_b.width

    return fmax(0, fmin(c_a_2, c_b_2) - fmax(c_a_1, c_b_1)) * fmax(0, fmin(r_a_2, r_b_2) - fmax(r_a_1, r_b_1))

cdef float rect_intersection_score(Detection rect_a, Detection rect_b):

    cdef:
        float intersection_area
        float union_area
        float smaller_area
        float area_a = rect_a.height * rect_a.width
        float area_b = rect_b.height * rect_b.width

    intersection_area = rect_intersection_area(rect_a, rect_b)

    union_area = area_a + area_b - intersection_area

    smaller_area = area_a if area_b > area_a else area_b

    return intersection_area / smaller_area


cdef class Cascade:
    """Class for cascade classifiers that are used for object detection."""

    cdef:
        public float eps
        public Py_ssize_t stages_amount
        public Py_ssize_t stumps_amount
        public Py_ssize_t features_amount
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
        eps : float
            Accuracy parameter. Increasing it, makes the classifier detect less
            false positives but at the same time the false negative score increases.
        """

        self._load_xml(xml_file, eps)


    cdef bint classify(self, float[:, ::1] int_img, Py_ssize_t row, Py_ssize_t col, float scale) nogil:
        """Classify the provided image patch i.e. check if the classifier
        detects an object in the given image patch.

        The function takes the original window size that is stored in the
        trained file, scales it and places in the specified part of the
        provided image, carries out classification and gives a binary result.

        Parameters
        ----------
        int_img : float[:, ::1]
            Memory-view to integral image.
        row : Py_ssize_t
            Row coordinate of the rectangle in the given image to classify.
            Top left corner of window.
        col : Py_ssize_t
            Column coordinate of the rectangle in the given image to classify.
            Top left corner of window.
        scale : float
            The scale by which the search window is multiplied.
            After multiplication the result is rounded to the lowest integer.

        Returns
        -------
        result : int
            The binary output that takes only 0 or 1. Gives 1 if the classifier
            detects the object in specified region and 0 otherwise.
        """

        cdef:
            float stage_threshold
            float stage_points
            int lbp_code
            int bit
            Py_ssize_t stage_number
            Py_ssize_t weak_classifier_number
            Py_ssize_t feature_number
            Py_ssize_t features_amount
            Py_ssize_t stumps_amount
            Py_ssize_t first_stump_idx
            Py_ssize_t lut_idx
            Py_ssize_t r, c, widht, height
            cnp.uint32_t[::1] current_lut
            Stage current_stage
            MBLBPStump current_stump
            MBLBP current_feature


        for stage_number in range(self.stages_amount):

            current_stage = self.stages[stage_number]
            first_stump_idx = current_stage.first_idx
            stage_points = 0

            for weak_classifier_number in range(current_stage.amount):

                current_stump = self.stumps[first_stump_idx + weak_classifier_number]

                current_feature = self.features[current_stump.feature_id]

                r = <Py_ssize_t>(current_feature.r * scale)
                c = <Py_ssize_t>(current_feature.c * scale)
                width = <Py_ssize_t>(current_feature.width * scale)
                height = <Py_ssize_t>(current_feature.height * scale)


                lbp_code = _multiblock_lbp(int_img,
                                           row + r,
                                           col + c,
                                           width,
                                           height)

                lut_idx = current_stump.lut_idx

                bit = (self.LUTs[lut_idx + (lbp_code >> 5)] >> (lbp_code & 31)) & 1

                stage_points += current_stump.left if bit else current_stump.right

            if stage_points < (current_stage.threshold - self.eps):

                return False

        return True

    def _get_valid_scale_factors(self, min_size, max_size, scale_step):
        """Get the valid scale multipliers for the original window size
        from the trained file.

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
        scale_step : float
            The scale by which the search window is multiplied
            on each iteration.

        Returns
        -------
        scale_factors : 1-D floats ndarray
            The scale factors that give the window sizes that are in the
            specified interval after multiplying the search window.
        """

        min_size = np.array(min_size)
        max_size = np.array(max_size)

        scale_factors = []
        current_scale = 1
        current_size = np.array((self.window_height, self.window_width))

        while (current_size <= max_size).all():

            if (current_size >= min_size).all():
                scale_factors.append(current_scale)

            current_scale = current_scale * scale_step
            current_size = current_size * scale_step

        return np.array(scale_factors, dtype=np.float32)

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

        img = rgb2gray(img)
        int_img = integral_image(img)
        int_img = np.ascontiguousarray(int_img, dtype=np.float32)

        return int_img


    def detect_multi_scale(self, img, float scale_factor, float step_ratio,
                           min_size, max_size, min_neighbour_amount=4):
        """Search for the object on multiple scales of input image.

        The function takes the input image, the scale factor by which the
        searching window is multiplied on each step, minimum window size
        and maximum window size that specify the interval for the search
        windows that are applied to the input image to detect objects.

        Parameters
        ----------
        img : 2-D or 3-D ndarray
            Ndarray that represents the input image.
        scale_factor : float
            The scale by which searching window is multiplied on each step.
        step_ratio : float
            The ratio by which the search step in multiplied on each scale
            of the image. 1 represents the exaustive search and usually is
            slow. By setting this parameter to higher values the results will
            be worse but the computation will be much faster. Usually, values
            in the interval [1, 1.5] give good results.
        min_size : typle (int, int)
            Minimum size of the search window.
        max_size : typle (int, int)
            Maximum size of the search window.

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
            Py_ssize_t amount_of_scales
            Py_ssize_t img_height
            Py_ssize_t img_width
            Py_ssize_t scale_number
            Py_ssize_t window_height = self.window_height
            Py_ssize_t window_width = self.window_width
            int result
            float[::1] scale_factors
            float[:, ::1] int_img
            float current_scale_factor
            vector[Detection] output
            Detection new_detection

        int_img = self._get_contiguous_integral_image(img)
        img_height = int_img.shape[0]
        img_width = int_img.shape[1]

        scale_factors = self._get_valid_scale_factors(min_size, max_size, scale_factor)
        amount_of_scales = scale_factors.shape[0]

        # Initialize lock to enable thread-safe writes to the array
        # in concurrent loop.
        cdef openmp.omp_lock_t mylock
        openmp.omp_init_lock(&mylock)


        # As the amount of work between the threads is not equal we use `dynamic`
        # schedule which enables them to use computing power on demand.
        for scale_number in prange(0, amount_of_scales, schedule='dynamic', nogil=True):

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

                    result = self.classify(int_img, current_row, current_col, scale_factors[scale_number])

                    if result:

                        new_detection = Detection()
                        new_detection.r = current_row
                        new_detection.c = current_col
                        new_detection.width = current_width
                        new_detection.height = current_height
                        openmp.omp_set_lock(&mylock)
                        output.push_back(new_detection)
                        openmp.omp_unset_lock(&mylock)

                    current_col = current_col + current_step

                current_row = current_row + current_step
                current_col = 0

        return list(_post_process_detections(output, min_neighbour_amount))

    def _load_xml(self, xml_file, eps=1e-5):
        """Load the parameters of cascade classifier into the class.

        The function takes the file with the parameters that represent
        trained cascade classifier and loads them into class for later
        use.

        Parameters
        ----------
        xml_file : filename or file object
            File that contains the cascade classifier.
        eps : float
            Accuracy parameter. Increasing it, makes the classifier detect less
            false positives but at the same time the false negative score increases.
        """

        cdef:
            Stage* stages_carr
            MBLBPStump* stumps_carr
            MBLBP* features_carr
            cnp.uint32_t* LUTs_carr

            float stage_threshold

            Py_ssize_t stage_number
            Py_ssize_t stages_amount
            Py_ssize_t window_height
            Py_ssize_t window_width

            Py_ssize_t weak_classifiers_amount
            Py_ssize_t weak_classifier_number

            Py_ssize_t feature_number
            Py_ssize_t features_amount
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
        stages_amount = int(tree.find('.//stageNum').text)
        window_height = int(tree.find('.//height').text)
        window_width = int(tree.find('.//width').text)
        features_amount = len(features)

        # Count the stumps.
        stumps_amount = 0
        for stage_number in range(stages_amount):
            current_stage = stages[stage_number]
            weak_classifiers_amount = int(current_stage.find('maxWeakCount').text)
            stumps_amount += weak_classifiers_amount

        # Allocate memory for data.
        features_carr = <MBLBP*>malloc(features_amount*sizeof(MBLBP))
        stumps_carr = <MBLBPStump*>malloc(stumps_amount*sizeof(MBLBPStump))
        stages_carr = <Stage*>malloc(stages_amount*sizeof(Stage))
        # Each look-up table consists of 8 u-int numbers.
        LUTs_carr = <cnp.uint32_t*>malloc(8*stumps_amount*sizeof(cnp.uint32_t))

        # Check if memory was allocated.
        if not (features_carr and stumps_carr and stages_carr and LUTs_carr):
            raise MemoryError()

        # Parse and load features in memory.
        for feature_number in range(features_amount):
            params = features[feature_number][0].text.split()
            params = map(lambda x: int(x), params)
            new_feature = MBLBP(params[1], params[0], params[2], params[3])
            features_carr[feature_number] = new_feature

        stump_lut_idx = 0
        stump_idx = 0

        # Parse and load stumps, stages.
        for stage_number in range(stages_amount):

            current_stage = stages[stage_number]

            # Parse and load current stage.
            stage_threshold = float(current_stage.find('stageThreshold').text)
            weak_classifiers_amount = int(current_stage.find('maxWeakCount').text)
            new_stage = Stage(stump_idx, weak_classifiers_amount, stage_threshold)
            stages_carr[stage_number] = new_stage

            weak_classifiers = current_stage.find('weakClassifiers')

            for weak_classifier_number in range(weak_classifiers_amount):

                current_weak_classifier = weak_classifiers[weak_classifier_number]

                # Stump's leaf values. First negative if image is probably not
                # a face. Second positive if image is probably a face.
                leaf_values = current_weak_classifier.find('leafValues').text
                leaf_values = map(lambda x: float(x), leaf_values.split())

                # Extract the elements only starting from second.
                # First two are useless
                internal_nodes = current_weak_classifier.find('internalNodes')
                internal_nodes = internal_nodes.text.split()[2:]

                # Extract the feature number and respective parameters.
                # The MBLBP position and size.
                feature_number = int(internal_nodes[0])
                lut_array = map(lambda x: int(x), internal_nodes[1:])
                lut = np.asarray(lut_array, dtype='uint32')

                # Copy array to the main LUT array
                for i in range(8):
                    LUTs_carr[stump_lut_idx + i] = lut[i]

                new_stump = MBLBPStump(feature_number, stump_lut_idx, leaf_values[0], leaf_values[1])
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
        self.stages_amount = stages_amount
        self.features_amount = features_amount
        self.stumps_amount = stumps_amount
