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
from libc.math cimport round
from libcpp.vector cimport vector

from cython.parallel import prange
from skimage.color import rgb2gray
from skimage.transform import integral_image
import xml.etree.ElementTree as ET
from ...feature._texture cimport _multiblock_lbp


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

cdef class Cascade:

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

        self._load_xml(xml_file, eps)


    cdef int classify(self, float[:, ::1] int_img, Py_ssize_t row, Py_ssize_t col, float scale) nogil:

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

                return 0

        return 1

    def _get_valid_scale_factors(self, min_size, max_size, scale):

        min_size = np.array(min_size)
        max_size = np.array(max_size)

        scale_factors = []
        current_scale = 1
        current_size = np.array((self.window_height, self.window_width))

        while (current_size <= max_size).all():

            if (current_size >= min_size).all():
                scale_factors.append(current_scale)

            current_scale = current_scale * scale
            current_size = current_size * scale

        return np.array(scale_factors, dtype=np.float32)

    def _get_contiguous_integral_image(self, img):

        img = rgb2gray(img)
        int_img = integral_image(img)
        int_img = np.ascontiguousarray(int_img, dtype=np.float32)

        return int_img


    def detect_multi_scale(self, img, float scale_factor, float step_ratio, min_size, max_size):

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
            vector[detection_container] output
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

                        new_detection = detection_container()
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

        return list(output)

    def _load_xml(self, xml_file, eps=1e-5):

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
