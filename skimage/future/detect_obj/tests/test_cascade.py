import numpy as np

from skimage.transform import rescale
from skimage.util import view_as_windows
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import skimage.future.detect_obj as detect_obj
from skimage.transform import integral_image

import skimage.data
import os


class TestCascade():

    def test_detector_with_naive_sliding_window(self):

        # Load the trained file from the module root.
        train_file_name = 'lbpcascade_frontalface.xml'
        current_path = os.path.abspath(os.path.dirname(__file__))
        train_file_path = os.path.join(current_path, os.pardir, train_file_name)

        # Initialize the detector cascade.
        detector = detect_obj.Cascade()
        detector.load_xml(train_file_path)

        img = skimage.data.astronaut()

        detected = detector.detect_multi_scale(img=img,
                                               scale_factor=1.2,
                                               min_size=(24, 24),
                                               max_size=(123, 123),
                                               step_ratio=1.5,
                                               amount_of_threads=4)

        # At least one face should be detected.
        assert detected


if __name__ == '__main__':
    np.testing.run_module_suite()
