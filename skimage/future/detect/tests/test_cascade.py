import numpy as np

import skimage.future.detect as detect
import skimage.data as data


def test_detector_astrout():

    # Load the trained file from the module root.
    trained_file = data.frontal_face_cascade_xml()

    # Initialize the detector cascade.
    detector = detect.Cascade(trained_file)

    img = data.astronaut()

    detected = detector.detect_multi_scale(img=img,
                                           scale_factor=1.2,
                                           step_ratio=1,
                                           min_size=(60, 60),
                                           max_size=(123, 123))

    assert len(detected) == 1, 'One face should be detected.'
