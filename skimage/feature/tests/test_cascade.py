import warnings

import skimage.data as data
from skimage.feature import Cascade



def test_detector_astronaut():

    # Load the trained file from the module root.
    trained_file = data.lbp_frontal_face_cascade_filename()

    # Initialize the detector cascade.
    with warnings.catch_warnings():
        # TODO: Ignore warning on overflow of negative values to positive in
        #       cast to uint32. Should determine the cause of negative values
        #       in trained_file
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        detector = Cascade(trained_file)

    img = data.astronaut()

    detected = detector.detect_multi_scale(img=img,
                                           scale_factor=1.2,
                                           step_ratio=1,
                                           min_size=(60, 60),
                                           max_size=(123, 123))

    assert len(detected) == 1, 'One face should be detected.'
