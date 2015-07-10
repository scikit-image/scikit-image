import numpy as np

import skimage.future.detect_obj as detect_obj
import skimage.data as data


class TestCascade():

    def test_detector(self):

        # Load the trained file from the module root.
        trained_file = data.xml.face_cascade_detector()

        # Initialize the detector cascade.
        detector = detect_obj.Cascade(trained_file)

        img = data.astronaut()

        detected = detector.detect_multi_scale(img=img,
                                               scale_factor=1.2,
                                               step_ratio=1.3,
                                               min_size=(24, 24),
                                               max_size=(123, 123))

        assert len(detected) == 2, 'Two faces on the image.'


if __name__ == '__main__':
    np.testing.run_module_suite()
