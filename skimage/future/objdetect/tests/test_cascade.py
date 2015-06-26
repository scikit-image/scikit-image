import numpy as np

from skimage.transform import rescale
from skimage.util import view_as_windows
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import skimage.future.objdetect as objdetect
from skimage.transform import integral_image

from skimage.color import rgb2gray
import skimage.data
import os


class TestCascade():

    def test_detector_with_naive_sliding_window(self):

        # Load the trained file from the module root.
        train_file_name = 'lbpcascade_frontalface.xml'
        current_path = os.path.abspath(os.path.dirname(__file__))
        train_file_path = os.path.join(current_path, os.pardir, train_file_name)

        # Initialize the detector cascade.
        detector = objdetect.Cascade()
        detector.load_xml(train_file_path)

        # Get the region of an image that contains face
        current_img = rgb2gray(skimage.data.astronaut()[30:200, 150:290])

        # Rescale to have the face in the same scale as the detector was trained on.
        current_img = rescale(current_img, 0.25, order=1)

        detected = []

        # Sliding window.
        views = view_as_windows(current_img, (24, 24))

        for row in xrange(views.shape[0]):
            for col in xrange(views.shape[1]):

                # Not efficient. Will be optimized.
                im = integral_image(views[row, col])
                im = np.ascontiguousarray(im, dtype=np.float32)

                if detector.evaluate(im):
                    detected.append([row, col])

        # At least one face should be detected.
        assert detected

        # plt.imshow(current_img)
        # img_desc= plt.gca()
        # plt.set_cmap('gray')
        #
        # for patch in detected:
        #     img_desc.add_patch(
        #         patches.Rectangle(
        #             (patch[1], patch[0]),
        #             24,
        #             24,
        #             fill=False,
        #             color='c'
        #         )
        #     )
        # plt.show()

if __name__ == '__main__':
    np.testing.run_module_suite()