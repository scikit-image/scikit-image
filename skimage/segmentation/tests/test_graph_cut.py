import numpy as np
from skimage.segmentation import normalized_cut
from skimage.segmentation import slic
from skimage.data import lena
from skimage.util import img_as_float


def test_all_finish_with_good_shape():
    img = img_as_float(lena()[::4, ::4])
    segments_slic = slic(img, ratio=10, n_segments=100, sigma=1)
    label_slic = normalized_cut(img, n_segments=5,
                                mask=segments_slic, random_state=0)
    # test callable mask
    label_callable = normalized_cut(img, n_segments=5,
                                    mask=lambda(image): slic(image, ratio=10,
                                                             n_segments=100,
                                                             sigma=1),
                                                             random_state=0)
    # test string mask
    label_string = normalized_cut(img, n_segments=5,
                                  mask="slic",
                                  n_init_cluster=100, random_state=0)
    assert (label_slic.shape == img.shape[0:2])
    assert (label_callable.shape == img.shape[0:2])
    assert (label_string.shape == img.shape[0:2])

    # test no mask is given
    label_no_mask = normalized_cut(img[::8, ::8],
                                   n_segments=5,
                                   mask=None, random_state=0)
    assert (label_no_mask.shape == img[::8, ::8].shape[0:2])


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
