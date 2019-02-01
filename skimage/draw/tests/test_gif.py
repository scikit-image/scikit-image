from skimage.draw import animate
from skimage import data
import numpy as np
from scipy import ndimage as ndi
from PIL import Image
import warnings
import os


def verify_gif(file_name):
    with Image.open(file_name) as im:
        im.verify()
        assert im.format == 'GIF', "File Created is not a gif"

    # 26 bytes is currently the smallest possible gif size
    # note: in order to reach 26 bytes, the gif needs to be
    #       one image, one pixel and the maker of the gif
    #       needs to remove optional common practices.
    assert 24 <= os.stat(file_name).st_size


def test(image_num=5, file_name="test_gif.gif"):
    np.random.seed(seed=7)

    img = data.coins()
    images = [None]*image_num
    images[0] = img

    for i in range(1, image_num):
        matrix = np.random.random((3, 3))

        matrix[2][0], matrix[2][1], matrix[2][2] = 0, 0, 1
        matrix[0][2] = (matrix[0][2]-0.5)*20
        matrix[1][2] = (matrix[1][2]-0.5)*20

        images[i] = ndi.affine_transform(images[i-1], matrix)

    animate(images, file_name=file_name)
    verify_gif(file_name)


def warning_free_test(image_num=5, file_name="test_gif.gif"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test(image_num=image_num, file_name=file_name)

warning_free_test(image_num=4)
