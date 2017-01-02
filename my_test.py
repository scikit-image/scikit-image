import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import warp, PiecewiseAffineTransform

path_image = '/datagrid/temporary/borovec/Dropbox/Workspace/py_RegistrationBenchmark/data/images/Rat_Kidney_HE.jpg'
image = np.array(Image.open(path_image))

tform = PiecewiseAffineTransform()

BASE = 150
corners = np.array([
    [BASE, BASE],
    [BASE, image.shape[0] - BASE],
    [image.shape[1] - BASE, BASE],
    [image.shape[1] - BASE, image.shape[0] - BASE],
])

tform.estimate(corners, corners)
img_warped = warp(image, tform, output_shape=(2000, 2000))

plt.imshow(img_warped), plt.show()