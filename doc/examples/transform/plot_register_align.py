from skimage.data import camera
from scipy.ndimage import shift as image_shift
from skimage.transform import register, matrix_to_p

img = camera()
img1 = image_shift(img, (0, 25))
img2 = image_shift(img, (0, -25))
print(matrix_to_p(register(img1, img2, draw=False)))
