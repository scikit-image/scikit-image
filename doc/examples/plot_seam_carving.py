from skimage import io, data
from skimage import transform
from skimage import color, filters
from matplotlib import pyplot as plt

def custom_sobel(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)

    return filters.sobel(img)

img = data.coins()
out  = transform.seam_carve(img, 'vertical', 80, energy_func = custom_sobel)
out  = transform.seam_carve(out, 'horizontal', 70, energy_func = custom_sobel)

io.imshow(out)
plt.figure()
io.imshow(img)
io.show()
