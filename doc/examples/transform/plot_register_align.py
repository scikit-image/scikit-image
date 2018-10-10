from skimage.data import camera
from scipy import ndimage as ndi
from skimage.transform import register_affine
from matplotlib import pyplot as plt


def draw_setup(base):
    fig, ax = plt.subplots()
    ax.imshow(base)
    plt.show(block=False)

    def _plot(img, matrix):
        img = ndi.affine_transform(img, matrix)
        plt.imshow(img)
        fig.canvas.draw()

    return _plot


img = camera()
img1 = ndi.shift(img, (0, 25))
img2 = ndi.shift(img, (0, -25))
print(register_affine(img1, img2, iter_callback=draw_setup(img2)))
