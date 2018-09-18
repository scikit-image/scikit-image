from skimage.data import camera
from scipy.ndimage import shift as image_shift
from skimage.transform import register, matrix_to_p, p_to_matrix, warp
from matplotlib import pyplot as plt


def draw_setup(base):
    fig, ax = plt.subplots()
    ax.imshow(base)
    plt.show(block=False)

    def _plot(img, p):
        matrix = p_to_matrix(p)
        img = warp(img, matrix)
        plt.imshow(img)
        fig.canvas.draw()

    return _plot


img = camera()
img1 = image_shift(img, (0, 25))
img2 = image_shift(img, (0, -25))
print(matrix_to_p(register(img1, img2, iter_callback=draw_setup(img2))))
