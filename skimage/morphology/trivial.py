import numpy as np
import matplotlib.pyplot as plt

from skel import prepare_image, compute_thin_image


def get_rhombus(n=64, L=22, width=7):
    img = np.zeros((n, n), dtype=np.int8)

    x = np.arange(L, dtype=int)
    y = L - x

    for w in range(width):
        img[x + n//2, y + n//2 + w] = 1
        img[-x + n//2, y + n//2 + w] = 1
        img[x + n//2, -y + n//2 + w] = 1
        img[-x + n//2, -y + n//2 + w] = 1

    return img

def get_strip():
    img = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    return img


def get_square(n=64, L=16, width=4):
    img = np.zeros((n, n), dtype=np.int8)

    x = np.arange(-L//2 + 1, L//2)
    for w in range(width):
        img[n//2 + x, n//2 + L//2 + w] = 1
        img[n//2 + x, n//2 - L//2 - w] = 1

        img[n//2 + L//2 - w, n//2 + x] = 1
        img[n//2 + x, n//2 - L//2 - w] = 1

    return img


def get_loop():
    img = np.loadtxt('Untitled.txt', dtype=np.uint8)
    return img


if __name__ == "__main__":

##    img = get_rhombus()
##    img = get_strip()
    img = get_loop()

    x, y = np.nonzero(img)
    plt.scatter(x, y, marker='s', color='b', s=40, alpha=0.3)

    # skeletonize
    img1 = prepare_image(img)
    img1 = compute_thin_image(img1)

    img1_2d = img1[1, 1:, 1:]
    x, y = np.nonzero(img1_2d)

    plt.scatter(x, y, marker='o', color='r')
    plt.show()
