"""
=================================
Retinex image enhancement
=================================

This example performs image enhancement using retinex:
Multicale Retinex (MSR) [1]_
and MSR combined with Simplest Color Balance (SCB) [2]_.

References
----------
.. [1] Jobson & Rahman, "A Multiscale Retinex for Bridging the Gap Between
    Color Images and the Human Observation of Scenes", IEEE TIP, 1997
.. [2] Petro, Sbert & Morel. "Multiscale retinex", IPOL, 2014

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import data, img_as_float
from skimage import exposure
from skimage.filters import gaussian

matplotlib.rcParams['font.size'] = 8


def ssr(img, sigma):
    """Single-Scale Retinex (SSR), Eq(1) of [1]"""
    return np.log10(img) - np.log10(gaussian(img, sigma))


def msr(img, sigmas):
    """Multicale Retinex (MSR), Eq(2) of [1]"""
    retinex = np.zeros_like(img)
    for sigma in sigmas:
        retinex += ssr(img, sigma)
    return retinex / len(sigmas)


def crf(img, alpha, beta):
    """Color restoration function (CRF), Eq(5) of [1]"""
    img_sum = np.sum(img, axis=2, keepdims=True)
    return beta * (np.log10(alpha * img) - np.log10(img_sum))


def scb(img, low_clip, high_clip):
    """Simplest color balance (SCB), [2]"""
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


def msrcr(img, sigmas, G=192, b=-30, alpha=125, beta=46, low_clip=None, high_clip=None):
    """Color Restoration with Multiscale Retinex (MSRCR)"""
    img = np.float64(img) + 1.0
    R = msr(img, sigmas)
    C = crf(img, alpha, beta)
    out = G * (C * R + b)  # Eq(6) of [1]

    for i in range(out.shape[2]):
        out[:, :, i] = 255 * (out[:, :, i] - np.min(out[:, :, i])) / \
            (np.max(out[:, :, i]) - np.min(out[:, :, i]))
    out = np.uint8(np.minimum(np.maximum(out, 0), 255))
    if low_clip and high_clip:
        out = scb(out, low_clip, high_clip)
    return out


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


# Load an example image
img = data.moon()[..., None]

sigmas = [15, 80, 250]

# Multicale Retinex (MSR)
img_msr = msrcr(img, sigmas)

# Multicale Retinex (MSR) with Simplest Color Balance (SCB)
img_msrscb = msrcr(img, sigmas, low_clip=0.01, high_clip=0.99)

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2, 3), dtype=object)
axes[0, 0] = plt.subplot(2, 3, 1)
axes[0, 1] = plt.subplot(2, 3, 2, sharex=axes[0, 0], sharey=axes[0, 0])
axes[0, 2] = plt.subplot(2, 3, 3, sharex=axes[0, 0], sharey=axes[0, 0])
axes[1, 0] = plt.subplot(2, 3, 4)
axes[1, 1] = plt.subplot(2, 3, 5)
axes[1, 2] = plt.subplot(2, 3, 6)

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_msr, axes[:, 1])
ax_img.set_title('MSR enhancement')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_msrscb, axes[:, 2])
ax_img.set_title('MSR+SCB enhancement')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.tight_layout()
plt.show()
