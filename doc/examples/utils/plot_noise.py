"""
==========
Blue noise
==========

Blue noise refers to sample sets that have random and yet uniform distributions
with absence of any spectral bias. Such noise is very useful in a variety of
graphics applications like rendering, dithering, stippling, etc.

This example shows the difference between uniform, grid jittered and blue noise
sampling.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.util.noise import blue_noise

def plot(ax, X, Y, title):
    ax.scatter(X, Y, s=10, facecolor='w', edgecolor='0.5')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12, 4))

# Uniform samples
P = np.random.uniform(0, 1, (1000,2))
plot(ax1, P[:,0], P[:,1], "Uniform sampling (n=%d)" % len(P))

# Jittered samples
n = 32
X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
X += 0.45*np.random.uniform(-1/n, 1/n, (n, n))
Y += 0.45*np.random.uniform(-1/n, 1/n, (n, n))
plot(ax2, X, Y, "Regular grid + jittering (n=%d)" % X.size)
     
# Blue noise samples
P = blue_noise((1.0, 1.0), radius=0.025, k=30)
plot(ax3, P[:,0], P[:,1], "Blue noise sampling (n=%d)" % len(P))
     
plt.tight_layout()
plt.show()
