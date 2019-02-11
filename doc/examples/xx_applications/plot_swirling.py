"""
======================
Swirling Checkersboard
======================

An animation of slowly increasing both the and radius
of the swirl.
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from skimage import data
from skimage.transform import swirl


FRAMES_PER_SECOND = 50
FRAME_TIME = 20  # 1000/FRAME_TIME
FILE_NAME = 'sphx_glr_plot_swirling_001.gif'

fig, ax = plt.subplots()
ax.set_title("Alice in wonderland")
img = data.checkerboard()

frames = [swirl(img, strength=i/50, radius=100+i//20) for i in range(0,2200,5)]
imgs = [[plt.imshow(frame)] for frame in frames]

anim = animation.ArtistAnimation(fig, imgs, interval=FRAME_TIME, repeat=True)
anim.save(FILE_NAME)

plt.show()

