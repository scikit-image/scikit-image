"""
===========
Slider demo
===========

Drag slider to adjust amplitude of sine curve.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.viewer.widgets import Slider


ax = plt.subplot2grid((10, 1), (0, 0), rowspan=8)
ax_slider = plt.subplot2grid((10, 1), (9, 0))

a0 = 5
x = np.arange(0.0, 1.0, 0.001)
y = np.sin(6 * np.pi * x)

line, = ax.plot(x, a0 * y, lw=2, color='red')
ax.axis([x.min(), x.max(), -10, 10])

def update(val):
    amp = samp.value
    line.set_ydata(amp * y)

samp = Slider(ax_slider, (0.1, 10.0), on_slide=update,
              label='Amplitude:', value=a0)

plt.show()

