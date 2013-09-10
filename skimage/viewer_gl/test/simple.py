"""
===============
Simple GLViewer Demo
===============

Steps the run and test
ipython --pylab=qt4 -i skimage/viewer_gl/test/simple.py

To dynamically update the contents of a layer:
arr = np.zeros_like(img, dtype=np.int32)
arr[:] = 0xFF0000FF
l2.setData(arr)

l2.data[:] = 0xFF00FFFF
l2.updateData()

To set the opacity
l2.setOpacity(0.25)
"""

import numpy as np
import Image
from skimage.viewer_gl.GLViewer import GLViewer, GLViewerEnum
from skimage import data_dir

img = Image.open(data_dir+'/'+'lena.png')
if img.mode != 'RGBA':
    img = img.convert('RGBA')

viewer = GLViewer(img.size)

img = np.array(img).view(np.uint32).squeeze()
dim = (img.shape[1], img.shape[0])

layer1 = np.zeros_like(img, dtype=np.int32)
layer2 = np.zeros_like(img, dtype=np.int32)

layer1[:] = 0xFFFF0000
layer1[0:img.shape[1], 0:img.shape[1]/2] = 0xFF00FFFF

layer2[:] = 0xFFFF00FF
layer2[0:img.shape[1]/2, 0:img.shape[1]] = 0xFF00FF00

def mouseDrag(pos1, pos2):
    if pos1 == pos2:
        return

    viewer.updateCanvas()

def mousePress(pos):
    mouseDrag(pos, None)

def keyPress(key):
    if key >= GLViewerEnum.Key_1 and key <= GLViewerEnum.Key_8:
        print key

l1 = viewer.addLayer('layer1', layer1, 0.5)
l2 = viewer.addLayer('layer2', layer2, 0.5)
l3 = viewer.addLayer('Image', img, 1.0)

viewer.setMousePress(mousePress)
viewer.setMouseDrag(mouseDrag)
viewer.setKeyPress(keyPress)

def slidevalue(value):
    print value

slider = viewer.addSlider('kernel_size')
slider.setRange(1, 50)
slider.setSingleStep(1)
slider.setTickInterval(100)
slider.valueChanged.connect(slidevalue)
slider.setValue(100)

viewer.window.resize(1000, 700)
viewer.window.show()

#viewer.run()
