import numpy as np
import Image

import pyopencl as cl
cm = cl.mem_flags

from skimage.viewer_glcl.CLViewer import CLViewer, CLViewerEnum
from skimage.viewer_glcl.Colorize import Colorize
from skimage.viewer_glcl.Brush import Brush
from skimage._shared.Buffer2D import Buffer2D

from skimage import data_dir

img = Image.open(data_dir+'/'+'lena.png')
if img.mode != 'RGBA':
    img = img.convert('RGBA')

viewer = CLViewer(img.size)

img = np.array(img).view(np.uint32).squeeze()
dim = (img.shape[1], img.shape[0])

dStrokes = Buffer2D(viewer.context, cl.mem_flags.READ_WRITE, dim, dtype=np.uint8)
brush = Brush(viewer.context, viewer.devices, dStrokes)

layer1 = np.zeros_like(img, dtype=np.int32)
layer2 = np.zeros_like(img, dtype=np.int32)

layer1[:] = 0
layer1[0:img.shape[1], 0:img.shape[1]/2] = 1

layer2[:] = 2
layer2[0:img.shape[1]/2, 0:img.shape[1]] = 4

def mouseDrag(pos1, pos2):
    if pos1 == pos2:
        return

    brush.draw(pos1, pos2)

    viewer.updateCanvas()

def mousePress(pos):
    mouseDrag(pos, None)

def keyPress(key):
    if key >= CLViewerEnum.Key_1 and key <= CLViewerEnum.Key_8:
        brush.setLabel(key - CLViewerEnum.Key_1 + 1)

dImg = Buffer2D(viewer.context, cm.READ_WRITE, hostbuf=img, pad=False)
dLabel = Buffer2D(viewer.context, cm.READ_WRITE, hostbuf=layer1, pad=False)
dLabel2 = Buffer2D(viewer.context, cm.READ_WRITE, hostbuf=layer2, pad=False)

filter = Colorize(viewer.context, (0, 8), Colorize.HUES.STANDARD)
viewer.addLayer('strokes', dStrokes, 0.5, filters=[filter])

filter = Colorize(viewer.context, (0, 4), Colorize.HUES.STANDARD)
viewer.addLayer('layer1', dLabel, 0.25, filters=[filter])
viewer.addLayer('layer2', dLabel2, 0.25, filters=[filter])
viewer.addLayer('Image', dImg, 1.0)

viewer.setMousePress(mousePress)
viewer.setMouseDrag(mouseDrag)
viewer.setKeyPress(keyPress)

viewer.window.resize(1000, 700)
viewer.window.show()

viewer.run()
