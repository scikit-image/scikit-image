import numpy as np
import Image

import pyopencl as cl

from skimage.viewer_glcl.CLViewer import CLViewer, CLViewerEnum
from skimage.viewer_glcl.Colorize import Colorize
from skimage.viewer_glcl.Brush import Brush
from skimage._shared.Buffer2D import Buffer2D
from skimage._shared.Image2D import Image2D
from skimage.util.clutil import roundUp, padArray2D
from skimage.segmentation import GrowCut_GPU

from PyQt4.QtCore import QTimer

from skimage import data_dir

cm = cl.mem_flags

img = Image.open(data_dir+'/trolls.png')
if img.mode != 'RGBA':
    img = img.convert('RGBA')

viewer = CLViewer(img.size)

shape = (img.size[1], img.size[0])
shape = roundUp(shape, GrowCut_GPU.lw)
dim = (shape[1], shape[0])

hImg = padArray2D(np.array(img).view(np.uint32).squeeze(), shape, 'edge')

dImg = Image2D(viewer.context,
    cl.mem_flags.READ_ONLY,
    cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8),
    dim
)
cl.enqueue_copy(viewer.queue, dImg, hImg, origin=(0, 0), region=dim).wait()

dStrokes = Buffer2D(viewer.context, cm.READ_WRITE, dim, dtype=np.uint8)

brush = Brush(viewer.context, viewer.devices, dStrokes)

growCut = GrowCut_GPU(viewer.context, viewer.devices, dImg, GrowCut_GPU.NEIGHBOURHOOD.VON_NEUMANN)

label = 1

iteration = 0
refresh = 100

timer = QTimer()

def update():
    global iteration

    growCut.evolve()

    if growCut.isComplete:
        viewer.updateCanvas()
        timer.stop()
        return

    if iteration % refresh == 0:
        viewer.updateCanvas()

    iteration += 1

def mouseDrag(pos1, pos2):
    if pos1 == pos2:
        return

    timer.stop()
    brush.draw(pos1, pos2)
    growCut.label(brush.d_points, brush.n_points, brush.label)

    viewer.updateCanvas()

    timer.start()

def mousePress(pos):
    mouseDrag(pos, None)

def keyPress(key):
    if key >= CLViewerEnum.Key_1 and key <= CLViewerEnum.Key_8:
        brush.setLabel(key - CLViewerEnum.Key_1 + 1)

#setup window
filter = Colorize(viewer.context, (0, 8), Colorize.HUES.STANDARD)
viewer.addLayer('labels', growCut.dLabelsIn, 0.5, filters=[filter])
#window.addLayer('labels', growCut.dLabelsOut, 0.5, filters=[filter])
viewer.addLayer('strokes', dStrokes, 0.5, filters=[filter])

filter = Colorize(viewer.context, (0, 1.0), hues=Colorize.HUES.REVERSED)
viewer.addLayer('strength', growCut.dStrengthIn, 0.25, filters=[filter])
viewer.addLayer('image', dImg)

viewer.addLayer('tiles', growCut.tilelist, filters=[growCut.tilelist])

viewer.setMousePress(mousePress)
viewer.setMouseDrag(mouseDrag)
viewer.setKeyPress(keyPress)

viewer.window.resize(1000, 700)
viewer.window.show()

timer = QTimer()
timer.timeout.connect(update)

viewer.run()
