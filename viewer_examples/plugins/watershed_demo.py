import matplotlib.pyplot as plt

from skimage import data
from skimage import filter
from skimage import morphology
from skimage.viewer import ImageViewer
from skimage.viewer.plugins.labelplugin import LabelPainter


class WatershedPlugin(LabelPainter):

    def help(self):
        helpstr = ("Watershed plugin",
                   "----------------",
                   "Use mouse to paint each region with a different label.",
                   "Press enter to display segmented image.")
        return '\n'.join(helpstr)

    def on_enter(self, overlay):
        viewer = self.image_viewer
        edge_image = filter.sobel(viewer.image)
        labels = morphology.watershed(edge_image, overlay)
        viewer.ax.imshow(labels, cmap=plt.cm.jet, alpha=0.5)
        viewer.redraw()
        self.close()


image = data.coins()
viewer = ImageViewer(image)
viewer += WatershedPlugin()
viewer.show()
