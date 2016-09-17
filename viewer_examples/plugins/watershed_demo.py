import matplotlib.pyplot as plt

from skimage import data
from skimage import filters
from skimage import morphology
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import history
from skimage.viewer.plugins.labelplugin import LabelPainter


class OKCancelButtons(history.OKCancelButtons):

    def update_original_image(self):
        # OKCancelButtons updates the original image with the filtered image
        # by default. Override this method to update the overlay.
        self.plugin._show_watershed()
        self.plugin.close()


class WatershedPlugin(LabelPainter):

    def help(self):
        helpstr = ("Watershed plugin",
                   "----------------",
                   "Use mouse to paint each region with a different label.",
                   "Press OK to display segmented image.")
        return '\n'.join(helpstr)

    def _show_watershed(self):
        viewer = self.image_viewer
        edge_image = filter.sobel(viewer.image)
        labels = morphology.watershed(edge_image, self.paint_tool.overlay)
        viewer.ax.imshow(labels, cmap=plt.cm.jet, alpha=0.5)
        viewer.redraw()


image = data.coins()
plugin = WatershedPlugin()
plugin += OKCancelButtons()

viewer = ImageViewer(image)
viewer += plugin
viewer.show()
