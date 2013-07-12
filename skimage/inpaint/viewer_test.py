import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import history
from skimage.viewer.plugins.labelplugin import LabelPainter
from test import demo_inpaint


class OKCancelButtons(history.OKCancelButtons):

    def update_original_image(self):
        # OKCancelButtons updates the original image with the filtered image
        # by default. Override this method to update the overlay.
        self.plugin._show_inpaint()
        self.plugin.close()


class InpaintPlugin(LabelPainter):

    def help(self):
        helpstr = ("Inpainting plugin",
                   "----------------",
                   "Use mouse to paint the mask region white",
                   "Press OK to display segmented image.")
        return '\n'.join(helpstr)

    def _show_inpaint(self):
        viewer = self.image_viewer
        image = viewer.image
        mask = self.paint_tool.overlay
        print mask.shape, mask.max(), mask.dtype

        demo_inpaint(image, mask)
        # viewer.ax.imshow(mask)
        # labels = morphology.watershed(edge_image, mask)
        # viewer.ax.imshow(labels, cmap=plt.cm.jet, alpha=0.5)
        viewer.redraw()


image = data.camera()
# paint_region = (slice(65, 70), slice(55, 75))
# image[paint_region] = 0
plugin = InpaintPlugin()
plugin += OKCancelButtons()
plt.gray()

viewer = ImageViewer(image)
viewer += plugin
viewer.show()
