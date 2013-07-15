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
        # image = viewer.image
        # mask = self.paint_tool.overlay
        painted = demo_inpaint(viewer.image, self.paint_tool.overlay)
        viewer.ax.imshow(painted, cmap=plt.cm.jet, alpha=0.5)
        viewer.redraw()


image = data.camera()
plugin = InpaintPlugin()
plugin += OKCancelButtons()
plt.gray()

viewer = ImageViewer(image)
viewer += plugin
viewer.show()
