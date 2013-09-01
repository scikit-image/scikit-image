import matplotlib.pyplot as plt
from skimage import data
from skimage.viewer import ImageViewer
from skimage.viewer.widgets import history
from skimage.viewer.plugins.labelplugin import LabelPainter
from skimage.filter.inpaint_texture import inpaint_efros


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
                   "Press OK to display the inpainted image.")
        return '\n'.join(helpstr)

    def _show_inpaint(self):
        viewer = self.image_viewer
        painted = inpaint_efros(viewer.image, self.paint_tool.overlay)
        viewer.ax.imshow(painted)
        viewer.redraw()


image = data.checkerboard()
plugin = InpaintPlugin()
plugin += OKCancelButtons()
plt.gray()

viewer = ImageViewer(image)
viewer += plugin
viewer.show()
