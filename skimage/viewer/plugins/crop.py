from .base import Plugin
from ..canvastools import RectangleTool
from ...viewer.widgets import SaveButtons


__all__ = ['Crop']


class Crop(Plugin):
    name = 'Crop'

    def __init__(self, maxdist=10, **kwargs):
        super(Crop, self).__init__(**kwargs)
        self.maxdist = maxdist
        self.add_widget(SaveButtons())
        print(self.help())

    def attach(self, image_viewer):
        super(Crop, self).attach(image_viewer)

        self.rect_tool = RectangleTool(image_viewer,
                                       maxdist=self.maxdist,
                                       on_enter=self.crop)
        self.artists.append(self.rect_tool)

    def help(self):
        helpstr = ("Crop tool",
                   "Select rectangular region and press enter to crop.")
        return '\n'.join(helpstr)

    def crop(self, extents):
        xmin, xmax, ymin, ymax = extents
        image = self.image_viewer.image[ymin:ymax+1, xmin:xmax+1]
        self.image_viewer.image = image
        self.image_viewer.ax.relim()
