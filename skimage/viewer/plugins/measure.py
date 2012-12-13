# -*- coding: utf-8 -*-
import numpy as np

from .base import Plugin
from ..widgets import Text
from ..canvastools import LineTool


__all__ = ['Measure']


rad2deg = 180 / np.pi


class Measure(Plugin):
    name = 'Measure'

    def __init__(self, maxdist=10, **kwargs):
        super(Measure, self).__init__(**kwargs)

        self.maxdist = maxdist
        self._length = Text('Length:')
        self._angle = Text('Angle:')

        self.add_widget(self._length)
        self.add_widget(self._angle)

        print self.help()

    def attach(self, image_viewer):
        super(Measure, self).attach(image_viewer)

        image = image_viewer.original_image
        h, w = image.shape
        x = [w / 3, 2 * w / 3]
        y = [h / 2] * 2
        self.line_tool = LineTool(self.image_viewer.ax, x, y,
                                  maxdist=self.maxdist,
                                  on_update=self.line_changed)
        # initialize displayed values
        self.line_changed(np.transpose((x, y)))

    def help(self):
        helpstr = ("Line profile tool",
                   "Select line to measure distance and angle.")
        return '\n'.join(helpstr)

    def line_changed(self, end_pts):
        x, y = np.transpose(end_pts)
        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        self._length.text = '%.1f' % np.hypot(dx, dy)
        self._angle.text = u'%.1f°' % (180 - np.arctan2(dy, dx) * rad2deg)
