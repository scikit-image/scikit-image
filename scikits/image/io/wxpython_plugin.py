import numpy as np
import plugin
from _plugin_util import prepare_for_display

try:
    import wx
except ImportError:
    pass
else:

    # idea shamelessly taken from here:
    # http://wiki.wxpython.org/WorkingWithImages

    class ImagePanel(wx.Panel):
        def __init__(self, parent, id):
            wx.Panel.__init__(self, parent, id)
            self.bitmap = None
            self.Bind(wx.EVT_PAINT, self.OnPaint)

        def display(self, npy_img):
            self.bitmap = self.get_bitmap(npy_img)
            self.Refresh(True)

        def OnPaint(self, evt):
            dc = wx.PaintDC(self)
            if self.bitmap:
                dc.DrawBitmap(self.bitmap, 0, 0)

        def get_bitmap(self, npy_img):
            width = npy_img.shape[1]
            height = npy_img.shape[0]
            wx_img = wx.EmptyImage(width, height)
            wx_img.SetData(npy_img.data)
            return wx.BitmapFromImage(wx_img)


    class ImageFrame(wx.Frame):
        def __init__(self, img):
            self.img = img
            width = img.shape[1]
            height = img.shape[0]
            wx.Frame.__init__(self, None, -1, 'wx', wx.DefaultPosition,
                                wx.Size(width, height))

            self.iPanel = ImagePanel(self, -1)
            self.iPanel.display(self.img)

    def wx_imshow(img):
        f = ImageFrame(prepare_for_display(img))
        f.CenterOnScreen()
        f.Show()

    plugin.register('wx', show=wx_imshow)
