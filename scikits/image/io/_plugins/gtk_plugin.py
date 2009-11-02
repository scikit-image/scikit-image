from util import prepare_for_display, window_manager
import plugin

try:
    import gtk
except ImportError:
    pass
else:

    window_manager.acquire()

    class ImageWindow(gtk.Window):
        def __init__(self, arr, mgr):
            gtk.Window.__init__(self)
            self.mgr = mgr
            self.mgr.add_window(self)

            self.connect("destroy", self.destroy)

            width = arr.shape[1]
            height = arr.shape[0]
            rstride = arr.strides[0]
            pb = gtk.gdk.pixbuf_new_from_data(arr.data, gtk.gdk.COLORSPACE_RGB,
                                              False, 8, width, height, rstride)
            self.img = gtk.Image()
            self.img.set_from_pixbuf(pb)

            self.add(self.img)
            self.img.show()

        def destroy(self, widget, data=None):
            self.mgr.remove_window(self)

    def all_gone():
        print 'all windows destroyed'

    def gtk_imshow(arr):
        arr = prepare_for_display(arr)

        iw = ImageWindow(arr, window_manager)
        iw.show()

    def show():
        window_manager.register_callback(gtk.main_quit)
        gtk.main()

    plugin.register('gtk', show=gtk_imshow)
    plugin.register('gtk', appshow=show)

