
class MPLWidgetCompatibility(object):
    """Base widget for compatibility with older versions of Matplotlib.

    * Matplotlib widgets before 1.1 are old-style classes (don't derive from
      `object` so properties don't work.
    * Matplotlib widgets before 1.2 don't derive from `AxesWidget` so this
      class adds the necessary attributes/methods for compatibility.
    * This can be replaced by calls to `matplotlib.widgets.AxesWidget` (or any
      of its subclasses) when Matplotlib 1.2 becomes the minimum requirement.
    """
    drawon = True
    eventson = True

    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.cids = []
        self.active = True

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.canvas.mpl_disconnect(c)

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active

