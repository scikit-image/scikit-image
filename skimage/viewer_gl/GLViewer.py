from PyQt4 import QtCore
from PyQt4.QtGui import QVBoxLayout, QScrollArea, QMainWindow, QSlider,\
    QListWidget, QAbstractItemView, QWidget, QGridLayout, QSplitter,\
    QPushButton, QListWidgetItem, QLabel, QApplication, QImage, QPixmap, QCursor

from skimage.viewer_gl.GLCanvas import GLCanvas

from PyQt4.QtCore import Qt as GLViewerEnum

class GLViewer(QApplication):
    class CenteredScrollArea(QScrollArea):
        def __init__(self, parent=None):
            QScrollArea.__init__(self, parent)

        def eventFilter(self, object, event):
            if object == self.widget() and event.type() == QtCore.QEvent.Resize:
                QScrollArea.eventFilter(self, object, event)
            else:
                QScrollArea.eventFilter(self, object, event)

            return True

    def __init__(self, dim):
        super(QApplication, self).__init__([])

        self.canvas = GLCanvas(dim)
        self.window = QMainWindow()

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setSingleStep(1)
        self.slider.setTickInterval(10)
        self.slider.valueChanged.connect(self.sigLayerOpacity)

        self.layerList = QListWidget(self.window)
        self.layerList.setAlternatingRowColors(True)
        self.layerList.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.layerList.setDropIndicatorShown(True)
        self.layerList.setFocusPolicy(QtCore.Qt.NoFocus)
        self.layerList.installEventFilter(self)
        self.layerList.itemClicked.connect(self.sigLayerClicked)

        self.canvas.mousePress = self.mousePress
        self.canvas.mouseDrag = self.mouseDrag

        self.scrollarea = QScrollArea()
        self.scrollarea.setWidget(self.canvas)
        self.scrollarea.setAlignment(QtCore.Qt.AlignCenter)

        layoutButtons = QVBoxLayout()
        self.widgetButtons = QWidget()
        self.widgetButtons.setLayout(layoutButtons)

        self.sliderZoom = QSlider(QtCore.Qt.Horizontal)
        self.sliderZoom.setRange(100, 400)
        self.sliderZoom.setSingleStep(1)
        self.sliderZoom.setTickInterval(100)
        self.sliderZoom.setTickPosition(QSlider.TicksBelow)
        self.sliderZoom.valueChanged.connect(self.sigZoom)
        self.sliderZoom.setValue(100)

        layout = QGridLayout()
        layout.addWidget(QLabel('Opacity'))
        layout.addWidget(self.slider)
        layout.addWidget(QLabel('Layers'))
        layout.addWidget(self.layerList)
        layout.addWidget(self.widgetButtons)
        layout.addWidget(QLabel('Zoom'))
        layout.addWidget(self.sliderZoom)

        side = QWidget()
        side.setLayout(layout)

        splitter = QSplitter()
        splitter.addWidget(side)
        splitter.addWidget(self.scrollarea)

        self.window.setCentralWidget(splitter)

        self.window.installEventFilter(self)

        splitter.setSizes([100, self.window.size().width() - 100])

    def addButton(self, name, action):
        btn = QPushButton(name)
        self.widgetButtons.layout().addWidget(btn)

        btn.clicked.connect(action)

    def addSlider(self, name):
        slider = QSlider(QtCore.Qt.Horizontal)
        slider.setTickPosition(QSlider.TicksBelow)

        self.widgetButtons.layout().addWidget(QLabel(name))
        self.widgetButtons.layout().addWidget(slider)

        return slider

    def setLayerMap(self, name, func):
        item = self.layerList.findItems(name, QtCore.Qt.MatchExactly)[0]

        item.map = func

    def setLayerOpacity(self, name, opacity):
        item = self.layerList.findItems(name, QtCore.Qt.MatchExactly)[0]

        item.view.opacity = opacity

        self.updateCanvas()

    def addLayer(self, name, view, opacity=None, filters=[]):
        layer = self.canvas.addLayer(view, opacity, filters=filters)

        item = QListWidgetItem(name)
        item.setText(name)
        item.setFlags(QtCore.Qt.ItemIsUserCheckable |
                      QtCore.Qt.ItemIsDragEnabled |
                      QtCore.Qt.ItemIsSelectable |
                      QtCore.Qt.ItemIsEnabled)
        item.setCheckState(QtCore.Qt.Checked)
        item.layer = layer

        self.layerList.addItem(item)

        return layer

    def sigLayerOpacity(self, value):
        if self.layerList.currentItem() == None:
            return

        item = self.layerList.currentItem().layer
        item.opacity = float(value) / 100

        self.updateCanvas()

    def sigZoom(self, value):
        self.setZoom(value)

    def sigLayerClicked(self, item):
        checked = item.checkState() == QtCore.Qt.Unchecked
        layer = item.layer

        if layer.enabled == checked:
            layer.enabled = not layer.enabled
            self.updateCanvas()
        else:
            self.slider.setValue(100 * layer.opacity)

    def updateCanvas(self):
        self.canvas.repaint()

    def setCursor(self, cursor):
        image = QImage(cursor.tostring(), cursor.size[0], cursor.size[1],
            QImage.Format_ARGB32)
        pixmap = QPixmap(image)
        cursor = QCursor(pixmap)

        self.canvas.setCursor(cursor)

    def setZoom(self, zoom):
        self.canvas.setZoom(float(zoom) / 100)

    def setMousePress(self, func):
        self.userMousePress = func

    def setMouseDrag(self, func):
        self.userMouseDrag = func

    def setKeyPress(self, func):
        self.userKeyPress = func

    def mousePress(self, pos):
        if hasattr(self, 'userMousePress'):
            self.userMousePress(pos)

    def mouseDrag(self, pos1, pos2):
        if hasattr(self, 'userMouseDrag'):
            self.userMouseDrag(pos1, pos2)

    def eventFilter(self, object, event):
        if hasattr(self, 'userKeyPress') and event.type() == QtCore.QEvent\
        .KeyPress:
            self.userKeyPress(event.key())

            return True

        return False

    def drawLayer(self):
        for i in range(self.layerList.count()):
            layer = self.layerList.item(i)

            if layer.map:
                layer.map()
            else:
                pass

    def run(self):
        self.exec_()
