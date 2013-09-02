from PyQt4.QtOpenGL import QGLWidget
from PyQt4.QtCore import QSize, Qt, QEvent

import numpy as np

try:
    from OpenGL.GL import *
except ImportError:
    raise ImportError("Error importing PyOpenGL")

szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

LWORKGROUP = (16, 16)

class GLCanvas(QGLWidget):
    class Layer:
        def __init__(self, parent, data, pos=None, enabled=True, opacity=1.0,
                     filters=[]):
            self.parent = parent
            self.data = data
            self.opacity = opacity if opacity != None else 1.0
            self.enabled = enabled
            self.pos = pos
            self.filters = filters
            self.size = data.shape[1]*data.shape[0]

            self.tex = glGenTextures(1)
            self.pbo = glGenBuffers(1)

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
            glBufferData(GL_PIXEL_UNPACK_BUFFER, self.size * 4, None,
                GL_STATIC_DRAW)

            self.setData(data)

        def updateData(self):
            glBindTexture(GL_TEXTURE_2D, self.tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.data.shape[1],
                self.data.shape[0], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self.pbo)
            glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, self.data)

            self.parent.update()

        def setOpacity(self, opacity):
            self.opacity = opacity
            self.parent.update()

        def setData(self, data):
            self.data = data
            self.updateData()

    def __init__(self, shape, parent=None):
        super(GLCanvas, self).__init__(parent)

        self.w = 0

        self.pbo = None
        self.tex = None

        self.width = shape[0]
        self.height = shape[1]
        self.shape = shape

        self.zoom = 1.0
        self.transX = 0
        self.transY = 1
        self.flag = 0

        self.viewport = None

        self.resize(self.zoom * self.width, self.zoom * self.height)

        self.initializeGL()

        self.installEventFilter(self)

        self.fbo = glGenFramebuffers(1)
        self.rbos = glGenRenderbuffers(2)

        self.rboRead = 0
        self.rboWrite = 1

        self.buffers = {}

        self.layers = []

        glFinish()
    def addLayer(self, data, opacity=None, filters=[]):
        if opacity == None:
            opacity = 1.0

        layer = GLCanvas.Layer(self, data, opacity=opacity, filters=filters)
        self.layers.append(layer)

        return layer

    def setZoom(self, value):
        self.zoom = value

        self.resize(self.zoom * self.width, self.zoom * self.height)

    def setPbo(self, pbo):
        self.pbo = pbo

    def minimumSizeHint(self):
        return QSize(self.zoom * self.width, self.zoom * self.height)

    def initializeGL(self):
        self.makeCurrent()

        glClearColor(1.0, 0.0, 0.0, 0.0)

    def swapRbos(self):
        self.rboRead = not self.rboRead
        self.rboWrite = not self.rboWrite

    def paintEvent(self, event):
        r = event.rect()

        self.transX = -r.x()
        self.transY = -int(self.zoom * self.height - r.height()) + r.y()

        self.viewport = (r.width(), r.height())

        self.rect = event.rect()

        self.makeCurrent()

        self.paintGL()

        if self.doubleBuffer():
            self.swapBuffers()

        glFlush()

    def paintGL(self):
        if len(self.layers) == 0:
            return

        glViewport(0, 0, self.width, self.height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, self.width, 0, self.height, -1, 1)
        #glTranslatef(-self.transX, -self.transY, 0)
        #glScalef(self.zoom, self.zoom, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glClear(GL_COLOR_BUFFER_BIT)

        visible = []

        for layer in self.layers:
            if not layer.enabled or layer.opacity == 0:
                continue

            visible.append(layer)

        #			if layer.opacity == 1.0:
        #				break

        glEnable(GL_TEXTURE_2D)
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE)

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        for layer in reversed(visible):
            glColor4f(1.0, 1.0, 1.0, layer.opacity);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, layer.pbo)
            glBindTexture(GL_TEXTURE_2D, layer.tex)
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                GL_RGBA, GL_UNSIGNED_BYTE, None)

            glBegin(GL_QUADS)
            glVertex2i(0,0)
            glTexCoord2i(0,0)
            glVertex2i(0, self.height)
            glTexCoord2i(1,0)
            glVertex2i(self.width, self.height)
            glTexCoord2i(1,1)
            glVertex2i(self.width, 0)
            glTexCoord2i(0,1)
            glEnd()

        glDisable(GL_BLEND)

    def eventFilter(self, object, event):
        if hasattr(self, 'mouseDrag') and (event.type() == QEvent.MouseMove
                                           and event.buttons() == Qt.LeftButton):
            point = (int(event.pos().x() / self.zoom),
                     int(event.pos().y() / self.zoom)
                    )

            self.mouseDrag(self.lastMousePos, point)

            self.lastMousePos = point

            return True
        if hasattr(self,
            'mousePress') and event.type() == QEvent.MouseButtonPress:
            point = (int(event.pos().x() / self.zoom),
                     int(event.pos().y() / self.zoom)
                    )

            self.mousePress(point)

            self.lastMousePos = point

            return True

        return False
