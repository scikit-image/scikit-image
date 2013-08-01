import pyopencl as cl
import numpy as np
import os
from skimage._shared.Buffer2D import Buffer2D
from skimage._shared.IncrementalTileList import IncrementalTileList
from skimage.util.clutil import roundUp, createProgram

LWORKGROUP = (16, 16)
LWORKGROUP_1D = (256, )

cm = cl.mem_flags
szFloat = np.dtype(np.float32).itemsize
szInt = np.dtype(np.int32).itemsize

TILEW = 16
TILEH = 16

WEIGHT = '(1.0f - X/1.7320508075688772f)'

#using a char array for labels
MAX_LABELS = 256

class GrowCut_GPU():
    lw = LWORKGROUP

    class NEIGHBOURHOOD:
        VON_NEUMANN = 0

    lWorksizeTiles16 = (16, 16)

    WEIGHT_DEFAULT = '(1.0f-X/1.7320508075688772f)'
    WEIGHT_POW2 = '(1.0f-pown(X/1.7320508075688772f,2))'
    WEIGHT_POW3 = '(1.0f-pown(X/1.7320508075688772f,3))'
    WEIGHT_POW1_5 = '(1.0f-pow(X/1.7320508075688772f,1.5))'
    WEIGHT_POW_SQRT = '(1.0f-sqrt(X/1.7320508075688772f))'

    def __init__(self, context, devices, img, neighbourhood=NEIGHBOURHOOD.VON_NEUMANN, weight=None):
        self.context = context
        self.queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

        if weight == None:
            weight = GrowCut_GPU.WEIGHT_DEFAULT

        if isinstance(img, cl.Image):
            self.dImg = img

            width = img.get_image_info(cl.image_info.WIDTH)
            height = img.get_image_info(cl.image_info.HEIGHT)

            dim = (width, height)
        else:
            raise NotImplementedError('Not implemented for {0}'.format(type(
                img)))

        self.tilelist = IncrementalTileList(context, devices, dim, (TILEW,
                                                                    TILEH))
        self.hHasConverged = np.empty((1,), np.int32)
        self.hHasConverged[0] = False

        self.dLabelsIn = Buffer2D(context, cm.READ_WRITE, dim, np.uint8)
        self.dLabelsOut = Buffer2D(context, cm.READ_WRITE, dim, np.uint8)
        self.dStrengthIn = Buffer2D(context, cm.READ_WRITE, dim, np.float32)
        self.dStrengthOut = Buffer2D(context, cm.READ_WRITE, dim, np.float32)
        self.dHasConverged = cl.Buffer(context, cm.READ_WRITE | cm.COPY_HOST_PTR, hostbuf=self.hHasConverged)

        self.args = [
            self.tilelist.d_list,
            self.dLabelsIn,
            self.dLabelsOut,
            self.dStrengthIn,
            self.dStrengthOut,
            self.dHasConverged,
            np.int32(self.tilelist.iteration),
            self.tilelist.d_tiles,
            cl.LocalMemory(szInt * 9),
            cl.LocalMemory(szInt * (TILEW + 2) * (TILEH + 2)),
            cl.LocalMemory(szFloat * (TILEW + 2) * (TILEH + 2)),
            #			cl.LocalMemory(4*szFloat*(TILEW+2)*(TILEH+2)),
            self.dImg,
            cl.Sampler(context, False, cl.addressing_mode.NONE, cl.filter_mode.NEAREST)
        ]

        self.gWorksize = roundUp(dim, self.lw)
        self.gWorksizeTiles16 = roundUp(dim, self.lWorksizeTiles16)

        options = [
            '-D TILESW=' + str(self.tilelist.dim[0]),
            '-D TILESH=' + str(self.tilelist.dim[1]),
            '-D IMAGEW=' + str(dim[0]),
            '-D IMAGEH=' + str(dim[1]),
            '-D TILEW=' + str(TILEW),
            '-D TILEH=' + str(TILEH),
            '-D G_NORM(X)=' + weight
        ]

        filename = os.path.join(os.path.dirname(__file__), '_growcut_gpu.cl')
        program = createProgram(context, devices, options, filename)

        if neighbourhood == GrowCut_GPU.NEIGHBOURHOOD.VON_NEUMANN:
            self.kernEvolve = cl.Kernel(program, 'evolveVonNeumann')
        elif neighbourhood == GrowCut_GPU.NEIGHBOURHOOD.MOORE:
            self.kernEvolve = cl.Kernel(program, 'evolveMoore')

        self.kernLabel = cl.Kernel(program, 'label')

        self.isComplete = False

    def label(self, d_points, n_points, label):
        gWorksize = roundUp((n_points, ), LWORKGROUP_1D)

        args = [
            self.dLabelsIn,
            self.dStrengthIn,
            d_points,
            np.uint8(label),
            np.int32(n_points),
            self.tilelist.d_tiles,
            np.int32(self.tilelist.iteration)
        ]

        self.kernLabel(self.queue, gWorksize, LWORKGROUP_1D, *args).wait()

    def evolve(self):
        self.isComplete = False

        self.tilelist.build()

        if self.tilelist.length == 0:
            self.isComplete = True
            return

        self.tilelist.increment()

        gWorksize = (TILEW * self.tilelist.length, TILEH)

        self.args[1] = self.dLabelsIn
        self.args[2] = self.dLabelsOut
        self.args[3] = self.dStrengthIn
        self.args[4] = self.dStrengthOut
        self.args[6] = np.int32(self.tilelist.iteration)

        self.kernEvolve(self.queue, gWorksize, self.lWorksizeTiles16, *self.args).wait()

        dTmp = self.dLabelsOut
        self.dLabelsOut = self.dLabelsIn
        self.dLabelsIn = dTmp

        dTmp = self.dStrengthOut
        self.dStrengthOut = self.dStrengthIn
        self.dStrengthIn = dTmp
