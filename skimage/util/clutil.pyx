#cython: boundscheck=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: wraparound=False

import pyopencl as cl

def pow2lt(x):
    i = 1;
    while(i <= x >> 1):
        i = i << 1;

    return i

def pow2gt(x):
    i = 1;
    while(i < x):
        i = i << 1;

    return i

def isPow2(x):
    return (x != 0) and ((x & (x - 1)) == 0)

def ceil_divi(dividend, divisor):
    return (dividend + divisor - 1) / divisor

def print_deviceAttr(device):
    attrs = [
        'GLOBAL_MEM_SIZE',
        'DRIVER_VERSION',
        'GLOBAL_MEM_CACHE_SIZE',
        'GLOBAL_MEM_CACHELINE_SIZE',
        'GLOBAL_MEM_SIZE',
        'GLOBAL_MEM_CACHE_TYPE',
        'IMAGE_SUPPORT',
        'IMAGE2D_MAX_HEIGHT',
        'IMAGE2D_MAX_WIDTH',
        'LOCAL_MEM_SIZE',
        'LOCAL_MEM_TYPE',
        'MAX_COMPUTE_UNITS',
        'MAX_WORK_GROUP_SIZE',
        'MAX_WORK_ITEM_SIZES',
        'MIN_DATA_TYPE_ALIGN_SIZE',
        'PREFERRED_VECTOR_WIDTH_FLOAT',
        'PREFERRED_VECTOR_WIDTH_CHAR',
        'PREFERRED_VECTOR_WIDTH_DOUBLE',
        'PREFERRED_VECTOR_WIDTH_HALF',
        'MAX_CONSTANT_BUFFER_SIZE',
        'MAX_MEM_ALLOC_SIZE',
    ]

    for attr in attrs:
        tmp = getattr(cl.device_info, attr)
        print '\t' + attr + ':\t' + str(device.get_info(tmp))

def kernelInfo(kernel, device):
    attrs = [
        'LOCAL_MEM_SIZE',
        'WORK_GROUP_SIZE',
        'PREFERRED_WORK_GROUP_SIZE_MULTIPLE',
        'COMPILE_WORK_GROUP_SIZE',
    ]

    for attr in attrs:
        tmp = getattr(cl.kernel_work_group_info, attr)
        print attr + ':\t' + str(kernel.get_work_group_info(tmp, device))

def platformInfo():
    platforms = cl.get_platforms();
    if len(platforms) == 0:
        print "Failed to find any OpenCL platforms."
        return None

    for platform in platforms:
        print platform

        for device in platform.get_devices():
            print cl.deviceInfo(device)

#
#  Create an OpenCL program from the kernel source file
#
def createProgram(context, devices, options, fileName):
    kernelFile = open(fileName, 'r')
    kernelStr = kernelFile.read()

    # Load the program source
    program = cl.Program(context, kernelStr)

    # Build the program and check for errors
    program.build(options, devices)

    return program

def localToGlobalWorkgroup(size, lWorkGroup):
    if len(size) != len(lWorkGroup):
        raise TypeError('dimensions so not match: {0}, {1}'.format(len(size),
            len(lWorkGroup)))

    return tuple([roundUp(l, d) for l, d in zip(lWorkGroup, size)])

def roundUp(size, multiple):
    if type(size) == int and type(multiple) == int:
        return roundUp_int(size, multiple)
    elif type(size) == tuple and type(multiple) == tuple:
        return roundUp_tuple(size, multiple)

cdef int roundUp_int(int size, int multiple):
    cdef int r = size % multiple

    if r == 0:
        return size
    else:
        return size + multiple - r

cdef int[:] roundUp_array(int[:] size, int[:] multiple):
    cdef int[:] out = size.copy()
    cdef int i, r

    for i in range(len(size)):
        r = size[i] % multiple[0];
        if r == 0:
            out[i] = size[0]
        else:
            out[i] = size[0] + multiple[0] - r

    return out

cdef tuple roundUp_tuple(tuple size, tuple multiple):
    cdef list out = [None] * len(size)
    cdef int i, r

    for i in range(len(size)):
        r = size[i] % multiple[i];
        if r == 0:
            out[i] = size[i]
        else:
            out[i] = size[i] + multiple[i] - r

    return tuple(out)

def padArray2D(arr, shape, mode):
    from numpy import pad

    padding = [(0, shape[0] - arr.shape[0]), (0, shape[1] - arr.shape[1])]

    return pad(arr, padding, mode)

def formatForCLImage2D(arr):
    import numpy as np

    return arr.view(np.int8).astype(np.float32).reshape(arr.shape + (-1,))

def compareFormat(format1, format2):
    if format1.channel_order != format2.channel_order:
        return False
    if format1.channel_data_type != format2.channel_data_type:
        return False

    return True

#align on 32B segment for 8-bit data
#align on 64B segment for 16-bit data
#align on 128B segment for 32, 64 and 128-bit data
def alignedDim(dim, dtype):
    import numpy as np

    if type(dim) == int:
        pass
    elif type(dim) == tuple and len(dim) == 2:
        dim = dim[0]
    else:
        raise NotImplemented

    if dtype == np.int32 or dtype == np.uint32:
        return roundUp(dim, 32)
    if dtype == np.int8 or dtype == np.uint8:
        return roundUp(dim, 32)
    elif dtype == np.float32:
        return roundUp(dim, 32)
    else:
        raise NotImplemented()
