import ctypes
import numpy
import sys
import os

lib_dirs = (os.path.dirname(__file__),
            '/lib',
            '/usr/lib',
            '/usr/local/lib',
            '/opt/local/lib',
            )

API = {
    'FreeImage_Load': (ctypes.c_voidp,
                       [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]),
    'FreeImage_GetWidth': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_GetHeight': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_GetImageType': (ctypes.c_uint,
                               [ctypes.c_void_p]),
    'FreeImage_GetBPP': (ctypes.c_uint,
                         [ctypes.c_void_p]),
    'FreeImage_GetPitch': (ctypes.c_uint,
                           [ctypes.c_void_p]),
    'FreeImage_GetBits': (ctypes.c_void_p,
                          [ctypes.c_void_p]),
    }

# Albert's ctypes pattern
def register_api(lib,api):
    for f, (restype, argtypes) in api.iteritems():
        func = getattr(lib, f)
        func.restype = restype
        func.argtypes = argtypes

_FI = None
for d in lib_dirs:
    for libname in ('libfreeimage', 'libFreeImage'):
        try:
            _FI = numpy.ctypeslib.load_library(libname, d)
        except OSError:
            pass
        else:
            break

    if _FI is not None:
        break

if not _FI:
    raise OSError('Could not find libFreeImage in any of the following '
                  'directories: \'%s\'' % '\', \''.join(lib_dirs))

register_api(_FI, API)

if sys.platform == 'win32':
    _functype = ctypes.WINFUNCTYPE
else:
    _functype = ctypes.CFUNCTYPE

@_functype(None, ctypes.c_int, ctypes.c_char_p)
def _error_handler(fif, message):
    raise RuntimeError('FreeImage error: %s' % message)

_FI.FreeImage_SetOutputMessage(_error_handler)

class FI_TYPES(object):
    FIT_UNKNOWN = 0
    FIT_BITMAP = 1
    FIT_UINT16 = 2
    FIT_INT16 = 3
    FIT_UINT32 = 4
    FIT_INT32 = 5
    FIT_FLOAT = 6
    FIT_DOUBLE = 7
    FIT_COMPLEX = 8
    FIT_RGB16 = 9
    FIT_RGBA16 = 10
    FIT_RGBF = 11
    FIT_RGBAF = 12

    dtypes = {
        FIT_BITMAP: numpy.uint8,
        FIT_UINT16: numpy.uint16,
        FIT_INT16: numpy.int16,
        FIT_UINT32: numpy.uint32,
        FIT_INT32: numpy.int32,
        FIT_FLOAT: numpy.float32,
        FIT_DOUBLE: numpy.float64,
        FIT_COMPLEX: numpy.complex128,
        FIT_RGB16: numpy.uint16,
        FIT_RGBA16: numpy.uint16,
        FIT_RGBF: numpy.float32,
        FIT_RGBAF: numpy.float32
        }

    fi_types = {
        (numpy.uint8, 1): FIT_BITMAP,
        (numpy.uint8, 3): FIT_BITMAP,
        (numpy.uint8, 4): FIT_BITMAP,
        (numpy.uint16, 1): FIT_UINT16,
        (numpy.int16, 1): FIT_INT16,
        (numpy.uint32, 1): FIT_UINT32,
        (numpy.int32, 1): FIT_INT32,
        (numpy.float32, 1): FIT_FLOAT,
        (numpy.float64, 1): FIT_DOUBLE,
        (numpy.complex128, 1): FIT_COMPLEX,
        (numpy.uint16, 3): FIT_RGB16,
        (numpy.uint16, 4): FIT_RGBA16,
        (numpy.float32, 3): FIT_RGBF,
        (numpy.float32, 4): FIT_RGBAF
        }

    extra_dims = {
        FIT_UINT16: [],
        FIT_INT16: [],
        FIT_UINT32: [],
        FIT_INT32: [],
        FIT_FLOAT: [],
        FIT_DOUBLE: [],
        FIT_COMPLEX: [],
        FIT_RGB16: [3],
        FIT_RGBA16: [4],
        FIT_RGBF: [3],
        FIT_RGBAF: [4]
        }

    @classmethod
    def get_type_and_shape(cls, bitmap):
        w = _FI.FreeImage_GetWidth(bitmap)
        h = _FI.FreeImage_GetHeight(bitmap)
        fi_type = _FI.FreeImage_GetImageType(bitmap)
        if not fi_type:
            raise ValueError('Unknown image pixel type')
        dtype = cls.dtypes[fi_type]
        if fi_type == cls.FIT_BITMAP:
            bpp = _FI.FreeImage_GetBPP(bitmap)
            if bpp == 8:
                extra_dims = []
            elif bpp == 24:
                extra_dims = [3]
            elif bpp == 32:
                extra_dims = [4]
            else:
                raise ValueError('Cannot convert %d BPP bitmap' % bpp)
        else:
            extra_dims = cls.extra_dims[fi_type]
        return numpy.dtype(dtype), extra_dims + [w, h]

class IO_FLAGS(object):
    #Bmp
    BMP_DEFAULT = 0
    BMP_SAVE_RLE = 1

    #Png
    PNG_DEFAULT = 0
    PNG_IGNOREGAMMA = 1

    #Gif
    GIF_DEFAULT = 0
    GIF_LOAD256 = 1
    GIF_PLAYBACK = 2

    #Ico
    ICO_DEFAULT = 0
    ICO_MAKEALPHA = 1

    #Tiff
    TIFF_DEFAULT = 0
    TIFF_CMYK = 0x0001
    TIFF_NONE = 0x0800
    TIFF_PACKBITS = 0x0100
    TIFF_DEFLATE = 0x0200
    TIFF_ADOBE_DEFLATE = 0x0400
    TIFF_CCITTFAX3 = 0x1000
    TIFF_CCITTFAX4 = 0x2000
    TIFF_LZW = 0x4000
    TIFF_JPEG = 0x8000

    #Jpeg
    JPEG_DEFAULT = 0
    JPEG_FAST = 1
    JPEG_ACCURATE = 2
    JPEG_QUALITYSUPERB = 0x80
    JPEG_QUALITYGOOD = 0x100
    JPEG_QUALITYNORMAL = 0x200
    JPEG_QUALITYAVERAGE = 0x400
    JPEG_QUALITYBAD = 0x800
    JPEG_CMYK = 0x1000
    JPEG_PROGRESSIVE = 0x2000

    #Others...
    CUT_DEFAULT = 0
    DDS_DEFAULT = 0
    HDR_DEFAULT = 0
    IFF_DEFAULT = 0
    KOALA_DEFAULT = 0
    LBM_DEFAULT = 0
    MNG_DEFAULT = 0
    PCD_DEFAULT = 0
    PCD_BASE = 1
    PCD_BASEDIV4 = 2
    PCD_BASEDIV16 = 3
    PCX_DEFAULT = 0
    PNM_DEFAULT = 0
    PNM_SAVE_RAW = 0
    PNM_SAVE_ASCII = 1
    PSD_DEFAULT = 0
    RAS_DEFAULT = 0
    TARGA_DEFAULT = 0
    TARGA_LOAD_RGB888 = 1
    WBMP_DEFAULT = 0
    XBM_DEFAULT = 0

class METADATA_MODELS(object):
    FIMD_NODATA = -1
    FIMD_COMMENTS = 0
    FIMD_EXIF_MAIN = 1
    FIMD_EXIF_EXIF = 2
    FIMD_EXIF_GPS = 3
    FIMD_EXIF_MAKERNOTE = 4
    FIMD_EXIF_INTEROP = 5
    FIMD_IPTC = 6
    FIMD_XMP = 7
    FIMD_GEOTIFF = 8
    FIMD_ANIMATION = 9
    FIMD_CUSTOM = 10

def read(filename, flags=0):
    """Read an image to a numpy array of shape (width, height) for
    greyscale images, or shape (width, height, nchannels) for RGB or
    RGBA images.

    """
    bitmap = _read_bitmap(filename, flags)
    try:
        return _array_from_bitmap(bitmap)
    finally:
        _FI.FreeImage_Unload(bitmap)

def read_multipage(filename, flags=0):
    """Read a multipage image to a list of numpy arrays, where each
    array is of shape (width, height) for greyscale images, or shape
    (nchannels, width, height) for RGB or RGBA images.

    """
    ftype = _FI.FreeImage_GetFileType(filename, 0)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s'%filename)
    create_new = False
    read_only = True
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename, create_new,
                                                read_only, keep_cache_in_memory,
                                                flags)
    if not multibitmap:
        raise ValueError('Could not open %s as multi-page image.'%filename)
    try:
        pages = _FI.FreeImage_GetPageCount(multibitmap)
        arrays = []
        for i in range(pages):
            bitmap = _FI.FreeImage_LockPage(multibitmap, i)
            try:
                arrays.append(_array_from_bitmap(bitmap))
            finally:
                _FI.FreeImage_UnlockPage(multibitmap, bitmap, False)
        return arrays
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, 0)

def _read_bitmap(filename, flags):
    """Load a file to a FreeImage bitmap pointer"""
    ftype = _FI.FreeImage_GetFileType(str(filename), 0)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s'%filename)
    bitmap = _FI.FreeImage_Load(ftype, filename, flags)
    if not bitmap:
        raise ValueError('Could not load file %s'%filename)
    return bitmap

def _wrap_bitmap_bits_in_array(bitmap, shape, dtype):
    """Return an ndarray view on the data in a FreeImage bitmap. Only
    valid for as long as the bitmap is loaded (if single page) / locked
    in memory (if multipage).

    """
    pitch = _FI.FreeImage_GetPitch(bitmap)
    height = shape[-1]
    byte_size = height * pitch
    itemsize = dtype.itemsize

    if len(shape) == 3:
        strides = (itemsize, shape[0]*itemsize, pitch)
    else:
        strides = (itemsize, pitch)
    bits = _FI.FreeImage_GetBits(bitmap)

    class DummyArray:
        __array_interface__ = {
            'data': (bits, False),
            'strides': strides,
            'typestr': dtype.str,
            'shape': tuple(shape),
            }

    # Still segfaulting on 64-bit machine because of illegal memory access

    return numpy.array(DummyArray())

def _array_from_bitmap(bitmap):
  """Convert a FreeImage bitmap pointer to a numpy array

  """
  dtype, shape = FI_TYPES.get_type_and_shape(bitmap)
  array = _wrap_bitmap_bits_in_array(bitmap, shape, dtype)
  # swizzle the color components and flip the scanlines to go from
  # FreeImage's BGR[A] and upside-down internal memory format to something
  # more normal
  if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
     dtype.type == numpy.uint8:
      b = array[0].copy()
      array[0] = array[2]
      array[2] = b

  array = array[..., ::-1]
  array = array.T

  return array.copy()

def string_tag(bitmap, key, model=METADATA_MODELS.FIMD_EXIF_MAIN):
    """Retrieve the value of a metadata tag with the given string key as a
    string."""
    tag = ctypes.c_int()
    if not _FI.FreeImage_GetMetadata(model, bitmap, str(key),
                                     ctypes.byref(tag)):
        return
    char_ptr = ctypes.c_char * _FI.FreeImage_GetTagLength(tag)
    return char_ptr.from_address(_FI.FreeImage_GetTagValue(tag)).raw()

def write(array, filename, flags=0):
    """Write a (width, height) or (nchannels, width, height) array to
    a greyscale, RGB, or RGBA image, with file type deduced from the
    filename.

    """
    filename = str(filename)
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('Cannot determine type for %s'%filename)
    bitmap, fi_type = _array_to_bitmap(array)
    try:
        if fi_type == FI_TYPES.FIT_BITMAP:
            can_write = _FI.FreeImage_FIFSupportsExportBPP(ftype,
                                      _FI.FreeImage_GetBPP(bitmap))
        else:
            can_write = _FI.FreeImage_FIFSupportsExportType(ftype, fi_type)
        if not can_write:
            raise TypeError('Cannot save image of this format '
                            'to this file type')
        res = _FI.FreeImage_Save(ftype, bitmap, filename, flags)
        if not res:
            raise RuntimeError('Could not save image properly.')
    finally:
      _FI.FreeImage_Unload(bitmap)

def write_multipage(arrays, filename, flags=0):
    """Write a list of (width, height) or (nchannels, width, height)
    arrays to a multipage greyscale, RGB, or RGBA image, with file type
    deduced from the filename.

    """
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s'%filename)
    create_new = True
    read_only = False
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename, create_new,
                                                read_only, keep_cache_in_memory,
                                                0)
    if not multibitmap:
        raise ValueError('Could not open %s for writing multi-page image.' %
                         filename)
    try:
        for array in arrays:
            bitmap = _array_to_bitmap(array)
            _FI.FreeImage_AppendPage(multibitmap, bitmap)
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, flags)

def _array_to_bitmap(array):
    """Allocate a FreeImage bitmap and copy a numpy array into it.

    """
    shape = array.shape
    dtype = array.dtype
    if len(shape) == 2:
        n_channels = 1
    else:
        n_channels = shape[0]
    try:
        fi_type = FI_TYPES.fi_types[(dtype.type, n_channels)]
    except KeyError:
        raise ValueError('Cannot write arrays of given type and shape.')
    width, height = shape[-2:]

    itemsize = array.dtype.itemsize
    bpp = 8 * itemsize * n_channels
    bitmap = _FI.FreeImage_AllocateT(fi_type, width, height, bpp, 0, 0, 0)
    if not bitmap:
        raise RuntimeError('Could not allocate image for storage')
    try:
        wrapped_array = _wrap_bitmap_bits_in_array(bitmap, shape, dtype)
        # swizzle the color components and flip the scanlines to go to
        # FreeImage's BGR[A] and upside-down internal memory format
        if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
               dtype.type == numpy.uint8:
            wrapped_array[0] = array[2,:,::-1]
            wrapped_array[1] = array[1,:,::-1]
            wrapped_array[2] = array[0,:,::-1]
            if shape[0] == 4:
                wrapped_array[3] = array[3,:,::-1]
        else:
            wrapped_array[:] = array[...,::-1]

        return bitmap, fi_type
    except:
      _FI.FreeImage_Unload(bitmap)
      raise


def imread(filename, as_grey=False, dtype=None):
    """Warning: currenly as_grey and dtype is simply ignored.

    """
    return read(filename)

def imsave(filename, arr):
    if arr.ndim == 3:
        arr = numpy.rollaxis(arr, 2, 0)
    write(arr, filename)
