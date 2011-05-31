import ctypes
import numpy
import sys
import os
import os.path
from numpy.compat import asbytes

def _load_library(libname, loader_path):
    """ A small fork of numpy.ctypeslib.load_library
    to support windll.
    """
    if ctypes.__version__ < '1.0.1':
        import warnings
        warnings.warn("All features of ctypes interface may not work " \
                          "with ctypes < 1.0.1")

    ext = os.path.splitext(libname)[1]
    if not ext:
        # Try to load library with platform-specific name, otherwise
        # default to libname.[so|pyd].  Sometimes, these files are built
        # erroneously on non-linux platforms.
        libname_ext = ['%s.so' % libname, '%s.pyd' % libname]
        if sys.platform == 'win32':
            libname_ext.insert(0, '%s.dll' % libname)
        elif sys.platform == 'darwin':
            libname_ext.insert(0, '%s.dylib' % libname)
    else:
        libname_ext = [libname]

    loader_path = os.path.abspath(loader_path)
    if not os.path.isdir(loader_path):
        libdir = os.path.dirname(loader_path)
    else:
        libdir = loader_path
        for ln in libname_ext:
            try:
                libpath = os.path.join(libdir, ln)
                if sys.platform == 'win32':
                    return ctypes.windll[libpath]
                else:
                    return ctypes.cdll[libpath]
            except OSError, e:
                pass

        raise e

lib_dirs = [os.path.dirname(__file__),
            '/lib',
            '/usr/lib',
            '/usr/local/lib',
            '/opt/local/lib',
            ]

if 'HOME' in os.environ:
    lib_dirs.append(os.path.join(os.environ['HOME'], 'lib'))

API = {
    'FreeImage_Load': (ctypes.c_void_p,
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
    for f, (restype, argtypes) in api.items():
        func = getattr(lib, f)
        func.restype = restype
        func.argtypes = argtypes

_FI = None
for d in lib_dirs:
    for libname in ('freeimage', 'FreeImage',
                    'libfreeimage', 'libFreeImage'):
        try:
            _FI = _load_library(libname, d)
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
    filename = asbytes(filename)
    ftype = _FI.FreeImage_GetFileType(filename, 0)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s' % filename)
    create_new = False
    read_only = True
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename, create_new,
                                                read_only, keep_cache_in_memory,
                                                flags)
    if not multibitmap:
        raise ValueError('Could not open %s as multi-page image.' % filename)
    try:
        multibitmap = ctypes.c_void_p(multibitmap)
        pages = _FI.FreeImage_GetPageCount(multibitmap)
        arrays = []
        for i in range(pages):
            bitmap = _FI.FreeImage_LockPage(multibitmap, i)
            bitmap = ctypes.c_void_p(bitmap)
            try:
                arrays.append(_array_from_bitmap(bitmap))
            finally:
                _FI.FreeImage_UnlockPage(multibitmap, bitmap, False)
        return arrays
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, 0)

def _read_bitmap(filename, flags):
    """Load a file to a FreeImage bitmap pointer"""
    filename = asbytes(filename)
    ftype = _FI.FreeImage_GetFileType(filename, 0)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s' % filename)
    bitmap = _FI.FreeImage_Load(ftype, filename, flags)
    if not bitmap:
        raise ValueError('Could not load file %s' % filename)
    return ctypes.c_void_p(bitmap)
    
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
  array = numpy.ndarray(shape, dtype=dtype, 
                        buffer=(ctypes.c_char*byte_size).from_address(bits),
                        strides=strides)
  return array

def _array_from_bitmap(bitmap):
    """Convert a FreeImage bitmap pointer to a numpy array

    """
    dtype, shape = FI_TYPES.get_type_and_shape(bitmap)
    array = _wrap_bitmap_bits_in_array(bitmap, shape, dtype)
    # swizzle the color components and flip the scanlines to go from
    # FreeImage's BGR[A] and upside-down internal memory format to something
    # more normal
    def n(arr):
        return arr[..., ::-1].T
    if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
       dtype.type == numpy.uint8:
        b = n(array[0])
        g = n(array[1])
        r = n(array[2])
        if shape[0] == 3:
            return numpy.dstack( (r, g, b) )
        elif shape[0] == 4:
            a = n(array[3])
            return numpy.dstack( (r, g, b, a) )
        else:
            raise ValueError('Cannot handle images of shape %s' % shape)

    # We need to copy because array does *not* own its memory
    # after bitmap is freed.
    return n(array).copy()

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
    """Write a (width, height) or (width, height, nchannels) array to
    a greyscale, RGB, or RGBA image, with file type deduced from the
    filename.

    """
    filename = asbytes(filename)
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('Cannot determine type for %s' % filename)
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
    filename = asbytes(filename)
    ftype = _FI.FreeImage_GetFIFFromFilename(filename)
    if ftype == -1:
        raise ValueError('Cannot determine type of file %s' % filename)
    create_new = True
    read_only = False
    keep_cache_in_memory = True
    multibitmap = _FI.FreeImage_OpenMultiBitmap(ftype, filename,
                                                create_new, read_only,
                                                keep_cache_in_memory, 0)
    if not multibitmap:
        raise ValueError('Could not open %s for writing multi-page image.' %
                         filename)
    try:
        multibitmap = ctypes.c_void_p(multibitmap)
        for array in arrays:
            bitmap, fi_type = _array_to_bitmap(array)
            _FI.FreeImage_AppendPage(multibitmap, bitmap)
    finally:
        _FI.FreeImage_CloseMultiBitmap(multibitmap, flags)

# 4-byte quads of 0,v,v,v from 0,0,0,0 to 0,255,255,255
_GREY_PALETTE = numpy.arange(0, 0x01000000, 0x00010101, dtype=numpy.uint32)

def _array_to_bitmap(array):
    """Allocate a FreeImage bitmap and copy a numpy array into it.

    """
    shape = array.shape
    dtype = array.dtype
    r,c = shape[:2]
    if len(shape) == 2:
        n_channels = 1
        w_shape = (c,r)
    elif len(shape) == 3:
        n_channels = shape[2]
        w_shape = (n_channels,c,r)
    else:
        n_channels = shape[0]
    try:
        fi_type = FI_TYPES.fi_types[(dtype.type, n_channels)]
    except KeyError:
        raise ValueError('Cannot write arrays of given type and shape.')

    itemsize = array.dtype.itemsize
    bpp = 8 * itemsize * n_channels
    bitmap = _FI.FreeImage_AllocateT(fi_type, c, r, bpp, 0, 0, 0)
    if not bitmap:
        raise RuntimeError('Could not allocate image for storage')
    try:
        def n(arr): # normalise to freeimage's in-memory format
            return arr.T[:,::-1]
        bitmap = ctypes.c_void_p(bitmap)
        wrapped_array = _wrap_bitmap_bits_in_array(bitmap, w_shape, dtype)
        # swizzle the color components and flip the scanlines to go to
        # FreeImage's BGR[A] and upside-down internal memory format
        if len(shape) == 3 and _FI.FreeImage_IsLittleEndian() and \
               dtype.type == numpy.uint8:
            wrapped_array[0] = n(array[:,:,2])
            wrapped_array[1] = n(array[:,:,1])
            wrapped_array[2] = n(array[:,:,0])
            if shape[2] == 4:
                wrapped_array[3] = n(array[:,:,3])
        else:
            wrapped_array[:] = n(array)
        if len(shape) == 2 and dtype.type == numpy.uint8:
            palette = _FI.FreeImage_GetPalette(bitmap)
            if not palette:
                raise RuntimeError('Could not get image palette')
            ctypes.memmove(palette, _GREY_PALETTE.ctypes.data, 1024)
        return bitmap, fi_type
    except:
      _FI.FreeImage_Unload(bitmap)
      raise


def imread(filename, as_grey=False):
    """
    img = imread(filename, as_grey=False)

    Reads an image from file `filename`

    Parameters
    ----------
      filename : file name
      as_grey : Whether to convert to grey scale image (default: no)
    Returns
    -------
      img : ndarray
    """
    img = read(filename)
    if as_grey and len(img) == 3:
        # these are the values that wikipedia says are typical
        transform = numpy.array([ 0.30,  0.59,  0.11])
        return numpy.dot(img, transform)
    return img

def imsave(filename, img):
    '''
    imsave(filename, img)

    Save image to disk

    Image type is inferred from filename

    Parameters
    ----------
      filename : file name
      img : image to be saved as nd array
    '''
    write(img, filename)
