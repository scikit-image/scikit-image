# a reimplementation of the opencv types.
# so we dont have to worry about having the opencv headers
# available at build time.

cdef struct _IplImage:
    int nSize               # sizeof(_IplImage)
    int ID                  # must be 0
    int nChannels           # number of channels 1, 2, 3 or 4
    int alphaChannel        # ignored by opencv
    int depth               # pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S, IPL_DEPTH_32S, IPL_DEPTH_32F, IPL_DEPTH_64F

    char colorModel[4]      # ignored by opencv
    char channelSeq[4]      # ignored by opencv
    int dataOrder           # must be 0

    int origin              # should be 0 for top-left origin

    int align               # ignored by opencv

    int width               # width in pixels
    int height              # height in pixels

    void *roi               # must be NULL
    void *maskROI           # must be NULL
    void *imageId           # must be NULL
    void *tileInfo          # must be NULL
    int imageSize           # image size in bytes

    char *imageData         # pointer to the data
    int widthStep           # row size in bytes (first stride of numpy array)
    int BorderMode[4]       # ignored by opencv
    int BorderConst[4]      # ignored by opencv
    char* imageDataOrigin   # pointer to origin of data. Used for deallocation, but python will handle this so we'll set it to void*


ctypedef _IplImage IplImage

cdef struct CvPoint2D32f:
    float x
    float y

cdef struct CvSize:
    int width
    int height

cdef struct CvTermCriteria:
    int type
    int max_iter
    double epsilon

