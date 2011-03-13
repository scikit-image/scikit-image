'''_ctmf.pyx - constant time per pixel median filtering

Reference: S. Perreault and P. Hebert, "Median Filtering in Constant Time",
IEEE Transactions on Image Processing, September 2007.

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky
'''
import numpy as np
cimport numpy as np
cimport cython

from libc.stdlib cimport malloc, free
from libc.string cimport memset

np.import_array()

##############################################################################
#
# median_filter - implementation of constant-time median filter with
#                 octagonal shape. The algorithm is derived from
#                 Perreault, "Median Filtering in Constant Time",
#                 IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 16, NO. 9,
#                 SEPTEMBER 2007.
#
# Inputs:
#    a 2d array of uint8's to be median filtered
#    a similarly shaped uint8 masking array with "1" indicating a significant
#         pixel and "0" indicating a pixel to be masked.
#
# Outputs:
#    a 2d median-filtered array
#
##############################################################################

DTYPE_UINT32 = np.uint32
DTYPE_BOOL = np.bool
ctypedef np.uint16_t pixel_count_t

###########
#
# Histograms
#
# There are five separate histograms for the octagonal filter and
# there are two levels (coarse = 16 values, fine = 256 values)
# per histogram. There are four histograms to maintain per position
# representing the four diagonals of the histogram plus one histogram
# for the straight side (which is used for adding and subtracting)
#
###########

cdef struct HistogramPiece:
    np.uint16_t coarse[16]
    np.uint16_t fine[256]

cdef struct Histogram:
    HistogramPiece top_left     # top-left corner
    HistogramPiece top_right    # top-right corner
    HistogramPiece edge         # leading/trailing edge
    HistogramPiece bottom_left  # bottom-left corner
    HistogramPiece bottom_right # bottom-right corner

# The pixel count has the number of pixels histogrammed in
# each of the five compartments for this position. This changes
# because of the mask
#
cdef struct PixelCount:
    pixel_count_t top_left
    pixel_count_t top_right
    pixel_count_t edge
    pixel_count_t bottom_left
    pixel_count_t bottom_right

#
# Stride + coordinates: the info we need when computing
# relative offsets from the octagon center
#
cdef struct SCoord:
    np.int32_t stride   # add the stride to the memory location
    np.int32_t x
    np.int32_t y

cdef struct Histograms:
    void *memory                # pointer to the allocated memory
    Histogram *histogram        # pointer to the histogram memory
    PixelCount *pixel_count     # pointer to the pixel count memory
    np.uint8_t *data            # pointer to the image data
    np.uint8_t *mask            # pointer to the image mask
    np.uint8_t *output          # pointer to the output array
    np.int32_t column_count     # number of columns represented by this
                                # structure
    np.int32_t stripe_length    # number of columns including "radius" before
                                # and after
    np.int32_t row_count        # number of rows available in image
    np.int32_t current_column   # the column being processed
    np.int32_t current_row      # the row being processed
    np.int32_t current_stride   # offset in data and mask to current location
    np.int32_t radius           # the "radius" of the octagon
    np.int32_t a_2              # 1/2 of the length of a side of the octagon
    #
    #
    # The strides are the offsets in the array to the points that need to
    # be added or removed from a histogram to shift from the previous row
    # to the current row.
    # Listed here going clockwise from the trailing edge's top.
    # (-) = needs to be removed
    # (+) = needs to be added
    #
    #          -        -
    #         1.=========2
    #        1.           2
    #       +.             +-   Y
    #      |.               3   |
    #      |.               3   |
    #      -.               X  \|/
    #       5.             4    v
    #        5.           4
    #         +.=========+
    #
    #          x -->
    #
    SCoord last_top_left     # (-) left side of octagon's top - 1 row
    SCoord top_left          # (+) -1 row from trailing edge top
    SCoord last_top_right    # (-) right side of octagon's top - 1 col - 1 row
    SCoord top_right         # (+) -1 col -1 row from leading edge top
    SCoord last_leading_edge # (-) leading edge (right) top stride - 1 row
    SCoord leading_edge      # (+) leading edge bottom stride
    SCoord last_bottom_right # (-) leading edge bottom - 1 col
    SCoord bottom_right      # (+) right side of octagon's bottom - 1 col
    SCoord last_bottom_left  # (-) trailing edge bottom - 1 col
    SCoord bottom_left       # (+) left side of octagon's bottom - 1 col

    np.int32_t row_stride          # stride between one row and the next
    np.int32_t col_stride          # stride between one column and the next
    # The accumulator holds the running histogram
    #
    HistogramPiece accumulator
    #
    # The running count of pixels in the accumulator
    #
    np.uint32_t accumulator_count
    #
    # The percent of pixels within the octagon whose value is
    # less than or equal to the median-filtered value (e.g. for
    # median, this is 50, for lower quartile it's 25)
    #
    np.int32_t percent
    #
    # last_update_column keeps track of the column # of the last update
    # to the fine histogram accumulator. Short-term, the median
    # stays in one coarse block so only one fine histogram might
    # need to be updated
    #
    np.int32_t last_update_column[16]

############################################################################
#
# allocate_histograms - allocates the Histograms structure for the run
#
############################################################################
cdef Histograms *allocate_histograms(np.int32_t rows,
                                     np.int32_t columns,
                                     np.int32_t row_stride,
                                     np.int32_t col_stride,
                                     np.int32_t radius,
                                     np.int32_t percent,
                                     np.uint8_t *data,
                                     np.uint8_t *mask,
                                     np.uint8_t *output):
    cdef:
        unsigned int adjusted_stripe_length = columns + 2*radius + 1
        unsigned int memory_size
        void *ptr
        Histograms *ph
        size_t roundoff
        int a
        SCoord *psc

    memory_size = (adjusted_stripe_length *
                   (sizeof(Histogram) + sizeof(PixelCount))+
                   sizeof(Histograms)+32)
    ptr = malloc(memory_size)
    memset(ptr, 0, memory_size)
    ph  = <Histograms *>ptr
    if not ptr:
        return ph
    ph.memory = ptr
    ptr = <void *>(ph+1)
    ph.pixel_count = <PixelCount *>ptr
    ptr = <void *>(ph.pixel_count + adjusted_stripe_length)
    #
    # Align histogram memory to a 32-byte boundary
    #
    roundoff = <size_t>ptr
    roundoff += 31
    roundoff -= roundoff % 32
    ptr = <void *>roundoff
    ph.histogram = <Histogram *>ptr
    #
    # Fill in the statistical things we keep around
    #
    ph.column_count = columns
    ph.row_count = rows
    ph.current_column = -radius
    ph.stripe_length = adjusted_stripe_length
    ph.current_row = 0
    ph.radius = radius
    ph.percent = percent
    ph.row_stride = row_stride
    ph.col_stride = col_stride
    ph.data = data
    ph.mask = mask
    ph.output = output
    #
    # Compute the coordinates of the significant points
    # (the SCoords)
    #
    # First, the length of a side of an octagon, compared
    # to what we call the radius is:
    #     2*r
    # ----------- =  a
    # (1+sqrt(2))
    #
    # a_2 is the offset from the center to each of the octagon
    # corners
    #
    a = <int>(<np.float64_t>radius * 2.0 / 2.414213)
    a_2 = a / 2
    if a_2 == 0:
        a_2 = 1
    ph.a_2 = a_2
    if radius <= a_2:
        radius = a_2+1
        ph.radius = radius

    ph.last_top_left.x = -a_2
    ph.last_top_left.y = -radius - 1

    ph.top_left.x = -radius
    ph.top_left.y = -a_2 - 1

    ph.last_top_right.x = a_2 - 1
    ph.last_top_right.y = -radius - 1

    ph.top_right.x = radius - 1
    ph.top_right.y = -a_2 - 1

    ph.last_leading_edge.x = radius
    ph.last_leading_edge.y = -a_2 - 1

    ph.leading_edge.x = radius
    ph.leading_edge.y = a_2

    ph.last_bottom_right.x = radius
    ph.last_bottom_right.y = a_2

    ph.bottom_right.x = a_2
    ph.bottom_right.y = radius

    ph.last_bottom_left.x = -radius-1
    ph.last_bottom_left.y = a_2

    ph.bottom_left.x = -a_2-1
    ph.bottom_left.y = radius

    #
    # Set the stride of each SCoord based on its x and y
    #
    set_stride(ph, &ph.last_top_left)
    set_stride(ph, &ph.top_left)
    set_stride(ph, &ph.last_top_right)
    set_stride(ph, &ph.top_right)
    set_stride(ph, &ph.last_leading_edge)
    set_stride(ph, &ph.leading_edge)
    set_stride(ph, &ph.last_bottom_left)
    set_stride(ph, &ph.bottom_left)
    set_stride(ph, &ph.last_bottom_right)
    set_stride(ph, &ph.bottom_right)

    return ph

############################################################################
#
# free_histograms - frees the Histograms structure
#
############################################################################
cdef void free_histograms(Histograms *ph):
    free(ph.memory)

############################################################################
#
# set_stride - set the stride of a SCoord from its X and Y
#
############################################################################

cdef void set_stride(Histograms *ph, SCoord *psc):
    psc.stride = psc.x * ph.col_stride + psc.y * ph.row_stride

############################################################################
#
# <tl,tr,bl,br>_colidx - convert a column index into the histogram
#                        index for a diagonal
#
# The top-right and bottom left diagonals for one row at one column
# become the diagonals for the next column to the right for the next row.
# Conversely, the top-left and bottom right become the diagonals for the
# previous column.
#
# These functions use the current row number to find the index of
# a particular histogram taking this into account. The indices progress
# forward or backward as you go to successive rows.
#
# The histogram array is, in effect, a circular buffer, so the start
# offset is immaterial - we take advantage of this to make sure that
# the numbers computed before taking the modulus are all positive, including
# those that might be done for columns to the left of 0. We add 3* the radius
# here to account for a row of -radius, a column of -radius and a request for
# a column that is "radius" to the left.
#
############################################################################
cdef inline np.int32_t tl_br_colidx(Histograms *ph, np.int32_t colidx):
    return (colidx + 3*ph.radius + ph.current_row) % ph.stripe_length

cdef inline np.int32_t tr_bl_colidx(Histograms *ph, np.int32_t colidx):
    return (colidx + 3*ph.radius + ph.row_count-ph.current_row) % \
           ph.stripe_length

cdef inline np.int32_t leading_edge_colidx(Histograms *ph, np.int32_t colidx):
    return (colidx + 5*ph.radius) % ph.stripe_length

cdef inline np.int32_t trailing_edge_colidx(Histograms *ph, np.int32_t colidx):
    return (colidx + 3*ph.radius - 1) % ph.stripe_length
#
# add16 - add 16 consecutive integers
#
# Add an array of 16 16-bit integers to an accumulator of 16 16-bit integers
#
# TO_DO - optimize using SIMD instructions
#
cdef inline void add16(np.uint16_t *dest, np.uint16_t *src):
    cdef int i
    for i in range(16):
        dest[i] += src[i]

cdef inline void sub16(np.uint16_t *dest, np.uint16_t *src):
    cdef int i
    for i in range(16):
        dest[i] -= src[i]

############################################################################
#
# accumulate_coarse_histogram - accumulate the coarse histogram
#                               at an index into the accumulator
#
# ph     - the Histograms structure that holds the accumulator
# colidx - the index of the column to add
#
############################################################################
cdef inline void accumulate_coarse_histogram(Histograms *ph, np.int32_t colidx):
    cdef:
        int offset

    offset = tr_bl_colidx(ph, colidx)
    if ph.pixel_count[offset].top_right > 0:
        add16(ph.accumulator.coarse, ph.histogram[offset].top_right.coarse)
        ph.accumulator_count += ph.pixel_count[offset].top_right
    offset = leading_edge_colidx(ph, colidx)
    if ph.pixel_count[offset].edge > 0:
        add16(ph.accumulator.coarse, ph.histogram[offset].edge.coarse)
        ph.accumulator_count += ph.pixel_count[offset].edge
    offset = tl_br_colidx(ph, colidx)
    if ph.pixel_count[offset].bottom_right > 0:
        add16(ph.accumulator.coarse, ph.histogram[offset].bottom_right.coarse)
        ph.accumulator_count += ph.pixel_count[offset].bottom_right

############################################################################
#
# deaccumulate_coarse_histogram - subtract the coarse histogram
#                                 for a given column
#
############################################################################
cdef inline void deaccumulate_coarse_histogram(Histograms *ph, np.int32_t colidx):
    cdef:
        int offset
    #
    # The trailing diagonals don't appear until here
    #
    if colidx <= ph.a_2:
        return
    offset = tl_br_colidx(ph, colidx)
    if ph.pixel_count[offset].top_left > 0:
        sub16(ph.accumulator.coarse, ph.histogram[offset].top_left.coarse)
        ph.accumulator_count -= ph.pixel_count[offset].top_left
    #
    # The trailing edge doesn't appear from the border until here
    #
    if colidx > ph.radius:
        offset = trailing_edge_colidx(ph, colidx)
        if ph.pixel_count[offset].edge > 0:
            sub16(ph.accumulator.coarse, ph.histogram[offset].edge.coarse)
            ph.accumulator_count -= ph.pixel_count[offset].edge
    offset = tr_bl_colidx(ph, colidx)
    if ph.pixel_count[offset].bottom_left > 0:
        sub16(ph.accumulator.coarse, ph.histogram[offset].bottom_left.coarse)
        ph.accumulator_count -= ph.pixel_count[offset].bottom_left

############################################################################
#
# accumulate_fine_histogram - accumulate one of the 16 fine histograms
#
############################################################################
cdef inline void accumulate_fine_histogram(Histograms *ph,
                                           np.int32_t colidx,
                                           np.uint32_t fineidx):
    cdef:
        int fineoffset = fineidx * 16
        int offset

    offset = tr_bl_colidx(ph, colidx)
    add16(ph.accumulator.fine + fineoffset,
          ph.histogram[offset].top_right.fine + fineoffset)

    offset = leading_edge_colidx(ph, colidx)
    add16(ph.accumulator.fine + fineoffset,
          ph.histogram[offset].edge.fine + fineoffset)

    offset = tl_br_colidx(ph, colidx)
    add16(ph.accumulator.fine + fineoffset,
          ph.histogram[offset].bottom_right.fine + fineoffset)

############################################################################
#
# deaccumulate_fine_histogram - subtract one of the 16 fine histograms
#
############################################################################
cdef inline void deaccumulate_fine_histogram(Histograms *ph,
                                             np.int32_t colidx,
                                             np.uint32_t fineidx):
    cdef:
        int fineoffset = fineidx * 16
        int offset

    #
    # The trailing diagonals don't appear until here
    #
    if colidx < ph.a_2:
        return

    offset = tl_br_colidx(ph, colidx)
    sub16(ph.accumulator.fine + fineoffset,
          ph.histogram[offset].top_left.fine + fineoffset)

    if colidx >= ph.radius:
        offset = trailing_edge_colidx(ph, colidx)
        sub16(ph.accumulator.fine+fineoffset,
              ph.histogram[offset].edge.fine + fineoffset)

    offset = tr_bl_colidx(ph, colidx)
    sub16(ph.accumulator.fine + fineoffset,
          ph.histogram[offset].bottom_left.fine + fineoffset)

############################################################################
#
# accumulate - add the leading edge and subtract the trailing edge
#
############################################################################

cdef inline void accumulate(Histograms *ph):
    cdef:
        int i
        int j
        np.int32_t accumulator
    accumulate_coarse_histogram(ph, ph.current_column)
    deaccumulate_coarse_histogram(ph, ph.current_column)

############################################################################
#
# update_fine - update one of the fine histograms to the current column
#
# The code has two choices:
#    redo the fine histogram from scratch - this involves accumulating
#         the entire histogram from the top_left.x to the top_right.x,
#         the center (edge) histogram from the trailing edge x to the
#         top_left.x and then computing a histogram of all points between
#         the trailing edge top, the point, (top_left.x,trailing edge top.y)
#         and the top_right and the corresponding triangle in the octagon's
#         lower half.
#
#    accumulate and deaccumulate within the fine histogram from the last
#    column computed.
#
#    The code below only implements the accumulate; redo and the code
#    to choose remains to be done.
############################################################################

cdef inline void update_fine(Histograms *ph, int fineidx):
    cdef:
        int first_update_column = ph.last_update_column[fineidx]+1
        int update_limit = ph.current_column+1
        int i

    for i in range(first_update_column, update_limit):
        accumulate_fine_histogram(ph, i, fineidx)
        deaccumulate_fine_histogram(ph, i, fineidx)
    ph.last_update_column[fineidx] = ph.current_column

############################################################################
#
# update_histogram - update the coarse and fine levels of a histogram
#                    based on addition of one value and subtraction of another
#
# ph         - Histograms pointer (for access to row_count, column_count)
# hist_piece - coarse and fine histogram to update
# pixel_count- pointer to pixel counter for histogram
# last_coord - coordinate and stride of pixel to remove
# coord      - coordinate and stride of pixel to add
#
############################################################################
cdef inline void update_histogram(Histograms *ph,
                                  HistogramPiece *hist_piece,
                                  pixel_count_t *pixel_count,
                                  SCoord *last_coord,
                                  SCoord *coord):
    cdef:
        np.int32_t current_column = ph.current_column
        np.int32_t current_row    = ph.current_row
        np.int32_t current_stride = ph.current_stride
        np.int32_t column_count   = ph.column_count
        np.int32_t row_count      = ph.row_count
        np.uint8_t value
        np.int32_t stride
        np.int32_t x
        np.int32_t y

    x = last_coord.x + current_column
    y = last_coord.y + current_row
    stride = current_stride+last_coord.stride

    if (x >= 0 and x < column_count and
        y >= 0 and y < row_count and
        ph.mask[stride]):
        value = ph.data[stride]
        pixel_count[0] -= 1
        hist_piece.fine[value] -= 1
        hist_piece.coarse[value / 16] -= 1

    x = coord.x + current_column
    y = coord.y + current_row
    stride = current_stride + coord.stride

    if (x >= 0 and x < column_count and
        y >= 0 and y < row_count and
        ph.mask[stride]):
        value = ph.data[stride]
        pixel_count[0] += 1
        hist_piece.fine[value] += 1
        hist_piece.coarse[value / 16] += 1

############################################################################
#
# update_current_location - update the histograms at the current location
#
############################################################################
cdef inline void update_current_location(Histograms *ph):
    cdef:
        np.int32_t current_column = ph.current_column
        np.int32_t radius = ph.radius
        np.int32_t top_left_off = tl_br_colidx(ph, current_column)
        np.int32_t top_right_off = tr_bl_colidx(ph, current_column)
        np.int32_t bottom_left_off = tr_bl_colidx(ph, current_column)
        np.int32_t bottom_right_off = tl_br_colidx(ph, current_column)
        np.int32_t leading_edge_off = leading_edge_colidx(ph, current_column)
        np.int32_t *coarse_histogram
        np.int32_t *fine_histogram
        np.int32_t last_xoff
        np.int32_t last_yoff
        np.int32_t last_stride
        np.int32_t xoff
        np.int32_t yoff
        np.int32_t stride

    update_histogram(ph, &ph.histogram[top_left_off].top_left,
                     &ph.pixel_count[top_left_off].top_left,
                     &ph.last_top_left,
                     &ph.top_left)

    update_histogram(ph, &ph.histogram[top_right_off].top_right,
                     &ph.pixel_count[top_right_off].top_right,
                     &ph.last_top_right,
                     &ph.top_right)

    update_histogram(ph, &ph.histogram[bottom_left_off].bottom_left,
                     &ph.pixel_count[bottom_left_off].bottom_left,
                     &ph.last_bottom_left,
                     &ph.bottom_left)

    update_histogram(ph, &ph.histogram[bottom_right_off].bottom_right,
                     &ph.pixel_count[bottom_right_off].bottom_right,
                     &ph.last_bottom_right,
                     &ph.bottom_right)

    update_histogram(ph, &ph.histogram[leading_edge_off].edge,
                     &ph.pixel_count[leading_edge_off].edge,
                     &ph.last_leading_edge,
                     &ph.leading_edge)

############################################################################
#
# find_median - search the current accumulator for the median
#
############################################################################

cdef inline np.uint8_t find_median(Histograms *ph):
    cdef:
        np.uint32_t pixels_below      # of pixels below the median
        int i
        int j
        int k
        np.uint32_t accumulator

    if ph.accumulator_count == 0:
        return 0
    pixels_below = ((ph.accumulator_count * ph.percent + 50)
                    / 100) # +50 for roundoff
    if pixels_below > 0:
        pixels_below -= 1

    accumulator = 0
    for i in range(16):
        accumulator += ph.accumulator.coarse[i]
        if accumulator > pixels_below:
            break

    accumulator -= ph.accumulator.coarse[i]
    update_fine(ph, i)
    for j in range(i*16,(i+1)*16):
        accumulator += ph.accumulator.fine[j]
        if accumulator > pixels_below:
            return <np.uint8_t> j

    return 0

############################################################################
#
# c_median_filter - median filter algorithm
#
# rows    - # of rows in each array
# columns - # of columns in each array
# row_stride - stride from one row to the next in each array
# col_stride - stride from one column to the next in each array
# radius - radius of circle inscribed into octagon
# percent - "median" cutoff: 50 = median, 25 = lower quartile, etc
# data - array of image pixels to be filtered
# mask - mask of significant pixels
# output - array to be filled with filtered pixels
#
############################################################################
cdef int c_median_filter(np.int32_t rows,
                         np.int32_t columns,
                         np.int32_t row_stride,
                         np.int32_t col_stride,
                         np.int32_t radius,
                         np.int32_t percent,
                         np.uint8_t *data,
                         np.uint8_t *mask,
                         np.uint8_t *output):
    cdef:
        Histograms *ph
        Histogram  *phistogram
        int row
        int col
        int i
        np.int32_t top_left_off
        np.int32_t top_right_off
        np.int32_t bottom_left_off
        np.int32_t bottom_right_off

    ph = allocate_histograms(rows, columns, row_stride, col_stride,
                             radius, percent, data, mask, output)
    if not ph:
       return 1

    for row in range(-radius, rows):
        #
        # Initialize the starting diagonal histograms to zero. The leading
        # and trailing histograms descend from above and so are initialized
        # when memory is initially set to zero. The diagonals move in
        # from the left (top left and bottom right) and right (top right
        # and bottom left). One of each needs to be initialized at the
        # start of each row
        #
        tl_br_off = tl_br_colidx(ph, -radius)
        tr_bl_off = tr_bl_colidx(ph, columns + radius - 1)

        memset(&ph.histogram[tl_br_off].top_left, 0, sizeof(HistogramPiece))
        memset(&ph.histogram[tl_br_off].bottom_right, 0, sizeof(HistogramPiece))
        memset(&ph.histogram[tr_bl_off].top_right, 0, sizeof(HistogramPiece))
        memset(&ph.histogram[tr_bl_off].bottom_left, 0, sizeof(HistogramPiece))

        ph.pixel_count[tl_br_off].top_left = 0
        ph.pixel_count[tl_br_off].bottom_right = 0
        ph.pixel_count[tr_bl_off].top_right = 0
        ph.pixel_count[tr_bl_off].bottom_left = 0

        #
        # Initialize the accumulator (octagon histogram) to zero
        #
        memset(&ph.accumulator, 0, sizeof(ph.accumulator))
        ph.accumulator_count = 0
        for i in range(16):
            ph.last_update_column[i] = -radius-1
        #
        # Initialize the current stride to the beginning of the row
        #
        ph.current_row = row
        #
        # Update locations and coarse accumulator for the octagon
        # for points before 0
        #
        for col in range(-radius, 0 if row >=0 else columns+radius):
            ph.current_column = col
            ph.current_stride = row * row_stride + col * col_stride
            update_current_location(ph)
            accumulate(ph)
        #
        # Update locations and coarse accumulator and compute
        # the median for points between 0 and "columns"
        #
        if row >= 0:
            for col in range(0, columns):
                ph.current_column = col
                ph.current_stride = row * row_stride + col * col_stride
                update_current_location(ph)
                accumulate(ph)
                ph.output[ph.current_stride] = find_median(ph)
            for col in range(columns, columns+radius):
                ph.current_column = col
                ph.current_stride = row * row_stride + col * col_stride
                update_current_location(ph)


    free_histograms(ph)
    return 0

def median_filter(
    np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] data,
    np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
    np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] output,
    int radius,
    np.int32_t percent):
    """Median filter with octagon shape and masking.

    Parameters
    ----------
    data : (M,N) ndarray, dtype uint8
        Input image.
    mask : (M,N) array, dtype uint8
        A value of 1 indicates a significant pixel, 0
        that a pixel is masked.
    output : (M,N) array, dtype uint8
        Array of same size as the input in which to store
        the filtered image.
    radius : int
        Radius of the inscribed circle to the octagon.
    percent : int, optional
        The unmasked pixels within the octagon are sorted, and the
        value at the `percent`-th index chosen.  For example, the
        default value of 50 chooses the median pixel.

    """
    if percent < 0:
        raise ValueError('Median filter percent = %d is less than zero' % \
                         percent)
    if percent > 100:
        raise ValueError('Median filter percent = %d is greater than 100' % \
                         percent)
    if data.shape[0] != mask.shape[0] or data.shape[1] != mask.shape[1]:
        raise ValueError('Data shape (%d, %d) is not mask shape (%d, %d)' %
                         (data.shape[0], data.shape[1],
                          mask.shape[0], mask.shape[1]))
    if data.shape[0] != output.shape[0] or data.shape[1] != output.shape[1]:
        raise ValueError('Data shape (%d, %d) is not output shape (%d, %d)' %
                         (data.shape[0], data.shape[1],
                          output.shape[0], output.shape[1]))
    if c_median_filter(data.shape[0], data.shape[1],
                       data.strides[0], data.strides[1],
                       radius, percent,
                       <np.uint8_t *>data.data,
                       <np.uint8_t *>mask.data,
                       <np.uint8_t *>output.data):
        raise MemoryError('Failed to allocate scratchpad memory')
