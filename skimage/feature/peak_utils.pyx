#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport malloc, realloc, free


ctypedef fused dtype_t:
    cnp.uint8_t
    cnp.uint16_t
    cnp.double_t


# -- Algorithm A ---------------------------------------------------------------

def maxima2d_A(dtype_t[:, ::1] image not None):
    """Detect local maxima in image.

    Notes
    -----
    Basic algorithm / idea:

    1. Iterate each pixel (row first). Find local maxima, including flat ones,
       in each row and mark those pixels.
    2. For each marked pixel, find all surrounding pixels with same value that
       are part of the same plateau (`_fill_plateau_A`).
       - If this plateau is bordered by higher value -> mark with 0
       - else plateau is true maxima -> mark with 1
    3. Return array with flags (only 0 and 1 remain).
    """
    cdef:
        unsigned char[:, ::1] flags
        Py_ssize_t* r_buffer
        Py_ssize_t* c_buffer
        Py_ssize_t candidate_count, buffer_size, r_max, c_max, r, c, c_ahead

    # Flags: 0 - no maximum, 2 - potential maximum, 1 - definite maximum
    flags = np.zeros_like(image, dtype=np.uint8)

    with nogil:
        candidate_count = 0  # Number of potential maxima
        r_max = image.shape[0] - 1
        c_max = image.shape[1] - 1

        # Find local maxima row-wise (ignore other dimension)
        for r in range(r_max + 1):
            c = 1
            while c < c_max:
                # If previous sample isn't larger -> possible peak
                if image[r, c - 1] <= image[r, c]:
                    # Find next sample that is unequal to current one or last
                    # sample in row
                    c_ahead = c + 1
                    while c_ahead < c_max and image[r, c_ahead] == image[r, c]:
                        c_ahead += 1
                    # If next sample is not larger -> maximum found
                    if image[r, c_ahead] <= image[r, c]:
                        # Mark samples as potential maximum
                        flags[r, c:c_ahead] = 2
                        candidate_count += c_ahead - c
                    # Skip already evaluated samples
                    c = c_ahead
                else:
                    c += 1

            # Evaluate first and last sample in each row. Only mark as potential
            # maximum if: bordering sample is 2 and has the same value or if
            # edge sample is larger than bordering sample
            if (
                flags[r, 1] == 2 and
                image[r, 1] == image[r, 0] or
                image[r, 1] < image[r, 0]
            ):
                flags[r, 0] = 2
                candidate_count += 1
            if (
                flags[r, c_max - 1] == 2 and
                image[r, c_max - 1] == image[r, c_max] or
                image[r, c_max - 1] < image[r, c_max]
            ):
                flags[r, c_max] = 2
                candidate_count += 1

        # Initialize a buffer used to queue positions while evaluating each
        # potential maximum (flagged with 2); the initial size is the number
        # of potential maxima / candidates.
        buffer_size = candidate_count
        r_buffer = <Py_ssize_t*>malloc(buffer_size * sizeof(Py_ssize_t))
        c_buffer = <Py_ssize_t*>malloc(buffer_size * sizeof(Py_ssize_t))
        if not r_buffer or not c_buffer:
            with gil:
                raise MemoryError()
        try:
            for r in range(r_max + 1):
                c = 0
                while c <= c_max:
                    # If sample is flagged as a potential maxima
                    if flags[r, c] == 2:
                        # Find all samples part of the plateau and fill with 0
                        # or 1 depending on whether it's a true maximum
                        _fill_plateau_A(image, flags, &r_buffer, &c_buffer,
                                        &buffer_size, r, c, r_max, c_max)
                    c += 1
        finally:
            free(r_buffer)
            free(c_buffer)

    # Return original python object which is base of memoryview
    return flags.base


cdef inline void \
    _fill_plateau_A(dtype_t[:, ::1] image, unsigned char[:, ::1] flags,
                    Py_ssize_t** r_buffer, Py_ssize_t** c_buffer,
                    Py_ssize_t* buffer_size, Py_ssize_t r, Py_ssize_t c,
                    Py_ssize_t r_max, Py_ssize_t c_max) nogil:
    """Fill with 1 if plateau is local maximum else with 0."""
    cdef:
        dtype_t h
        Py_ssize_t p_next, p_valid
        unsigned char true_maximum
        Py_ssize_t steps_r[8]
        Py_ssize_t steps_c[8]

    h = image[r, c]
    true_maximum = 1 # Boolean flag
    # Steps that shift a position to all 8 neighbouring samples
    steps_r = ( 0, -1, 0, 0, 1, 1,  0,  0)
    steps_c = (-1,  0, 1, 1, 0, 0, -1, -1)

    # Push starting position to buffer
    p_next = 0  # Pointer to next evaluated position
    p_valid = 0  # Pointer to most recent position in buffer (end of valid area)
    r_buffer[0][p_valid] = r  # Dereference pointer with "[0]"
    c_buffer[0][p_valid] = c
    # Mark as true maximum according to initial guess, also works as a flag that
    # a sample is queued / in the buffer
    flags[r, c] = 1

    # Break loop if all queued positions were evaluated
    while p_next <= p_valid:
        # Load next position in buffer and shift pointer to next element
        r = r_buffer[0][p_next]
        c = c_buffer[0][p_next]
        p_next += 1

        # Look at all  8 neighbouring samples
        for i in range(8):
            r += steps_r[i]
            c += steps_c[i]
            if not (0 <= r <= r_max and 0 <= c <= c_max):
                continue  # Skip "neighbours" outside image

            # If neighbour wasn't queued already (!= 1) and is part of the
            # current plateau (== h)
            if flags[r, c] != 1 and image[r, c] == h:
                # Push neighbour to buffer
                p_valid += 1
                if p_valid >= buffer_size[0]:
                    _double_buffer(r_buffer, c_buffer, buffer_size)
                r_buffer[0][p_valid] = r
                c_buffer[0][p_valid] = c
                # Flag as true maximum (initial guess) so that it isn't queued
                # multiple times
                flags[r, c] = 1

            # If neighbouring sample is larger -> plateau isn't a maximum
            elif image[r, c] > h:
                true_maximum = 0

    if not true_maximum:
        # Initial guess was wrong -> replace 1 with 0 for plateau
        while p_valid >= 0:
            flags[r_buffer[0][p_valid], c_buffer[0][p_valid]] = 0
            p_valid -= 1



# -- Algorithm B ---------------------------------------------------------------

def maxima2d_B(dtype_t[:, ::1] image not None):
    """Detect local maxima in image.

    Notes
    -----
    Basic algorithm / idea:

    1. Iterate each pixel (row first). Find all surrounding pixels with same
       value that are part of the plateau (`_fill_plateau_B`).
       - If this plateau is bordered by higher value -> mark with 0
       - else plateau is true maxima -> mark with 1
    2. Return array with flags (only 0 and 1 remain).
    """
    cdef:
        unsigned char[:, ::1] flags
        Py_ssize_t* r_buffer
        Py_ssize_t* c_buffer
        Py_ssize_t buffer_size
        Py_ssize_t r_max, c_max, r, c, c_ahead

    # Flags: 0 - no maximum, 2 - potential maximum, 1 - definite maximum
    flags = np.ones_like(image, dtype=np.uint8) * 2

    with nogil:
        r_max = image.shape[0] - 1
        c_max = image.shape[1] - 1

        # Initialize a buffer used to queue positions while evaluating each
        # sample; the initial size is just a guess
        buffer_size = 64
        r_buffer = <Py_ssize_t *>malloc(buffer_size * sizeof(Py_ssize_t))
        c_buffer = <Py_ssize_t *>malloc(buffer_size * sizeof(Py_ssize_t))
        if not r_buffer or not c_buffer:
            with gil:
                raise MemoryError()

        try:
            for r in range(r_max + 1):
                c = 0
                while c <= c_max:
                    # If status of sample isn't known already (not 0 or 1)
                    if flags[r, c] == 2:
                        # Find all samples part of the plateau and fill with 0
                        # or 1 depending on whether it's a true maximum
                        _fill_plateau_B(image, flags, &r_buffer, &c_buffer,
                                        &buffer_size, r, c, r_max, c_max)
                    c += 1
        finally:
            free(r_buffer)
            free(c_buffer)
    # Return original python object which is base of memoryview
    return flags.base


cdef inline void \
    _fill_plateau_B(dtype_t[:, ::1] image, unsigned char[:, ::1] flags,
                    Py_ssize_t** r_buffer, Py_ssize_t** c_buffer,
                    Py_ssize_t* buffer_size, Py_ssize_t r, Py_ssize_t c,
                    Py_ssize_t r_max, Py_ssize_t c_max) nogil:
    """Fill with 1 if plateau is local maximum else with 0."""
    cdef:
        dtype_t h
        unsigned char true_maximum
        Py_ssize_t steps_r[8]
        Py_ssize_t steps_c[8]
        Py_ssize_t p_next, p_valid

    h = image[r, c]
    true_maximum = 1  # Boolean flag
    # Steps that shift a position to all 8 neighbouring samples
    steps_r = ( 0, -1, 0, 0, 1, 1,  0,  0)
    steps_c = (-1,  0, 1, 1, 0, 0, -1, -1)

    # Push starting position to buffer
    p_next = 0  # Pointer to next evaluated position
    p_valid = 0  # Pointer to most recent position in buffer (end of valid area)
    r_buffer[0][p_valid] = r  # Dereference pointer with "[0]"
    c_buffer[0][p_valid] = c
    # Assume that majority of samples isn't part of a maximum, thus mark
    # initially with 0, also works as a flag that a sample is queued already
    flags[r, c] = 0

    # Break loop if all pending positions were evaluated
    while p_next <= p_valid:
        # Load next position in buffer and shift pointer to next element
        r = r_buffer[0][p_next]
        c = c_buffer[0][p_next]
        p_next += 1

        # Look at all  8 neighbouring samples
        for i in range(8):
            r += steps_r[i]
            c += steps_c[i]
            if not (0 <= r <= r_max and 0 <= c <= c_max):
                continue  # Skip "neighbours" outside image

            # If neighbour wasn't queued already (== 2) and is part of the
            # current plateau (== h)
            if flags[r, c] == 2 and image[r, c] == h:
                # Push neighbour to buffer
                p_valid += 1
                if p_valid >= buffer_size[0]:
                    _double_buffer(r_buffer, c_buffer, buffer_size)
                r_buffer[0][p_valid] = r
                c_buffer[0][p_valid] = c
                # Flag as "not maximum" (initial guess) so that it isn't queued
                # multiple times
                flags[r, c] = 0

            # If neighbouring sample is larger -> plateau isn't a maximum
            elif image[r, c] > h:
                true_maximum = 0

    if true_maximum:
        # Initial guess was wrong -> replace 0 with 1 for plateau
        while p_valid >= 0:
            flags[r_buffer[0][p_valid], c_buffer[0][p_valid]] = 1
            p_valid -= 1


# -- Helper --------------------------------------------------------------------

cdef void _double_buffer(Py_ssize_t** r_buffer, Py_ssize_t** c_buffer,
                         Py_ssize_t* buffer_size) nogil:
    """Safely double buffer size."""
    cdef:
        Py_ssize_t* new_r_buffer
        Py_ssize_t* new_c_buffer
    buffer_size[0] *= 2
    new_r_buffer = <Py_ssize_t *>realloc(r_buffer[0], buffer_size[0] * sizeof(Py_ssize_t))
    new_c_buffer = <Py_ssize_t *>realloc(c_buffer[0], buffer_size[0] * sizeof(Py_ssize_t))
    if not new_r_buffer or not new_c_buffer:
        with gil:
            raise MemoryError()
    # Only replace pointer to buffer if reallocation was successful
    r_buffer[0] = new_r_buffer
    c_buffer[0] = new_c_buffer
