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

ctypedef struct QueueItem:
    Py_ssize_t r
    Py_ssize_t c


cdef:
    struct Queue:
        # A queue whose items can be restored after consumption.
        QueueItem* _buffer_ptr
        Py_ssize_t _buffer_size, _index_valid, _index_consumed

    void q_init(Queue* self, Py_ssize_t buffer_size) nogil:
        """Initialize the queue and its buffer."""
        self._buffer_ptr = <QueueItem*>malloc(buffer_size * sizeof(QueueItem))
        if not self._buffer_ptr:
            with gil:
                raise MemoryError("couldn't allocate buffer")
        self._buffer_size = buffer_size
        self._index_consumed = -1
        self._index_valid = -1

    void q_free_buffer(Queue* self) nogil:
        """Free the buffer of the queue."""
        free(self._buffer_ptr)

    inline void q_restore(Queue* self) nogil:
        """Restore all consumed items to the queue."""
        self._index_consumed = -1

    inline void q_clear(Queue* self) nogil:#
        """Remove all items in the queue."""
        self._index_consumed = -1
        self._index_valid = -1

    inline void q_push(Queue* self, QueueItem* item_ptr) nogil:
        """Enqueue a new item."""
        self._index_valid += 1
        if self._buffer_size <= self._index_valid:
            _q_grow_buffer(self)
        self._buffer_ptr[self._index_valid] = item_ptr[0]

    inline unsigned char q_pop(Queue* self, QueueItem* item_ptr) nogil:
        """Dequeue / consume an item."""
        if 0 <= self._index_consumed + 1 <= self._index_valid:
            self._index_consumed += 1
            item_ptr[0] = self._buffer_ptr[self._index_consumed]
            return 1
        return 0

    void _q_grow_buffer(Queue* self) nogil:
        """Double the memory used for the buffer."""
        cdef QueueItem* new_buffer

        # TODO prevent integer overflow!
        self._buffer_size *= 2
        new_buffer_ptr = <QueueItem*>realloc(
            self._buffer_ptr,
            self._buffer_size * sizeof(QueueItem)
        )
        if not new_buffer_ptr:
            with gil:
                raise MemoryError("couldn't reallocate buffer")
        self._buffer_ptr = new_buffer_ptr


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
        Queue queue
        unsigned char[:, ::1] flags
        Py_ssize_t r_max, c_max, r, c, c_ahead

    # Flags: 0 - no maximum, 2 - potential maximum, 1 - definite maximum
    flags = np.zeros_like(image, dtype=np.uint8)

    with nogil:
        r_max = image.shape[0] - 1
        c_max = image.shape[1] - 1

        # Compare values row-wise (ignore other dimension)
        for r in range(1, r_max):
            c = 1
            while c < c_max:
                # If previous sample isn't larger -> possible peak
                if image[r, c - 1] < image[r, c]:
                    # Find next sample that is unequal to current one or last
                    # sample in row
                    c_ahead = c + 1
                    while c_ahead < c_max and image[r, c_ahead] == image[r, c]:
                        c_ahead += 1
                    # If next sample is not larger -> maximum found
                    if image[r, c_ahead] < image[r, c]:
                        # Mark samples as potential maximum
                        flags[r, c:c_ahead] = 2
                    # Skip already evaluated samples
                    c = c_ahead
                else:
                    c += 1

        # Initialize a buffer used to queue positions while evaluating each
        # potential maximum (flagged with 2); the initial size is the number
        # of potential maxima
        q_init(&queue, 64)
        try:
            for r in range(1, r_max ):
                c = 1
                while c < c_max:
                    # If sample is flagged as a potential maxima
                    if flags[r, c] == 2:
                        # Find all samples part of the plateau and fill with 0
                        # or 1 depending on whether it's a true maximum
                        _fill_plateau_A(image, flags, &queue, r, c, r_max, c_max)
                    c += 1
        finally:
            q_free_buffer(&queue)

    # Return original python object which is base of memoryview
    return flags.base


cdef inline void \
    _fill_plateau_A(dtype_t[:, ::1] image, unsigned char[:, ::1] flags,
                    Queue* queue_ptr, Py_ssize_t r, Py_ssize_t c,
                    Py_ssize_t r_max, Py_ssize_t c_max) nogil:
    """Fill with 1 if plateau is local maximum else with 0."""
    cdef:
        dtype_t h
        unsigned char true_maximum
        Py_ssize_t steps_r[8]
        Py_ssize_t steps_c[8]
        QueueItem item

    h = image[r, c]
    true_maximum = 1 # Boolean flag
    # Steps that shift a position to all 8 neighbouring samples
    steps_r = ( 0, -1, 0, 0, 1, 1,  0,  0)
    steps_c = (-1,  0, 1, 1, 0, 0, -1, -1)
    item.r = r
    item.c = c
    q_clear(queue_ptr)

    # Push starting position to buffer
    q_push(queue_ptr, &item)
    # Mark as true maximum according to initial guess, also works as a flag that
    # a sample is queued / in the buffer
    flags[r, c] = 1

    # Break loop if all queued positions were evaluated
    while q_pop(queue_ptr, &item):

        # Look at all  8 neighbouring samples
        for i in range(8):
            item.r += steps_r[i]
            item.c += steps_c[i]
            if not (0 <= item.r <= r_max and 0 <= item.c <= c_max):
                continue  # Skip "neighbours" outside image

            # If neighbour wasn't queued already (!= 1) and is part of the
            # current plateau (== h)
            if flags[item.r, item.c] != 1 and image[item.r, item.c] == h:
                if (
                    item.r == 0 or item.r == r_max or
                    item.c == 0 or item.c == c_max
                ):
                    true_maximum = 0

                # Push neighbour to buffer
                q_push(queue_ptr, &item)
                # Flag as true maximum (initial guess) so that it isn't queued
                # multiple times
                flags[item.r, item.c] = 1

            # If neighbouring sample is larger -> plateau isn't a maximum
            elif image[item.r, item.c] > h:
                true_maximum = 0

    if not true_maximum:
        q_restore(queue_ptr)
        # Initial guess was wrong -> replace 1 with 0 for plateau
        while q_pop(queue_ptr, &item):
            flags[item.r, item.c] = 0



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
        Queue queue
        Py_ssize_t r_max, c_max, r, c, c_ahead

    # Flags: 0 - no maximum, 2 - potential maximum, 1 - definite maximum
    flags = np.empty_like(image, dtype=np.uint8)
    flags[:, :] = 2

    with nogil:
        r_max = image.shape[0] - 1
        c_max = image.shape[1] - 1

        # Initialize a buffer used to queue positions while evaluating each
        # sample; the initial size is just a guess
        q_init(&queue, 64)
        try:
            for r in range(r_max + 1):
                c = 0
                while c <= c_max:
                    # If status of sample isn't known already (not 0 or 1)
                    if flags[r, c] == 2:
                        # Find all samples part of the plateau and fill with 0
                        # or 1 depending on whether it's a true maximum
                        _fill_plateau_B(image, flags, &queue, r, c, r_max, c_max)
                    c += 1
        finally:
            q_free_buffer(&queue)
    # Return original python object which is base of memoryview
    return flags.base


cdef inline void \
    _fill_plateau_B(dtype_t[:, ::1] image, unsigned char[:, ::1] flags,
                    Queue* queue_ptr, Py_ssize_t r, Py_ssize_t c,
                    Py_ssize_t r_max, Py_ssize_t c_max) nogil:
    """Fill with 1 if plateau is local maximum else with 0."""
    cdef:
        dtype_t h
        unsigned char true_maximum
        Py_ssize_t steps_r[8]
        Py_ssize_t steps_c[8]
        QueueItem item

    h = image[r, c]
    if r == 0 or r == r_max or c == 0 or c == c_max:
        true_maximum = 0
    else:
        true_maximum = 1  # Boolean flag

    # Steps that shift a position to all 8 neighbouring samples
    steps_r = ( 0, -1, 0, 0, 1, 1,  0,  0)
    steps_c = (-1,  0, 1, 1, 0, 0, -1, -1)
    q_clear(queue_ptr)

    # Push starting position to buffer
    item.r = r
    item.c = c
    q_push(queue_ptr, &item)
    # Assume that majority of samples isn't part of a maximum, thus mark
    # initially with 0, also works as a flag that a sample is queued already
    flags[r, c] = 0

    # Break loop if all pending positions were evaluated
    while q_pop(queue_ptr, &item):
        # Look at all  8 neighbouring samples
        for i in range(8):
            item.r += steps_r[i]
            item.c += steps_c[i]

            if not (0 <= item.r <= r_max and 0 <= item.c <= c_max):
                continue  # Skip "neighbours" outside image

            # If neighbour wasn't queued already (== 2) and is part of the
            # current plateau (== h)
            if flags[item.r, item.c] == 2 and image[item.r, item.c] == h:
                if (
                    item.r == 0 or item.r == r_max or
                    item.c == 0 or item.c == c_max
                ):
                    true_maximum = 0
                # Push neighbour to buffer if plateau might still be a
                # maximum
                q_push(queue_ptr, &item)
                # Flag as "not maximum" (initial guess) so that it isn't queued
                # multiple times
                flags[item.r, item.c] = 0

            # If neighbouring sample is larger -> plateau isn't a maximum
            elif image[item.r, item.c] > h:
                true_maximum = 0

    if true_maximum:
        q_restore(queue_ptr)
        # Initial guess was wrong -> replace 0 with 1 for plateau
        while q_pop(queue_ptr, &item):
            flags[item.r, item.c] = 1
