include "npy_iterator.pxi"


def run_iterator_tests():
    arr = np.arange(720).reshape(6, 5, 4, 3, 2).astype(np.int)

    cdef Iterator iter = Iterator(arr)
    cdef int sum

    sum = 0

    while iter.has_next():
        sum += iter.get_int()
        iter.next()

    del iter
    assert np.sum(arr) == sum
