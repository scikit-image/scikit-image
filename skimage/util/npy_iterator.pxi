from cpython.ref cimport PyObject
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport free

cnp.import_array()


cdef extern from "numpy/ndarraytypes.h":
    ctypedef struct PyArrayIterObject:
        int index
        int size
        char* dataptr
        cnp.npy_intp ndim


cdef extern from "numpy/arrayobject.h":
    cdef PyObject* PyArray_IterNew(PyObject* arr)
    ctypedef struct PyArrayObject:
        pass
    cdef void PyArray_ITER_NEXT(PyArrayIterObject* iter)


cdef class Iterator:
    """ A cython numpy array iterator.

    This class wraps the Numpy array iterator API in a cython interface.
    You can intialize this with any numpy array object and use the
    corresponding `get_*` function to get each element.

    Examples
    --------

    >>> cdef iter = Iterator(numpy.arange(9).reshape(3,3).astype(np.int))
    >>> while iter.has_next():
    >>>     print(iter.get_int())
    >>>     iter.next()
    >>> del iter # Delete iterator after finishing with it

    """
    cdef PyArrayObject* array
    cdef PyArrayIterObject* iter;
    cdef PyObject* obj
    cdef Py_ssize_t ndim

    def __init__(self, arr):
        """
        Initialize the iterator to iterate over an array

        Paramaters
        ----------
        arr : ndarray
            The array to be iterated over.
        """
        self.iter = <PyArrayIterObject*>PyArray_IterNew(<PyObject*>arr);
        self.ndim = arr.ndim

    cdef inline bint has_next(self):
        """ Indicates whether or not the iterator has finshed iteration.

        Returns
        -------
        is_next : boolean
            `True` if there are more elements to be iterated over. `False`
            otherwise
        """
        return (self.iter.index < self.iter.size)

    cdef next(self):
        """ Points the iterator to the next element of the array.

        The behaviour of this function when `has_next` is `False`, is undefined
        and may result in a crash.
        """
        PyArray_ITER_NEXT(self.iter)

    cdef inline cnp.float_t get_float(self):
        """ Returns the current element as `float`. """
        cdef cnp.float_t *address = <cnp.float_t*>self.iter.dataptr
        return address[0]

    cdef inline cnp.int_t get_int(self):
        """ Returns the current element as `int`. """
        cdef cnp.int_t *address = <cnp.int_t*>self.iter.dataptr
        return address[0]

    def __dealloc__(self):
        """ Frees up the memory used by the iterator. """
        free(self.iter)
