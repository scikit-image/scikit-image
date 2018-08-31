cdef extern from "conditional_omp.h":
    ctypedef struct omp_lock_t:
        pass
    extern void omp_init_lock(omp_lock_t *) nogil
    extern void omp_destroy_lock(omp_lock_t *) nogil
    extern void omp_set_lock(omp_lock_t *) nogil
    extern void omp_unset_lock(omp_lock_t *) nogil
    extern int omp_test_lock(omp_lock_t *) nogil
    cdef int have_openmp