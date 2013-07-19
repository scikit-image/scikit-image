#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=False
#cython: wraparound=True

from __future__ import print_function

import numpy as np

from libc.math cimport M_PI, lround
from libc.stdlib cimport malloc, free, abort
cimport numpy as cnp


ctypedef void * QueueValue

cdef extern from "queue.h":
    ctypedef struct Queue:
        pass
    Queue *queue_new()
    void queue_free(Queue *queue)
    int queue_push_head(Queue *queue, QueueValue data)
    QueueValue queue_pop_head(Queue *queue)
    QueueValue queue_peek_head(Queue *queue)
    int queue_push_tail(Queue *queue, QueueValue data)
    QueueValue queue_pop_tail(Queue *queue)
    QueueValue queue_peek_tail(Queue *queue)
    int queue_is_empty(Queue *queue)


cdef enum:
    UINT16_MAX = 1 << 16 - 1
    UNDEFINED = -(1 << 30)

PERIODS_UNDEFINED = UNDEFINED


cdef struct branch_cut:
    # Positions of the cuts are after the pixel with the same indices in
    # in the phase image
    cnp.uint8_t vcut         # cut normal to 1st dimension
    cnp.uint8_t hcut         # cut normal to 0th dimension
    cnp.uint16_t visit_code  # variable to mark an intersection as visited
    cnp.uint32_t residue_no  # index into an array of residues


cdef struct QueuedLocation:
    Py_ssize_t i
    Py_ssize_t j
    QueuedLocation *came_from


branch_cut_dtype = np.dtype([('vcut', np.uint8), ('hcut', np.uint8),
                             ('visit_code', np.uint16),
                             ('residue_no', np.uint32)])


cdef inline double _phase_difference(double from_, double to):
    cdef double d = to - from_
    if d > M_PI:
        d -= 2 * M_PI
    elif d < -M_PI:
        d += 2 * M_PI
    return d


cdef inline int _phase_period_increment(double from_, double to):
    cdef double d = to - from_
    if d > M_PI:
        return -1
    elif d < -M_PI:
        return 1
    else:
        return 0


def find_phase_residues_cy(double[:, ::1] image):
    residues_array = np.zeros((image.shape[0], image.shape[1]),
                              dtype=np.int8, order='C')
    cdef:
        cnp.int8_t[:, ::1] residues = residues_array
        Py_ssize_t i, j
        double s
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = (_phase_difference(image[i - 1, j - 1], image[i - 1, j])
                 + _phase_difference(image[i - 1, j], image[i, j])
                 + _phase_difference(image[i, j], image[i, j - 1])
                 + _phase_difference(image[i, j - 1], image[i - 1, j - 1]))
            residues[i, j] = lround(s / (2 * M_PI))
    return residues_array


def _prepare_branch_cuts_cy(branch_cut[:, ::1] branch_cuts,
                         cnp.int_t[::1] residue_storage, Py_ssize_t index,
                         cnp.int8_t[:, ::1] residues,
                         cnp.uint8_t[:, ::1] mask):
    '''Prepare the branch_cuts structures for branch cut finding.

    Parameters
    ----------
    branch_cuts : output parameter, will be modified in-place
        Branch cut struct
    residue_storage : output parameter, will be modified in-place
        Storage for residues
    index :
        Element of ``residue_storage`` where the first residue should be stored
    residues :
        Residues as determined by find_phase_residues
    mask :
        Mask of the ``residues`` array; masked entries will not be stored.
    '''
    cdef:
        Py_ssize_t i, j
    for i in range(residues.shape[0]):
        for j in range(residues.shape[1]):
            if residues[i, j] != 0 and mask[i, j] == 0:
                # Found an unmasked residue
                residue_storage[index] = residues[i, j]
                branch_cuts[i, j].residue_no = index
                index += 1
    return None


cdef inline Py_ssize_t normalize_coordinate(int c, Py_ssize_t dim):
    if c < 0:
        c += dim
    elif c >= dim:
        c -= dim
    return c


cdef inline Py_ssize_t maybe_add_location(int i, int j,
                                          QueuedLocation *coming_from,
                                          branch_cut[:, ::1] branch_cuts,
                                          QueuedLocation * location_buffer,
                                          Py_ssize_t location_index,
                                          Queue *queue,
                                          cnp.uint16_t visit_code):
    cdef:
        QueuedLocation *l
    i = normalize_coordinate(i, branch_cuts.shape[0])
    j = normalize_coordinate(j, branch_cuts.shape[1])
    print('\t\tNew location: [%d, %d]' % (i, j))
    if branch_cuts[i, j].visit_code == visit_code:
        # We've seen this location before, don't add it
        return location_index
    else:
        # Create a new location object
        l = &location_buffer[location_index]
        l.i = i
        l.j = j
        l.came_from = coming_from
        # Add the location to the queue
        if edge_is_set(branch_cuts, l, coming_from):
            # The edge between these locations is already set, so we schedule
            # the new location for immediate processing
            queue_push_head(queue, <QueueValue> l)
            print('\t\t\tAdding location to head of queue: [%d, %d]' % (i, j))
        else:
            # The between these locations was not set, so we schedule the
            # new location as in an ordinary BFS
            queue_push_tail(queue, <QueueValue> l)
            print('\t\t\tAdding location to tail of queue: [%d, %d]' % (i, j))
        # Mark this location as visited
        branch_cuts[i, j].visit_code = visit_code
        return location_index + 1


cdef inline long edge_index(long a, long b):
    if abs(a - b) == 1:
        return a if a < b else b
    else:
        # We are at the boundary of the image and wrapping around; return the
        # largest index
        return a if a > b else b


cdef inline cnp.uint8_t edge_is_set(branch_cut[:, ::1] branch_cuts,
                                  QueuedLocation *la, QueuedLocation *lb):
    # Is the edge set between to residues?
    if la.i != lb.i:
        # Edge along 0th dimension (vertical)
        return branch_cuts[edge_index(la.i, lb.i), la.j].vcut
    else:
        # Edge along 1st dimension (horizontal)
        return branch_cuts[la.i, edge_index(la.j, lb.j)].hcut


cdef void set_edges_to_root(QueuedLocation *location,
                            branch_cut[:, ::1] branch_cuts):
    print('\t\t\tSetting edges to root')
    cdef Py_ssize_t num_edges = 0
    cdef QueuedLocation *l = location
    while l.came_from != NULL:
        print('\t\t\t\t%x -> %x' % (<long> l, <long> l.came_from))
        print('\t\t\t\tSetting edge %d from [%d, %d] to [%d, %d]'
              % (num_edges, l.i, l.j, l.came_from.i, l.came_from.j))
        if l.i != l.came_from.i:
            # Edge along 0th dimension (vertical)
            branch_cuts[edge_index(l.i, l.came_from.i), l.j].vcut = 1
        else:
            # Edge along 1st dimension (horizontal)
            branch_cuts[l.i, edge_index(l.j, l.came_from.j)].hcut = 1
        num_edges += 1
        if num_edges > branch_cuts.shape[0] + branch_cuts.shape[1]:
            print('Programming error')
            break
        l = l.came_from


def find_branch_cuts_cy(branch_cut[:, ::1] branch_cuts,
                     cnp.int_t[::1] residue_storage):
    cdef:
        Py_ssize_t i, j, size, residue_no, vi, vj
        Queue *queue
        Py_ssize_t location_index
        QueuedLocation *location_buffer
        QueuedLocation *l
        QueueValue qv
        int residue, net_residue
        cnp.uint16_t visit_code
    size = branch_cuts.shape[0] * branch_cuts.shape[1]
    queue = NULL
    visit_code = 1
    location_buffer = <QueuedLocation *> malloc(size * sizeof(QueuedLocation))
    print('bytes for location_buffer: %d' % (size * sizeof(QueuedLocation)))
    print('location_buffer: %x' % (<long> location_buffer))

    # Scan for residues
    for i in range(branch_cuts.shape[0]):
        for j in range(branch_cuts.shape[1]):
            print('Scan reached [%d, %d]' %  (i, j))
            if (branch_cuts[i, j].residue_no != 0
                and residue_storage[branch_cuts[i, j].residue_no] != 0):
                print('\tFound unmatched residue')
                # Found a residue that has not yet been matched
                # Initialize a queue
                queue = queue_new()
                location_index = 0
                net_residue = 0

                # Push starting point into the queue
                l = &location_buffer[location_index]
                l.i = i
                l.j = j
                l.came_from = NULL
                queue_push_tail(queue, <QueueValue> l)
                branch_cuts[l.i, l.j].visit_code = visit_code
                location_index += 1

                # Breadth first search for residues
                while not queue_is_empty(queue):
                    # Process next location in the queue
                    l = <QueuedLocation *> queue_pop_head(queue)
                    residue_no = branch_cuts[l.i, l.j].residue_no
                    residue = residue_storage[residue_no]
                    if residue != 0:
                        print('\t\tFound residue at [%d, %d] with value %d'
                              % (l.i, l.j, residue))
                        set_edges_to_root(l, branch_cuts)
                        net_residue += residue
                        print('\t\tNet residue: %d' % (net_residue,))
                        residue_storage[residue_no] = 0
                        if net_residue == 0:
                            print('\t\tMatched the residue found; '
                                  'continuing scan for more residues')
                            break    # All residues matched, success!

                    # Add all neighbors that have not been seen
                    location_index = maybe_add_location(l.i - 1, l.j, l,
                                                        branch_cuts,
                                                        location_buffer,
                                                        location_index, queue,
                                                        visit_code)
                    print('\t\t\tlocation_index = %d' % location_index)
                    location_index = maybe_add_location(l.i + 1, l.j, l,
                                                        branch_cuts,
                                                        location_buffer,
                                                        location_index, queue,
                                                        visit_code)
                    print('\t\t\tlocation_index = %d' % location_index)
                    location_index = maybe_add_location(l.i, l.j - 1, l,
                                                        branch_cuts,
                                                        location_buffer,
                                                        location_index, queue,
                                                        visit_code)
                    print('\t\t\tlocation_index = %d' % location_index)
                    location_index = maybe_add_location(l.i, l.j + 1, l,
                                                        branch_cuts,
                                                        location_buffer,
                                                        location_index, queue,
                                                        visit_code)
                    print('\t\t\tlocation_index = %d' % location_index)

                if queue != NULL:
                    queue_free(queue)
                    queue = NULL
                if visit_code == UINT16_MAX:
                    branch_cuts[...].visit_code = 0
                    visit_code = 1
                else:
                    visit_code += 1

    print('freeing location_buffer')
    if location_buffer != NULL:
        free(<void *> location_buffer)
        location_buffer = NULL
    print('free\'d: location_buffer')
    return np.asarray(branch_cuts)


cdef inline cnp.uint8_t cut_between_pixels(cnp.uint8_t[:, ::1] vcut,
                                           cnp.uint8_t[:, ::1] hcut,
                                           Py_ssize_t ia, Py_ssize_t ja,
                                           Py_ssize_t ib, Py_ssize_t jb):
    # Is there a cut between two pixels?
    if ia != ib:
        # Cut normal to 0th dimension; horizontal cut
        return hcut[edge_index(ia, ib), ja]
    else:
        # Cut normal to 1st dimension: vertical cut
        return vcut[ia, edge_index(ib, jb)]


cdef inline Py_ssize_t maybe_add_pixel(cnp.float64_t[:, ::1] image,
                                       cnp.uint8_t[:, ::1] image_mask,
                                       cnp.int64_t[:, ::1] periods,
                                       cnp.uint8_t[:, ::1] vcut,
                                       cnp.uint8_t[:, ::1] hcut,
                                       Queue * queue,
                                       QueuedLocation * location_buffer,
                                       Py_ssize_t location_index,
                                       QueuedLocation * coming_from,
                                       long i, long j):
    cdef:
        QueuedLocation *l
    i = normalize_coordinate(i, image.shape[0])
    j = normalize_coordinate(j, image.shape[1])
    print('\tConsidering [%d, %d]' % (i, j))
    print('\t\tperiods = %d' % periods[i, j])
    if periods[i, j] != UNDEFINED:
        # Pixel has already been visited
        print('\t\tAlready processed')
        return location_index
    elif image_mask[i, j] == 1:
        # Masked pixel
        print('\t\tMasked')
        return location_index
    elif cut_between_pixels(vcut, hcut, i, j, coming_from.i, coming_from.j):
        # Cut between these pixels, unreachable
        print('\t\tCut between pixels')
        return location_index
    else:
        # Unwrap phase of the new location
        print('\t\tAdding at location_index = %d' % location_index)
        periods[i, j] = (periods[coming_from.i, coming_from.j]
                         + _phase_period_increment(image[coming_from.i,
                                                         coming_from.j],
                                                   image[i, j]))
        print('\t\tPhase periods at [%d, %d]: %d' % (i, j, periods[i, j]))
        # Add the new location to the queue
        if location_index >= (image.shape[0] * image.shape[1]):
            print('Illegal location_index: %d' % location_index)
            abort()
        l = &location_buffer[location_index]
        l.i = i
        l.j = j
        queue_push_tail(queue, <QueueValue> l)
        return location_index + 1


def integrate_phase(cnp.float64_t[:, ::1] image, cnp.uint8_t[:, ::1] image_mask,
                    cnp.int64_t[:, ::1] periods,
                    cnp.uint8_t[:, ::1] vcut, cnp.uint8_t[:, ::1] hcut,
                    Py_ssize_t initial_i, Py_ssize_t initial_j):
    cdef:
        Py_ssize_t i, j, location_index
        Queue * queue
        QueuedLocation *location_buffer
        QueuedLocation *l


    print('Code for UNDEFINED in periods: %d' % UNDEFINED)
    print('Code for UINT16_MAX in periods: %d' % UINT16_MAX)
    size = image.shape[0] * image.shape[1]
    queue = NULL
    queue = queue_new()
    print('queue: %x' % (<long> queue))
    location_buffer = <QueuedLocation *> malloc(size * sizeof(QueuedLocation))
    print('location_buffer: %x' % (<long> location_buffer))
    location_index = 0

    # Push start point into queue
    l = &location_buffer[location_index]
    l.i = initial_i
    l.j = initial_j
    location_index += 1
    periods[l.i, l.j] = 0
    queue_push_tail(queue, <QueueValue> l)
    # Unwrap all reachable pixels
    while not queue_is_empty(queue):
        l = <QueuedLocation *> queue_pop_head(queue)
        print('At pixel [%d, %d]' % (l.i, l.j))
        location_index = maybe_add_pixel(image, image_mask, periods, vcut, hcut,
                                         queue, location_buffer, location_index,
                                         l, l.i, l.j - 1)
        location_index = maybe_add_pixel(image, image_mask, periods, vcut, hcut,
                                         queue, location_buffer, location_index,
                                         l, l.i + 1, l.j)
        location_index = maybe_add_pixel(image, image_mask, periods, vcut, hcut,
                                         queue, location_buffer, location_index,
                                         l, l.i, l.j + 1)
        location_index = maybe_add_pixel(image, image_mask, periods, vcut, hcut,
                                         queue, location_buffer, location_index,
                                         l, l.i - 1, l.j)

    # Cleanup
    if location_buffer != NULL:
        free(location_buffer)
        location_buffer = NULL
    if queue != NULL:
        queue_free(queue)
        queue = NULL

    return np.asarray(periods)
