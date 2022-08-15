import numpy as np
import math
from functools import reduce


def _shift_indices(a1, a2, n_vertices, ex_start=True, ex_end=True):
    """Shift the position of the indices in a way that wraps them connecting
    the last index with the first one.

    Parameters
    ----------
    a1 : int
        The first index after wrapping and shifting the indices.
    a2 : int
        The first index after wrapping and shifting the indices.
    n_vertices : int
        The number of vertices in the polygon.
    ex_start : bool
        If the first index (a1) will be excluded from the
        selected sequence of indices.
    ex_end : bool
        If the last index (a2) will be excluded from the
        selected sequence of indices.

    Returns
    -------
    shifted_indices : numpy.ndarray
        The sequence of wrapped and shifted indices between a1 and a2.

    """
    shifted_indices = np.mod(a1 + np.arange(n_vertices), n_vertices)
    r = a2 - a1 + 1 + (0 if a2 > a1 else n_vertices)
    return shifted_indices[ex_start:r - ex_end]


def _get_cut_id(a1, a2, left2right=True):
    """ Generate an identifier key for the cut between a1 and a2.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    left2right : bool, optional
        The direction of the condition tested for the validity of this cut.
        The default direction is from left to right on the polygon vertices.

    Returns
    -------
    cut_id : tuple
        A tuple containing the cut indices a1 and a2,
        and the direction of the vertices of the polygon
        between indices a1 and a2.
    """
    cut_id = (a1, a2, 'L' if left2right else 'R')
    return cut_id


def _get_cut(a, vert_info, next_cut=True, same_ray=True, sign=0):
    """Look for the next/previous valid vertex that can be used as cut
    along with vertex `a`.

    Parameters
    ----------
    a : int
        An index of a vertex on the polygon used as reference to look for a cut
        that satisfies the given conditions.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    next_cut : bool, optional
        If the cut is looked for after index `a` from left to right on
        the polygon.
        Use False to look for the previous cut instead.
    same_ray : bool, optional
        If the next/previous cut index is being looked for on the same ray
        where `a` belongs.
    sign : int, optional
        The orientation (positive 1 /negative -1 / ambiguous 0) of the
        vertex index according to the ray that generated it.

    Returns
    -------
    cut_id : int
        The next/previous vertex index that satisfies the specifications given.
    """
    n_vertices = vert_info.shape[0]
    ray_a = int(vert_info[a, 2])

    if same_ray:
        criterion_0 = np.ones(n_vertices, dtype=bool)
        criterion_1 = vert_info[:, 2].astype(np.int32) == ray_a

        shifted_indices = np.argsort(vert_info[:, -1])
        pos_shift = np.where(vert_info[shifted_indices, -1].astype(np.int32)
                             == int(vert_info[a, -1]))[0][-1 * next_cut]

        shifted_indices = shifted_indices[
            np.mod(
                pos_shift + next_cut + np.arange(n_vertices),
                n_vertices)]
    else:
        shifted_indices = np.mod(
            a + (1 if next_cut else 0) + np.arange(n_vertices),
            n_vertices)
        criterion_0 = vert_info[:, 2].astype(np.int32) >= 0
        criterion_1 = vert_info[:, 2].astype(np.int32) != ray_a

    if sign:
        criterion_2 = vert_info[:, -2].astype(np.int32) == sign
    else:
        criterion_2 = np.ones(n_vertices, dtype=bool)

    criteria = criterion_0 * criterion_1 * criterion_2
    cut_id = np.where(criteria[shifted_indices])[0]

    if len(cut_id) == 0:
        return None

    cut_id = shifted_indices[cut_id[0 if next_cut else -1]]

    return cut_id


def _check_adjacency(a1, a2, vert_info, left2right=True):
    """Check whether cut point a2 is adjacent to cut point a1 to the left/right.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    left2right : bool, optional
        The direction of the condition tested for the validity of this cut.
        The default direction is from left to right on the polygon vertices

    Returns
    -------
    is_adjacent : bool
        Whether indices a1 and a2 are adjacent when looked from left to right,
        or right to left.

    """
    n_vertices = vert_info.shape[0]

    if left2right:
        shifted_indices = _shift_indices(a1, a2, n_vertices,
                                         ex_start=True,
                                         ex_end=True)
    else:
        shifted_indices = _shift_indices(a2, a1, n_vertices,
                                         ex_start=True,
                                         ex_end=True)

    # Both points are non-adjacent if there is at least one cut between them
    is_adjacent = not np.any(
        vert_info[shifted_indices, 2].astype(np.int32) >= 0)
    return is_adjacent


def _condition_1(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                 check_left=True):
    """ First condition that a cut has to satisfy in order to be
    left or right valid, according to `check_left`.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut
        validity has already been expanded.
    children_ids : list
        A list of tuples containing the indices of the Exp cuts that
        this cut depends on, and an identifier of this condition number.
    """
    children_ids = []
    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None, children_ids

    is_valid = _check_adjacency(a1, a2, vert_info, left2right=check_left)
    if is_valid:
        children_ids.append([(-1, -1, 'self',
                              '1%s' % ('L' if check_left else 'R'))])

    return is_valid, children_ids


def _condition_2(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                 check_left=True):
    """ Second condition that a cut has to satisfy in order to be
    left or right valid, according to `check_left`.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut
        validity has already been expanded.
    children_ids : list
        A list of tuples containing the indices of the Exp cuts that
        this cut depends on, and an identifier of this condition number.
    """
    children_ids = []
    is_valid = None

    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None, children_ids

    b1 = _get_cut(a1, vert_info,
                  next_cut=check_left,
                  same_ray=False,
                  sign=1)
    b2 = _get_cut(a2, vert_info,
                  next_cut=not check_left,
                  same_ray=False,
                  sign=-1)

    if b1 is None or b2 is None or \
       not (_check_adjacency(a1, b1, vert_info, left2right=check_left) and
            _check_adjacency(b2, a2, vert_info, left2right=check_left)):
        return None, children_ids

    ray_b1 = int(vert_info[b1, 2])
    ray_b2 = int(vert_info[b2, 2])

    if ray_b1 != ray_b2:
        return None, children_ids

    is_valid, child_id = _check_validity(b1, b2, vert_info,
                                         max_crest_cuts,
                                         min_crest_cuts,
                                         visited,
                                         check_left=check_left)

    child_id = (*child_id, '2%s' % ('L' if check_left else 'R'))

    children_ids.append([child_id])
    is_valid = None \
        if isinstance(is_valid, str) and is_valid == 'Exp'\
        else is_valid

    return is_valid, children_ids


def _condition_3(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                 check_left=True):
    """ Third condition that a cut has to satisfy in order to be
    left or right valid, according to `check_left`.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut
        validity has already been expanded.
    children_ids : list
        A list of tuples containing the indices of the Exp cuts that
        this cut depends on, and an identifier of this condition number.
    """
    children_ids = []

    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None, children_ids

    a1_pos = int(vert_info[a1, -1])
    a2_pos = int(vert_info[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition.
    if abs(a1_pos - a2_pos) - 1 < 2:
        return None, children_ids

    # This condition is applied only if there are four or more
    # cut points in the same ray.
    ver_in_ray_ids = list(np.where(vert_info[:, 2].astype(np.int32)
                                   == int(vert_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    ray_a = int(vert_info[a1, 2])
    is_valid = None

    for mc in filter(lambda mc: mc[1] == ray_a,
                     min_crest_cuts if check_left else max_crest_cuts):
        a3 = mc[2 if check_left else 3]
        a4 = mc[3 if check_left else 2]

        # Check that all intersection points are different
        if len(set({a1, a2, a3, a4})) < 4:
            continue

        # Check the order of the intersection vertices on the current ray
        if not (vert_info[a1, -1]
                < vert_info[a3, -1]
                < vert_info[a4, -1]
                < vert_info[a2, -1]):
            continue

        # Check if point a3' and a1 are left/right valid
        is_valid_1, child_id_1 = _check_validity(a1, a3, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=check_left)

        child_id_1 = (*child_id_1, '3%s' % ('L' if check_left else 'R'))

        # Check if point a2' and a4 are left/right valid
        is_valid_2, child_id_2 = _check_validity(a4, a2, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=check_left)

        child_id_2 = (*child_id_2, '3%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])
        is_valid_2 = None \
            if isinstance(is_valid_2, str) and is_valid_2 == 'Exp'\
            else is_valid_2
        is_valid_1 = None \
            if isinstance(is_valid_1, str) and is_valid_1 == 'Exp'\
            else is_valid_1

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid, children_ids


def _condition_4(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                 check_left=True):
    """ Fourth condition that a cut has to satisfy in order to be
    left or right valid, according to `check_left`.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this cut
        validity has already been expanded.
    children_ids : list
        A list of tuples containing the indices of the Exp cuts that
        this cut depends on, and an identifier of this condition number.
    """
    children_ids = []

    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more
    # cut points in the same ray.
    ver_in_ray_ids = list(np.where(
        vert_info[:, 2].astype(np.int32) == int(vert_info[a1, 2]))[0])

    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    a3 = _get_cut(a2, vert_info, next_cut=True, same_ray=True, sign=1)
    if a3 is None:
        return None, children_ids

    # Verify that a2' and a3 are minimal crest cuts
    if not any(filter(
        lambda mc:
            mc[3 if check_left else 2] == a2 and
            mc[2 if check_left else 3] == a3,
            max_crest_cuts if check_left else min_crest_cuts)):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    n_vertices_on_ray = np.max(
        vert_info[vert_info[:, 2].astype(np.int32)
                  == int(vert_info[a3, 2]), -1])
    set_id = 0

    while int(vert_info[a_p, -1]) < n_vertices_on_ray:
        set_id += 1

        a_p = _get_cut(a_p, vert_info, next_cut=True, same_ray=True, sign=-1)
        if a_p is None:
            break

        # Check if that point and a1 are left/right valid
        is_valid_1, child_id_1 = _check_validity(a1, a_p, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=check_left)

        child_id_1 = (*child_id_1, '4%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are right/left valid
        is_valid_2, child_id_2 = _check_validity(a3, a_p, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=not check_left)

        child_id_2 = (*child_id_2, '4%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])

        is_valid_1 = None \
            if isinstance(is_valid_1, str) and is_valid_1 == 'Exp'\
            else is_valid_1

        is_valid_2 = None \
            if isinstance(is_valid_2, str) and is_valid_2 == 'Exp'\
            else is_valid_2

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid, children_ids


def _condition_5(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                 check_left=True):
    """ Fifth condition that a cut has to satisfy in order to be
    left or right valid, according to `check_left`.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is left/right valid, or None if this
        cut validity has already been expanded.
    children_ids : list
        A list of tuples containing the indices of the Exp cuts that
        this cut depends on, and an identifier of this condition number .
    """
    children_ids = []

    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None, children_ids

    # This condition is applied only if there are four or more
    # cut points in the same ray.
    ver_in_ray_ids = list(
        np.where(vert_info[:, 2].astype(np.int32) == int(vert_info[a1, 2]))[0])

    if len(ver_in_ray_ids) < 4:
        return None, children_ids

    a3 = _get_cut(a1, vert_info, next_cut=False, same_ray=True, sign=-1)
    if a3 is None:
        return None, children_ids

    # Verify that a1 and a3' are minimal crest cuts
    if not any(filter(
        lambda mc:
        mc[3 if check_left else 2] == a3 and mc[2 if check_left else 3] == a1,
            max_crest_cuts if check_left else min_crest_cuts)):
        return None, children_ids

    # Get all points to the right of a3 in the same ray
    a_p = a3
    is_valid = None
    set_id = 0
    while int(vert_info[a_p, -1]) > 0:
        set_id += 1

        a_p = _get_cut(a_p, vert_info, next_cut=False, same_ray=True, sign=1)
        if a_p is None:
            break

        # Check if that point and a1 are right valid
        is_valid_1, child_id_1 = _check_validity(a_p, a2, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=check_left)

        child_id_1 = (*child_id_1, '5%s' % ('L' if check_left else 'R'))

        # Check if that point and a3 are left valid
        is_valid_2, child_id_2 = _check_validity(a_p, a3, vert_info,
                                                 max_crest_cuts,
                                                 min_crest_cuts,
                                                 visited,
                                                 check_left=not check_left)

        child_id_2 = (*child_id_2, '5%s' % ('L' if check_left else 'R'))

        children_ids.append([child_id_1, child_id_2])

        is_valid_1 = None \
            if isinstance(is_valid_1, str) and is_valid_1 == 'Exp' \
            else is_valid_1

        is_valid_2 = None \
            if isinstance(is_valid_2, str) and is_valid_2 == 'Exp' \
            else is_valid_2

        if is_valid_1 is None and is_valid_2 is None:
            continue
        elif is_valid_2 is None:
            pair_is_valid = is_valid_1
        elif is_valid_1 is None:
            pair_is_valid = is_valid_2
        else:
            pair_is_valid = is_valid_1 & is_valid_2

        if is_valid is None:
            is_valid = pair_is_valid
        else:
            is_valid &= pair_is_valid

    return is_valid, children_ids


def _invalidity_condition_1(a1, a2, vert_info, check_left=True):
    """ First condition that can turn a cut to be left/right invalid.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with
        this condition.
        This returns None when this condition can not be tested.
    """
    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None

    a1_pos = int(vert_info[a1, -1])
    a2_pos = int(vert_info[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(
        np.where(vert_info[:, 2].astype(np.int32) == int(vert_info[a1, 2]))[0])

    if len(ver_in_ray_ids) < 3:
        return None

    a3 = _get_cut(a1, vert_info, next_cut=check_left, same_ray=True, sign=-1)

    if a3 is None:
        return None

    if not _check_adjacency(a3, a1, vert_info, left2right=check_left):
        return None

    is_valid = not (vert_info[a1, -1] < vert_info[a3, -1] < vert_info[a2, -1])

    return is_valid


def _invalidity_condition_2(a1, a2, vert_info, check_left=True):
    """ Second condition that can turn a cut to be left/right invalid.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with
        this condition.
        This returns None when this condition can not be tested.
    """
    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None

    a1_pos = int(vert_info[a1, -1])
    a2_pos = int(vert_info[a2, -1])

    # If there are no vertices between a1 and 2, continue with other condition
    if a1_pos - a2_pos != 2:
        return None

    ver_in_ray_ids = list(
        np.where(vert_info[:, 2].astype(np.int32) == int(vert_info[a1, 2]))[0])
    if len(ver_in_ray_ids) < 3:
        return None

    a3 = _get_cut(a2, vert_info, next_cut=not check_left, same_ray=True,
                  sign=1)

    if a3 is None:
        return None

    if not _check_adjacency(a2, a3, vert_info, left2right=check_left):
        return None

    is_valid = not (vert_info[a1, -1] < vert_info[a3, -1] < vert_info[a2, -1])
    return is_valid


def _invalidity_condition_3(a1, a2, vert_info, check_left=True,
                            tolerance=1e-4):
    """ Third condition that can turn a cut to be left invalid.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.
    tolerance : float, optional
        A tolerance used to fetermine if two intersection points are the same.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 has no invalidity according with
        this condition.
        This returns None when this condition can not be tested.
    """
    if vert_info[a1, 3] < 0.5 or vert_info[a2, 3] > 0.5:
        return None

    b2 = _get_cut(a1, vert_info, next_cut=check_left, same_ray=False, sign=0)
    b1 = _get_cut(a2, vert_info, next_cut=not check_left, same_ray=False,
                  sign=0)

    if b1 is None or b2 is None:
        return None

    if not (_check_adjacency(b1, a2, vert_info, left2right=check_left) and
            _check_adjacency(a1, b2, vert_info, left2right=check_left)):
        return None

    n_vertices = vert_info.shape[0]
    # Get all intersections between b2 and a1, and between a2 and b1.
    if check_left:
        shifted_indices_1 = _shift_indices(a1, b2, n_vertices,
                                           ex_start=True,
                                           ex_end=True)

        shifted_indices_2 = _shift_indices(b1, a2, n_vertices,
                                           ex_start=True,
                                           ex_end=True)
    else:
        shifted_indices_1 = _shift_indices(b2, a1, n_vertices,
                                           ex_start=True,
                                           ex_end=True)

        shifted_indices_2 = _shift_indices(a2, b1, n_vertices,
                                           ex_start=True,
                                           ex_end=True)

    b2_a1_int = list(np.where(
        vert_info[shifted_indices_1, 2].astype(np.int32) < -1)[0])
    a2_b1_int = list(np.where(
        vert_info[shifted_indices_2, 2].astype(np.int32) < -1)[0])

    # This condition does not apply if only one of the two segments
    # contain an intersection.
    if len(b2_a1_int) == 0 or len(a2_b1_int) == 0:
        return None

    int_1 = vert_info[shifted_indices_1[b2_a1_int], :2]
    int_1 = int_1 / np.linalg.norm(int_1, axis=1)[..., np.newaxis]
    int_2 = vert_info[shifted_indices_2[a2_b1_int], :2]
    int_2 = int_2 / np.linalg.norm(int_2, axis=1)[..., np.newaxis]

    # If there is at least one intersection point that is the same for
    # both segments, this cut is left invalid.
    is_valid = not (np.matmul(int_1, int_2.transpose()) >= 1 - tolerance).any()
    return is_valid


all_valid_conditions = [_condition_5,
                        _condition_4,
                        _condition_3,
                        _condition_2,
                        _condition_1]

all_invalid_conditions = [_invalidity_condition_1,
                          _invalidity_condition_2,
                          _invalidity_condition_3]


def _check_validity(a1, a2, vert_info, max_crest_cuts, min_crest_cuts, visited,
                    check_left=True):
    """ Checks recursively the validity of a cut between indices `a1` and `a2`
    of the polygon.

    Parameters
    ----------
    a1 : int
        The first index of the polygon for the cut between a1 and a2.
    a2 : int
        The last index of the polygon for the cut between a1 and a2.
    vert_info : numpy.ndarray
        The array containing the information about the polygon and the
        characteristics of each vertex.
    max_crest_cuts : list
        List of tuples containing the indices that correspond to maximum crests
        on the polygon when walked from left to right.
    min_crest_cuts : list
        List of tuples containing the indices that correspond to minimum crests
        on the polygon when walked from left to right.
    visited : dict
        A dictionary containing the dependencies and validities of the
        already visited cuts.
    check_left : bool, optional
        If the validity of the cut is being tested from left to right,
        or right to left.

    Returns
    -------
    is_valid : bool or None
        If the cut between a1 and a2 is valid or is being Exp
        This returns None when its cuts dependencies have been Exp before.
    cut_id : tuple
        A tuple containing the indices a1 and a2,
        and the direction of the vertices of the polygon between indices a1 and
         a2 of the current cut.

    """
    cut_id = _get_cut_id(a1, a2, left2right=check_left)

    # Check if this node has been visited before
    is_valid, children_ids = visited.get(cut_id, [None, []])

    if is_valid is None:
        visited[cut_id] = ['Exp', children_ids]
        is_valid = True

        for condition in all_invalid_conditions:
            resp = condition(a1, a2, vert_info, check_left=check_left)
            if resp is not None:
                is_valid &= resp
            if not is_valid:
                break

        if is_valid:
            # This cut is not left valid until the contrary is proven
            is_valid = None
            for condition in all_valid_conditions:
                resp, cond_children_ids = \
                    condition(a1, a2, vert_info,
                              max_crest_cuts=max_crest_cuts,
                              min_crest_cuts=min_crest_cuts,
                              visited=visited,
                              check_left=check_left)

                if resp is not None:
                    children_ids += cond_children_ids
                    if is_valid is None:
                        is_valid = resp
                    else:
                        is_valid |= resp

        visited[cut_id] = [is_valid, children_ids]

    return is_valid, cut_id


def _traverse_tree(root, visited, path=None):
    """ Traverse the tree of validity of the cuts that were tested previously
    with `_check_validity`.

    Parameters
    ----------
    root : tuple
        A tuple containing the indices a1 and a2,
        and the direction of the vertices of the polygon between indices a1 and
        a2 of the current cut used as root.
    visited : dict
        A dictionary containing the dependencies and validities of the
        visited cuts.
    path : list or None
        A list with the cut identifiers of the already visited cuts.
        This prevents infinite recursion on cyclic graphs.

    Returns
    -------
    validity : bool
        The validity of the current branch being traversed.

    """
    root = root if len(root) == 3 else root[:-1]

    if path is None:
        path = []
    elif root in path:
        return None

    validity, cond_dep = visited.get(root, (True, []))
    if len(cond_dep) == 0:
        return validity

    validity = False
    # Make a copy from the conditions dependency of this node.
    # This way the original list can be shrunked if needed.
    for sib_cond in list(cond_dep):
        # This path is valid only if all sibling conditions are valid
        sibling_validity = all(map(lambda sib_path:
                                   _traverse_tree(sib_path, visited,
                                                  list(path) + [root]),
                                   sib_cond))

        if not sibling_validity:
            visited[root][1].remove(sib_cond)

        validity |= sibling_validity

    visited[root][0] = validity
    return validity


def _immerse_tree(root, visited, n_vertices, polys_idx):
    """ Traverses the valid paths according to the visited dictionary to get
    the set of non-overlapping sub polygons.
    This traverses only one of the possible valid immersions for simplicity.

    Parameters
    ----------
    root : tuple
        A tuple containing the indices a1 and a2,
        and the direction of the vertices of the polygon between indices a1 and
        a2 of the current cut used as root.
    visited : dict
        A dictionary containing the dependencies and validities of the
        visited cuts.
    n_vertices : int
        The number of vertices in the polygon.
    polys_idx : list
        A list of the indices of the polygons that are being discovered.

    Returns
    -------
    immersion : dict
        The children sub tree of valid conditions.
    sub_poly : list
        A list of indices of all sub polygons generated from
        children conditions.

    """
    root = root if len(root) == 3 else root[:-1]
    immersion = {root: {}}
    _, conditions = visited.get(root, None)

    a1, a2 = root[:2]
    left2right = root[2][-1] == 'L'
    sub_poly = []

    if conditions[0][0][0] < 0:
        # The base case is reached is when a cut that is left/right
        # valid by condition 1
        if left2right:
            sub_poly.append(_shift_indices(a1, a2, n_vertices,
                                           ex_start=False,
                                           ex_end=False))
        else:
            sub_poly.append(_shift_indices(a2, a1, n_vertices,
                                           ex_start=False,
                                           ex_end=False))

        return immersion, sub_poly

    for sib_cond in conditions:
        for child_1 in sib_cond:
            child_owner, child_conditions = visited.get(child_1[:-1],
                                                        (None, None))
            # Set the owner of this branch to the current visited node.
            if child_owner is None and child_conditions is not None:
                visited[child_1[:-1]][0] = root

        child_1 = sib_cond[0]
        child_cond_id = child_1[-1][0]

        # If this branch has been already added to the poygon list by another
        # node, continue with the next children condition.
        cut_owner, _ = visited.get(child_1[:-1], (None, None))
        if cut_owner is not None and cut_owner != root:
            continue

        if len(sib_cond) == 1:
            child_left2right = child_1[-1][1] == 'L'

            sub_immersion, child_poly = _immerse_tree(child_1, visited,
                                                      n_vertices,
                                                      polys_idx)
            immersion[root].update(sub_immersion)

            # If this cut was selected by satisfying condition 2,
            # append the child path to the parent path.
            b1, b2 = child_1[:2]
            if child_cond_id == '2':
                if child_left2right:
                    ex_child_end = False \
                        if len(child_poly) == 0 \
                        else child_poly[0][0] in [a1, b1]

                    ex_child_start = False \
                        if len(child_poly) == 0 \
                        else child_poly[-1][-1] in [a2, b2]

                    sub_poly.append(_shift_indices(a1, b1, n_vertices,
                                                   ex_start=False,
                                                   ex_end=ex_child_end))
                    sub_poly += child_poly
                    sub_poly.append(_shift_indices(b2, a2, n_vertices,
                                                   ex_start=ex_child_start,
                                                   ex_end=False))
                else:
                    ex_child_end = False \
                        if len(child_poly) == 0\
                        else child_poly[0][0] in [a2, b2]

                    ex_child_start = False \
                        if len(child_poly) == 0\
                        else child_poly[-1][-1] in [a1, b1]

                    sub_poly.append(_shift_indices(a2, b2, n_vertices,
                                                   ex_start=False,
                                                   ex_end=ex_child_end))
                    sub_poly += child_poly
                    sub_poly.append(_shift_indices(b1, a1, n_vertices,
                                                   ex_start=ex_child_start,
                                                   ex_end=False))
            else:
                sub_poly = child_poly

        else:
            # Children cuts that where generated using rules 3, 4, and 5
            # have a special way to be merged with their respective parents.
            child_2 = sib_cond[1]
            child_cond_id = child_1[-1][0]
            child_left2right = child_1[-1][1] == 'L'

            if child_cond_id == '3':
                sub_immersion, child_poly = _immerse_tree(child_1, visited,
                                                          n_vertices,
                                                          polys_idx)
                if len(child_poly):
                    # If the children branch has not been added to the
                    # sub polygons list, append it.
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_immersion, child_poly = _immerse_tree(child_2, visited,
                                                          n_vertices,
                                                          polys_idx)

                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                if child_left2right:
                    sub_poly.append(_shift_indices(child_1[1], child_2[0],
                                                   n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))
                else:
                    sub_poly.append(_shift_indices(child_2[0], child_1[1],
                                                   n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))

            elif child_cond_id == '4':
                sub_immersion, child_poly = _immerse_tree(child_1, visited,
                                                          n_vertices,
                                                          polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly

                if child_left2right:
                    sub_poly.append(_shift_indices(child_2[0], a2, n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))
                else:
                    sub_poly.append(_shift_indices(a2, child_2[0], n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))

                if len(child_poly):
                    polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = _immerse_tree(child_2, visited,
                                                          n_vertices,
                                                          polys_idx)

                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_poly = []

            elif child_cond_id == '5':
                sub_immersion, child_poly = _immerse_tree(child_1, visited,
                                                          n_vertices,
                                                          polys_idx)
                immersion[root].update(sub_immersion)
                sub_poly += child_poly

                if child_left2right:
                    sub_poly.append(_shift_indices(a1, child_2[1], n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))
                else:
                    sub_poly.append(_shift_indices(child_2[1], a1, n_vertices,
                                                   ex_start=False,
                                                   ex_end=False))

                if len(child_poly):
                    polys_idx.append(np.concatenate(sub_poly, axis=0))

                sub_immersion, child_poly = _immerse_tree(child_2, visited,
                                                          n_vertices,
                                                          polys_idx)

                if len(child_poly):
                    immersion[root].update(sub_immersion)
                    polys_idx.append(np.concatenate(child_poly, axis=0))

                sub_poly = []
        break

    return immersion, sub_poly


def _get_root_indices(vert_info):
    """ Finds the indices of the vertices that define the root cut used to
    recurse the polygon subdivision algorithm.

    Parameters
    ----------
    vert_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices and
        additional information about each vertex.

    Returns
    -------
    left_idx : int
        The positional index of the root left vertex
    right_idx : int
        The positional index of the root right vertex

    """
    n_vertices = vert_info.shape[0]

    # Non-intersection points:
    org_ids = np.where(vert_info[:, 2].astype(np.int32) == -1)[0]

    root_vertex = org_ids[np.argmax(vert_info[org_ids, 1])]

    shifted_indices = np.mod(root_vertex + 1 + np.arange(n_vertices),
                             n_vertices)

    right_idx = np.where(
        vert_info[shifted_indices, 2].astype(np.int32) == 0)[0][0]
    right_idx = shifted_indices[right_idx]

    shifted_indices = np.mod(root_vertex + np.arange(n_vertices),
                             n_vertices)

    left_idx = np.where(
        vert_info[shifted_indices, 2].astype(np.int32) == 0)[0][-1]
    left_idx = shifted_indices[left_idx]

    return left_idx, right_idx


def _get_crest_ids(vertices, tolerance=1e-3):
    """ Finds the positional indices of the crest points.
    Only crests where there is a left turn on the polygon are considered.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    tolerance : float, optional
        A tolerance to determine if two vertices are at the same height.

    Returns
    -------
    max_crest_ids : list
        A list of indices of the maximum crest vertices.
    min_crest_ids : list
        A list of indices of the minimum crest vertices.
    max_crest : int
        The positional index of the maximum crest vertex.
    min_crest : int
        The positional index of the minimum crest vertex.

    """
    # Get the direction of the polygon when traversing its perimeter.
    edges = vertices[1:, :2] - vertices[:-1, :2]
    edges = np.vstack((vertices[0, :2] - vertices[-1, :2], edges))
    edges = edges / np.linalg.norm(edges, axis=1).reshape(-1, 1)

    norm_left = np.hstack((-edges[:, np.newaxis, 1], edges[:, np.newaxis, 0]))

    cos_angle = np.sum(norm_left[:-1] * edges[1:], axis=1)
    cos_angle = np.append(cos_angle, np.sum(norm_left[-1] * edges[0]))

    dir_left = cos_angle > 0.0

    # Only crest points of left turns are considered
    diff_y_prev = vertices[1:, 1] - vertices[:-1, 1]
    diff_y_next = -diff_y_prev
    diff_y_prev = np.insert(diff_y_prev, 0, vertices[0, 1] - vertices[-1, 1])
    diff_y_next = np.append(diff_y_next, vertices[-1, 1] - vertices[0, 1])

    u_prev = diff_y_prev > tolerance
    u_next = diff_y_next > tolerance
    d_prev = diff_y_prev < -tolerance
    d_next = diff_y_next < -tolerance

    ey_next = np.fabs(diff_y_next) <= tolerance

    # Determines if the path is climbing up.
    clmb_up = reduce(lambda l1, l2:
                     l1 + [
                        (l2[0] and l2[2])
                        or
                        (l1[-1] and not l2[0] and not l2[1])],
                     zip(u_prev, d_prev, ey_next), [False])[1:]

    # Determines if the path is climbing down.
    clmb_dwn = reduce(lambda l1, l2:
                      l1 + [
                        (l2[0] and l2[2])
                        or
                        (l1[-1] and not any(l2[:2]))],
                      zip(d_prev, u_prev, ey_next), [False])[1:]

    # Find maximum crests on left turns only.
    max_crest_ids = np.nonzero(dir_left *
                               (u_prev * u_next + u_prev * clmb_up))[0]

    # Find minimum crests on left turns only.
    min_crest_ids = np.nonzero(dir_left *
                               (d_prev * d_next + d_prev * clmb_dwn))[0]

    if len(max_crest_ids) > 0:
        max_crest = max_crest_ids[np.argmax(vertices[max_crest_ids, 1])]
    else:
        max_crest = None

    if len(min_crest_ids) > 0:
        min_crest = min_crest_ids[np.argmin(vertices[min_crest_ids, 1])]
    else:
        min_crest = None

    return max_crest_ids, min_crest_ids, max_crest, min_crest


def _check_clockwise(vertices):
    """ Check if the polygon vertices are in a clockwise ordering.
    The polygon sub division algorithm works for clockwise polygns.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.

    Returns
    -------
    is_clockwise : bool or None
        Whether the polygon is in clockwise direction or not.
        This is None when no crest points were detected.

    """
    signed_area = np.sum(vertices[:-1, 0] * vertices[1:, 1] -
                         vertices[:-1, 1] * vertices[1:, 0]) + \
        vertices[-1, 0] * vertices[0, 1] - vertices[-1, 1] * vertices[0, 0]
    is_clockwise = signed_area < 0
    return is_clockwise


def _get_crest_cuts(vert_info, crest_ids):
    """ Finds the positional indices of the intersection vertices
    that are closest to each crest point.

    Parameters
    ----------
    vert_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices with
        additional information of each vertex.

    Returns
    -------
    closest_cuts : list
        A list of tuples with the index of the crest point, the positional
        index of the two intersection vertices that are closest to it,
        and their corresponding ray index.

    """
    n_vertices = vert_info.shape[0]
    closest_cuts = []

    for i in crest_ids:
        shifted_indices = np.mod(i + np.arange(n_vertices), n_vertices)
        prev_id = np.where(
            vert_info[shifted_indices, 2].astype(np.int32) >= 0)[0][-1]
        prev_id = shifted_indices[prev_id]

        shifted_indices = np.mod(i + 1 + np.arange(n_vertices), n_vertices)
        next_id = np.where(
            vert_info[shifted_indices, 2].astype(np.int32) >= 0)[0][0]
        next_id = shifted_indices[next_id]

        r = int(vert_info[prev_id, 2])

        closest_cuts.append((i, r, prev_id, next_id))

    return closest_cuts


def _get_self_intersections(vertices):
    """ Computes the rays formulae of all edges to identify
    self-intersections later.
    This will iterate over all edges to generate the ray formulae.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.

    Returns
    -------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each
        edge on the polygon.
        The tuple contains the source coordinate and the vectorial
        direction of the ray.
        It also states that the rays were not computed from a crest point
        (last element in the tuple = False).

    """
    n_vertices = vertices.shape[0]
    rays_formulae = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices
        px, py = vertices[i, :]
        qx, qy = vertices[j, :]
        vx, vy = qx - px, qy - py

        rays_formulae.append(((px, py, 1), (vx, vy, 0), False))

    return rays_formulae


def _compute_rays(vertices, crest_ids, epsilon=1e-1):
    """ Computes the rays that cross each crest point, offsetted by epsilon.
    The rays are returned in general line form.
    For maximum crests, give a negative epsion instead.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    crest_ids : list
        A list of positional indices that correspond to maximum/minimum
        crest vertices.
    epsilon : float, optional
        An offset added to the crest point in the y-axis.

    Returns
    -------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each
        crest point.
        The tuple contains the source coordinate and the vectorial direction
        of the ray.
        It also states that the rays were computed from a crest point
        (last element in the tuple = True).

    """
    rays_formulae = []
    existing_heights = []
    for ids in crest_ids:
        ray_src = np.array(
            (vertices[ids, 0] - 1.0, vertices[ids, 1] + epsilon, 1))
        ray_dir = np.array((1.0, 0.0, 0.0))
        if len(existing_heights) == 0:
            existing_heights.append(ray_src[1])
            rays_formulae.append((ray_src, ray_dir, True))

        elif ray_src[1] not in existing_heights:
            rays_formulae.append((ray_src, ray_dir, True))

    return rays_formulae


def _sort_rays(rays_formulae, max_crest_y, tolerance=1e-3):
    """ Sorts the rays according to its relative position to the crest point.
    It also filters any repeated rays.

    Parameters
    ----------
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray for each
        crest point.
    max_crest_y : float
        The coordinate in the y-axis of the topmost crest vertex.
    tolerance : float, optional
        The tolerance used to remove any ray that is at less distance than
        `toelrance` from any other existing ray.

    Returns
    -------
    sorted_rays_formulae : list
        The list of unique rays formulae sorted according to their position in
        the y-axis.

    """
    if len(rays_formulae) == 1:
        return rays_formulae

    sorted_rays_formulae = [rays_formulae[i]
                            for i in np.array(
                                list(map(lambda ray:
                                         math.fabs(ray[0][1] - max_crest_y),
                                         rays_formulae))).argsort()]

    curr_id = len(sorted_rays_formulae) - 1
    while curr_id > 0:
        curr_y = sorted_rays_formulae[curr_id][0][1]
        same_ray = any(filter(lambda r:
                              math.fabs(r[0][1] - curr_y) < tolerance,
                              sorted_rays_formulae[:curr_id]))
        # If there is at least one ray close (< tolerance) to this,
        # remove the current ray.
        if same_ray:
            sorted_rays_formulae.pop(curr_id)
        curr_id -= 1

    return sorted_rays_formulae


def _find_intersections(vertices, rays_formulae, tolerance=1e-3):
    """ Walks the polygon to find self-intersections and
    intersections a set of rays.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray.
    tolerance : float, optional
        The tolerance used to determine if a ray intersects an edge of
        the polygon.

    Returns
    -------
    cut_coords : numpy.ndarray
        A two-dimensional array with coordinates of the intersection vertices.
    valid_edges : numpy.ndarray
        A two-dimensional array with the information of the edges that were
        cut by a ray.
    t_coefs : numpy.ndarray
        The coefficients of the parametric rays that define the position of
        the intersection vertex on its respective edge.

    """
    n_vertices = vertices.shape[0]

    # Find the cuts made by each ray to each line defined between two points in
    # the polygon.
    valid_edges = []
    valid_cuts = []
    valid_t_coefs = []
    for i in range(n_vertices):
        j = (i + 1) % n_vertices

        vx = vertices[j, 0] - vertices[i, 0]
        vy = vertices[j, 1] - vertices[i, 1]

        px = vertices[i, 0]
        py = vertices[i, 1]

        # Only add non-singular equation systems i.e. edges that are actually
        # crossed by each ray.
        for k, \
            ((rs_x, rs_y, _),
             (rd_x, rd_y, _),
             is_ray) in enumerate(rays_formulae):

            if not is_ray and (i == k or j == k):
                continue

            mat = np.array([[vx, -rd_x], [vy, -rd_y]])
            vec = np.array([rs_x - px, rs_y - py])

            if math.fabs(mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]) < 1e-3:
                continue

            # Find the parameter `t` that defines the intersection between the
            # urrent polygon edge and the testing ray.
            t_coefs = np.linalg.solve(mat, vec)

            # Determine if the intersection is inside the current edge or not.
            if is_ray:
                inter_in_edge = tolerance <= t_coefs[0] <= 1.0-tolerance
            else:
                inter_in_edge = tolerance <= t_coefs[0] <= 1.0-tolerance \
                                and \
                                tolerance <= t_coefs[1] <= 1.0-tolerance

            # Add this cut if it is on an edge of the polygon
            if inter_in_edge:
                valid_edges.append((i, j, k if is_ray else -2))
                valid_cuts.append((px + vx * t_coefs[0], py + vy * t_coefs[0]))
                valid_t_coefs.append(t_coefs[0])

    valid_edges = np.array(valid_edges, dtype=np.int32)
    cut_coords = np.array(valid_cuts)
    t_coefs = np.array(valid_t_coefs)

    return cut_coords, valid_edges, t_coefs


def _sort_ray_cuts(vert_info, rays_formulae):
    """ Adds the positional ordering of the intersection vertices that are
    on rays. It also assigns their corresponding sign according to the
    direction of the polygon when it is walked from left to right.

    Parameters
    ----------
    vert_info : numpy.ndarray
        An array containing the coordinates of the polygon vertices with
        additional information about each vertex.
    rays_formulae : list
        A list of tuples containing the parametric formula of a ray.

    Returns
    -------
    vert_info : numpy.ndarray
        The set of vetices coordinates with the updated information about the
        position of intersection vertices.

    """
    all_idx_per_ray = []
    all_ord_per_ray = []
    all_sym_per_ray = []
    for k, ((_, rs_y, _), _, _) in enumerate(rays_formulae):
        sel_k = np.nonzero(vert_info[:, -1].astype(np.int32) == k)[0]
        if len(sel_k) == 0:
            continue

        # Determine the intersection's symbol using the y coordinate of the
        # previous point of each intersection (sel_k - 1).
        inter_symbols = (vert_info[sel_k - 1, 1] < rs_y) * 2 - 1

        rank_ord = np.empty(len(sel_k))
        rank_ord[np.argsort(vert_info[sel_k, 0])] = list(range(len(sel_k)))

        all_idx_per_ray += list(rank_ord)
        all_ord_per_ray += list(sel_k)
        all_sym_per_ray += list(inter_symbols)

    vert_info = np.hstack((vert_info, np.zeros((vert_info.shape[0], 2))))

    # Fourth column contains the symbol of the cut, and the sixth column the
    # index of that cut on the corresponding ray.
    vert_info[all_ord_per_ray, -2] = all_sym_per_ray
    vert_info[all_ord_per_ray, -1] = all_idx_per_ray

    return vert_info


def _merge_new_vertices(vertices, cut_coords, valid_edges, t_coefs):
    """ Merges the new vertices computed from self-intersections and
    intersections of the polygon with any ray.
    The newly inserted vertices are sorted according to the coefficient used
    to compute them.

    Parameters
    ----------
    vertices : numpy.ndarray
        An array containing the coordinates of the polygon vertices.
    cut_coords : numpy.ndarray
        A two-dimensional array with coordinates of the intersection vertices.
    valid_edges : numpy.ndarray
        A two-dimensional array with the information of the edges that were
        cut by a ray.
    t_coefs : numpy.ndarray
        The coefficients of the parametric rays that define the position of the
        intersection vertex on its respective edge.

    Returns
    -------
    vert_info : numpy.ndarray
        The set of vetices coordinates with the updated information about the
        position of intersection vertices.

    """
    vert_info = []
    last_j = 0

    for i in np.unique(valid_edges[:, 0]):
        vert_info.append(
            np.hstack((vertices[last_j:i+1, :], -np.ones((i-last_j+1, 1)))))
        sel_i = np.nonzero(valid_edges[:, 0] == i)[0]
        sel_r = valid_edges[sel_i, 2]

        ord_idx = np.argsort(t_coefs[sel_i])

        sel_i = sel_i[ord_idx]
        sel_r = sel_r[ord_idx]

        vert_info.append(np.hstack((cut_coords[sel_i], sel_r.reshape(-1, 1))))
        last_j = i + 1

    n_vertices = vertices.shape[0]
    vert_info.append(
        np.hstack((vertices[last_j:, :], -np.ones((n_vertices-last_j, 1)))))
    vert_info = np.vstack(vert_info)

    return vert_info


def divide_selfoverlapping(coords):
    """ Divide a self-overlapping polygon into non self-overlapping polygons.
    This implements the algorithm proposed in [1].

    Parameters
    ----------
    coords : (N, 2) array
        Coordinates of the polygon vertices.

    Returns
    -------
    sub_polys : list
        A list of ndarrays with the coordinates of the non
        self-overlapping polygons obtained from dividing the original polygon.

    References
    ----------
    .. [1] Uddipan Mukherjee. (2014). Self-overlapping curves:
           Analysis and applications. Computer-Aided Design, 46, 227-232.
           :DOI: https://doi.org/10.1016/j.cad.2013.08.037
    """
    # Change the order of the axis to x, y from rr, cc.
    vertices = coords[:, [1, 0]]
    max_crest_ids, min_crest_ids, max_crest, min_crest = \
        _get_crest_ids(vertices, tolerance=2*np.finfo(np.float32).eps)

    # Check if the polygon vertices are given in counter-clockwise direction
    is_clockwise = _check_clockwise(vertices)

    if is_clockwise is None:
        # If the polygon does not have any crest point, it is because
        # it is not self-overlapping.
        return [vertices[:, [1, 0]]]

    if not is_clockwise:
        vertices = vertices[::-1, :]
        max_crest_ids, min_crest_ids, max_crest, min_crest = \
            _get_crest_ids(vertices, tolerance=2*np.finfo(np.float32).eps)

    if max_crest is None and min_crest is None:
        # If the polygon does not have any crest point, it is because
        # it is not self-overlapping.
        return [vertices[::(-1)**(not is_clockwise), [1, 0]]]

    rays_max = _compute_rays(vertices, max_crest_ids, epsilon=-1e-4)
    rays_min = _compute_rays(vertices, min_crest_ids, epsilon=1e-4)

    rays_formulae = _sort_rays(rays_max + rays_min, vertices[:, 1].max(),
                               tolerance=2e-4)
    self_inter_formulae = _get_self_intersections(vertices)

    cut_coords, valid_edges, t_coefs = \
        _find_intersections(vertices, rays_formulae + self_inter_formulae,
                            tolerance=2*np.finfo(np.float32).eps)
    vert_info = _merge_new_vertices(vertices, cut_coords, valid_edges, t_coefs)
    vert_info = _sort_ray_cuts(vert_info, rays_formulae)

    # Get the first point at the left of the crest point
    new_max_crest_ids, new_min_crest_ids, _, _ = \
        _get_crest_ids(vert_info, tolerance=2*np.finfo(np.float32).eps)
    new_max_crest_cuts = _get_crest_cuts(vert_info, new_max_crest_ids)
    new_min_crest_cuts = _get_crest_cuts(vert_info, new_min_crest_ids)

    left_idx, right_idx = _get_root_indices(vert_info)

    # The root is left valid by construction.
    # Therefore, the right validity of the root cut is checked and then all the
    # possible valid cuts are computed.
    visited = {}
    _, root_id = _check_validity(left_idx, right_idx, vert_info,
                                 new_max_crest_cuts,
                                 new_min_crest_cuts,
                                 visited,
                                 check_left=False)

    # Update the visited dictionary to leave only valid paths
    polygon_is_valid = _traverse_tree(root_id, visited)

    if not polygon_is_valid:
        # If the polygon cannot be sub divided into polygons that are not
        # self-overlapping, return the original polygon.
        return [vertices[::(-1)**(not is_clockwise), [1, 0]]]

    # Remove invalid cuts from the `visited` dictionary. This dictionary
    # is reused to identify branches that have been already added to
    # the sub polygons list.
    for k in list(visited.keys()):
        if visited[k][0]:
            visited[k][0] = None
        else:
            del visited[k]

    # Perform a single immersion on the validity tree to get the first valid
    # path that cuts the polygon into non self-overlapping sub polygons.
    sub_polys_ids = []
    _, sub_poly = _immerse_tree(root_id, visited, vert_info.shape[0],
                                sub_polys_ids)

    # Add the root cut of the immersion tree
    n_vertices = vert_info.shape[0]
    shifted_indices = np.mod(left_idx + np.arange(n_vertices), n_vertices)
    r = right_idx - left_idx + 1 + (0 if right_idx > left_idx else n_vertices)
    sub_poly = [shifted_indices[:r]] + sub_poly
    sub_polys_ids.insert(0, np.concatenate(sub_poly, axis=0))

    polys = []
    for poly_ids in sub_polys_ids:
        # Revert the ordering of the columns to be rr, cc instead of x, y
        new_sub_poly = vert_info[poly_ids, :]
        new_sub_poly = new_sub_poly[::(-1)**(not is_clockwise), [1, 0]]

        polys.append(new_sub_poly)

    return polys
