import numpy as np
from .._shared.utils import deprecated
from ._remap import _map_array


def join_segmentations(s1, s2):
    """Return the join of the two input segmentations.

    The join J of S1 and S2 is defined as the segmentation in which two
    voxels are in the same segment if and only if they are in the same
    segment in *both* S1 and S2.

    Parameters
    ----------
    s1, s2 : numpy arrays
        s1 and s2 are label fields of the same shape.

    Returns
    -------
    j : numpy array
        The join segmentation of s1 and s2.

    Examples
    --------
    >>> from skimage.segmentation import join_segmentations
    >>> s1 = np.array([[0, 0, 1, 1],
    ...                [0, 2, 1, 1],
    ...                [2, 2, 2, 1]])
    >>> s2 = np.array([[0, 1, 1, 0],
    ...                [0, 1, 1, 0],
    ...                [0, 1, 1, 1]])
    >>> join_segmentations(s1, s2)
    array([[0, 1, 3, 2],
           [0, 5, 3, 2],
           [4, 5, 5, 3]])
    """
    if s1.shape != s2.shape:
        raise ValueError("Cannot join segmentations of different shape. " +
                         "s1.shape: %s, s2.shape: %s" % (s1.shape, s2.shape))
    s1 = relabel_sequential(s1)[0]
    s2 = relabel_sequential(s2)[0]
    j = (s2.max() + 1) * s1 + s2
    j = relabel_sequential(j)[0]
    return j


@deprecated('relabel_sequential')
def relabel_from_one(label_field):
    """Convert labels in an arbitrary label field to {1, ... number_of_labels}.

    This function is deprecated, see ``relabel_sequential`` for more.
    """
    return relabel_sequential(label_field, offset=1)


def relabel_sequential(label_field, offset=1):
    """Relabel arbitrary labels to {`offset`, ... `offset` + number_of_labels}.

    This function also returns the forward map (mapping the original labels to
    the reduced labels) and the inverse map (mapping the reduced labels back
    to the original ones).

    Parameters
    ----------
    label_field : numpy array of int, arbitrary shape
        An array of labels, which must be non-negative integers.
    offset : int, optional
        The return labels will start at `offset`, which should be
        strictly positive.

    Returns
    -------
    relabeled : numpy array of int, same shape as `label_field`
        The input label field with labels mapped to
        {offset, ..., number_of_labels + offset - 1}.
        The data type will be the same as `label_field`, except when
        offset + number_of_labels causes overflow of the current data type.
    forward_map : numpy array of int, shape ``(label_field.max() + 1,)``
        The map from the original label space to the returned label
        space. Can be used to re-apply the same mapping. See examples
        for usage. The data type will be the same as `relabeled`.
    inverse_map : 1D numpy array of int, of length offset + number of labels
        The map from the new label space to the original space. This
        can be used to reconstruct the original label field from the
        relabeled one. The data type will be the same as `relabeled`.

    Notes
    -----
    The label 0 is assumed to denote the background and is never remapped.

    The forward map can be extremely big for some inputs, since its
    length is given by the maximum of the label field. However, in most
    situations, ``label_field.max()`` is much smaller than
    ``label_field.size``, and in these cases the forward map is
    guaranteed to be smaller than either the input or output images.

    Examples
    --------
    >>> from skimage.segmentation import relabel_sequential
    >>> label_field = np.array([1, 1, 5, 5, 8, 99, 42])
    >>> relab, fw, inv = relabel_sequential(label_field)
    >>> relab
    array([1, 1, 2, 2, 3, 5, 4])
    >>> fw
    array([0, 1, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5])
    >>> inv
    array([ 0,  1,  5,  8, 42, 99])
    >>> (fw[label_field] == relab).all()
    True
    >>> (inv[relab] == label_field).all()
    True
    >>> relab, fw, inv = relabel_sequential(label_field, offset=5)
    >>> relab
    array([5, 5, 6, 6, 7, 9, 8])
    """
    offset = int(offset)
    # current version can return signed (if it fits in input dtype and that one is signed)
    # but will return unsigned if a dtype change is necessary
    in_vals = np.unique(label_field)
    out_vals = np.arange(offset, offset+len(in_vals))
    input_type = label_field.dtype
    required_type = np.min_scalar_type(out_vals[-1]) # what happens to signed/unsigned ?
    output_type = (input_type if input_type.itemsize > required_type.itemsize
                   else required_type)
    out_array = np.empty(label_field.shape, dtype=output_type)
    map_array(label_field, in_vals, out_vals, out=out_array)
    fw_map = ArrayMap(in_vals, out_vals)
    inv_map = ArrayMap(out_vals, in_vals)
    return out_array, fw_map, inv_map


def map_array(input_arr, input_vals, output_vals, out=None):
    """Map values from input array from input_vals to output_vals.

    Parameters
    ----------
    input_arr : array of int
        The input label image.
    input_vals : array of int
        The values to map from.
    input_vals: 1d array of input values (integer)
    output_vals: 1d array of output values
    out: the output array. Created if not provided
    """

    # We want to reshape to 1D to make the numba loop as simple
    # as possible
    orig_shape = input_arr.shape
    # numpy doc for ravel says 
    # "When a view is desired in as many cases as possible, 
    # arr.reshape(-1) may be preferable."
    input_arr = input_arr.reshape(-1)
    if out is None:
        out= np.empty_like(input_arr, dtype=output_vals.dtype)
    out = out.reshape(-1)

    _map_array(input_arr, out, input_vals, output_vals)
    return out.reshape(orig_shape)


class ArrayMap:
    def __init__(self, inval, outval):
        self.inval = inval
        self.outval = outval
        self._max_lines = 4

    def __repr__(self):
        return f'ArrayMap({repr(self.inval)}, {repr(self.outval)})'

    def __str__(self):
        if len(self.inval) <= self._max_lines:
            rows = range(len(self.inval))
            string = '\n'.join(
                ['ArrayMap:'] +
                [f'  {self.inval[i]} → {self.outval[i]}' for i in rows]
            )
        else:
            rows0 = list(range(0, self._max_lines // 2))
            rows1 = list(range(-self.max_lines // 2, None))
            string = '\n'.join(
                ['ArrayMap:'] +
                [f'  {self.inval[i]} → {self.outval[i]}' for i in rows0] +
                ['  ...'] +
                [f'  {self.inval[i]} → {self.outval[i]}' for i in rows1]
            )
        return string

    def __call__(self, arr):
        return self.__getitem__(arr)

    def __getitem__(self, arr):
        return map_array(arr, self.inval, self.outval)

