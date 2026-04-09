"""Sparse table morphology for efficient grayscale morphological operations.

This module implements the sparse table algorithm described in an
unpublished contribution to OpenCV's ximgproc module, ported to
Python/NumPy.
"""

import functools
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["FootprintDecomp", "decomp_footprint"]


@dataclass
class FootprintDecomp:
    """Pre-computed decomposition of a morphological footprint.

    Use :func:`decomp_footprint` to construct this object.
    Pass it to :func:`erode` or :func:`dilate` for efficient repeated operations
    on images with the same footprint.

    Attributes
    ----------
    rows : int
        Number of rows in the original footprint.
    cols : int
        Number of columns in the original footprint.
    dyadic_rects : list of list of list of (int, int)
        ``dyadic_rects[row_depth][col_depth]`` is a list of ``(row, col)``
        origins for dyadic rectangles of height ``2**row_depth`` and width
        ``2**col_depth`` that together cover the footprint at that depth level.
    plan_row : ndarray of bool, shape (max_row_depth, max_col_depth)
        Build schedule for the sparse table; ``True`` means expand in the
        row direction at depth level ``(row_depth, col_depth)``.
    plan_col : ndarray of bool, shape (max_row_depth, max_col_depth)
        Build schedule for the sparse table; ``True`` means expand in the
        column direction at depth level ``(row_depth, col_depth)``.
    max_stack_depth : ndarray of int, shape (max_row_depth, max_col_depth)
        Strahler number of each node: the minimum stack depth needed to DFS
        all leaves from that node when the lighter subtree is visited first.
        Used by :func:`_morph_op` to decide which child to push first (LIFO),
        minimising peak stack depth.
    """

    rows: int
    cols: int
    dyadic_rects: list
    plan_row: np.ndarray
    plan_col: np.ndarray
    max_stack_depth: np.ndarray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log2(n: int) -> int:
    """Integer floor-log2; returns -1 for n == 0."""
    return n.bit_length() - 1


def _max_run_length_row(footprint: np.ndarray) -> int:
    """Maximum consecutive run of nonzero values in any single column."""
    if footprint.size == 0:
        return 0
    fp = (footprint != 0).astype(np.int8)
    # Pad sentinel rows so boundary runs are detected by np.diff
    padded = np.zeros((fp.shape[0] + 2, fp.shape[1]), dtype=np.int8)
    padded[1:-1] = fp
    diff = np.diff(padded, axis=0)          # shape (rows+1, cols)
    starts_r, starts_c = np.nonzero(diff == 1)
    ends_r, ends_c = np.nonzero(diff == -1)
    if starts_r.size == 0:
        return 0
    # Sort by column then row so starts pair with their matching ends
    order_s = np.lexsort((starts_r, starts_c))
    order_e = np.lexsort((ends_r, ends_c))
    return int((ends_r[order_e] - starts_r[order_s]).max())


def _max_run_length_col(footprint: np.ndarray) -> int:
    """Maximum consecutive run of nonzero values in any single row."""
    if footprint.size == 0:
        return 0
    fp = (footprint != 0).astype(np.int8)
    # Pad sentinel cols so boundary runs are detected by np.diff
    padded = np.zeros((fp.shape[0], fp.shape[1] + 2), dtype=np.int8)
    padded[:, 1:-1] = fp
    diff = np.diff(padded, axis=1)          # shape (rows, cols+1)
    starts_r, starts_c = np.nonzero(diff == 1)
    ends_r, ends_c = np.nonzero(diff == -1)
    if starts_r.size == 0:
        return 0
    # Sort by row then col so starts pair with their matching ends
    order_s = np.lexsort((starts_c, starts_r))
    order_e = np.lexsort((ends_c, ends_r))
    return int((ends_c[order_e] - starts_c[order_s]).max())


def _find_dyadic_rect_origins(
    st_node: np.ndarray, row_depth: int, col_depth: int
) -> list:
    """Find top-left origins of dyadic rectangles covering ``st_node``.

    A dyadic rectangle at depth ``(row_depth, col_depth)`` has height
    ``2**row_depth`` and width ``2**col_depth``.  Only "corner" cells that are
    not redundantly covered by a deeper (larger) rectangle are selected.

    Parameters
    ----------
    st_node : ndarray of uint8
        Current sparse table node (a reduced view of the footprint).
    row_depth, col_depth : int
        Depth indices that define the rectangle size.

    Returns
    -------
    list of (int, int)
        Top-left origins of selected dyadic rectangles at this depth level.
    """
    row_ofst = 1 << row_depth
    col_ofst = 1 << col_depth
    n_rows, n_cols = st_node.shape

    active = st_node != 0  # (n_rows, n_cols) bool

    # Cells surrounded on both sides in the column direction
    col_surrounded = np.zeros((n_rows, n_cols), dtype=bool)
    if n_cols >= 3:
        col_surrounded[:, 1:-1] = active[:, :-2] & active[:, 2:]

    # Cells surrounded on both sides in the row direction
    row_surrounded = np.zeros((n_rows, n_cols), dtype=bool)
    if n_rows >= 3:
        row_surrounded[1:-1, :] = active[:-2, :] & active[2:, :]

    interior = col_surrounded | row_surrounded

    # Cells whose col-offset or row-offset neighbor is active
    # (a larger rectangle already covers them)
    has_right = np.zeros((n_rows, n_cols), dtype=bool)
    has_left = np.zeros((n_rows, n_cols), dtype=bool)
    if col_ofst < n_cols:
        has_right[:, :n_cols - col_ofst] = active[:, col_ofst:]
        has_left[:, col_ofst:] = active[:, :n_cols - col_ofst]

    has_down = np.zeros((n_rows, n_cols), dtype=bool)
    has_up = np.zeros((n_rows, n_cols), dtype=bool)
    if row_ofst < n_rows:
        has_down[:n_rows - row_ofst, :] = active[row_ofst:, :]
        has_up[row_ofst:, :] = active[:n_rows - row_ofst, :]

    selected = active & ~interior & ~(has_right | has_left | has_down | has_up)
    rows_idx, cols_idx = np.nonzero(selected)
    return list(zip(rows_idx.tolist(), cols_idx.tolist()))


def _decomp_rect_footprint(
    rows: int, cols: int, max_row_depth: int, max_col_depth: int
) -> list:
    """Compute dyadic_rects analytically for a full rectangular footprint.

    For a full rectangle, ``_find_dyadic_rect_origins`` returns non-empty
    origins at exactly one depth level ``(rd*, cd*)`` where
    ``rd* = floor(log2(rows))`` and ``cd* = floor(log2(cols))``.
    This avoids the ``O(R * C * log^2)`` cost of :func:`_gen_dyadic_cover`.

    Parameters
    ----------
    rows, cols : int
        Shape of the footprint (both must be >= 1).
    max_row_depth, max_col_depth : int
        Depth limits (``floor(log2(rows)) + 1`` and ``floor(log2(cols)) + 1``).

    Returns
    -------
    list of list of list of (int, int)
        Same structure as :func:`_gen_dyadic_cover`.
    """
    rd_star = max_row_depth - 1   # floor(log2(rows))
    cd_star = max_col_depth - 1   # floor(log2(cols))

    # st_node shape at (rd*, cd*): (rows - 2**rd* + 1, cols - 2**cd* + 1)
    n_rows = rows - (1 << rd_star) + 1
    n_cols = cols - (1 << cd_star) + 1

    # Origins: corners of the (n_rows × n_cols) grid at depth (rd*, cd*)
    origins: list = [(0, 0)]
    if n_cols > 1:
        origins.append((0, n_cols - 1))
    if n_rows > 1:
        origins.append((n_rows - 1, 0))
    if n_rows > 1 and n_cols > 1:
        origins.append((n_rows - 1, n_cols - 1))

    return [
        [origins if (rd == rd_star and cd == cd_star) else []
         for cd in range(max_col_depth)]
        for rd in range(max_row_depth)
    ]


def _gen_dyadic_cover(
    footprint: np.ndarray, max_row_depth: int, max_col_depth: int
) -> list:
    """Generate dyadic rectangle covers for all depth combinations.

    For each ``(row_depth, col_depth)`` pair, computes the set of dyadic
    rectangle origins whose union covers the footprint's active cells.

    Parameters
    ----------
    footprint : ndarray of uint8
    max_row_depth, max_col_depth : int

    Returns
    -------
    dyadic_rects : list of list of list of (int, int)
        Top-left origins of dyadic rectangles grouped by depth level.
        At depth ``(row_depth, col_depth)``, each rectangle has height
        ``2**row_depth`` and width ``2**col_depth``.
    """
    dyadic_rects: list = []
    st_node_cache = footprint.astype(np.uint8, copy=True)

    for row_depth in range(max_row_depth):
        st_node = st_node_cache.copy()
        row_rects: list = []
        dyadic_rects.append(row_rects)

        for col_depth in range(max_col_depth):
            row_rects.append(
                _find_dyadic_rect_origins(st_node, row_depth, col_depth)
            )
            col_step = 1 << col_depth
            if st_node.shape[1] - col_step < 0:
                # Pad remaining col_depths with empty lists
                for _ in range(col_depth + 1, max_col_depth):
                    row_rects.append([])
                break
            # Collapse: st_node[:, c] = min(st_node[:, c], st_node[:, c+col_step])
            st_node = np.minimum(
                st_node[:, : st_node.shape[1] - col_step],
                st_node[:, col_step:],
            )

        row_step = 1 << row_depth
        if st_node_cache.shape[0] - row_step < 0:
            # Pad remaining row_depths with empty lists
            for _ in range(row_depth + 1, max_row_depth):
                dyadic_rects.append([[] for _ in range(max_col_depth)])
            break
        # Collapse: st_node_cache[r, :] = min(cache[r, :], cache[r+row_step, :])
        st_node_cache = np.minimum(
            st_node_cache[: st_node_cache.shape[0] - row_step, :],
            st_node_cache[row_step:, :],
        )

    return dyadic_rects


def _solve_rsap_greedy(initial_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Rectilinear Steiner Arborescence Problem greedily.

    Finds a spanning tree over the nodes marked in ``initial_map`` that
    determines the order in which sparse table nodes are computed.

    Parameters
    ----------
    initial_map : ndarray of uint8, shape (max_row_depth, max_col_depth)
        Cells with value 1 are nodes that need to be in the spanning tree.

    Returns
    -------
    plan_row : ndarray of bool, same shape as ``initial_map``
        Build schedule for the sparse table; ``True`` means expand in the
        row direction at depth level ``(row_depth, col_depth)``.
    plan_col : ndarray of bool, same shape as ``initial_map``
        Build schedule for the sparse table; ``True`` means expand in the
        column direction at depth level ``(row_depth, col_depth)``.
    max_stack_depth : ndarray of int32, same shape as ``initial_map``
        Strahler number of each node: the minimum stack depth required to
        DFS all leaves from that node when the lighter subtree is always
        visited first.
    """
    # pos: list of (col, row) in the depth-index space
    pos = [
        (c, r)
        for r in range(initial_map.shape[0])
        for c in range(initial_map.shape[1])
        if initial_map[r, c] == 1
    ]
    plan_row = np.zeros(initial_map.shape, dtype=bool)
    plan_col = np.zeros(initial_map.shape, dtype=bool)

    while len(pos) > 1:
        max_cost = -1
        max_i = max_j = 0
        max_x = max_y = 0

        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                _x = min(pos[i][0], pos[j][0])  # shared col
                _y = min(pos[i][1], pos[j][1])  # shared row
                cost = _x + _y
                if cost > max_cost:
                    max_cost = cost
                    max_i, max_j = i, j
                    max_x, max_y = _x, _y

        # Draw path from pos[max_i] to shared corner (max_x, max_y)
        for col in range(pos[max_i][0] - 1, max_x - 1, -1):
            plan_col[pos[max_i][1], col] = True
        for row in range(pos[max_i][1] - 1, max_y - 1, -1):
            plan_row[row, max_x] = True

        # Draw path from pos[max_j] to shared corner (max_x, max_y)
        for col in range(pos[max_j][0] - 1, max_x - 1, -1):
            plan_col[pos[max_j][1], col] = True
        for row in range(pos[max_j][1] - 1, max_y - 1, -1):
            plan_row[row, max_x] = True

        pos[max_i] = (max_x, max_y)
        pos[max_j] = pos[-1]
        pos.pop()

    # Compute Strahler numbers bottom-up.
    R, C = initial_map.shape
    max_stack_depth = np.ones((R, C), dtype=np.int32)
    for r in range(R - 1, -1, -1):
        for c in range(C - 1, -1, -1):
            pr = plan_row[r, c]
            pc = plan_col[r, c]
            if pr and pc:
                a = max_stack_depth[r + 1, c]
                b = max_stack_depth[r, c + 1]
                max_stack_depth[r, c] = max(a, b) if a != b else a + 1
            elif pr:
                max_stack_depth[r, c] = max_stack_depth[r + 1, c]
            elif pc:
                max_stack_depth[r, c] = max_stack_depth[r, c + 1]
            # else: leaf → stays 1

    return plan_row, plan_col, max_stack_depth


def _plan_st_build(
    dyadic_rects: list,
    max_row_depth: int,
    max_col_depth: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plan the order in which sparse table nodes are built.

    Constructs a map of required nodes (those with at least one dyadic
    rectangle) and solves the RSAP to find the build order.

    Parameters
    ----------
    dyadic_rects : list
        Output of :func:`_gen_dyadic_cover`.
    max_row_depth, max_col_depth : int

    Returns
    -------
    plan_row, plan_col : ndarray of bool
        See :func:`_solve_rsap_greedy`.
    max_stack_depth : ndarray of int32
        See :func:`_solve_rsap_greedy`.  Used to order DFS traversal so the
        shallower subtree is processed first, minimising peak stack depth.
    """
    st_map = np.zeros((max_row_depth, max_col_depth), dtype=np.uint8)
    for rd in range(max_row_depth):
        for cd in range(max_col_depth):
            if len(dyadic_rects[rd][cd]) > 0:
                st_map[rd, cd] = 1
    st_map[0, 0] = 1  # root node is always required
    return _solve_rsap_greedy(st_map)


def _mirror_dyadic_rects(
    dyadic_rects: list, rows: int, cols: int
) -> list:
    """Mirror dyadic rect origins for a 180-degree footprint rotation.

    For a dyadic rectangle of height ``2**row_depth`` and width
    ``2**col_depth`` with top-left origin ``(r, c)`` in a footprint of shape
    ``(rows, cols)``, the mirrored origin is
    ``(rows - 2**row_depth - r, cols - 2**col_depth - c)``.

    Parameters
    ----------
    dyadic_rects : list of list of list of (int, int)
        Output of :func:`_gen_dyadic_cover`.
    rows, cols : int
        Shape of the footprint.

    Returns
    -------
    list of list of list of (int, int)
        Mirrored dyadic rect origins with the same nested structure.
    """
    mirrored: list = []
    for row_depth, row_rects in enumerate(dyadic_rects):
        h = 1 << row_depth
        mirrored_row: list = []
        for col_depth, origins in enumerate(row_rects):
            w = 1 << col_depth
            mirrored_row.append(
                [(rows - h - r, cols - w - c) for r, c in origins]
            )
        mirrored.append(mirrored_row)
    return mirrored


# ---------------------------------------------------------------------------
# Morphological operation engine
# ---------------------------------------------------------------------------


# scipy/skimage2 mode names → numpy.pad mode names
_SCIPY_TO_NUMPY_PAD_MODE = {
    "constant": "constant",
    "reflect": "symmetric",
    "mirror": "reflect",
    "nearest": "edge",
    "wrap": "wrap",
}


def _morph_op(
    ufunc,
    neutral,
    image: np.ndarray,
    rows: int,
    cols: int,
    dyadic_rects: list,
    plan_row: np.ndarray,
    plan_col: np.ndarray,
    max_stack_depth: np.ndarray,
    anchor: Tuple[int, int],
    mode: str,
    cval: float,
) -> np.ndarray:
    """Core morphological operation using a pre-computed sparse table decomposition.

    Pads the image, initializes the output with the neutral element, then
    traverses the sparse table DFS to accumulate results from dyadic rectangles.

    Parameters
    ----------
    ufunc : numpy ufunc
        ``np.minimum`` for erosion, ``np.maximum`` for dilation.
    neutral : scalar
        Neutral element for ``ufunc`` (max dtype value for min, min for max).
    image : ndarray
        Input image.
    rows, cols : int
        Footprint shape.
    dyadic_rects : list
        Pre-computed rectangle origins from :func:`_gen_dyadic_cover`.
    plan_row, plan_col : ndarray of bool
        Build plan from :func:`_plan_st_build`.
    anchor : (int, int)
        ``(row, col)`` position of the footprint origin in the image.
    mode : str
        Border padding mode (scipy convention).
    cval : float
        Constant border value when ``mode='constant'``.

    Returns
    -------
    ndarray
        Morphological operation result, same shape and dtype as ``image``.
    """
    if rows * cols == 1:
        return image.copy()

    anchor_row, anchor_col = anchor
    pad_top = anchor_row
    pad_bottom = rows - 1 - anchor_row
    pad_left = anchor_col
    pad_right = cols - 1 - anchor_col

    pad_mode = _SCIPY_TO_NUMPY_PAD_MODE.get(mode, mode)

    dst = np.empty_like(image)

    if pad_mode == "constant":
        expanded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=cval,
        )
    else:
        expanded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode=pad_mode,
        )

    dst[:] = neutral
    h, w = dst.shape
    # Range trimming is only valid when padding is constant with the neutral
    # element: in other modes the padded border contains real pixel values,
    # not neutral, so we must not skip those columns/rows.
    _trim_ok = pad_mode == "constant" and cval == neutral

    # DFS over the sparse table build plan.
    stack = [(expanded, 0, 0)]
    del expanded

    while stack:
        st, rd, cd = stack.pop()

        for row, col in dyadic_rects[rd][cd]:
            ufunc(dst, st[row : row + h, col : col + w], out=dst)

        has_row = plan_row[rd, cd]
        has_col = plan_col[rd, cd]

        if has_row and has_col:
            ofs_r = 1 << rd
            ofs_c = 1 << cd
            # The heavier subtree (by Strahler number) is pushed first so
            # the lighter one is processed first, minimising peak stack depth.
            if max_stack_depth[rd + 1, cd] >= max_stack_depth[rd, cd + 1]:
                row_st = ufunc(st[:-ofs_r, :], st[ofs_r:, :])
                # NOTE: numpy ufunc allocates an internal buffer for overlapping slices
                #       (unavoidable peak memory cost)
                if _trim_ok:
                    _rl = max(0, anchor_row - ofs_r + 1)
                    _rh = min(st.shape[0], anchor_row + h)
                    ufunc(st[_rl:_rh, :-ofs_c], st[_rl:_rh, ofs_c:], out=st[_rl:_rh, :-ofs_c])
                else:
                    ufunc(st[:, :-ofs_c], st[:, ofs_c:], out=st[:, :-ofs_c])
                stack.append((row_st, rd + 1, cd))
                stack.append((st[:, :-ofs_c], rd, cd + 1))
            else:
                col_st = ufunc(st[:, :-ofs_c], st[:, ofs_c:])
                if _trim_ok:
                    _cl = max(0, anchor_col - ofs_c + 1)
                    _ch = min(st.shape[1], anchor_col + w)
                    ufunc(st[:-ofs_r, _cl:_ch], st[ofs_r:, _cl:_ch], out=st[:-ofs_r, _cl:_ch])
                else:
                    ufunc(st[:-ofs_r, :], st[ofs_r:, :], out=st[:-ofs_r, :])  # see above
                stack.append((col_st, rd, cd + 1))
                stack.append((st[:-ofs_r, :], rd + 1, cd))
        elif has_row:
            ofs_r = 1 << rd
            if _trim_ok:
                _cl = max(0, anchor_col - (1 << cd) + 1)
                _ch = min(st.shape[1], anchor_col + w)
                ufunc(st[:-ofs_r, _cl:_ch], st[ofs_r:, _cl:_ch], out=st[:-ofs_r, _cl:_ch])
            else:
                ufunc(st[:-ofs_r, :], st[ofs_r:, :], out=st[:-ofs_r, :])  # see above
            stack.append((st[:-ofs_r, :], rd + 1, cd))
        elif has_col:
            ofs_c = 1 << cd
            if _trim_ok:
                _rl = max(0, anchor_row - (1 << rd) + 1)
                _rh = min(st.shape[0], anchor_row + h)
                ufunc(st[_rl:_rh, :-ofs_c], st[_rl:_rh, ofs_c:], out=st[_rl:_rh, :-ofs_c])
            else:
                ufunc(st[:, :-ofs_c], st[:, ofs_c:], out=st[:, :-ofs_c])  # see above
            stack.append((st[:, :-ofs_c], rd, cd + 1))

        del st

    return dst


def _neutral_cval(dtype: np.dtype, op: str) -> float:
    """Return the neutral element for the given dtype and morphological operation.

    Parameters
    ----------
    dtype : numpy dtype
        Image dtype.
    op : {'min', 'max'}
        ``'min'`` for erosion, ``'max'`` for dilation.

    Returns
    -------
    float
        Maximum dtype value for ``'min'`` (erosion), minimum for ``'max'``
        (dilation), so border pixels do not influence the result.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.max if op == "min" else info.min
    else:
        maxval = np.finfo(dtype).max
        return maxval if op == "min" else -maxval


@functools.lru_cache(maxsize=32)
def _decomp_footprint_cached(
    fp_bytes: bytes, shape: tuple, dtype: np.dtype
) -> FootprintDecomp:
    """Cached core of :func:`decomp_footprint`.

    Parameters are the normalized (immutable) representation of a footprint
    so that identical footprints share the same ``FootprintDecomp`` object.

    Parameters
    ----------
    fp_bytes : bytes
        Raw bytes of the uint8 footprint array.
    shape : tuple of int
        Shape ``(rows, cols)`` of the footprint.
    dtype : numpy dtype
        Always ``np.dtype('uint8')``; kept as part of the key for clarity.

    Returns
    -------
    FootprintDecomp
        Pre-computed decomposition.  Do not mutate the returned object.
    """
    fp = np.frombuffer(fp_bytes, dtype=dtype).reshape(shape)

    max_row_depth = _log2(_max_run_length_row(fp)) + 1
    max_col_depth = _log2(_max_run_length_col(fp)) + 1

    if fp.all():
        # Fast path: full rectangle — compute dyadic_rects analytically in O(log^2)
        # instead of the O(R * C * log^2) _gen_dyadic_cover loop.
        dyadic_rects = _decomp_rect_footprint(
            fp.shape[0], fp.shape[1], max_row_depth, max_col_depth
        )
    else:
        dyadic_rects = _gen_dyadic_cover(fp, max_row_depth, max_col_depth)

    plan_row, plan_col, max_stack_depth = _plan_st_build(
        dyadic_rects, max_row_depth, max_col_depth
    )

    return FootprintDecomp(
        rows=fp.shape[0],
        cols=fp.shape[1],
        dyadic_rects=dyadic_rects,
        plan_row=plan_row,
        plan_col=plan_col,
        max_stack_depth=max_stack_depth,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decomp_footprint(footprint: np.ndarray) -> FootprintDecomp:
    """Decompose a morphological footprint for sparse table operations.

    Pre-computes the dyadic rectangle cover and build plan for ``footprint``
    so that :func:`erode` and :func:`dilate` can apply the morphological
    operation efficiently, especially when the same footprint is reused across
    many images.

    Results are cached (up to 32 distinct footprints) so repeated calls with
    the same footprint are essentially free.

    Parameters
    ----------
    footprint : ndarray of bool or uint8, shape (M, N)
        The structuring element.  Nonzero values indicate active cells.

    Returns
    -------
    FootprintDecomp
        Pre-computed decomposition.  Do not mutate the returned object.
    """
    fp = np.asarray(footprint, dtype=np.uint8)

    # Handle empty footprint
    if fp.size == 0:
        fp = np.ones((1, 1), dtype=np.uint8)

    # Ensure at least one nonzero element
    if not np.any(fp):
        fp = fp.copy()
        fp[0, 0] = 1

    return _decomp_footprint_cached(fp.tobytes(), fp.shape, fp.dtype)


def erode(
    image: np.ndarray,
    decomp: FootprintDecomp,
    *,
    mode: str = "constant",
    cval: float | None = None,
) -> np.ndarray:
    """Erode an image using a pre-computed sparse table decomposition.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    decomp : FootprintDecomp
        Pre-computed decomposition from :func:`decomp_footprint`.
    mode : str, optional
        Border handling mode passed to :func:`numpy.pad`.
        Default is ``'constant'``.
    cval : scalar or None, optional
        Constant border value when ``mode='constant'``.
        ``None`` (default) uses the maximum value for the image dtype,
        which is the neutral element for the min operation.

    Returns
    -------
    ndarray
        Eroded image, same shape and dtype as ``image``.
    """
    neutral = _neutral_cval(image.dtype, "min")
    if cval is None:
        cval = neutral
    # Erosion anchor: center of footprint.
    # Change here when explicit anchor support is added.
    anchor = (decomp.rows // 2, decomp.cols // 2)
    return _morph_op(
        np.minimum, neutral, image,
        decomp.rows, decomp.cols,
        decomp.dyadic_rects, decomp.plan_row, decomp.plan_col, decomp.max_stack_depth,
        anchor, mode, cval,
    )


def dilate(
    image: np.ndarray,
    decomp: FootprintDecomp,
    *,
    mode: str = "constant",
    cval: float | None = None,
) -> np.ndarray:
    """Dilate an image using a pre-computed sparse table decomposition.

    Parameters
    ----------
    image : ndarray, shape (M, N)
        Input image.
    decomp : FootprintDecomp
        Pre-computed decomposition from :func:`decomp_footprint`.
    mode : str, optional
        Border handling mode passed to :func:`numpy.pad`.
        Default is ``'constant'``.
    cval : scalar or None, optional
        Constant border value when ``mode='constant'``.
        ``None`` (default) uses the minimum value for the image dtype,
        which is the neutral element for the max operation.

    Returns
    -------
    ndarray
        Dilated image, same shape and dtype as ``image``.
    """
    neutral = _neutral_cval(image.dtype, "max")
    if cval is None:
        cval = neutral
    mirrored_rects = _mirror_dyadic_rects(
        decomp.dyadic_rects, decomp.rows, decomp.cols
    )
    # Dilation anchor: mirrored center (differs from erosion for even-size fp).
    # Change here when explicit anchor support is added.
    anchor = (decomp.rows - 1 - decomp.rows // 2, decomp.cols - 1 - decomp.cols // 2)
    return _morph_op(
        np.maximum, neutral, image,
        decomp.rows, decomp.cols,
        mirrored_rects, decomp.plan_row, decomp.plan_col, decomp.max_stack_depth,
        anchor, mode, cval,
    )
