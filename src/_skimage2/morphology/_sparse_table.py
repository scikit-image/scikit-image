"""Sparse table morphology for efficient grayscale morphological operations.

This module ports the sparse table algorithm from OpenCV's ximgproc module
(sparse_table_morphology.cpp) to Python/NumPy.
"""

from dataclasses import dataclass
from typing import List, Tuple

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
        ``plan_row[rd, cd]`` is ``True`` when ``st[rd+1][cd]`` should be
        computed from ``st[rd][cd]`` (expand in the row direction).
    plan_col : ndarray of bool, shape (max_row_depth, max_col_depth)
        ``plan_col[rd, cd]`` is ``True`` when ``st[rd][cd+1]`` should be
        computed from ``st[rd][cd]`` (expand in the column direction).
    anchor : (int, int)
        Anchor position ``(row, col)`` within the footprint.
    iterations : int
        Number of times the operation is applied.
    """

    rows: int
    cols: int
    dyadic_rects: List[List[List[Tuple[int, int]]]]
    plan_row: np.ndarray
    plan_col: np.ndarray
    anchor: Tuple[int, int]
    iterations: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log2(n: int) -> int:
    """Integer floor-log2; returns -1 for n == 0."""
    ans = -1
    while n > 0:
        n //= 2
        ans += 1
    return ans


def _max_run_length_row(footprint: np.ndarray) -> int:
    """Maximum consecutive run of nonzero values in any single column.

    Scanning each column top-to-bottom gives the longest contiguous span of
    active cells in the row direction, which determines ``max_row_depth``.
    """
    max_len = 0
    for c in range(footprint.shape[1]):
        cnt = 0
        for r in range(footprint.shape[0]):
            if footprint[r, c] == 0:
                max_len = max(max_len, cnt)
                cnt = 0
            else:
                cnt += 1
        max_len = max(max_len, cnt)
    return max_len


def _max_run_length_col(footprint: np.ndarray) -> int:
    """Maximum consecutive run of nonzero values in any single row.

    Scanning each row left-to-right gives the longest contiguous span of
    active cells in the column direction, which determines ``max_col_depth``.
    """
    max_len = 0
    for r in range(footprint.shape[0]):
        cnt = 0
        for c in range(footprint.shape[1]):
            if footprint[r, c] == 0:
                max_len = max(max_len, cnt)
                cnt = 0
            else:
                cnt += 1
        max_len = max(max_len, cnt)
    return max_len


def _find_dyadic_rect_origins(
    st_node: np.ndarray, row_depth: int, col_depth: int
) -> List[Tuple[int, int]]:
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
        ``(row, col)`` top-left origins of the selected dyadic rectangles.
    """
    row_ofst = 1 << row_depth
    col_ofst = 1 << col_depth
    origins = []
    n_rows, n_cols = st_node.shape

    for row in range(n_rows):
        for col in range(n_cols):
            if st_node[row, col] == 0:
                continue

            # Skip interior cells (surrounded on both sides in either axis)
            if (
                col > 0
                and st_node[row, col - 1] == 1
                and col + 1 < n_cols
                and st_node[row, col + 1] == 1
            ):
                continue
            if (
                row > 0
                and st_node[row - 1, col] == 1
                and row + 1 < n_rows
                and st_node[row + 1, col] == 1
            ):
                continue

            # Skip if a cell at distance 2**depth is also active
            # (a larger rectangle already covers this one)
            if col + col_ofst < n_cols and st_node[row, col + col_ofst] == 1:
                continue
            if col - col_ofst >= 0 and st_node[row, col - col_ofst] == 1:
                continue
            if row + row_ofst < n_rows and st_node[row + row_ofst, col] == 1:
                continue
            if row - row_ofst >= 0 and st_node[row - row_ofst, col] == 1:
                continue

            origins.append((row, col))

    return origins


def _gen_dyadic_cover(
    footprint: np.ndarray, max_row_depth: int, max_col_depth: int
) -> List[List[List[Tuple[int, int]]]]:
    """Generate dyadic rectangle covers for all depth combinations.

    For each ``(row_depth, col_depth)`` pair, computes the set of dyadic
    rectangle origins whose union covers the footprint's active cells.

    The algorithm progressively collapses the footprint via element-wise
    minimum (building up the sparse table structure) to identify which
    positions still have active cells at each depth level.

    Parameters
    ----------
    footprint : ndarray of uint8
    max_row_depth, max_col_depth : int

    Returns
    -------
    dyadic_rects : list of list of list of (int, int)
        ``dyadic_rects[row_depth][col_depth]`` → ``[(row, col), ...]``
    """
    dyadic_rects: List[List[List[Tuple[int, int]]]] = []
    st_node_cache = footprint.astype(np.uint8, copy=True)

    for row_depth in range(max_row_depth):
        st_node = st_node_cache.copy()
        row_rects: List[List[Tuple[int, int]]] = []
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
        ``plan_row[rd, cd]`` is ``True`` when ``st[rd+1][cd]`` is computed
        from ``st[rd][cd]`` (row direction edge).
    plan_col : ndarray of bool, same shape as ``initial_map``
        ``plan_col[rd, cd]`` is ``True`` when ``st[rd][cd+1]`` is computed
        from ``st[rd][cd]`` (column direction edge).
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

    return plan_row, plan_col


def _plan_st_build(
    dyadic_rects: List[List[List[Tuple[int, int]]]],
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
    """
    st_map = np.zeros((max_row_depth, max_col_depth), dtype=np.uint8)
    for rd in range(max_row_depth):
        for cd in range(max_col_depth):
            if len(dyadic_rects[rd][cd]) > 0:
                st_map[rd, cd] = 1
    st_map[0, 0] = 1  # root node is always required
    return _solve_rsap_greedy(st_map)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decomp_footprint(
    footprint: np.ndarray,
    anchor: Tuple[int, int] | None = None,
    iterations: int = 1,
) -> FootprintDecomp:
    """Decompose a morphological footprint for sparse table operations.

    Pre-computes the dyadic rectangle cover and build plan for ``footprint``
    so that :func:`erode` and :func:`dilate` can apply the morphological
    operation efficiently, especially when the same footprint is reused across
    many images.

    Parameters
    ----------
    footprint : ndarray of bool or uint8, shape (M, N)
        The structuring element.  Nonzero values indicate active cells.
    anchor : (int, int) or None, optional
        Anchor position ``(row, col)`` within the footprint.
        ``None`` (default) places the anchor at the footprint center.
    iterations : int, optional
        Number of times erosion/dilation is applied. Default is 1.

    Returns
    -------
    FootprintDecomp
        Pre-computed decomposition.
    """
    fp = np.asarray(footprint, dtype=np.uint8)

    # Handle empty footprint (match OpenCV behaviour)
    if fp.size == 0:
        size = 1 + iterations * 2
        fp = np.ones((size, size), dtype=np.uint8)
        anchor = (iterations, iterations)
        iterations = 1

    # Ensure at least one nonzero element
    if not np.any(fp):
        fp = fp.copy()
        fp[0, 0] = 1

    max_row_depth = _log2(_max_run_length_row(fp)) + 1
    max_col_depth = _log2(_max_run_length_col(fp)) + 1

    dyadic_rects = _gen_dyadic_cover(fp, max_row_depth, max_col_depth)
    plan_row, plan_col = _plan_st_build(dyadic_rects, max_row_depth, max_col_depth)

    if anchor is None:
        anchor = (fp.shape[0] // 2, fp.shape[1] // 2)

    return FootprintDecomp(
        rows=fp.shape[0],
        cols=fp.shape[1],
        dyadic_rects=dyadic_rects,
        plan_row=plan_row,
        plan_col=plan_col,
        anchor=anchor,
        iterations=iterations,
    )


# ---------------------------------------------------------------------------
# Morphological operation engine
# ---------------------------------------------------------------------------


def _apply_morph_dfs(
    ufunc,
    st: np.ndarray,
    dst: np.ndarray,
    dyadic_rects: List[List[List[Tuple[int, int]]]],
    plan_row: np.ndarray,
    plan_col: np.ndarray,
    row_depth: int,
    col_depth: int,
) -> None:
    """Apply morphological operation via sparse table DFS traversal.

    Parameters
    ----------
    ufunc : numpy ufunc
        ``np.minimum`` for erosion, ``np.maximum`` for dilation.
    st : ndarray
        Current sparse table node (expanded source image).
    dst : ndarray
        Destination image updated in-place.
    dyadic_rects : list
        Pre-computed rectangle origins from :func:`_gen_dyadic_cover`.
    plan_row, plan_col : ndarray of bool
        Build plan from :func:`_plan_st_build`.
    row_depth, col_depth : int
        Current depth indices.
    """
    h, w = dst.shape

    # Apply all dyadic rectangles at the current depth level
    for row, col in dyadic_rects[row_depth][col_depth]:
        roi = st[row : row + h, col : col + w]
        ufunc(dst, roi, out=dst)

    if plan_col[row_depth, col_depth]:
        # Expand in the column direction: create a narrower sparse table node.
        # st2 is a new array; the original st is unchanged so the row branch
        # below still sees the full-width st.
        ofs = 1 << col_depth
        st2 = ufunc(st[:, :-ofs], st[:, ofs:])
        _apply_morph_dfs(
            ufunc, st2, dst, dyadic_rects, plan_row, plan_col,
            row_depth, col_depth + 1,
        )

    if plan_row[row_depth, col_depth]:
        # Expand in the row direction: rebind local st to a shorter array.
        # Rebinding a local name in Python does not affect the caller's
        # reference, so this is safe even though st is passed by reference.
        ofs = 1 << row_depth
        st = ufunc(st[:-ofs, :], st[ofs:, :])
        _apply_morph_dfs(
            ufunc, st, dst, dyadic_rects, plan_row, plan_col,
            row_depth + 1, col_depth,
        )


def _morph_op(
    ufunc,
    nil,
    image: np.ndarray,
    decomp: FootprintDecomp,
    mode: str,
    cval: float,
) -> np.ndarray:
    """Core morphological operation using a pre-computed :class:`FootprintDecomp`."""
    if decomp.iterations == 0 or decomp.rows * decomp.cols == 1:
        return image.copy()

    anchor_row, anchor_col = decomp.anchor
    pad_top = anchor_row
    pad_bottom = decomp.rows - 1 - anchor_row
    pad_left = anchor_col
    pad_right = decomp.cols - 1 - anchor_col

    src = image
    dst = np.empty_like(image)

    for _ in range(decomp.iterations):
        if mode == "constant":
            expanded = np.pad(
                src,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=cval,
            )
        else:
            expanded = np.pad(
                src,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode=mode,
            )

        dst[:] = nil
        _apply_morph_dfs(
            ufunc, expanded, dst,
            decomp.dyadic_rects, decomp.plan_row, decomp.plan_col,
            0, 0,
        )
        src = dst

    return dst


def _neutral_cval(dtype: np.dtype, op: str) -> float:
    """Return the neutral constant-border value for the given dtype and operation."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return info.max if op == "min" else info.min
    else:
        maxval = np.finfo(dtype).max
        return maxval if op == "min" else -maxval


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
    nil = _neutral_cval(image.dtype, "min")
    if cval is None:
        cval = nil
    return _morph_op(np.minimum, nil, image, decomp, mode, cval)


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
    nil = _neutral_cval(image.dtype, "max")
    if cval is None:
        cval = nil
    return _morph_op(np.maximum, nil, image, decomp, mode, cval)
