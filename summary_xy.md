# Summary of functions using "xy" coordinate convention

This report details functions and methods in the scikit-image library that use the "xy" coordinate convention, where 'x' corresponds to columns (axis 1) and 'y' corresponds to rows (axis 0).

## `skimage.feature.corner`

- **`structure_tensor`**

  - **Reasoning:** Accepts an `order` parameter that can be set to 'xy'. When set, the function uses the last axis (x) first for gradient computation.
  - **GitHub Links:**
    - [Docstring for `order` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L80)
    - [Code using `order='xy'`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L125)

- **`hessian_matrix`**
  - **Reasoning:** Accepts an `order` parameter that can be set to 'xy'. This changes the order of the returned tensor elements from row/column-based ('rc') to x/y-based (Hxx, Hxy, Hyy).
  - **GitHub Links:**
    - [Docstring for `order` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L316)
    - [Code using `order='xy'`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L351)

## `skimage.filters.rank`

- **Rank Filters** (`autolevel`, `equalize`, `gradient`, `maximum`, `mean`, etc.)
  - **Reasoning:** Many functions in the `skimage.filters.rank` module accept `shift_x` and `shift_y` parameters. These parameters offset the center of the footprint in a manner consistent with the 'xy' coordinate convention (shift_x for columns, shift_y for rows).
  - **GitHub Link:**
    - [Docstring for `shift_x`, `shift_y`](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/filters/rank/generic.py#L108)

## `skimage.transform`

### `_warps` submodule

- **`warp`**

  - **Reasoning:** The `inverse_map` callable is expected to process coordinates in `(col, row)` format, which corresponds to `(x, y)`.
  - **GitHub Link:**
    - [Docstring for `inverse_map` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L1024)

- **`warp_coords`**

  - **Reasoning:** The `coord_map` callable is documented to work with `(col, row)` pairs, which is `(x, y)`.
  - **GitHub Link:**
    - [Docstring for `coord_map` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L782)

- **`swirl`**

  - **Reasoning:** The `center` parameter is documented as a `(column, row)` tuple, which is an `(x, y)` convention.
  - **GitHub Link:**
    - [Docstring for `center` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L707)

- **`rotate`**
  - **Reasoning:** The `center` parameter is documented in terms of `(cols, rows)`, which corresponds to an `(x, y)` convention.
  - **GitHub Link:**
    - [Docstring for `center` parameter](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/_warps.py#L465)

### `hough_transform` submodule

- **`hough_line`**

  - **Reasoning:** The docstring explicitly states that the X and Y axes are horizontal and vertical respectively, which is the 'xy' convention.
  - **GitHub Link:**
    - [Notes in Docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L261)

- **`probabilistic_hough_line`**
  - **Reasoning:** The function is documented to return a list of lines in the format `((x0, y0), (x1, y1))`.
  - **GitHub Link:**
    - [Returns section of Docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/transform/hough_transform.py#L330)

## `skimage.measure.fit`

- **`LineModelND`**

  - **Reasoning:** The `predict_x` and `predict_y` methods clearly indicate an assumption of x and y coordinates, distinct from row/column indices.
  - **GitHub Links:**
    - [`predict_x` method](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/fit.py#L384)
    - [`predict_y` method](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/fit.py#L415)

- **`CircleModel`**

  - **Reasoning:** The model is designed to work with `(x, y)` data, and the `predict_xy` method explicitly returns coordinates in this format.
  - **GitHub Link:**
    - [`predict_xy` method](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/fit.py#L683)

- **`EllipseModel`**
  - **Reasoning:** Similar to `CircleModel`, this class operates on `(x, y)` data and has a `predict_xy` method for generating points on the ellipse.
  - **GitHub Link:**
    - [`predict_xy` method](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/fit.py#L984)

# Summary of functions using "ij" (Numpy) coordinate convention

This section details functions and methods that use the "ij" convention, where the first coordinate is the row (axis 0) and the second is the column (axis 1).

## `skimage.measure.regionprops`

- **`regionprops`**
  - **Reasoning:** Properties like `bbox`, `centroid`, and `coords` return coordinates in `(row, col)` order. `moments` are calculated using row/column indices. `orientation` is defined relative to the 0th axis (rows).
  - **GitHub Links:**
    - [`bbox` property](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_regionprops.py#L309)
    - [`centroid` property](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_regionprops.py#L320)
    - [`coords` property](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/measure/_regionprops.py#L335)

## `skimage.feature`

### `blob` submodule

- **`blob_dog`, `blob_log`, `blob_doh`**
  - **Reasoning:** Documentation and implementation confirm return values are `(row, col, sigma)` (or `pln, row, col` for 3D). `blob_doh` mentions `(y, x)` in docstring but `y` corresponds to row and `x` to column.
  - **GitHub Links:**
    - [`blob_dog` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/blob.py#L254)
    - [`blob_log` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/blob.py#L450)
    - [`blob_doh` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/blob.py#L645)

### `peak` submodule

- **`peak_local_max`**
  - **Reasoning:** Returns coordinates of peaks as `(row, col)` indices.
  - **GitHub Link:**
    - [`peak_local_max` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/peak.py#L187)

### `corner` submodule

- **`corner_peaks`**

  - **Reasoning:** Returns coordinates as `(row, col)`.
  - **GitHub Link:**
    - [`corner_peaks` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L1163)

- **`corner_subpix`**
  - **Reasoning:** Input `corners` and output `positions` are `(row, col)`.
  - **GitHub Link:**
    - [`corner_subpix` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/corner.py#L960)

### `util` submodule

- **`plot_matched_features`**
  - **Reasoning:** Input keypoints are `(row, col)`. Implementation maps column index to x-axis for plotting.
  - **GitHub Link:**
    - [`plot_matched_features` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/util.py#L68)

### `texture` submodule

- **`multiblock_lbp`**
  - **Reasoning:** Parameters `r` and `c` refer to row and column coordinates.
  - **GitHub Link:**
    - [`multiblock_lbp` docstring](https://github.com/scikit-image/scikit-image/blob/main/src/skimage/feature/texture.py#L414)
