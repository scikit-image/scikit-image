# Style guide

This is a living document that collects conventions for the project's code and documentation.
It is a reference for core team members and interested contributors.

If something is not covered here, fall back to existing conventions of the scientific Python ecosystem and the Python community in general.
These are non-binding general guidelines but arguments should be made for exceptions.

## API design

- Callables should always return an object or collection of the same type.
  Prefer returning a single object or collection of objects of the same type if possible.
  Consider creating a similarly named function to return a different type if necessary.

- Mandatory parameters that a function operates on or is named after can be passed as _positional or keyword_ parameters.
  Use _keyword-only_ parameters for the remaining ones, especially if they are optional.
  Examples:

  ```python
  def regionprops(label_image, *, intensity_image=None, ...): ...

  def pearson_corr_coeff(image0, image1, *, mask=None): ...

  def ransac(data, model_class, *, min_samples, residual_threshold, ...): ...
  ```

- Functions should support all input image dtypes. Use utility functions such
  as `img_as_float` to help convert to an appropriate type. The output
  format can be whatever is most efficient. This allows us to string together
  several functions into a pipeline like this:

  ```python
  hough(canny(my_image))
  ```

- Every object that is not exposed as public API should be prefixed with an underscore.
  This allows differentiating internal and public functions at a glance.

## Testing

- All code paths and parameter combinations should be covered by tests.
  Testing input validation, errors and warnings is optional but encouraged – especially when non-trivial conditional statements or control flow is involved.

## NumPy

- Use numpy data types instead of strings (`np.uint8` instead of `"uint8"`).

## Documentation

- All code should be documented, to the same [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>) as NumPy and SciPy.

- For new functionality, always add an example to the gallery.

- Refer to array dimensions as _i, j, k_, not as _x, y, z_ or _(plane,) row, column_.
  Exceptions can be made where the concept of a plane, rows or columns is intrinsic to the algorithm.
  See {ref}`Coordinate conventions <numpy-images-coordinate-conventions>` in the user guide for more information.

- When documenting array parameters, use `image : ndarray of shape (M, N)` and then refer to `M` and `N` in the docstring, if necessary.

## Import conventions

- Use the following import conventions:

  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy as sp
  import skimage as ski
  import skimage2 as ski2

  sp.ndimage.label(...)
  ski.measure.label(...)
  ski2.measure.label(...)
  ```

- Use relative module imports like `from .._shared import xyz` rather than
  `from skimage._shared import xyz`.

## Cython and compiled code

- For Cython functions:

  - Release the GIL whenever possible, using `with nogil:`.
  - Wrap Cython code in a pure Python function, which defines the
    API. This improves compatibility with code introspection tools,
    which are often not aware of Cython code.

- Use `Py_ssize_t` as data type for all indexing, shape and size variables
  in C/C++ and Cython code.
