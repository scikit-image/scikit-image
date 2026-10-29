(guidelines)=

# Guidelines

- All code should have tests (see [test coverage](testing.md#test-coverage) for more details).
- All code should be documented, to the same
  [standard](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard) as NumPy and SciPy.
- For new functionality, always add an example to the gallery (see
  [Gallery](building_docs.md#gallery) for more details).
- No changes are ever merged without review and approval by two core team members.
  There are two exceptions to this rule. First, pull requests which affect
  only the documentation require review and approval by only one core team
  member in most cases. If the maintainer feels the changes are large or
  likely to be controversial, two reviews should still be encouraged. The
  second case is that of minor fixes which restore CI to a working state,
  because these should be merged fairly quickly. Reach out on the
  [developer forum](https://discuss.scientific-python.org/c/contributor/skimage) if
  you get no response to your pull request.
  **Never merge your own pull request.**

## Stylistic Guidelines

- Use the following import conventions:

  ```
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy as sp
  import skimage as ski

  sp.ndimage.label(...)
  ski.measure.label(...)
  ```

- Use numpy data types instead of strings (`np.uint8` instead of
  `"uint8"`).

- When documenting array parameters, use `image : ndarray of shape (M, N)`
  and then refer to `M` and `N` in the docstring, if necessary.

- Refer to array dimensions as (plane), row, column, not as x, y, z. See
  {ref}`Coordinate conventions <numpy-images-coordinate-conventions>`
  in the user guide for more information.

- Functions should support all input image dtypes. Use utility functions such
  as `img_as_float` to help convert to an appropriate type. The output
  format can be whatever is most efficient. This allows us to string together
  several functions into a pipeline, e.g.:

  ```
  hough(canny(my_image))
  ```

- Use relative module imports, i.e. `from .._shared import xyz` rather than
  `from _skimage2._shared import xyz`.

- For Cython functions:

  - Release the GIL whenever possible, using `with nogil:`.
  - Wrap Cython code in a pure Python function, which defines the
    API. This improves compatibility with code introspection tools,
    which are often not aware of Cython code.

- Use `Py_ssize_t` as data type for all indexing, shape and size variables
  in C/C++ and Cython code.
