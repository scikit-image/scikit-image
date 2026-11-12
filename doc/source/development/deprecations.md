(deprecation-cycle)=

# Deprecation cycle (advanced)

If the way a function is called has to be changed, a deprecation cycle
must be followed to warn users.

A deprecation cycle is _not_ necessary when:

- adding a new function, or
- adding a new keyword argument to the _end_ of a function signature, or
- fixing unexpected or incorrect behavior.

A deprecation cycle is necessary when:

- renaming keyword arguments, or
- changing the order of arguments or keywords, or
- adding arguments to a function, or
- changing a function's name or location, or
- changing the default value of function arguments or keywords.

Typically, deprecation warnings are in place for two releases, before
a change is made.

For example, consider the modification of a default value in
a function signature. In version N, we have:

```python
def some_function(image, rescale=True):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image unless ``False`` is given.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    out = do_something(image, rescale=rescale)
    return out
```

In version N+1, we will change this to:

```python
def some_function(image, rescale=None):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image unless ``False`` is given.

        .. warning:: The default value will change from ``True`` to
                     ``False`` in skimage N+3.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    if rescale is None:
        warn('The default value of rescale will change '
             'to `False` in version N+3.', stacklevel=2)
        rescale = True
    out = do_something(image, rescale=rescale)
    return out
```

And, in version N+3:

```python
def some_function(image, rescale=False):
    """Do something.

    Parameters
    ----------
    image : ndarray
        Input image.
    rescale : bool, optional
        Rescale the image if ``True`` is given.

    Returns
    -------
    out : ndarray
        The resulting image.
    """
    out = do_something(image, rescale=rescale)
    return out
```

Here is the process for a 3-release deprecation cycle:

- Set the default to `None`, and modify the
  docstring to specify that the default is `True`.
- In the function, \_if\_ rescale is `None`, set it to `True` and warn that the
  default will change to `False` in version N+3.
- In `doc/release/release_dev.rst`, under deprecations, add "In
  `some_function`, the `rescale` argument will default to `False` in N+3."
- In `TODO.txt`, create an item in the section related to version
  N+3 and write "change rescale default to False in some_function".

Note that the 3-release deprecation cycle is not a strict rule and, in some
cases, developers can agree on a different procedure.

## Raising Warnings

`skimage` raises `FutureWarning`s to highlight changes in its
API, e.g.:

```python
from skimage._shared._warnings import warn_external
warn_external(
    "Automatic detection of the color channel was deprecated in "
    "v0.19, and `channel_axis=None` will be the new default in "
    "v0.22. Set `channel_axis=-1` explicitly to silence this "
    "warning.",
    category=FutureWarning,
)
```

## Deprecating Keywords and Functions

When removing keywords or entire functions, the
`_skimage2._shared.utils.deprecate_parameter` and
`_skimage2._shared.utils.deprecate_func` utility functions can be used
to perform the above procedure.
