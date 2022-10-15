=================================
Getting help on using ``skimage``
=================================

Besides the user guide, there exist other opportunities to get help on
using ``skimage``.

Examples gallery
----------------

The :ref:`examples_gallery` gallery provides graphical examples of
typical image processing tasks. By a quick glance at the different
thumbnails, the user may find an example close to a typical use case of
interest. Each graphical example page displays an introductory paragraph,
a figure, and the source code that generated the figure. Downloading the
Python source code enables one to modify quickly the example into a case
closer to one's image processing applications.

Users are warmly encouraged to report on their use of ``skimage`` on the
`forum <https://discuss.scientific-python.org/c/contributor/skimage>`_, in
order to propose more examples in the future.

Search field
------------

The ``quick search`` field located in the navigation bar of the html
documentation can be used to search for specific keywords (segmentation,
rescaling, denoising, etc.).

API Discovery
-------------

NumPy provides a ``lookfor`` function to search API functions.
By default ``lookfor`` will search the NumPy API.
NumPy lookfor example:
```np.lookfor('eigenvector') ```

But it can be used to search in modules, by passing in the module
name as a string:

``` np.lookfor('boundaries', 'skimage') ```

or the module itself.
```
> import skimage
> np.lookfor('boundaries', skimage)
```

Docstrings
----------

Docstrings of ``skimage`` functions are formatted using `Numpy's
documentation standard
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_,
starting with a ``Parameters`` section for the arguments and a
``Returns`` section for the objects returned by the function. Also, most
functions include one or more examples.


Ask for help
------------

If you still have questions, reach out through

- our `user forum <https://forum.image.sc/tags/scikit-image>`_
- our `developer forum
  <https://discuss.scientific-python.org/c/contributor/skimage>`_
- our `chat channel <https://skimage.zulipchat.com/>`_
- `Stack Overflow <https://stackoverflow.com/questions/tagged/scikit-image>`_
