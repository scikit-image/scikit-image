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
:ref:`mailing_list`, in order to propose more examples in the future.
Contributing examples to the gallery can be done on github (see
:doc:`../contribute`).

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


.. _mailing_list:

Mailing-list
------------

The scikit-image mailing-list is scikit-image@python.org (users
should `join
<https://mail.python.org/mailman3/lists/scikit-image.python.org/>`_ before posting). This
mailing-list is shared by users and developers, and it is the right
place to ask any question about ``skimage``, or in general, image
processing using Python.  Posting snippets of code with minimal examples
ensures to get more relevant and focused answers.

We would love to hear from how you use ``skimage`` for your work on the
mailing-list!
