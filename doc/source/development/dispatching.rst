Dispatching
===========

The dispatching mechanism allows users to use alternative implementations of the algorithms
available in scikit-image. The alternative implementations are provided by separate
Python packages maintained outside of scikit-image. Example use-cases are support for array
libraries other than Numpy or optimised implementations for particular hardware.

This section of the documentation describes how to create an alternative implementation (a "backend")
and how the dispatching mechanism in scikit-image works.

.. important::
    The dispatching API is experimental and is not ready for production.
    It is made available as an early prototype so that developers can gain experience
    with the system.

    Expect the dispatching API to change without notice.

Using backends
--------------

Scikit-image provides an API that allows the dispatching of algorithms to specific backend(s).
By default, this dispatching is **disabled**. To enable and customize backend dispatching, you
can use the `SKIMAGE_BACKEND_PRIORITY` environment variable.

The `SKIMAGE_BACKEND_PRIORITY` environment variable can be configured in the following ways:

- Using a single backend:

    ```python
    os.environ["SKIMAGE_BACKEND_PRIORITY"] = "backend_name"
    ```

- Using multiple backends:

    ```python
    os.environ["SKIMAGE_BACKEND_PRIORITY"] = "backend_name_1, backend_name_2, backend_name_3"
    ```

    Here, the first backend (`backend_name_1`) will be queried for the implementation of an algorithm.
    If it does not implement that algorithm, then the next backend in the list (`backend_name_2`) will be
    checked, and so on, until the we encounter a backend that does have the implementation for the algorithm.
    If none of the backends in the list implement the algorithm, then the scikit-image's original
    implementations is executed.

- Disabling backend dispatching:

    ```python
    os.environ["SKIMAGE_BACKEND_PRIORITY"] = "False"
    ```

Note that if no backend(s) in the `SKIMAGE_BACKEND_PRIORITY`,

- are installed on your local machine, or
- provide an alternate implementation for an algorithm,  

then scikit-image will fallback to its native implementation of the algorithm.  

Additionally, if an error is raised during the execution of a backend implementation,
this fallback will **not** occur, and the error will be propagated.

Creating a scikit-image backend
-------------------------------

An alternative implementation ("backend") is a good place to provide optimized implementations
for particular hardware, support array libraries other than Numpy or explore novel ideas that
are not (yet) a good fit for the core scikit-image library.

To create a backend you have to create a new Python package that registers two particular
`entry points <https://packaging.python.org/en/latest/specifications/pyproject-toml/#entry-points>`_.
:ref:`backend-infos-entry-point` should resolve to a function with no
arguments that returns an instance of the :py:class:`skimage.util._backends.BackendInformation` class
or something that behaves like it.
:ref:`backend-entry-point` should resolve to a namespace
that contains two functions: `can_has(name, *args, **kwargs)` and `get_implementation(name)`.

.. _backend-infos-entry-point:

The `skimage_backend_infos` entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The information your backend provides via this entry point is used to get things like its
name, homepage and which functions it implements. The main requirement for this entry point
is that it is fast to import and has no dependencies other than `scikit-image`.

The return value of the function should be an instance of the
`skimage.util._backends.BackendInformation` class or something that behaves like it.

The reason this entry point has to be fast is that the list of implemented functions
it provides is used to make a decision on which backends to try for a particular scikit-image
function. This means it is loaded unconditionally as soon as a backend is installed.

To help make your implementation fast avoid computing the list of implemented functions
dynamically or performing other expensive operations.

.. _backend-entry-point:

The `skimage_backends` entry point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This entry point should point to a namespace that contains two functions:
`can_has(name, *args, **kwargs)` and `get_implementation(name)`. They will be used to
determine if the backend can be called with the provided arguments and to get the
actual implementation.

When a user calls a function that the backend listed as one that it implements (via
the `skimage_backend_infos` entry point) the
`can_has` function will be called with the function name and the arguments the user
provided when calling the function. The `can_has` function
should use this information to determine if the backend wants to be called or if a
different backend should be tried. A backend might implement a particular function but
only want to handle calls where the input arrays are of a particuler type or size.

If your backend can not handle a particular call the `can_has` function should return `False` as
quickly as possible. This means you should perform fast checks first and more expensive
checks later in your `can_has` function.

If the `can_has` function indicates that the backend wants to handle the call then the
`get_implementation(name)` function is called to get the implementation. This should
return the backend function that implements the behaviour of the function `name` in scikit-image.
The `name` parameter will contain the public module name and the function name separated by a
colon. For example, the `name` for the `canny` function from the `feature` module would
be `skimage.feature:canny`.

Once the implementation has been retrieved from the backend it will be called with the
arguments the user provided and it is expected to return the result of the computation.

When returning an array it has to be of the same type as the array(s) passed in to the
function by the user. This means a backend implementation can convert the input to a different
array type, but it has to convert the result back to the original array type.


An example backend
~~~~~~~~~~~~~~~~~~

To make the ideas describe above more concrete take a look at `an example backend that implements
a single function <https://github.com/betatim/scikit-image-backend-phony>`_.
This example gives you an idea of how everything fits together and to see the dispatching
in action. It is designed to make it easy to understand and experiment with.
