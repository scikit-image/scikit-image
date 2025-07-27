Dispatching to scikit-image backends
====================================

.. important::
    This backend API is experimental and is not ready for production.
    It is made available as an early prototype so that developers can gain experience
    with the system.

    Expect the dispatching API to change without notice.

Scikit-image offers an API that enables dispatching algorithms to specific backends,
allowing users to run alternative implementations outside of scikit-image while maintaining
the same user interface. These backend packages may provide optimized versions of scikit-image
algorithms, support different array types beyond NumPy, be implemented in other languages
like C++ or Rust, or be tailored for specific hardware such as GPUs.

Using backends
--------------

First, install the scikit-image backend you want to use. By default, the backend
dispatching is **disabled**. To enable it you can either:

- set the environment variable ``SKIMAGE_DISPATCHING = "True"``, or
- use the ``skimage.set_backends(dispatch=True)`` function in your script.

By default, backends are prioritized alphabetically by name. But, you can further configure the backend
priority or a single backend in the following ways:

- Setting a single backend::

        skimage.set_backends("backend_name", dispatch=True)

- Setting backend priority::

        skimage.set_backends("backend_1", "backend_2", "backend_3")

  Here, the first backend (``backend_1``) will be queried for the implementation of an algorithm.
  If it does not implement that algorithm, then the next backend in the list (``backend_2``) will be
  checked, and so on, until the we encounter a backend that does have the implementation for the algorithm.
  If none of the backends in the list implement the algorithm, then the scikit-image's original
  implementation is executed.

- Dispatching within a context manager::

        with skimage.set_backends("backend_1", "backend_2", dispatch=True):
            skimage.metrics.mean_squared_error(img1, img2, additional_arg="foo")

  Here, ``additional_arg`` is an additional parameter supported by ``backend_1`` and ``backend_2``.
  It is recommended to wrap such function calls within a context manager to prevent errors or
  unexpected behavior when falling back to the default scikit-image implementation.

- You can also set the backend(s) without modifying your existing scikit-image code, like this::

        $ export SKIMAGE_DISPATCHING="True" && export SKIMAGE_BACKEND_PRIORITY="backend_1, backend_2" && python scikit_image_code.py


To disable backend dispatching run::

        skimage.set_backends(dispatch=False)


Note that if a ``set_backends`` instance is active in a runtime, then the values
stored in the above two environment variables will be ignored. To delete an
active ``set_backends`` instance run::

        skimage.set_backends.delete_active_instance()


Also, if no backend(s) in the given backend priority,

- are installed on your local machine, or
- provide an alternate implementation for an algorithm,

then scikit-image will fallback to its native implementation of the algorithm.

Additionally, if an error is raised during the execution of a backend implementation,
this fallback will **not** occur, and the error will be propagated.

To know how you can create your own backend refer the :doc:`Developer guide on dispatching <../development/dispatching>`.
