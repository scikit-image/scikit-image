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

Firstly, you need to install the scikit-image backend package you need to use.

By default, the backend dispatching is **disabled**. To enable and customize backend dispatching, set
the ``SKIMAGE_DISPATCHING`` environment variable to ``"True"``.

By default, backends are prioritized alphabetically by name. But, you can further configure the backend
priority or a single backend using the ``SKIMAGE_BACKEND_PRIORITY`` environment variable at runtime in
the following ways:

- Setting a single backend::

        os.environ["SKIMAGE_BACKEND_PRIORITY"] = "backend_name"

- Setting backend priority::

        os.environ["SKIMAGE_BACKEND_PRIORITY"] = "backend_name_1, backend_name_2, backend_name_3"

  Here, the first backend (``backend_name_1``) will be queried for the implementation of an algorithm.
  If it does not implement that algorithm, then the next backend in the list (``backend_name_2``) will be
  checked, and so on, until the we encounter a backend that does have the implementation for the algorithm.
  If none of the backends in the list implement the algorithm, then the scikit-image's original
  implementation is executed.

You can also set the backend(s) without modifying your existing scikit-image code file, like this::

        $ export SKIMAGE_DISPATCHING="True" && export SKIMAGE_BACKEND_PRIORITY="backend_name1, backend_name_2" && python scikit_image_code.py


To disable backend dispatching::

        os.environ["SKIMAGE_DISPATCHING"] = "False"


Note that if no backend(s) in ``SKIMAGE_BACKEND_PRIORITY``,

- are installed on your local machine, or
- provide an alternate implementation for an algorithm,

then scikit-image will fallback to its native implementation of the algorithm.

Additionally, if an error is raised during the execution of a backend implementation,
this fallback will **not** occur, and the error will be propagated.

To know how you can create your own backend refer the :doc:`Developer guide on dispatching <../development/dispatching>`.
