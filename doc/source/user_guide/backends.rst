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

By default, the backend dispatching is **disabled**. To enable and customize backend dispatching, you
can use the `SKIMAGE_BACKENDS` environment variable.

The `SKIMAGE_BACKENDS` environment variable can be configured at runtime in the following ways:

- Using a single backend:

    ```python
    os.environ["SKIMAGE_BACKENDS"] = "backend_name"
    ```

- Using multiple backends:

    ```python
    os.environ["SKIMAGE_BACKENDS"] = "backend_name_1, backend_name_2, backend_name_3"
    ```

    Here, the first backend (`backend_name_1`) will be queried for the implementation of an algorithm.
    If it does not implement that algorithm, then the next backend in the list (`backend_name_2`) will be
    checked, and so on, until the we encounter a backend that does have the implementation for the algorithm.
    If none of the backends in the list implement the algorithm, then the scikit-image's original
    implementations is executed.

- Disabling backend dispatching:

    ```python
    os.environ["SKIMAGE_BACKENDS"] = "False"
    ```

You can also set the backend(s) without modifying your existing scikit-image code file, like this:

    ```sh
    $ export SKIMAGE_BACKENDS="backend_name1, backend_name_2" && python scikit_image_code.py
    ```

Note that if no backend(s) in the `SKIMAGE_BACKENDS`,

- are installed on your local machine, or
- provide an alternate implementation for an algorithm,

then scikit-image will fallback to its native implementation of the algorithm.

Additionally, if an error is raised during the execution of a backend implementation,
this fallback will **not** occur, and the error will be propagated.

To know how you can create your own backend refer the :doc:`Developer guide on dispatching <../development/dispatching>`.
