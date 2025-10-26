========================
How to parallelize loops
========================

In image processing, we frequently apply the same algorithm
on a large batch of images. In this paragraph, we propose to
use `joblib <https://joblib.readthedocs.io>`_ to parallelize
loops. Here is an example of such repetitive tasks:

.. code-block:: python

    import skimage as ski

    def task(image):
        """
        Apply some functions and return an image.
        """
        image = ski.restoration.denoise_tv_chambolle(
            image[0][0], weight=0.1, channel_axis=-1
        )
        fd, hog_image = ski.feature.hog(
            ski.color.rgb2gray(image),
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            visualize=True
        )
        return hog_image


    # Prepare images
    hubble = ski.data.hubble_deep_field()
    width = 10
    pics = ski.util.view_as_windows(
        hubble, (width, hubble.shape[1], hubble.shape[2]), step=width
    )

To call the function ``task`` on each element of the list ``pics``, it is
usual to write a for loop. To measure the execution time of this loop, you can
use ipython and measure the execution time with ``%timeit``.

.. code-block:: python

    def classic_loop():
        for image in pics:
            task(image)


    %timeit classic_loop()

Another equivalent way to code this loop is to use a comprehension list which has the same efficiency.

.. code-block:: python

    def comprehension_loop():
        [task(image) for image in pics]

    %timeit comprehension_loop()

:func:`skimage.util.apply_parallel` can map a function in parallel across an array.
In this guide, it is used to apply the ``task`` function to each element of a 5D array(image) in parallel.

.. note::
    :func:`skimage.util.apply_parallel` needs the optional dependency dask_ to be installed.
    
.. _dask: https://www.dask.org

.. code-block:: python

    from skimage.util.apply_parallel import apply_parallel
    def apply_parallel_loop():
        [apply_parallel(task, image, dtype=image.dtype,
                        mode="nearest", channel_axis=-1) for image in pics]

    %timeit apply_parallel_loop()

``joblib`` is a library providing an easy way to parallelize for loops once we have a comprehension list.
The number of jobs can be specified.

.. code-block:: python

    from joblib import Parallel, delayed
    def joblib_loop():
        Parallel(n_jobs=4)(delayed(task)(i) for i in pics)

    %timeit joblib_loop()
