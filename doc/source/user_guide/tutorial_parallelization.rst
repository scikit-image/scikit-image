=========================
Parallel image processing
=========================

Sometimes, the need arises to process batches of images, or images
that do not fit into memory.  We recommend `joblib
<https://pythonhosted.org/joblib/>`__ or `dask
<https://dask.pydata.org>`__ for such tasks.

Batch processing with ``joblib``
--------------------------------

Consider the case where you have a directory with, say 2000, images.
On each of these, you want to calculate the optimal threshold:

.. code-block:: python

    from skimage import io, filters, color
    import glob

    # Load all images into memory
    image_files = sorted(glob.glob('*.png'))
    images = [io.imread(image) for image in image_files]

    def threshold(image):
        return filters.threshold_otsu(color.rgb2gray(image))

    # Process all images serially and store the result
    thresholds = [threshold(image) for image in images]

The above code has two flaws: first, it attempts to load *all* image
data into memory at once.  Second, it processes the images serially,
not utilizing more than one CPU core.

Instead, we can parallelize the batch processing as follows:

.. code-block:: python

    from joblib import Parallel, delayed

    image_files = sorted(glob.glob('*.png'))

    def load_and_threshold(image_fn):
        image = io.imread(image_fn)
        return filters.threshold_otsu(color.rgb2gray(image))

    threshold_tasks = [delayed(load_and_threshold)(fn) for fn in image_files]

    p = Parallel(n_jobs=-1, backend='threading')
    thresholds = p(threshold_tasks)

The ``delayed`` call is to prevent ``load_and_threshold`` from
executing immediately while setting up the Parallel processing.
Instead, it becomes a task that can be scheduled.  Setting ``n_jobs``
to -1 tells ``joblib`` to use however many cores are available.

Note that speedups can only be expected in the case where the
individual operations are somewhat long running (i.e., thresholding is
not the best example).  Otherwise, the overhead of scheduling and
executing tasks can be more than what is gained by executing in
parallel.


Batch processing with ``dask``
------------------------------

In ``dask``, the above processing looks as follows:

.. code-block:: python

    from dask import delayed, compute

    image_files = sorted(glob.glob('*.png'))

    def load_and_threshold(image_fn):
        image = io.imread(image_fn)
        return filters.threshold_otsu(color.rgb2gray(image))

    threshold_tasks = [delayed(load_and_threshold)(fn) for fn in image_files]
    thresholds = compute(threshold_tasks)

``dask`` can also deploy tasks across multiple nodes (computers) by
using the `distributed <http://distributed.readthedocs.io>`__
scheduler.


Tiled processing of large images with ``dask``
----------------------------------------------

When a large image is processed, we can often improve performance by
a) parallelizing computation and b) reducing the amount of memory
utilized at any point in time.
<<<<<<< HEAD
=======

``dask`` has tbe
>>>>>>> 2a1333e0ea0097b4da3529d7c4431910cadfd5cd

.. code-block:: python

    from skimage import data, color, util
    from skimage.restoration import denoise_tv_chambolle
    from skimage.feature import hog

    def task(image):
        """
        Apply some functions and return an image.
        """
        image = denoise_tv_chambolle(image[0][0], weight=0.1, multichannel=True)
        fd, hog_image = hog(color.rgb2gray(image), orientations=8,
                            pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                            visualize=True)
        return hog_image


    # Prepare images
    hubble = data.hubble_deep_field()
    width = 10
    pics = util.view_as_windows(hubble, (width, hubble.shape[1], hubble.shape[2]), step=width)

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

``joblib`` is a library providing an easy way to parallelize for loops once we have a comprehension list.
The number of jobs can be specified.

.. code-block:: python

    from joblib import Parallel, delayed
    def joblib_loop():
        Parallel(n_jobs=4)(delayed(task)(i) for i in pics)

    %timeit joblib_loop()
