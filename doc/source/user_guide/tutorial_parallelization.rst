========================
How to parallelize loops
========================

In image processing, we frequently apply the same algorithm
on a large batch of images. In this paragraph, we propose to
use `joblib <https://pythonhosted.org/joblib/>`_ to parallelize
loops. Here is an example of such repetitive tasks:

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
