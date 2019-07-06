
Batch processing with Dask
==========================

This example shows how to use `Dask <http://dask.pydata.org/>`__ to
parallelize a batch processing job. Parallelized code doesn’t have to
look too different than serial code in python.

Our job is organized as follows

1. Load our images as (dask) arrays.
2. Apply scikit-image filters to our images.
3. Compute and return some important metric about the image.
4. Save resulting images to disk.

The main difference in using dask is that the computation is done in a
so called *lazy* fashion. Lazy means that no results are computed until
a call to the ``compute`` function is made. The advantage of
parallelization comes when you have access to a machine with many cores.
Most modern computers, including laptops and tablet, now have at least
two cores.

In the example below, the same computation will be done twice, once
serially, and a second time using dask. You should experiment with the
size and number of the images to compare the performance improvements
for different cases.

.. code:: ipython3

    import tempfile
    import skimage.filters as filters
    import numpy as np
    import os
    import imageio
    from skimage import img_as_ubyte
    
    # The number of images we wish to analyze
    N_images = 20
    # The shape of each generated image
    shape = (1024, 1024)
    # The directory where we wish to save the final results.
    # The code will ask your operating system for a temporary directory
    save_directory = tempfile.mkdtemp()

Serial processing
-----------------

Loading function
~~~~~~~~~~~~~~~~

Unfortunately, parallel computing gives the best results when using
large datasets. Currently, scikit-image doesn’t provide any dataset that
is large enough to show the true advantages of dask. To overcome this
limitation, we will use a random number generator to create the data. We
encourage you to adapt this function to load your dataset from disk.

.. code:: ipython3

    def load_my_data(index, shape, max_index=255):
        """A simple mock loading function.
    
        Return an image that is made of Gaussian noise centered about
        `index/max_index` with standard deviation equal to 20/max_index
        """
        # Make sure to use independend random number generators otherwise
        # parallel code might have conflicts
        r = np.random.RandomState(index)
        image =  r.normal(loc=((index+1) % max_index)/max_index, scale=20/max_index, size=shape)
        return np.clip(image, 0, 1)

.. code:: ipython3

    %%time
    # Step 1. Load our images as arrays.
    images = []
    for i in range(0, N_images):
        image = load_my_data(i, shape=shape)
        images.append(image)

.. code:: ipython3

    %%time
    input_variances = []
    output_variances = []
    output_images = []
    
    for image in images:
        input_variance = np.var(image)
        # We process the input image to generate our output
        # Step 2. Apply a scikit-image filter
        output_image = filters.gaussian(image, 10)
        # Step 3. Compute and return some important metric
        output_variance = np.var(output_image)
        output_images.append(output_image)
    
        # Store all the results
        input_variances.append(input_variance)
        output_variances.append(output_variance)

.. code:: ipython3

    %%time
    # Step 4. Save the resulting images to disk.
    for i, image in enumerate(output_images):    
        image_ubyte = img_as_ubyte(image)
        filename = os.path.join(save_directory,
                                'image_{i:2d}.bmp'.format(i=i))
        imageio.imwrite(filename, image_ubyte)

A note on memory usage
~~~~~~~~~~~~~~~~~~~~~~

We find that in many cases this kind of organization very useful during
prototyping stages:

1. All images can be easily accessed from the variables ``images``
2. Inspection of their metadata (``dtype``, ``shape``) is readily
   acheived.
3. There is no need to rewrite the code between the *prototyping* stage
   and the *useful* execution stage where you might increase ``N`` from
   ``10`` to ``1000``\ s. This leads to fewer bugs.

Unfortunately, loading images can become a daunting task since realistic
images, stored as PNGs or JPGs can often acheive compression ratios of
10:1. 1GB of images on your disk, might become 10GB or more when loaded
as full numpy arrays in python. As such, it might be useful to refactor
your code in a single loop that only keeps one image loaded at the same
time.

Parallel computation with dask
------------------------------

For this tutorial, we will make use of the ``delayed`` module in Dask.
By default, the delayed module will start multiple python processes,
each computing part of the desired computation.

Instead of calling our functions ``load_my_data``, ``var``,
``gaussian``, and ``imwrite`` directly, we will be calling delayed
versions of them that will eventually be executed when we issue a
``compute`` instruction.

For example, instead of calling the function ``load_my_data(5)``, we
will call the function ``delayed(load_my_data)(5)``

.. code:: ipython3

    %%time
    from dask import delayed
    images = []
    for i in range(0, N_images):
        # Step 1. Load our images as delayed arrays.
        image = delayed(load_my_data)(i, shape=shape)
        images.append(image)
        
    print(images[0])

This loop returned almost immediately! This isn’t because the data has
been loaded into memory. Rather dask provided us a ``Delayed`` object
that promises to execute ``load_my_data`` in the future. We proceed to
wrapping our calls to ``np.var`` and to ``scikit-image`` in delayed
calls.

If you need to access a particular image you can do so by slicing the
list with ``images[index]``. Calling ``compute`` will cause the image to
be loaded and stored into memory.

.. code:: python

   image_of_interest = images[index].compute()

Note that on calls to ``compute``, Dask is re-computing all of the
computation (in this case, just loading). This is likely acceptable
since computing individual images might be rather quick compared.

.. code:: ipython3

    %%time
    input_variances = []
    output_variances = []
    output_images = []
    
    for image in images:
        input_variance = delayed(np.var)(image)
        # We process the input image to generate our output
        # Step 2. Apply a scikit-image filter
        output_image = delayed(filters.gaussian)(image, 10)
        # Step 3. Compute and return some important metric
        output_variance = delayed(np.var)(output_image)
        output_images.append(output_image)
    
        # Store all the results
        input_variances.append(input_variance)
        output_variances.append(output_variance)

.. code:: ipython3

    %%time
    # Step 4. Delay saving resulting images to disk.
    saved_list = []
    for i, image in enumerate(output_images):
        image_ubyte = delayed(img_as_ubyte)(image)
        filename = os.path.join(save_directory,
                                'image_{i:2d}.bmp'.format(i=i))
        saved_list.append(delayed(imageio.imwrite)(filename, image_ubyte))

We can now visualize what the computation looks like. Visualizing the
whole batch might be a little daunting so we will instead visualize the
first 3 elements of our computation. You will need to install
``python-graphviz`` for these next few lines to execute.

.. code:: ipython3

    import dask
    # uncomment the line below to visualize the graph
    # dask.visualize(input_variances[:3], output_variances[:3], saved_list[:3])

Visualizing the computation graph is a good way to double check your
code. Here, the important aspect of the graph is that the data paths for
the analysis is completely independent from one image to the next. This
is what will help us get the most from parallelization.

Computing the result
--------------------

Finally, we will issue a call to ``dask.compute`` for all the outputs of
interest. For our specific comptuation, we don’t actually care about
getting the ``output_images``. Rather, we simply want to ensure that
they are computed. For that, we will ask for the results of
``intput_variances``, ``output_variances``, and ``saved_list``.
``saved_list`` will actually be a list of ``None`` elements, but it will
ensure that that branch of the computation graph is executed.

.. code:: ipython3

    %%time
    input_variances, output_variances, saved_list = dask.compute(
        input_variances, output_variances, saved_list)

Discussion
----------

The table below summarizes the results of running the program above
changing the values of ``N`` and ``shape`` for a few typical examples.
All examples were run on a computer with the following specifications:

-  Processor: i7-7700HQ, 4 cores, hyperthreading,
-  RAM: 16GB of RAM
-  Storage: Samsung 960 PRO SSD.

===== =========== ==================== ======================= =======
N     shape       Wall time serial (s) Wall time with Dask (s) Speedup
===== =========== ==================== ======================= =======
10    2048 x 2048 6.4                  2.56                    2.5 x
50    2048 x 2048 32.5                 11.1                    3 x
50    512 x 512   1.85                 1.71                    1.1 x
500   512 x 512   18                   13.9                    1.3 x
50000 32 x 32     43                   4 + 7 + 5 + 88          0.4 x
5000  32 x 32     4.35                 .4 + .6 + .5 + 8.6      0.42 x
===== =========== ==================== ======================= =======

If we were computationally bound, the best case theoretical speedup will
be around 4x as the particular processor used has 4 cores, each with
their own arithmetic logical unit able to perform independent
computation.

Was it surprising to you that not all computation was able to benefit
from parallelization with Dask? When trying to accelerate your program,
it isn’t always obvious how speedups affect different workloads. Here
are a few things you can do to help make this more systematic.

Benchmark your code
~~~~~~~~~~~~~~~~~~~

The results above do not show uniform improvement when using Dask. For
very small images, this kind of parallelization actually hurts
performance! Make sure you first get a good feeling for the amount of
time it takes for your code to run before starting to optimize for
speed. Rigorous benchmarks might seem ideal, but they often aren’t
practical when rapidly developing something. The ``%time`` or
``%timeit`` magic commands in IPython can really help.

You can’t always assume that the rate limiting step will be the same for
different types of images and different image sizes. As such, it may be
helpful to benchmark your whole analysis pipeline.

i/o speed
~~~~~~~~~

Are you using a slow storage medium such as a hard disk? Upgrading to a
Solid State Drive (SSD) might be the easiest and cheapest way to speed
up your whole workflow. While a hard-drive might be fast at copying
large files from one directory to an other, it can be slow accessing
multiple files at once. Solid state drives overcome these problems and
have become relatively inexpensive in recent years.

Numpy and scipy already try to parrallelize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Is numpy’s parallelization enough? In the example above, numpy and scipy
do parallize the computation of the variance. We encourage you to look
at your CPU usage and observe how multiple cores are working together
during the computation loop of the serial code.

Image loading
~~~~~~~~~~~~~

Do you need to load all your images at once? If not, you can sometimes
combine the 3 steps (load, analyze, save) into a single step discarding
the images once they have been loaded.

This can have dramatic effects on the program’s memory usage. If your
RAM fills up before the computation, it is almost guaranteed that your
code will run many times slower than it should simply because your
computer is moving memory back to your disk without warning you. Before
starting to paralleize your code, we encourage you try this strategy to
see if it helps your analysis.
