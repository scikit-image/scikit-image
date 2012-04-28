Benchmarking using `vbench`
===========================

This code runs performance benchmarks using vbench_.

To build these benchmarks, just run `make` in this directory. This will run the
benchmarks defined in the `./suite/` directory and create a `source` directory,
which details the benchmarking results using reStructuredText files.

For testing, you'll want to replace the start date in `settings.py` with
a recent date. (Even so, it still takes a long time since vbench_ will clean
and rebuild skimage for each commit.) Note that vbench_ saves previous
benchmarks so only commits added since the last benchmarking are run.


.. _vbench: https://github.com/pydata/vbench


Build Requirements
------------------

The main requirement is vbench_, but that in turn requires the following:

* `pandas <http://pandas.pydata.org/>`__
* `matplotlib <http://matplotlib.sf.net>`__
* `sqlalchemy <http://www.sqlalchemy.org/>`__
* `pytz <http://pytz.sourceforge.net/>`__


Adding benchmarks
-----------------

Currently, only a few example benchmarks are defined. You can help by adding
benchmarks to `./suite/`, which should look similar to the following:

.. code-block:: python

   from vbench.api import Benchmark

   setup = """
   from skimage import filter
   from skimage import data
   """

   sobel = "filter.sobel(data.camera())"
   filter_sobel = Benchmark(sobel, setup, name="sobel_test")

.. note::

   Each benchmark has to be saved to a unique variable (a list won't work).

