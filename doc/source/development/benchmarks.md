# Benchmarks

While not mandatory for most pull requests, we ask that performance-related
PRs include a benchmark in order to clearly depict the use case that is being
optimized for.

In this section we will review how to setup the benchmarks,
and three commands `spin asv -- dev`, `spin asv -- run` and
`spin asv -- continuous`.

## Prerequisites

Begin by installing [airspeed velocity](https://asv.readthedocs.io/en/stable/)
in your development environment. Prior to installation, be sure to activate your
development environment, then if using `venv` you may install the requirement with:

```
source skimage-dev/bin/activate
pip install asv
```

If you are using conda, then the command:

```
conda activate skimage-dev
conda install asv
```

is more appropriate. Once installed, it is useful to run the command:

```
spin asv -- machine
```

To let airspeed velocity know more information about your machine.

## Writing a benchmark

To write benchmark, add a file in the `benchmarks` directory which contains a
a class with one `setup` method and at least one method prefixed with `time_`.

The `time_` method should only contain code you wish to benchmark.
Therefore it is useful to move everything that prepares the benchmark scenario
into the `setup` method. This function is called before calling a `time_`
method and its execution time is not factored into the benchmarks.

Take for example the `TransformSuite` benchmark:

```python
import numpy as np
from skimage import transform

class TransformSuite:
    """Benchmark for transform routines in scikit-image."""

    def setup(self):
        self.image = np.zeros((2000, 2000))
        idx = np.arange(500, 1500)
        self.image[idx[::-1], idx] = 255
        self.image[idx, idx] = 255

    def time_hough_line(self):
        result1, result2, result3 = transform.hough_line(self.image)
```

Here, the creation of the image is completed in the `setup` method, and not
included in the reported time of the benchmark.

It is also possible to benchmark features such as peak memory usage. To learn
more about the features, please refer to the official
[airspeed velocity documentation](https://asv.readthedocs.io/en/latest/writing_benchmarks.html).

Also, the benchmark files need to be importable when benchmarking old versions
of scikit-image. So if anything from scikit-image is imported at the top level,
it should be done as:

```python
try:
    from skimage import metrics
except ImportError:
    pass
```

The benchmarks themselves don't need any guarding against missing features,
only the top-level imports.

To allow tests of newer functions to be marked as "n/a" (not available)
rather than "failed" for older versions, the setup method itself can raise a
NotImplemented error. See the following example for the registration module:

```python
try:
    from skimage import registration
except ImportError:
    raise NotImplementedError("registration module not available")
```

## Testing the benchmarks locally

Prior to running the true benchmark, it is often worthwhile to test that the
code is free of typos. To do so, you may use the command:

```
spin asv -- dev -b TransformSuite
```

Where the `TransformSuite` above will be run once in your current environment
to test that everything is in order.

## Running your benchmark

The command above is fast, but doesn't test the performance of the code
adequately. To do that you may want to run the benchmark in your current
environment to see the performance of your change as you are developing new
features. The command `asv run -E existing` will specify that you wish to run
the benchmark in your existing environment. This will save a significant amount
of time since building scikit-image can be a time consuming task:

```
spin asv -- run -E existing -b TransformSuite
```

## Comparing results to main

Often, the goal of a PR is to compare the results of the modifications in terms
speed to a snapshot of the code that is in the main branch of the
`scikit-image` repository. The command `asv continuous` is of help here:

```
spin asv -- continuous main -b TransformSuite
```

This call will build out the environments specified in the `asv.conf.json`
file and compare the performance of the benchmark between your current commit
and the code in the main branch.

The output may look something like:

```
$ spin asv -- continuous main -b TransformSuite
· Creating environments
· Discovering benchmarks
·· Uninstalling from conda-py3.7-cython-numpy1.15-scipy
·· Installing 544c0fe3 <benchmark_docs> into conda-py3.7-cython-numpy1.15-scipy.
· Running 4 total benchmarks (2 commits * 2 environments * 1 benchmarks)
[  0.00%] · For scikit-image commit 37c764cb <benchmark_docs~1> (round 1/2):
[...]
[100.00%] ··· ...ansform.TransformSuite.time_hough_line           33.2±2ms

BENCHMARKS NOT SIGNIFICANTLY CHANGED.
```

In this case, the differences between HEAD and main are not significant
enough for airspeed velocity to report.

It is also possible to get a comparison of results for two specific revisions
for which benchmark results have previously been run via the `asv compare`
command:

```
spin asv -- compare v0.14.5 v0.17.2
```

Finally, one can also run ASV benchmarks only for a specific commit hash or
release tag by appending `^!` to the commit or tag name. For example to run
the skimage.filter module benchmarks on release v0.17.2:

```
spin asv -- run -b Filter v0.17.2^!
```
