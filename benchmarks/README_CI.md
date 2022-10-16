# Benchmark CI

<!-- Author: @jaimergp -->
<!-- Last updated: 2021.07.06 -->
<!-- Describes the work done as part of https://github.com/scikit-image/scikit-image/pull/5424 -->

## How it works

The `asv` suite can be run for any PR on GitHub Actions (check workflow `.github/workflows/benchmarks.yml`) by adding a `run-benchmark` label to said PR. This will trigger a job that will run the benchmarking suite for the current PR head (merged commit) against the PR base (usually `main`).

We use `asv continuous` to run the job, which runs a relative performance measurement. This means that there's no state to be saved and that regressions are only caught in terms of performance ratio (absolute numbers are available but they are not useful since we do not use stable hardware over time). `asv continuous` will:

- Compile `scikit-image` for _both_ commits. We use `ccache` to speed up the process, and `mamba` is used to create the build environments.
- Run the benchmark suite for both commits, _twice_ (since `processes=2` by default).
- Generate a report table with performance ratios:
  - `ratio=1.0` -> performance didn't change.
  - `ratio<1.0` -> PR made it slower.
  - `ratio>1.0` -> PR made it faster.

Due to the sensitivity of the test, we cannot guarantee that false positives are not produced. In practice, values between `(0.7, 1.5)` are to be considered part of the measurement noise. When in doubt, running the benchmark suite one more time will provide more information about the test being a false positive or not.

## Running the benchmarks on GitHub Actions

1. On a PR, add the label `run-benchmark`.
2. The CI job will be started. Checks will appear in the usual dashboard panel above the comment box.
3. If more commits are added, the label checks will be grouped with the last commit checks _before_ you added the label.
4. Alternatively, you can always go to the `Actions` tab in the repo and [filter for `workflow:Benchmark`](https://github.com/scikit-image/scikit-image/actions?query=workflow%3ABenchmark). Your username will be assigned to the `actor` field, so you can also filter the results with that if you need it.

## The artifacts

The CI job will also generate an artifact. This is the `.asv/results` directory compressed in a zip file. Its contents include:

- `fv-xxxxx-xx/`. A directory for the machine that ran the suite. It contains three files:
  - `<baseline>.json`, `<contender>.json`: the benchmark results for each commit, with stats.
  - `machine.json`: details about the hardware.
- `benchmarks.json`: metadata about the current benchmark suite.
- `benchmarks.log`: the CI logs for this run.
- This README.

## Re-running the analysis

Although the CI logs should be enough to get an idea of what happened (check the table at the end), one can use `asv` to run the analysis routines again.

1. Uncompress the artifact contents in the repo, under `.asv/results`. This is, you should see `.asv/results/benchmarks.log`, not `.asv/results/something_else/benchmarks.log`. Write down the machine directory name for later.
2. Run `asv show` to see your available results. You will see something like this:

```
$> asv show

Commits with results:

Machine    : Jaimes-MBP
Environment: conda-py3.9-cython-numpy1.20-scipy

    00875e67

Machine    : fv-az95-499
Environment: conda-py3.7-cython-numpy1.17-pooch-scipy

    8db28f02
    3a305096
```

3. We are interested in the commits for `fv-az95-499` (the CI machine for this run). We can compare them with `asv compare` and some extra options. `--sort ratio` will show largest ratios first, instead of alphabetical order. `--split` will produce three tables: improved, worsened, no changes. `--factor 1.5` tells `asv` to only complain if deviations are above a 1.5 ratio. `-m` is used to indicate the machine ID (use the one you wrote down in step 1). Finally, specify your commit hashes: baseline first, then contender!

```
$> asv compare --sort ratio --split --factor 1.5 -m fv-az95-499 8db28f02 3a305096

Benchmarks that have stayed the same:

       before           after         ratio
     [8db28f02]       [3a305096]
     <ci-benchmark-check~9^2>
              n/a              n/a      n/a  benchmark_restoration.RollingBall.time_rollingball_ndim
      1.23±0.04ms       1.37±0.1ms     1.12  benchmark_transform_warp.WarpSuite.time_to_float64(<class 'numpy.float64'>, 128, 3)
       5.07±0.1μs       5.59±0.4μs     1.10  benchmark_transform_warp.ResizeLocalMeanSuite.time_resize_local_mean(<class 'numpy.float32'>, (192, 192, 192), (192, 192, 192))
      1.23±0.02ms       1.33±0.1ms     1.08  benchmark_transform_warp.WarpSuite.time_same_type(<class 'numpy.float32'>, 128, 3)
       9.45±0.2ms       10.1±0.5ms     1.07  benchmark_rank.Rank3DSuite.time_3d_filters('majority', (32, 32, 32))
       23.0±0.9ms         24.6±1ms     1.07  benchmark_interpolation.InterpolationResize.time_resize((80, 80, 80), 0, 'symmetric', <class 'numpy.float64'>, True)
         38.7±1ms         41.1±1ms     1.06  benchmark_transform_warp.ResizeLocalMeanSuite.time_resize_local_mean(<class 'numpy.float32'>, (2048, 2048), (192, 192, 192))
       4.97±0.2μs       5.24±0.2μs     1.05  benchmark_transform_warp.ResizeLocalMeanSuite.time_resize_local_mean(<class 'numpy.float32'>, (2048, 2048), (2048, 2048))
       4.21±0.2ms       4.42±0.3ms     1.05  benchmark_rank.Rank3DSuite.time_3d_filters('gradient', (32, 32, 32))

...
```

If you want more details on a specific test, you can use `asv show`. Use `-b pattern` to filter which tests to show, and then specify a commit hash to inspect:

```
$> asv show -b time_to_float64 8db28f02

Commit: 8db28f02 <ci-benchmark-check~9^2>

benchmark_transform_warp.WarpSuite.time_to_float64 [fv-az95-499/conda-py3.7-cython-numpy1.17-pooch-scipy]
  ok
  =============== ============= ========== ============= ========== ============ ========== ============ ========== ============
  --                                                                N / order
  --------------- --------------------------------------------------------------------------------------------------------------
      dtype_in       128 / 0     128 / 1      128 / 3     1024 / 0    1024 / 1    1024 / 3    4096 / 0    4096 / 1    4096 / 3
  =============== ============= ========== ============= ========== ============ ========== ============ ========== ============
    numpy.uint8    2.56±0.09ms   523±30μs   1.28±0.05ms   130±3ms     28.7±2ms    81.9±3ms   2.42±0.01s   659±5ms    1.48±0.01s
    numpy.uint16   2.48±0.03ms   530±10μs   1.28±0.02ms   130±1ms    30.4±0.7ms   81.1±2ms    2.44±0s     653±3ms    1.47±0.02s
   numpy.float32    2.59±0.1ms   518±20μs   1.27±0.01ms   127±3ms     26.6±1ms    74.8±2ms   2.50±0.01s   546±10ms   1.33±0.02s
   numpy.float64   2.48±0.04ms   513±50μs   1.23±0.04ms   134±3ms     30.7±2ms    85.4±2ms   2.55±0.01s   632±4ms    1.45±0.01s
  =============== ============= ========== ============= ========== ============ ========== ============ ========== ============
  started: 2021-07-06 06:14:36, duration: 1.99m
```

## Other details

### Skipping slow or demanding tests

To minimize the time required to run the full suite, we trimmed the parameter matrix in some cases and, in others, directly skipped tests that ran for too long or require too much memory. Unlike `pytest`, `asv` does not have a notion of marks. However, you can `raise NotImplementedError` in the setup step to skip a test. In that vein, a new private function is defined at `benchmarks.__init__`: `_skip_slow`. This will check if the `ASV_SKIP_SLOW` environment variable has been defined. If set to `1`, it will raise `NotImplementedError` and skip the test. To implement this behavior in other tests, you can add the following attribute:

```python
from . import _skip_slow  # this function is defined in benchmarks.__init__

def time_something_slow():
    pass

time_something.setup = _skip_slow
```
