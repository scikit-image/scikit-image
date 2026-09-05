# Benchmark CI

<!-- Author: @jaimergp -->
<!-- Last updated: 2026.08.02 -->
<!-- Describes the work done as part of https://github.com/scikit-image/scikit-image/pull/5424 -->

## How it works

The `asv` suite runs automatically on every PR, scoped to whichever benchmark module(s) cover the `skimage` subpackage(s) touched (see `.github/scripts/resolve-benchmark-params.py`). A PR touching only `skimage/restoration/` runs just `benchmark_restoration.py`; a PR touching no mapped subpackage runs none. A label whose name contains `benchmark` (e.g. `run-benchmark`) overrides this and runs the full suite, for changes to shared code that could affect benchmarks outside the touched subpackage.

The suite also runs nightly at 07:00 UTC against `main` (see "Full nightly runs" below), and can be triggered manually from the `workflow_dispatch` entry point on the `Actions` tab. `workflow_dispatch` offers a `baseline` choice: `parent-commit` (default) or `previous-nightly`, falling back to the immediate parent if no nightly run has succeeded yet. Merges to `main` don't trigger a run on their own; the nightly run covers that ground without the CI cost of running on every merge. A failing nightly or dispatched run on `main`, including a detected regression, opens or updates a `CI failure`-labeled issue, the same convention the repo's other main-branch checks use.

`asv continuous` runs a relative performance measurement: no state is saved, and a regression is only a ratio, since we don't have stable hardware over time to make absolute numbers meaningful.

Before `asv` runs, the baseline and contender commits build as wheels in two parallel jobs (`build-baseline`/`build-contender` in `.github/workflows/benchmarks.yaml`), reusing the repo's `_build_linux_for_python_x.yaml` build workflow instead of compiling sequentially inside `asv`. `asv`'s `build_command` copies the matching prebuilt wheel into place. `asv continuous` then:

- Installs the prebuilt wheel for each commit.
- Runs the suite for both commits, twice per commit (`processes=2`), trading time for statistical robustness (see `ASV_FACTOR`/`ASV_PROCESSES` below).
- Reports a performance ratio per benchmark: 1.0 unchanged, below 1.0 slower, above 1.0 faster.

Values within roughly `(0.91, 1.1)` are measurement noise (the `ASV_FACTOR` cutoff), not a reliable signal. When in doubt, rerun the suite.

## How the run is parameterized

`resolve-benchmark-params.py` decides everything about a run from the trigger event, reading the event payload from `GITHUB_EVENT_PATH` instead of the workflow's `env:` block. It publishes two things:

- Three job outputs the earlier jobs need: `should_run`, `python_version` (from `asv.conf.json`, so it can't drift), and `baseline_sha`.
- `benchmark-params.json`, an artifact holding the settings only the benchmark job needs: `ASV_FACTOR`, `ASV_PROCESSES`, `ASV_SKIP_SLOW`, `ASV_FULL_PARAMS`, `BASELINE_SHA`, `BASELINE_LABEL`, `CONTENDER_LABEL`, `BENCH_FILTER`. `prepare-benchmarks.sh` copies these into `$GITHUB_ENV` for `run-benchmarks.sh` and the benchmark processes.

Its keys are the environment variable names, so adding a parameter is one entry in `resolve-benchmark-params.py`, not an edit to the workflow in several places.

`MODULE_MAP` in the same script maps each subpackage — in any of the three package trees (`src/skimage/`, `src/_skimage2/`, `src/skimage2/`) — to the benchmark modules covering it, and is what scopes a pull request check. The diff is taken from the merge base, so a PR that has fallen behind `main` is scoped by its own changes, not by what `main` merged since it branched.

The benchmark job splits along that seam: `prepare-benchmarks.sh` handles everything before measurement (parameters, thread pinning, pointing `build_command` at the prebuilt wheels, `asv machine`), and `run-benchmarks.sh` is just the `asv continuous` call and its pass/fail check.

## Running the benchmarks on GitHub Actions

1. Opening or updating a PR that touches a mapped subpackage runs that module's benchmarks automatically. Checks appear above the comment box.
2. A label whose name contains `benchmark` (e.g. `run-benchmark`) forces the full suite, and stays in effect for that PR's later runs too.
3. Filter the `Actions` tab for [`workflow:Benchmark`](https://github.com/scikit-image/scikit-image/actions?query=workflow%3ABenchmark); your username is the `actor`.

## Full nightly runs

Every night at 07:00 UTC, a scheduled run compares `main`'s tip against the commit the last successful nightly compared, not just its immediate parent, so a busy day of merges is covered cumulatively. It runs the complete suite: `ASV_SKIP_SLOW=0` includes the slow benchmarks, `ASV_FULL_PARAMS=1` uses each benchmark's full parameter matrix instead of the reduced one PR checks use. This is the only check unscoped by path or trimmed by time, so it's where a regression a fast PR check missed, including one from a merge to `main` itself, would surface.

## The artifacts

The job also uploads `.asv/results` compressed into a zip. Its contents include:

- `fv-xxxxx-xx/`. A directory for the machine that ran the suite. It contains three files:
  - `<baseline>.json`, `<contender>.json`: the benchmark results for each commit, with stats.
  - `machine.json`: details about the hardware.
- `benchmarks.json`: metadata about the current benchmark suite.
- `benchmarks.log`: the CI logs for this run.
- This README.

## Re-running the analysis

Although the CI logs usually show enough to see what happened (check the table at the end), `asv` can rerun the analysis.

1. Uncompress the artifact contents in the repo, under `.asv/results` (that is, `.asv/results/benchmarks.log`, not `.asv/results/something_else/benchmarks.log`). Write down the machine directory name for later.
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

3. Compare the commits for `fv-az95-499` (the CI machine for this run) with `asv compare` and some extra options. `--sort ratio` shows the largest ratios first instead of alphabetical order. `--split` produces three tables: improved, worsened, no changes. `--factor 1.1` only complains about deviations above a 1.1 ratio. `-m` gives the machine ID (the one you wrote down in step 1). Give your commit hashes baseline first, then contender.

```
$> asv compare --sort ratio --split --factor 1.1 -m fv-az95-499 8db28f02 3a305096

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

For more details on a specific test, use `asv show`. Filter which tests to show with `-b pattern`, then specify a commit hash to inspect:

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

To keep the full suite fast, we trimmed some parameter matrices and skipped tests that run too long or need too much memory. Unlike `pytest`, `asv` has no concept of marks; instead, raise `NotImplementedError` in a setup step. `benchmarks/__init__.py` ships `_skip_slow`, which does that when `ASV_SKIP_SLOW` is set to `1`. Attach it as a setup method or attribute:

```python
from . import _skip_slow

def time_something_slow():
    pass

time_something.setup = _skip_slow
```
