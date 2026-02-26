(skimage2-migration)=

# Migration guide: from skimage to skimage2

:::{hint}
This document is a work in progress and still subject to change.
:::

scikit-image is preparing to release version 2.0 as a new package: `skimage2`.
Alongside skimage2, we will release version 1.0.0. Versions 1.x will be using the current API.
Versions 1.1.x will throw a `FutureWarning` upon import, as a means to notify users that
they should either upgrade to skimage2 or pin to version 1.0.x.

We have undertaken this to make some long-outstanding, backward-incomptible changes to the scikit-image API.
Most changes were difficult or impossible to make using deprecations alone.
To honor the Hinsen principle (that is, never change results silently unless to fix a bug), we introduce a new package, which gives users an explicit way of upgrading.
Users also have the option to use the two versions side-by-side while they do so.

You can find a more detailed description of our motivation and discussion leading up to this in {doc}`SKIP 4 <../skips/4-transition-to-v2>`.

(enable-skimage2-warnings)=

## Enable skimage2-related warnings

Even before skimage2 is released, you may enable skimage2-related warnings to prepare for code changes early on.
Run the following [warnings filter](https://docs.python.org/3/library/warnings.html#the-warnings-filter) before you use scikit-image in your code:

```python
import warnings
import skimage as ski
warnings.filterwarnings(action="default", category=ski.util.PendingSkimage2Change)
```

This will raise a warning in code that needs to be modified to continue functioning with the new, skimage2 API.

## Updating existing code

When switching to the new `skimage2` namespace, some code will need to be updated to continue working the way it did before.

:::{note}
For a while, you will be able to use `skimage` and `skimage2` (the 2.0 API) side-by-side, to facilitate porting.
The new API may, for the same function call, return different results—e.g., because of a change in a keyword argument default value.
By importing functionality from `skimage2`, you explicitly opt in to the new behavior.
:::

### `skimage.data.binary_blobs`

This function is replaced by `skimage2.data.binary_blobs` with a new signature.
The optional parameters `length` and `n_dim` are replaced with a new required parameter `shape`, which allows generating non-square outputs.
Optional parameter `blob_size_fraction` is replaced with required parameter `blob_size`, whose behavior is independent of the output image size.
The default value of `boundary_mode` is changed from `'nearest'` to `'wrap'`.

To keep the old (`skimage`, v1.x) behavior, use

```python
import skimage2 as ski2

ski2.data.binary_blobs(
    shape=(length,) * n_dim,
    blob_size=blob_size_fraction * length,
    boundary_mode='nearest',
)
```

with `length`, `n_dim`, and `blob_size_fraction` containing values used with the old signature.
Other parameters -- including `boundary_mode` if you already set it explicitly -- can be left unchanged.

### `skimage.feature.peak_local_max`

This function is replaced by `skimage2.feature.peak_local_max` with new behavior:

- Parameter `p_norm` defaults to 2 (Euclidean distance), was `numpy.inf` (Chebyshev distance)
- Parameter `exclude_border` defaults to 1, was `True`
- Parameter `exclude_border` no longer accepts `False` and `True`, pass 0 instead of `False`, or `min_distance` instead of `True`
- Parameters after `image` are keyword-only

To keep the old behavior when switching to `skimage2`, update your call according to the following cases:

:::{list-table}
:header-rows: 1

- - In `skimage`
  - In `skimage2`

- - `exclude_border` not passed (default)
  - Assign it the same value as `min_distance` which may be its default value `1`.

- - `exclude_border=True`
  - Same as above in the default case.

- - `exclude_border=False`
  - Use `min_distance=0`.

- - `exclude_border=<int>`
  - No change necessary.

- - `p_norm` not passed (default)
  - Pass the previous default explicitly with `p_norm=numpy.inf`.

- - `p_norm=<float>`
  - No change necessary.

:::

Other keyword parameters can be left unchanged.

Examples:

```python
ski.morphology.peak_local_max(image)
ski2.morphology.peak_local_max(image, exclude_border=1, p_norm=np.inf)

ski.morphology.peak_local_max(image, min_distance=10)
ski2.morphology.peak_local_max(
    image, min_distance=10, exclude_border=10, p_norm=np.inf
)
```

### Grayscale morphological operators in `skimage.morphology`

The functions

- `skimage.morphology.erosion`
- `skimage.morphology.dilation`
- `skimage.morphology.opening`
- `skimage.morphology.closing`
- `skimage.morphology.white_tophat`
- `skimage.morphology.black_tophat`

are deprecated in favor of counterparts in `skimage2.morphology` with new behavior:

- All functions now default to `mode='ignore'` (was `mode='reflect'`).
- Additionally, `skimage.morphology.dilation`, `skimage.morphology.closing`, and `skimage.morphology.black_tophat` now also mirror the footprint (invert its order in each dimension).
  This is does only impact behavior for asymmetric/eccentric footprints.

:::{admonition} Background for changes
:class: note dropdown

The new behavior ensures that `closing` and `opening` (the composition of `erosion` and `dilation`) behave _extensive_ and _anti-extensive_ by default.
It also aligns the behavior for asymmetric/eccentric footprints with SciPy's `scipy.ndimage.grey_*` functions.

Refer to [gh-6665](https://github.com/scikit-image/scikit-image/issues/6665), [gh-6676](https://github.com/scikit-image/scikit-image/issues/6676), [gh-8046](https://github.com/scikit-image/scikit-image/pull/8046,) and [gh-8060](https://github.com/scikit-image/scikit-image/pull/8060) for more details.
:::

To keep the old (`skimage`, v1.x) behavior:

- Set `mode='reflect` explicitly.
  If you set it explicitly before, the behavior is unchanged.

- If you use an asymmetric `footprint` with `dilation`, `erosion` or `black_tophat`, modify it like this before passing it to the `skimage2.morphology` counterpart:
  ```python
  footprint = ski2.morphology.pad_footprint(footprint, pad_end=False)
  footprint = ski2.morphology.mirror_footprint(footprint)
  ```

For example:

```python
ski.morphology.dilation(image, footprint=eccentric)

# Replace above with
eccentric = ski2.morphology.pad_footprint(eccentric, pad_end=False)
eccentric = ski2.morphology.mirror_footprint(eccentric)
ski2.morphology.dilation(image, footprint=eccentric, mode="reflect")
```

## Deprecations prior to skimage2

We have already introduced a number of changes and deprecations to our API.
These are part of the API cleanup for skimage2 but are not breaking.
You will simply notice these as the classical deprecation warnings that you are already used to.
We list them here, because updating your code to the new API will make it easier to transition to skimage2.

_To be defined._
