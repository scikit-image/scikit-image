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

(threshold-blob-funcs)=

### Range-preserving thresholding in `feature.blob_*` functions

**Affects:**

- {func}`skimage.feature.blob_dog`
- {func}`skimage.feature.blob_doh`
- {func}`skimage.feature.blob_log`

**Description:**

Starting with **version 0.26**, we no longer automatically rescale input images to fit the interval `[0, 1]` (for unsigned integers) or `[-1, 1]` (for signed integers).
This affects the now deprecated `threshold` parameter, whose value from before would no longer be correct, now that we avoid scaling the input images.
The value of `threshold` would need to vary, depending on the image dtype.
Instead, the `threshold_rel` parameter is now the preferred way to set the threshold in a data-type agnostic way.
Setting the threshold to a fixed value is still possible, using the newly-introduced `threshold_abs` keyword.
`threshold_abs` will always be compared against a range-preserving version of `image` and behave consistently regardless of `image.dtype`.

Starting with **version 2.0** and the new `skimage2` namespace, the default values of these parameters are updated too.
The default of `threshold` will be set to `None` and `threshold_rel` will be set to the old value of `threshold`.
This will make the defaults useful to a wider set of input images and should approximate the old behavior for many unsigned integer images.

In **version 2.2** (or later), the old `threshold` parameter will be removed completely.

**How to update:**

To discover all calls that need to be updated, [enable skimage2-related warnings](#enable-skimage2-warnings).

Since the old dtype-specific behavior might have been unexpected, we suggest setting `threshold_abs=None` and finding an appropriate relative value for `threshold_rel`.

However, if you want to maintain the exact old behavior across all releases:

- Set `threshold_rel=None` (to ignore the new default in version 2.0).
- Set `threshold_abs` to the old value of `threshold`.
- Wrap the input `image` parameter with {func}`skimage.util.img_as_float` before passing it to the function.
  This will conditionally rescale the input range of the `image` as before depending on its dtype.

## Deprecations prior to skimage2

We have already introduced a number of changes and deprecations to our API.
These are part of the API cleanup for skimage2 but are not breaking.
You will simply notice these as the classical deprecation warnings that you are already used to.
We list them here, because updating your code to the new API will make it easier to transition to skimage2.

_To be defined._
