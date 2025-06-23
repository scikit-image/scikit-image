(skimage2-migration)=

# Migration guide: from skimage to skimage2

:::{hint}
This document is a work in progress and still subject to change.
:::

scikit-image is preparing to release version 2.0 as a new package: `skimage2`.
Alongside skimage2, we will release version 1.0.0. Versions 1.x will be using the current API.
Versions 1.1.x will throw a FutureWarning upon import, as a means to notify users that
they should either upgrade to skimage2 or pin to version 1.0.x.

We have undertaken this to make some long-outstanding, backward-incomptible changes to the scikit-image API.
Most changes were difficult or impossible to make using deprecations alone.
To honor the Hinsen principle—that is, never change results silently, unless to fix a bug—we introduce a new package, which gives users an explicit way of upgrading.
Users also have the option to use the two versions side-by-side while they do so.

You can find a more detailed description of our motivation and discussion leading up to this in {doc}`SKIP 4 <../skips/4-transition-to-v2>`.

(enable-skimage2-warnings)=

## Enable skimage2 related warnings

Even before the release of skimage2, you can enable early warnings to prepare your code.
Insert the following [warnings filter](https://docs.python.org/3/library/warnings.html#the-warnings-filter) at the start of your code:

```python
import warnings
import skimage as ski
warnings.filterwarnings(action="default", category=ski.util.PendingSkimage2Change)
```

This will raise warnings for all code that uses scikit-image's API in a way that will silently break or change behavior starting with skimage2.

## Breaking changes in skimage2

These changes require an intervention in order to keep the code working as before after switching to version 2.0 and the new namespace `skimage2`.

(threshold-blob-funcs)=

### Deprecate range sensitive threshold in `feature.blob_*` functions

**Affects:**

- `skimage.feature.blob_dog`
- `skimage.feature.blob_doh`
- `skimage.feature.blob_log`

**Description:**

Starting with version 0.26, the value range of the parameter `image` is always preserved.
When `image` had an integer dtype, its value range was scaled (`uint -> [0, 1]`, `int -> [-1, 1]`).
This affected the now deprecated `threshold` parameter, whose absolute value would compare differently to a scaled or untouched value range.
With this deprecation, the new parameter `threshold_abs` is preferred instead.
`threshold_abs` will always be compared against a range preserving version of `image` and behave consistent regardless of `image.dtype`.

Starting with version 2.0 and the new `skimage2` namespace, the default values of these functions are updated too.
The default of `threshold` will be set to `None` and `threshold_rel` will be set to the old value of `threshold`.
Calls to these functions that rely on default values may change behavior, so we recommend setting `threshold_abs` and `threshold_rel` explicitly.

**How to update:**

We recommend setting both parameters `threshold_abs` and `threshold_rel` explicitly if you want to preserve old behavior.
You can also [enable skimage2 warnings]{#enable-skimage2-warnings} to that should instruct you how to use the new API.

- Remove all uses of the deprecated parameter `threshold`.

- If `image` is of integer dtype, adjust the old value of `threshold`

  ```python
  threshold_abs = threshold * scaling_factor
  ```

  and pass it to `threshold_abs`.
  For `blob_dog` and `blob_log`, use `scaling_factor = np.iinfo(image.dtype).max`.
  For `blob_doh`, you need the squared version: `scaling_factor = np.iinfo(image.dtype).max ** 2`.
  If `image` is of floating dtype, simply set `threshold_abs` to the old value of `threshold`.

- Set `threshold_rel=None` explicitly if you are not passing an explicit value already.
  E.g., `blob_doh(image)` becomes `blob_doh(image, threshold_rel=None)` but `blob_doh(image, threshold_rel=0.6)` does not need to be modified.

## Changes un-related to skimage2

We have already introduced a number of changes and deprecations to our API.
These are part of the API cleanup for skimage2 but are not breaking.
You will simply notice these as the classical deprecation warnings that you are already used to.
We list them here, because updating your code to the new API will make it easier to transition to the skimage2 API.

_To be defined._
