(skimage2_migration)=

# Migration guide: skimage 0.x to skimage2

:::{hint}
This document is a work in progress and still subject to change.
:::

scikit-image is preparing to release version 2.0 as a new package: `skimage2`.
Alongside skimage2, we will release version 1.0.0, the last using the current API.
Version 1.0.1 will emit a notification (FutureWarning) on import, telling users to either upgrade to skimage2 or pin to version 1.0.0.

We do this to make some long-outstanding, backward-incomptible changes to our API.
Most changes were difficult or impossible to make using deprecations alone.
To honor the Hinsen principle—that is, we never change results silently, unless it is to fix a bug—we introduce a new package, which gives users an explicit way of upgrading.
Users also have the option to use the two versions side-by-side while they do so.

You can find a more detailed description of our motivation and discussion leading up to this in {doc}`SKIP 4 <../skips/4-transition-to-v2>`.

## Changes in skimage2

_To be defined._

## Changes pre skimage2

We have already introduced a number of changes and deprecations to our API.
These will only be completed in during the transition to skimage2 and will continue to work in all versions pre-skimage2.
However, updating your code to the new API will make it easier to transition to the skimage2 API.

_To be defined._
