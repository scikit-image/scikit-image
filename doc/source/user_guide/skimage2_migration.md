(skimage2_migration)=

# Transition to skimage2

:::{hint}
This document is a work in progress and still subject to change.
:::

scikit-image is preparing to release version 2.0 under the new namespace `skimage2`.
This will affect both its import name and its package name on PyPI and elsewhere.
Together with skimage2, we will release version 1.0.0 and 1.0.1 with the old API.
Version 1.0.1 will emit a notification (FutureWarning) on import, telling users to either upgrade to skimage2 or pin to version 1.0.0.

We do this to clean up long-standing issue without our API.
Some of these issues involve changes to behavior that are difficult to address with conventional deprecations.
Since we don't want to change behavior silently, we will introduce the new namespace which gives users an explicit way to upgrade to new behavior.

You can find a more detailed description of our motivation and discussion leading up to this in {doc}`SKIP 4 <../skips/4-transition-to-v2>`.

## Changes in skimage2

_To be defined._

## Changes pre skimage2

We have already introduced a number of changes and deprecations to our API.
These will only be completed in during the transition to skimage2 and will continue to work in all versions pre-skimage2.
However, updating your code to the new API will make it easier to transition to the skimage2 API.

_To be defined._
