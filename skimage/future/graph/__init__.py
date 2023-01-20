# Remove this package in the release after v0.20

raise ModuleNotFoundError(
    "The `skimage.future.graph` submodule was moved to `skimage.graph` in "
    "v0.20. `ncut` was removed in favor of the identical function "
    "`cut_normalized`. Please update your import paths accordingly."
)
