def seam_carve(*args, **kwargs):
    """Seam carving has been removed because it is a patented algorithm.

    See https://github.com/scikit-image/scikit-image/issues/3646
    """

    message = (
    """Seam carving has been removed because it is a patented algorithm.

    See https://github.com/scikit-image/scikit-image/issues/3646

    This message will be removed completely in scikit-image version 0.16.
    """
    )

    raise NotImplementedError(message)
