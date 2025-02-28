import numpy as np


def process_image_patches(image, patch_size, padding, processing_function):
    """
    Apply a processing function to sliding square patches of an image.

    The function divides the input image into patches of size ``patch_size x patch_size``
    and applies the given processing function to each patch. If ``padding`` is True,
    the image is padded with zeros (black) so that its dimensions become multiples of
    ``patch_size``. Otherwise, patches on the borders may have smaller dimensions.

    Parameters
    ----------
    image : ndarray
        Input image. It can be a 2D array (grayscale) or a 3D array (e.g., color).
    patch_size : int
        Size of the square patch (e.g., 254, 512, etc.).
    padding : bool
        If True, pad the image with zeros so that its dimensions are exact multiples of
        ``patch_size``.
    processing_function : callable
        A function that takes a patch (ndarray) as input and returns a processed patch.
        The returned patch should have the same dimensions as the input patch.

    Returns
    -------
    output : ndarray
        The image with the processed patches. If ``padding`` is True, the output image
        will include the padded regions; otherwise, it retains the original dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> def invert_patch(patch):
    ...     # Example processing function: invert pixel values (for images in [0, 255])
    ...     return 255 - patch
    >>> # Create a random grayscale image with values in [0, 255]
    >>> image = np.random.randint(0, 256, (500, 750), dtype=np.uint8)
    >>> processed = process_image_patches(image, patch_size=256, padding=True,
    ...                                    processing_function=invert_patch)
    """
    # Retrieve original image dimensions
    h, w = image.shape[:2]

    # If padding is enabled, calculate the necessary padding for both dimensions
    if padding:
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size

        if image.ndim == 3:
            image_padded = np.pad(
                image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0,
            )
        else:
            image_padded = np.pad(
                image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0
            )
    else:
        image_padded = image

    new_h, new_w = image_padded.shape[:2]
    # Create an output array to store the processed image
    output = np.copy(image_padded)

    # Iterate over the image in steps of patch_size to process each patch
    for i in range(0, new_h, patch_size):
        for j in range(0, new_w, patch_size):
            patch = image_padded[i : i + patch_size, j : j + patch_size]
            processed_patch = processing_function(patch)
            output[i : i + patch_size, j : j + patch_size] = processed_patch

    return output
