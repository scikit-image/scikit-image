import numpy as np
import pytest
from numpy.testing import assert_array_equal

# If your function is defined in a separate module (e.g., patch_processing.py),
# you would import it like:
# from patch_processing import process_image_patches
# For this example, we assume the function is available in the namespace.


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


# ---------------------------------------------------------------------------
# Dummy processing functions for testing
# ---------------------------------------------------------------------------


def identity(patch):
    """Return the patch unchanged."""
    return patch


def invert_patch(patch):
    """Return the inverted patch assuming 8-bit values."""
    return 255 - patch


# ---------------------------------------------------------------------------
# Tests for 2D (grayscale) images
# ---------------------------------------------------------------------------


def test_identity_no_padding():
    """Test that the identity function returns the original image (no padding)."""
    image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    patch_size = 4
    result = process_image_patches(
        image, patch_size, padding=False, processing_function=identity
    )
    # With no padding, the output should be identical to the input
    assert_array_equal(result, image)


def test_identity_with_padding():
    """Test that identity processing with padding returns a padded image
    with the original region unchanged and padded regions equal to zero."""
    image = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
    patch_size = 4
    result = process_image_patches(
        image, patch_size, padding=True, processing_function=identity
    )
    # Compute expected padded dimensions
    new_h = ((image.shape[0] + patch_size - 1) // patch_size) * patch_size
    new_w = ((image.shape[1] + patch_size - 1) // patch_size) * patch_size
    assert result.shape == (new_h, new_w)
    # Original region should match
    assert_array_equal(result[: image.shape[0], : image.shape[1]], image)
    # Padded regions should be zeros (because np.pad was called with constant 0)
    if new_h > image.shape[0]:
        assert_array_equal(
            result[image.shape[0] :, :],
            np.zeros((new_h - image.shape[0], new_w), dtype=np.uint8),
        )
    if new_w > image.shape[1]:
        assert_array_equal(
            result[:, image.shape[1] :],
            np.zeros((new_h, new_w - image.shape[1]), dtype=np.uint8),
        )


def test_invert_no_padding():
    """Test that inverting an image (without padding) produces 255 - image."""
    image = np.random.randint(0, 256, (12, 12), dtype=np.uint8)
    patch_size = 5
    result = process_image_patches(
        image, patch_size, padding=False, processing_function=invert_patch
    )
    expected = 255 - image
    assert_array_equal(result, expected)


def test_invert_with_padding():
    """Test that inverting an image (with padding) produces the correct result
    on the original region and pads with the inversion of 0 (i.e., 255)."""
    image = np.random.randint(0, 256, (13, 17), dtype=np.uint8)
    patch_size = 8
    result = process_image_patches(
        image, patch_size, padding=True, processing_function=invert_patch
    )
    # Compute expected padded dimensions
    new_h = ((image.shape[0] + patch_size - 1) // patch_size) * patch_size
    new_w = ((image.shape[1] + patch_size - 1) // patch_size) * patch_size
    # Expected original region
    expected_original = 255 - image
    assert_array_equal(result[: image.shape[0], : image.shape[1]], expected_original)
    # Padded regions (processing 0 with invert_patch gives 255)
    if new_h > image.shape[0]:
        padded_bottom = result[image.shape[0] :, :]
        assert_array_equal(
            padded_bottom, np.full(padded_bottom.shape, 255, dtype=np.uint8)
        )
    if new_w > image.shape[1]:
        padded_right = result[:, image.shape[1] :]
        assert_array_equal(
            padded_right, np.full(padded_right.shape, 255, dtype=np.uint8)
        )


# ---------------------------------------------------------------------------
# Tests for 3D (color) images
# ---------------------------------------------------------------------------


def test_color_image_no_padding():
    """Test that inverting a color image (without padding) works per channel."""
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    patch_size = 4
    result = process_image_patches(
        image, patch_size, padding=False, processing_function=invert_patch
    )
    expected = 255 - image
    assert_array_equal(result, expected)


def test_color_image_with_padding():
    """Test that inverting a color image (with padding) works per channel and pads correctly."""
    image = np.random.randint(0, 256, (11, 13, 3), dtype=np.uint8)
    patch_size = 5
    result = process_image_patches(
        image, patch_size, padding=True, processing_function=invert_patch
    )
    new_h = ((image.shape[0] + patch_size - 1) // patch_size) * patch_size
    new_w = ((image.shape[1] + patch_size - 1) // patch_size) * patch_size
    expected_original = 255 - image
    assert_array_equal(result[: image.shape[0], : image.shape[1], :], expected_original)
    # Check padded regions are correctly processed (inversion of 0 gives 255)
    if new_h > image.shape[0]:
        padded_bottom = result[image.shape[0] :, :, :]
        assert_array_equal(
            padded_bottom, np.full(padded_bottom.shape, 255, dtype=np.uint8)
        )
    if new_w > image.shape[1]:
        padded_right = result[:, image.shape[1] :, :]
        assert_array_equal(
            padded_right, np.full(padded_right.shape, 255, dtype=np.uint8)
        )


# ---------------------------------------------------------------------------
# Additional tests can be added here to cover edge cases
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__])
