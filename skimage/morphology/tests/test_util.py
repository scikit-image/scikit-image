"""Test for the `_util`."""


import pytest
import numpy as np

from skimage.morphology import _util


class TestOffsetsToRaveledNeighbors:

    @pytest.mark.parametrize("shape", [
        (111,), (33, 44), (22, 55, 11), (6, 5, 4, 3)
    ])
    @pytest.mark.parametrize("order", ["C", "F"])
    def test_highest_connectivity(self, shape, order):
        """
        Check a scenarios where selem is always of the highest connectivity
        and all dimensions are > 2.
        """
        selem = np.ones((3,) * len(shape))
        center = (1,) * len(shape)
        offsets = _util._offsets_to_raveled_neighbors(
            shape, selem, center, order
        )

        # Assert only neighbors are present (no center)
        assert len(offsets) == selem.sum() - 1
        # Assert uniqueness
        assert len(set(offsets)) == offsets.size
        # selem of hightest connectivity is symmetric around center
        # -> offsets build pairs of with same value but different signs
        assert all(-x in offsets for x in offsets)

        # Construct image whose values are the Manhattan distance to its center
        image_center = tuple(s // 2 for s in shape)
        grid = np.meshgrid(
            *[np.abs(np.arange(s, dtype=np.intp) - c)
              for s, c in zip(shape, image_center)],
            indexing="ij"
        )
        image = np.sum(grid, axis=0)

        image_raveled = image.ravel(order)
        image_center_raveled = np.ravel_multi_index(
            image_center, shape, order=order
        )

        # Sample raveled image around its center
        samples = []
        for offset in offsets:
            index = image_center_raveled + offset
            samples.append(image_raveled[index])

        # Assert that center with value 0 wasn't selected
        assert np.min(samples) == 1
        # Assert that only neighbors where selected
        # (highest value == connectivity)
        assert np.max(samples) == len(shape)
        # Assert that nearest neighbors are selected first
        assert list(sorted(samples)) == samples
