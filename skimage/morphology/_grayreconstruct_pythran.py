"""
`reconstruction_loop` originally part of CellProfiler,
code licensed under both GPL and BSD licenses.

Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np

# pythran export reconstruction_loop(uint32[] or uint32[::],
#         int32[],
#         int32[],
#         int32[],
#         int64, int)
#
# pythran export reconstruction_loop(uint64[] or uint64[::],
#         int32[],
#         int32[],
#         int32[],
#         int64, int)


def reconstruction_loop(aranks, aprev, anext, astrides,
                        current_idx, image_stride):
    """The inner loop for reconstruction.

    This algorithm uses the rank-order of pixels. If low intensity pixels have
    a low rank and high intensity pixels have a high rank, then this loop
    performs reconstruction by dilation. If this ranking is reversed, the
    result is reconstruction by erosion.

    For each pixel in the seed image, check its neighbors. If its neighbor's
    rank is below that of the current pixel, replace the neighbor's rank with
    the rank of the current pixel. This dilation is limited by the mask, i.e.
    the rank at each pixel cannot exceed the mask as that pixel.

    Parameters
    ----------
    aranks : array
        The rank order of the flattened seed and mask images.
    aprev, anext: arrays
        Indices of previous and next pixels in rank sorted order.
    astrides : array
        Strides to neighbors of the current pixel.
    current_idx : int
        Index of highest-ranked pixel used as starting point in loop.
    image_stride : int
        Stride between seed image and mask image in `aranks`.
    """
    nstrides = astrides.shape[0]
    assert image_stride >= 0

    while current_idx >= 0:
        if current_idx < image_stride:
            current_rank = aranks[current_idx]
            if current_rank == 0:
                break
            for i in range(nstrides):
                neighbor_idx = current_idx + astrides[i]
                assert neighbor_idx >= 0
                neighbor_rank = aranks[neighbor_idx]
                # Only propagate neighbors ranked below the current rank
                if neighbor_rank < current_rank:
                    mask_rank = aranks[neighbor_idx + image_stride]
                    # Only propagate neighbors ranked below the mask rank
                    if neighbor_rank < mask_rank:
                        # Raise the neighbor to the mask rank if
                        # the mask ranked below the current rank
                        if mask_rank < current_rank:
                            current_link = neighbor_idx + image_stride
                            aranks[neighbor_idx] = mask_rank
                        else:
                            current_link = current_idx
                            aranks[neighbor_idx] = current_rank
                        # unlink the neighbor
                        nprev = aprev[neighbor_idx]
                        nnext = anext[neighbor_idx]
                        anext[nprev] = nnext
                        if nnext != -1:
                            aprev[nnext] = nprev
                        # link to the neighbor after the current link
                        nnext = anext[current_link]
                        anext[neighbor_idx] = nnext
                        aprev[neighbor_idx] = current_link
                        if nnext >= 0:
                            aprev[nnext] = neighbor_idx
                            anext[current_link] = neighbor_idx
        current_idx = anext[current_idx]
