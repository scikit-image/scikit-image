"""
=======================================================
Decomposition of flat footprints (structuring elements)
=======================================================

Many footprints (structuring elements) can be decomposed into an equivalent
series of smaller structuring elements. The term "flat" refers to a footprints
that only contains values of 0 or 1 (i.e. all methods in
``skimage.morphology.footprints``). Binary dilation operations have an
associative and distributive property such that often allows decomposition into
an equivalent series of smaller footprints. Most often this is done to provide
a performance benefit.

As a concrete example, dilation with a square footprint of size (15, 15) is
equivalent to dilation with a rectangle of size (15, 1) followed by a
dilation with a rectangle of size (1, 15). It is also equivalent to 7
consecutive dilations with a shape (3, 3) square.

There are many possible decompositions and which one performs best may be
architecture-dependent.

scikit-image currently provides two forms of automated decomposition. For the
cases of ``square``, ``rectangle`` and ``cube`` footprints, there is an option
for a "separable" decomposition (size > 1 along only 1 axis at a time).

For some other symmetric convex shapes such as ``diamond``, ``octahedron`` and
``octagon`` there is no separable decomposition, but it is possible to provide
a "sequence" decomposition based on a series of small footprints of shape
``(3,) * ndim``.

For simplicity of implementation, all decompositions use only odd-sized
footprints with their origin located at the center of the footprint.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D

from skimage.morphology import (cube, diamond, octagon, octahedron, rectangle,
                                square)

# Generate 2D and 3D structuring elements.
footprint_dict = {
    "square(11) (separable)": (square(11, decomposition=None),
                               square(11, decomposition="separable")),
    "square(11) (sequence)": (square(11, decomposition=None),
                              square(11, decomposition="sequence")),
    "rectangle(7, 11) (separable)": (rectangle(7, 11, decomposition=None),
                                     rectangle(7, 11,
                                               decomposition="separable")),
    "rectangle(7, 11) (sequence)": (rectangle(7, 11, decomposition=None),
                                    rectangle(7, 11,
                                              decomposition="sequence")),
    "diamond(5) (sequence)": (diamond(5, decomposition=None),
                              diamond(5, decomposition="sequence")),
    "octagon(7, 4) (sequence)": (octagon(7, 4, decomposition=None),
                                 octagon(7, 4, decomposition="sequence")),
    "cube(11) (separable)": (cube(11, decomposition=None),
                             cube(11, decomposition="separable")),
    "cube(11) (sequence)": (cube(11, decomposition=None),
                            cube(11, decomposition="sequence")),
    "octahedron(7) (sequence)": (octahedron(7, decomposition=None),
                                 octahedron(7, decomposition="sequence")),
}

# Visualize the elements

# use a similar dark blue for the 2d plots as for the 3d voxel plots
cmap = colors.ListedColormap(['white', (0.1216, 0.4706, 0.70588)])
fontdict = dict(fontsize=16, fontweight='bold')
for title, (footprint, footprint_sequence) in footprint_dict.items():
    fig = plt.figure(figsize=(12, 4))
    ndim = footprint.ndim
    num_seq = len(footprint_sequence)
    if ndim == 2:
        ax = fig.add_subplot(1, num_seq + 1, num_seq + 1)
        ax.imshow(footprint, cmap=cmap, vmin=0, vmax=1)
    else:
        ax = fig.add_subplot(1, num_seq + 1, num_seq + 1,
                             projection=Axes3D.name)
        ax.voxels(footprint, cmap=cmap)

    ax.set_title(title.split(' (')[0], fontdict=fontdict)
    ax.set_axis_off()
    for n, (fp, num_reps) in enumerate(footprint_sequence):
        npad = [((footprint.shape[d] - fp.shape[d]) // 2, ) * 2
                for d in range(ndim)]
        fp = np.pad(fp, npad, mode='constant')
        if ndim == 2:
            ax = fig.add_subplot(1, num_seq + 1, n + 1)
            ax.imshow(fp, cmap=cmap, vmin=0, vmax=1)
        else:
            ax = fig.add_subplot(1, num_seq + 1, n + 1, projection=Axes3D.name)
            ax.voxels(fp, cmap=cmap)
        title = f"element {n + 1} of {num_seq}\n({num_reps} iteration"
        title += "s)" if num_reps > 1 else ")"
        ax.set_title(title, fontdict=fontdict)
        ax.set_axis_off()
        ax.set_xlabel(f'num_reps = {num_reps}')
    fig.tight_layout()

    # draw dividing line between seqeuence element plots and composite plot
    line_pos = num_seq / (num_seq + 1)
    line = plt.Line2D([line_pos, line_pos], [0, 1], color="black")
    fig.add_artist(line)

plt.show()
