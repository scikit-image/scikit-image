"""
=========================
Measure region properties
=========================

This example shows how to measure properties of labelled image regions. We
first analyze an image with two ellipses. Below we show how to explore
interactively the properties of labelled objects.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

image = np.zeros((600, 600))

rr, cc = ellipse(300, 350, 100, 220)
image[rr, cc] = 1

image = rotate(image, angle=15, order=0)

rr, cc = ellipse(100, 100, 60, 50)
image[rr, cc] = 1

label_img = label(image)
regions = regionprops(label_img)

#####################################################################
# We use the :py:func:`skimage.measure.regionprops` result to draw certain
# properties on each region. For example, in red, we plot the major and minor
# axes of each ellipse.

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()

#####################################################################
# We use the :py:func:`skimage.measure.regionprops_table` function to compute
# (selected) properties for each region. Note that
# ``skimage.measure.regionprops_table`` actually computes the properties,
# whereas ``skimage.measure.regionprops`` computes them when they come in use
# (lazy evaluation).

props = regionprops_table(
    label_img,
    properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length'),
)

#####################################################################
# We now display a table of these selected properties (one region per row),
# the ``skimage.measure.regionprops_table`` result being a pandas-compatible
# dict.

pd.DataFrame(props)

#####################################################################
# It is also possible to explore interactively the properties of labelled
# objects by visualizing them in the hover information of the labels.
# This example uses plotly in order to display properties when
# hovering over the objects.
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from skimage import data, filters, measure, morphology

img = data.coins()
# Binary image, post-process the binary mask and compute labels
threshold = filters.threshold_otsu(img)
mask = img > threshold
mask = morphology.remove_small_objects(mask, max_size=49)
mask = morphology.remove_small_holes(mask, max_size=49)
labels = measure.label(mask)

property_names = ["label", "area", "eccentricity", "perimeter", "intensity_mean"]

# Get region properties
props = measure.regionprops(labels, img)

# Filter out big regions
filtered_props = [prop for prop in props if prop.area < 5000]

fig = px.imshow(img, binary_string=True)
fig.update_traces(hoverinfo="skip")  # hover is only for label info

# Prepare data for scatter plot
scatter_data = []
for prop in filtered_props:
    label_i = prop.label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y, x = contour.T
    hoverinfo = "<br>".join(
        [
            f'<b>{prop_name}: {getattr(prop, prop_name):.2f}</b>'
            for prop_name in property_names
        ]
    )
    scatter_data.append({"x": x, "y": y, "label": label_i, "hoverinfo": hoverinfo})

# For each label, add a filled scatter trace for its contour,
# and display the properties of the label in the hover of this trace.
for data_i in scatter_data:
    fig.add_trace(
        go.Scatter(
            x=data_i["x"],
            y=data_i["y"],
            name=str(data_i["label"]),
            mode="lines",
            fill="toself",
            showlegend=False,
            hovertemplate=data_i["hoverinfo"],
            hoveron="points+fills",
        )
    )

fig.update_layout(autosize=True)
pio.show(fig)
