"""
===================================================
Explore and visualize region properties with pandas
===================================================

This toy example shows how to compute the area of every labelled region in a
series of 10 images. The blob-like regions are generated synthetically. As the
fraction of image pixels covered by the blobs increases, the number of blobs
(regions) decreases, and the size (area) of a single region can get larger and
larger. The area values are available in a pandas-compatible format, which
makes for convenient data analysis and visualization.

Beyond area, many other region properties are available.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage import data, measure


fractions = np.linspace(0.05, 0.5, 10)

images = [data.binary_blobs(volume_fraction=f) for f in fractions]

labeled_images = [measure.label(image) for image in images]

properties = ['label', 'area']

tables = [measure.regionprops_table(image, properties=properties)
          for image in labeled_images]
tables = [pd.DataFrame(table) for table in tables]

for fraction, table in zip(fractions, tables):
    table['volume fraction'] = fraction

areas = pd.concat(tables, axis=0)

# Divide the figure into a 2x1 grid
fig, (im, ax) = plt.subplots(2, 1)
# Turn off axes for top subplot
im.axis('off')
# Plot area vs volume fraction
areas.plot(x='volume fraction', y='area', kind='scatter', ax=ax)
# Show image with lowest volume fraction
ax1 = fig.add_subplot(221)
ax1.imshow(images[0])
# Show image with highest volume fraction
ax2 = fig.add_subplot(222)
ax2.imshow(images[-1])
plt.show()

"""
In the scatterplot, many points seem to be overlapping at low area values.
To get a better sense of the distribution, we may want to add some 'jitter'
to the visualization. To this end, we import `seaborn`, the Python library
dedicated to statistical data visualization, and use `stripplot` with
argument `jitter=True`.
"""

import seaborn as sns

sns_fig, sns_ax = plt.subplots()
sns.stripplot(x='volume fraction', y='area', data=areas, jitter=True,
              ax=sns_ax)
x_format = sns_ax.xaxis.get_major_formatter()
x_format.seq = ['{:0.2f}'.format(float(s)) for s in x_format.seq]
sns_ax.xaxis.set_major_formatter(x_format)
plt.show()
