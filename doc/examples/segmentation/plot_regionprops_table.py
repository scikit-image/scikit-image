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

areas.plot(x='volume fraction', y='area', kind='scatter')
plt.show()
