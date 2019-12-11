"""
=========================
Tabulate areas of regions
=========================

This toy example shows how to compute the area of every labelled region in a
series of 3D images. The blob-like regions are generated synthetically. As the
fraction of image pixels covered by the blobs increases, the number of blobs
(regions) decreases, and the size (area) of a single region can get larger and
larger. The area values are available in a pandas-compatible format, which
makes for convenient data analysis and visualization.

Beyond area, many other region properties are available.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from skimage import data, measure


fractions = np.linspace(0.05, 0.5, 10)

images = [data.binary_blobs(length=128, n_dim=3, volume_fraction=f)
          for f in fractions]

labeled_images = [measure.label(image) for image in images]

properties = ['label', 'area']

tables = [measure.regionprops_table(image, properties=properties)
          for image in labeled_images]
tables = [pd.DataFrame(table) for table in tables]

for fraction, table in zip(fractions, tables):
    table['volume fraction'] = fraction

areas = pd.concat(tables, axis=0)

ax = sns.stripplot(data=areas, x='volume fraction', y='area', jitter=0.2)
x_format = ax.xaxis.get_major_formatter()
x_format.seq = ["{:0.2f}".format(float(s)) for s in x_format.seq]
ax.xaxis.set_major_formatter(x_format)
plt.show()
