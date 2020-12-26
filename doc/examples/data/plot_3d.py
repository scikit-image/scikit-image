"""
==========================================
Datasets with 3 or more spatial dimensions
==========================================

Most scikit-image functions are compatible with 3-D datasets, ie images with
3 spatial dimensions (as opposed to 2-D multichannel arrays, which also have
three axes). func:`skimage.data.cells3d` returns a 3D fluorescence microscopy
image of cells. The returned data is a 3D multichannel array with dimensions
provided in ``(z, c, y, x)`` order. Channel 0 contains cell membranes, channel
1 contains nuclei.

The example below shows how to explore this dataset. This 3-D image can be used
to test the various functions of scikit-image.
"""
from skimage import data
import plotly.express as px
import plotly

img = data.cells3d()[20:]
fig = px.imshow(img, facet_col=1, animation_frame=0,
                binary_string=True, binary_format='jpg')
fig.layout.annotations[0]['text'] = 'Cell membranes'
fig.layout.annotations[1]['text'] = 'Nuclei'
plotly.io.show(fig)
