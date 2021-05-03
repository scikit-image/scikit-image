"""
==========================================
Datasets with 3 or more spatial dimensions
==========================================

Most scikit-image functions are compatible with 3D datasets, i.e., images with
3 spatial dimensions (to be distinguished from 2D multichannel images, which 
are also arrays with
three axes). :func:`skimage.data.cells3d` returns a 3D fluorescence microscopy
image of cells. The returned dataset is a 3D multichannel image with dimensions
provided in ``(z, c, y, x)`` order. Channel 0 contains cell membranes, while channel
1 contains nuclei.

The example below shows how to explore this dataset. This 3D image can be used
to test the various functions of scikit-image.
"""
from skimage import data
import plotly
import plotly.express as px

img = data.cells3d()[20:]
fig = px.imshow(img, facet_col=1, animation_frame=0,
                binary_string=True, binary_format='jpg')
fig.layout.annotations[0]['text'] = 'Cell membranes'
fig.layout.annotations[1]['text'] = 'Nuclei'
plotly.io.show(fig)
