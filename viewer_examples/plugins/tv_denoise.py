from skimage import data
from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_float
from numpy import random, clip

from skimage.viewer import ImageViewer
from skimage.viewer.widgets import (Slider, CheckBox, OKCancelButtons,
                                    SaveButtons)
from skimage.viewer.plugins.base import Plugin


image = img_as_float(data.chelsea())
sigma = 30/255.

image = image + random.normal(loc=0, scale=sigma, size=image.shape)
image = clip(image, 0, 1)
viewer = ImageViewer(image)

plugin = Plugin(image_filter=denoise_tv_chambolle)
plugin += Slider('weight', 0.01, 5, value=0.3, value_type='float')
plugin += Slider('n_iter_max', 1, 100, value=20, value_type='int')
plugin += CheckBox('multichannel', value=True)
plugin += SaveButtons()
plugin += OKCancelButtons()

viewer += plugin
viewer.show()
