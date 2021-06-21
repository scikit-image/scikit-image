import numpy as np
import skimage


def _channel_kwarg(is_multichannel=False):
    if np.lib.NumpyVersion(skimage.__version__) < '0.19.0':
        return dict(multichannel=is_multichannel)
    else:
        return dict(channel_axis=-1 if is_multichannel else None)
