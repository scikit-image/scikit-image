import os
import numpy as np
import skimage


def _channel_kwarg(is_multichannel=False):
    if np.lib.NumpyVersion(skimage.__version__) < '0.19.0':
        return dict(multichannel=is_multichannel)
    else:
        return dict(channel_axis=-1 if is_multichannel else None)

def _skip_slow():
    if os.environ.get("ASV_SKIP_SLOW", "0") == "1":
        raise NotImplementedError("Skipping this test...")
