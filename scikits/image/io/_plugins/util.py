import numpy as np

# utilities to make life easier for plugin writers.

class WindowManager(object):
    ''' A class to keep track of spawned windows,
    and make any needed callback once all the windows,
    are closed.'''
    def __init__(self):
        self._windows = []
        self._callback = None
        self._callback_args = ()
        self._callback_kwargs = {}

        self._gui_lock = False

    def _check_locked(self):
        if not self._gui_lock:
            raise RuntimeError(\
            'Must first acquire the gui lock before using this image manager ')

    def acquire(self):
        print 'lock requested while: ', self._gui_lock
        if self._gui_lock:
            raise RuntimeError(\
            'The gui lock can only be acquired by one toolkit per session')
        else:
            self._gui_lock = True

    def add_window(self, win):
        self._windows.append(win)

    def remove_window(self, win):
        try:
            self._windows.remove(win)
        except ValueError:
            print 'Unable to find referenced window in tracked windows.'
            print 'Ignoring...'
        else:
            print len(self._windows)
            if len(self._windows)==0:
                self._exec_callback()

    def register_callback(self, cb, *cbargs, **cbkwargs):
        self._callback = cb
        print self._callback
        self._callback_args = cbargs
        self._callback_kwargs = cbkwargs

    def _exec_callback(self):
        if self._callback:
            print 'calling callback'
            self._callback(*self._callback_args, **self._callback_kwargs)

window_manager = WindowManager()

def prepare_for_display(npy_img):
    '''Convert a 2D or 3D numpy array of any dtype into a
    3D numpy array with dtype uint8. This array will
    be suitable for use in passing to gui toolkits for
    image display purposes.

    Parameters
    ----------
    npy_img : ndarray, 2D or 3D
        The image to convert for display

    Returns
    -------
    out : ndarray, 3D dtype=np.uint8
        The converted image. This is guaranteed to be a contiguous array.

    Notes
    -----
    If the input image is floating point, it is assumed that the data
    is in the range of 0.0 - 1.0. No check is made to assert this
    condition. The image is then scaled to be in the range 0 - 255
    and then cast to np.uint8

    For all other dtypes, the array is simply cast to np.uint8

    If a 2D array is passed, the single channel is replicated
    to the 2nd and 3rd channels.

    If the array contains an alpha channel, this channel is
    ignored.

    '''
    if len(npy_img.shape) < 2:
        raise ValueError('Image must be 2D or 3D array')

    height = npy_img.shape[0]
    width = npy_img.shape[1]

    out = np.empty((height, width, 3), dtype=np.uint8)

    if len(npy_img.shape) == 2 or \
       (len(npy_img.shape) == 3 and npy_img.shape[2] == 1):
        if npy_img.dtype in [np.float32, np.float64]:
            out[:,:,0] = npy_img*255
            out[:,:,1] = out[:,:,0]
            out[:,:,2] = out[:,:,0]
        else:
            out[:,:,0] = npy_img
            out[:,:,1] = npy_img
            out[:,:,2] = npy_img

    elif len(npy_img.shape) == 3:
        if npy_img.shape[2] == 3 or npy_img.shape[2] == 4:
            if npy_img.dtype in [np.float32, np.float64]:
                out[:,:,:3] = (npy_img[:,:,:3])*255
            else:
                out[:,:,:3] = npy_img[:,:,:3]
        else:
            raise ValueError('Image must have 1, 3, or 4 channels')

    else:
        raise ValueError('Image must have 2 or 3 dimensions')

    return out