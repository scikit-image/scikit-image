def imread(fname, as_grey=False, dtype=None):
    assert fname == 'test.png'
    assert as_grey == True
    assert dtype == 'i4'

def imsave(fname, arr):
    assert fname == 'test.png'
    assert arr == [1, 2, 3]

def imshow(arr, plugin_arg=None):
    assert arr == [1, 2, 3]
    assert plugin_arg == (1, 2)

def imread_collection(x, conserve_memory=True):
    assert conserve_memory == False
    assert x == '*.png'
