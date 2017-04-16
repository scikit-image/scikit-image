from vbench.api import Benchmark

common_setup ="""from vb_common import *
"""

setup = common_setup + """
from skimage.util.dtype import *
from skimage import data

img_ubyte = data.camera()
img_int = img_as_int(img_ubyte)
img_uint = img_as_uint(img_ubyte)
img_float = img_as_float(img_ubyte)

"""

# NOTE: These tests can't be defined in a loop for some reason.
#       Each benchmark has to be saved to a unique variable (a list won't work)

img_as_ubyte_on_ubyte = Benchmark('img_as_ubyte(img_ubyte)', setup,
                                  name='img_as_ubyte_on_ubyte')

img_as_ubyte_on_int = Benchmark('img_as_ubyte(img_int)', setup,
                                name='img_as_ubyte_on_int')

img_as_ubyte_on_uint = Benchmark('img_as_ubyte(img_uint)', setup,
                                 name='img_as_ubyte_on_uint')

img_as_ubyte_on_float = Benchmark('img_as_ubyte(img_float)', setup,
                                  name='img_as_ubyte_on_float')


img_as_int_on_ubyte = Benchmark('img_as_int(img_ubyte)', setup,
                                  name='img_as_int_on_ubyte')

img_as_int_on_int = Benchmark('img_as_int(img_int)', setup,
                                name='img_as_int_on_int')

img_as_int_on_uint = Benchmark('img_as_int(img_uint)', setup,
                                 name='img_as_int_on_uint')

img_as_int_on_float = Benchmark('img_as_int(img_float)', setup,
                                  name='img_as_int_on_float')


img_as_uint_on_ubyte = Benchmark('img_as_uint(img_ubyte)', setup,
                                  name='img_as_uint_on_ubyte')

img_as_uint_on_int = Benchmark('img_as_uint(img_int)', setup,
                                name='img_as_uint_on_int')

img_as_uint_on_uint = Benchmark('img_as_uint(img_uint)', setup,
                                 name='img_as_uint_on_uint')

img_as_uint_on_float = Benchmark('img_as_uint(img_float)', setup,
                                  name='img_as_uint_on_float')


img_as_float_on_ubyte = Benchmark('img_as_float(img_ubyte)', setup,
                                  name='img_as_float_on_ubyte')

img_as_float_on_int = Benchmark('img_as_float(img_int)', setup,
                                name='img_as_float_on_int')

img_as_float_on_uint = Benchmark('img_as_float(img_uint)', setup,
                                 name='img_as_float_on_uint')

img_as_float_on_float = Benchmark('img_as_float(img_float)', setup,
                                  name='img_as_float_on_float')

