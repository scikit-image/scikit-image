from .version import __version__

import logging
logging.basicConfig(format='%(levelname)s:%(name)s | %(message)s')
logger = logging.getLogger(__name__)
#logger.setLevel(logging.WARNING)

from gputools.config.config import init_device, get_device
from gputools.config import config

from gputools.utils.utils import pad_to_shape, pad_to_power2
from gputools.utils.utils import remove_cache_dir
from gputools.utils.tile_iterator import tile_iterator


from gputools.core.ocltypes import OCLArray, OCLImage
from gputools.core.oclprogram import OCLProgram
from gputools.core.oclalgos import OCLReductionKernel, OCLElementwiseKernel
from gputools.core.oclmultireduction import OCLMultiReductionKernel


from gputools.fft.oclfft_convolve import fft_convolve
from gputools.fft.oclfft import fft, fft_plan
from gputools.fft.fftshift import fftshift


from gputools.convolve.convolve_sep import convolve_sep2, convolve_sep3
from gputools.convolve import min_filter, max_filter, uniform_filter,blur, gaussian_filter,median_filter
from gputools.convolve.convolve import convolve
from gputools.convolve import convolve_spatial2, convolve_spatial3


from gputools.noise import perlin2, perlin3

from gputools import denoise
from gputools import deconv
from gputools import convolve
from gputools import transforms

from gputools import noise

from gputools.transforms import scale
from gputools.transforms import affine, rotate, shift, map_coordinates, geometric_transform

