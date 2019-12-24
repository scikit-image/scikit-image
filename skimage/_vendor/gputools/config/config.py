from __future__ import print_function, unicode_literals, absolute_import, division

import logging

logging.basicConfig(level=logging.INFO)

import os
from gputools.config.myconfigparser import MyConfigParser
from gputools.core.ocldevice import OCLDevice


__CONFIGFILE__ = os.path.expanduser("~/.gputools")

config_parser = MyConfigParser(__CONFIGFILE__)

#Using os here allows us to configure multiple python gputools processes 
#in a multi GPU environnement.
#Simply changing the device with init_device keeps the first device in the process. And that takes some memory on the gpu
defaults = {
    "id_device": os.getenv('gputools_id_device',0),
    "id_platform": os.getenv('gputools_id_platform',0),
    "use_gpu": os.getenv('gputools_use_gpu',1)
}

def _get_param(name, type):
    return type(config_parser.get(name, defaults[name]))


__ID_DEVICE__ = _get_param("id_device", int)
__ID_PLATFORM__ = _get_param("id_platform", int)
__USE_GPU__ = _get_param("use_gpu", int)


class _ocl_globals(object):
    device = OCLDevice(id_platform=__ID_PLATFORM__,
                       id_device=__ID_DEVICE__,
                       use_gpu=__USE_GPU__)


def init_device(**kwargs):
    """same arguments as OCLDevice.__init__
    e.g.
    id_platform = 0
    id_device = 1
    ....
    """
    new_device = OCLDevice(**kwargs)

    # just change globals if new_device is different from old
    if _ocl_globals.device.device != new_device.device:
        _ocl_globals.device = new_device


def get_device():
    return _ocl_globals.device



if __name__ == '__main__':
    print(get_device())
    print(__ID_DEVICE__)
