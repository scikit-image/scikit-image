"""
@author: mweigert

A basic wrapper class around pyopencl to handle image manipulation via OpenCL
basic usage:


    #create a device
    dev = OCLDevice(useGPU=True, useDevice = 0, printInfo = True)

"""

from __future__ import absolute_import, print_function

import logging

logger = logging.getLogger(__name__)

import pyopencl

__all__ = ["OCLDevice"]


class OCLDevice:
    """ a wrapper class representing a CPU/GPU device"""

    def __init__(self, id_platform=-1,
                 id_device=-1,
                 use_gpu=True,
                 print_info=False,
                 context_properties=None,
                 initCL=True, **kwargs):
        """ same kwargs as initCL """
        if initCL:
            self.init_cl(id_platform=id_platform,
                         id_device=id_device,
                         use_gpu=use_gpu,
                         print_info=print_info,
                         context_properties=context_properties,
                         **kwargs)

    @classmethod
    def device_priority(cls, device_with_type_tuple):
        """used to sort devices
        device_with_type_tuple = (device, device_type)
        """
        device, device_type = device_with_type_tuple
        return (device_type is pyopencl.device_type.GPU,
                device.get_info(pyopencl.device_info.GLOBAL_MEM_SIZE),
                )

    def init_cl(self,
                id_platform=-1,
                id_device=-1,
                use_gpu=True,
                print_info=False,
                context_properties=None):

        platforms = pyopencl.get_platforms()
        if len(platforms)==0:
            raise Exception("Failed to find any OpenCL platforms.")

        device_types = [pyopencl.device_type.GPU, pyopencl.device_type.CPU]

        # get all platforms and devices
        all_platforms_devs = dict([((_ip, _id, t), d)
                                   for _ip, p in enumerate(platforms)
                                   for _it, t in enumerate(device_types)
                                   for _id, d in enumerate(p.get_devices(t))])

        if len(all_platforms_devs)==0:
            raise Exception("Failed to find any OpenCL platform or device.")

        device_type = pyopencl.device_type.GPU if use_gpu else pyopencl.device_type.CPU

        device = None

        # try to get the prefered platform...
        # otherwise choose the best one
        try:
            device = all_platforms_devs[(id_platform, id_device, device_type)]
        except KeyError:
            logger.warning("prefered platform/device (%s/%s) not available (device type = %s) \n"
                           "...choosing the best from the rest"%
                           (id_platform, id_device, device_type))
            # get the best available device
            device, _ = max([(d, t) for (_ip, _id, t), d in all_platforms_devs.items()],
                            key=OCLDevice.device_priority)

        if device is None:
            raise Exception("Failed to find a valid device")

        self.context = pyopencl.Context(devices=[device],
                                        properties=context_properties)

        self.device = device

        self.queue = pyopencl.CommandQueue(self.context,
                                           properties=pyopencl.command_queue_properties.PROFILING_ENABLE)

        self.imageformats = pyopencl.get_supported_image_formats(self.context,
                                                                 pyopencl.mem_flags.READ_WRITE,
                                                                 pyopencl.mem_object_type.IMAGE3D)

        logger.info("intialized, device: {}".format(self.device))
        if print_info:
            self.print_info()

    def get_all_info(self):
        platforms = pyopencl.get_platforms()
        s = "\n-------- available devices -----------\n"
        for p in platforms:
            s += "platform: \t%s\n"%p.name
            printNames = [["CPU", pyopencl.device_type.CPU],
                          ["GPU", pyopencl.device_type.GPU]]
            for name, identifier in printNames:
                s += "device type: \t%s\n"%name
                try:
                    for d in p.get_devices(identifier):
                        s += "\t%s \n"%d.name
                except:
                    s += "nothing found: \t%s\n"%name

        infoKeys = ['NAME', 'GLOBAL_MEM_SIZE',
                    'GLOBAL_MEM_SIZE', 'MAX_MEM_ALLOC_SIZE',
                    'LOCAL_MEM_SIZE', 'IMAGE2D_MAX_WIDTH',
                    'IMAGE2D_MAX_HEIGHT', 'IMAGE3D_MAX_WIDTH',
                    'IMAGE3D_MAX_HEIGHT', 'IMAGE3D_MAX_DEPTH',
                    'MAX_WORK_GROUP_SIZE', 'MAX_WORK_ITEM_SIZES']

        s += "\n-------- currently used device -------\n"

        for k in infoKeys:
            s += "%s: \t  %s\n"%(k, self.get_info(k))
        return s

    def print_info(self):
        print(self.get_all_info())

    def get_info(self, info_str="MAX_MEM_ALLOC_SIZE"):
        return self.device.get_info(getattr(pyopencl.device_info, info_str))

    def get_extensions(self):
        return self.device.extensions.strip().split(' ')
    def __repr__(self):
        return self.get_all_info()


if __name__=='__main__':
    logger.setLevel(logging.WARNING)
    dev = OCLDevice(id_platform=0, id_device=0, useGPU=False)
    dev.print_info()
