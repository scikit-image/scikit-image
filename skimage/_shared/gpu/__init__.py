use_gpu = False
try: 
    import pyopencl
    import skimage._vendor.gputools
    from pyopencl.tools import get_test_platforms_and_devices
    print("OpenCL devices detected: ",get_test_platforms_and_devices(),"using GPU.")
    use_gpu = True

except (ImportError, pyopencl.LogicError) as e: #TODO see what the actual error thrown here is
    print("Error occured with GPU imports, is your system configured correctly?")
    print(e)
    use_gpu = False
use_gpu = True