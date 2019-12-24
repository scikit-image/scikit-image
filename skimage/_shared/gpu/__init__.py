import os


try:
    if os.environ['SKIMAGE_USE_GPU'].upper() == 'TRUE':
        use_gpu = True
    else:
        use_gpu = False
except KeyError:
    print('SKIMAGE_USE_GPU environt variable not set: defaulting to false')
    use_gpu = False
try:
    if use_gpu == True:
        import pyopencl
        import skimage._vendor.gputools
        from pyopencl.tools import get_test_platforms_and_devices
        print("OpenCL devices detected: ", get_test_platforms_and_devices(), "using GPU.")

except (ImportError, pyopencl.LogicError) as e:
    print("Error occured with GPU imports, is your system configured correctly?")
    print(e)
    use_gpu = False
