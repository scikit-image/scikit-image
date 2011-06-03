import pyopencl as cl
import numpy as np

platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)
program = cl.Program(context,
"""
__kernel void sobel_filter_uint(__global uint* input_image, __global int* output_image, int const axis) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint width = get_global_size(0);
    uint height = get_global_size(1);
    int Gx = 0;
    int Gy = Gx;
    int index = y + x * height;
    output_image[index] = 0;	
    if( x >= 1 && x < (width - 1) && y >= 1 && y < (height - 1)) {
	    uint i00 = input_image[index - 1 - height];
	    uint i10 = input_image[index - height];
	    uint i20 = input_image[index + 1 - height];
	    uint i01 = input_image[index - 1];
	    uint i11 = input_image[index];
	    uint i21 = input_image[index + 1];
	    uint i02 = input_image[index - 1 + height];
	    uint i12 = input_image[index + height];
	    uint i22 = input_image[index + 1 + height];
	    if (axis == -1) {
            Gx = i00 + 2 * i10 + i20 - i02 - 2 * i12 - i22;
            Gy = i00 - i20 + 2 * i01 - 2 * i21 + i02 - i22;
    	    output_image[index] = hypot(Gx, Gy);
	    } 
	    else if (axis == 0) {
            Gx = i00 + 2 * i10 + i20 - i02 - 2 * i12 - i22;
            output_image[index] = Gx;
        } 
        else if (axis == 1) {
            Gy = i00 - i20 + 2 * i01 - 2 * i21 + i02 - i22;
    	    output_image[index] = Gy;
        }
    }
}

__kernel void sobel_filter_float32(__global float* input_image, __global float* output_image, int const axis) {
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint width = get_global_size(0);
    uint height = get_global_size(1);
    int Gx = 0;
    int Gy = Gx;
    int index = y + x * height;
    output_image[index] = 0;	
    if( x >= 1 && x < (width - 1) && y >= 1 && y < (height - 1)) {
	    uint i00 = input_image[index - 1 - height];
	    uint i10 = input_image[index - height];
	    uint i20 = input_image[index + 1 - height];
	    uint i01 = input_image[index - 1];
	    uint i11 = input_image[index];
	    uint i21 = input_image[index + 1];
	    uint i02 = input_image[index - 1 + height];
	    uint i12 = input_image[index + height];
	    uint i22 = input_image[index + 1 + height];
	    if (axis == -1) {
            Gx = i00 + 2 * i10 + i20 - i02 - 2 * i12 - i22;
            Gy = i00 - i20 + 2 * i01 - 2 * i21 + i02 - i22;
    	    output_image[index] = hypot(Gx, Gy);
	    } 
	    else if (axis == 0) {
            Gx = i00 + 2 * i10 + i20 - i02 - 2 * i12 - i22;
            output_image[index] = Gx;
        } 
        else if (axis == 1) {
            Gy = i00 - i20 + 2 * i01 - 2 * i21 + i02 - i22;
    	    output_image[index] = Gy;
        }
    }
}
""")
    
try:
    print "building"
    program.build()
except:
    print("Error:")
    print(program.get_build_info(context.devices[0], cl.program_build_info.LOG))
    raise

def sobel(image, axis=None, output=None):
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)    
    if image.dtype == np.uint8:
        image = image.astype(np.uint32)        
        output_type = np.int32
    elif image.dtype == np.float32:
        output_type = np.float32
    else:
        raise TypeError, "Input type of uint8 or float32 expected."
    
    if not output:
        output = np.empty(image.shape, dtype=output_type)

    mf = cl.mem_flags
    input_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image)
    output_buf = cl.Buffer(context, mf.WRITE_ONLY, output.nbytes)
    if axis is None:
        axis = -1
    if image.dtype == np.uint32:
        program.sobel_filter_uint(queue, image.shape, None, input_buf, output_buf, np.int32(axis))
    elif image.dtype == np.float32:
        program.sobel_filter_float32(queue, image.shape, None, input_buf, output_buf, np.int32(axis))
    cl.enqueue_read_buffer(queue, output_buf, output).wait()
    return output



if __name__ == "__main__":
    from scikits.image import data_dir
    from scikits.image import io
    from scikits.image.color import rgb2gray
    import os, time
    image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.uint8)
    #image = rgb2gray(io.imread(os.path.join(data_dir,"lena512.png"))).astype(np.float32)
    image = np.zeros((2000,2000), dtype=np.float32)
    t = time.time()
    output = sobel(image, axis=None)
    print time.time() -t 
    io.use_plugin("gtk")
    io.imshow(output.astype(np.uint8))
    #io.show()
    
    
    
