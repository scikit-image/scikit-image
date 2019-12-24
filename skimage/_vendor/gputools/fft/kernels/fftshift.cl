/*

fftshift implementation for opencl

adopted from:

cufftShift: high performance CUDA-accelerated FFT-shift library.
Proc High Performance Computing Symposium.
2014.


*/

#include <pyopencl-complex.h>


__kernel void fftshift_1_f(__global float* src,
                           __global float * dest,
                           const int N,
                           const int stride
                           ){

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);


    int index1 = i+j*stride + k*stride*N;
    int index2 = i+(j+N/2)*stride + k*stride*N;

    float val1 = src[index1];
    float val2 = src[index2];


    //swap halves
    dest[index1] = val2;
    dest[index2] = val1;


}

__kernel void fftshift_1_c(__global cfloat_t* src,
                           __global cfloat_t* dest,
                           const int N,
                           const int stride
                           ){

    int i = get_global_id(0);
    int j = get_global_id(1);
    int k = get_global_id(2);


    int index1 = i+j*stride + k*stride*N;
    int index2 = i+(j+N/2)*stride + k*stride*N;

    cfloat_t val1 = src[index1];
    cfloat_t val2 = src[index2];


    //swap halves
    dest[index1] = val2;
    dest[index2] = val1;


}

/*
__kernel void fftshift_1_f(__global float* src,
                           __global float * dest,
                           const int N,
                           const int stride1,
                           const int stride2,
                           const int offset){

    int i = get_global_id(0);
    int j = get_global_id(1);


    int index1 = offset + i*stride1 + j*stride2;
    int index2 = offset + (i+N/2)*stride1 + j*stride2;

    float val1 = src[index1];
    float val2 = src[index2];


    //swap halves
    dest[index1] = val2;
    dest[index2] = val1;

    //dest[index2] = src[index1];


}

*/

/*
__kernel void fftshift_1_f(__global float* src,
                           __global float * dest,
                           const int N){

    int i = get_global_id(0);

    float val1 = src[i];
    float val2 = src[i+N/2];

    //swap halves
    dest[i] = val2;
    dest[i+N/2] = val1;
}


__kernel void fftshift_1_c(__global cfloat_t* src,
                           __global cfloat_t* dest,
                            const int N){

    int i = get_global_id(0);

    cfloat_t val1 = src[i];
    cfloat_t val2 = src[i+N/2];

    //swap halves
    dest[i] = val2;
    dest[i+N/2] = val1;

}



__kernel void fftshift_2_f(__global float* src,
                           __global float * dest,
                           const int Nx,
                           const int Ny){

    int i = get_global_id(0);
    int j = get_global_id(1);

    int index = i+j*Nx;

    int offset1 = (Nx*Ny+Nx)/2;
    int offset2 = (Nx*Ny-Nx)/2;

    if (i<Nx/2){

        if (j<(Ny/2)){

            float val1 = src[index];
            float val2 = src[index + offset1];

            //swap halves
            dest[index] = val2;
            dest[index+offset1] = val1;

        }
    }
    else{

        if (j<(Ny/2)){

            float val1 = src[index];
            float val2 = src[index + offset2];

            //swap halves
            dest[index] = val2;
            dest[index+offset2] = val1;

        }
    }

    //dest[index] = 1.;

}
*/