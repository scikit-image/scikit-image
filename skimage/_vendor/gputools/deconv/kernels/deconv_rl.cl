#include <pyopencl-complex.h>



__kernel void multiply_complex(__global cfloat_t* a,__global cfloat_t* b,__global cfloat_t* output){

  uint i = get_global_id(0);
  output[i] = cfloat_mul(a[i], b[i]);
 
}


__kernel void multiply_complex_inplace(__global cfloat_t* a,__global cfloat_t* b){

  uint i = get_global_id(0);
  a[i] = cfloat_mul(a[i], b[i]);
 
}


__kernel void divide_complex_inplace(__global cfloat_t* a,__global cfloat_t* b){

  uint i = get_global_id(0);
  b[i] = cfloat_divide(a[i], b[i]);
 
}

