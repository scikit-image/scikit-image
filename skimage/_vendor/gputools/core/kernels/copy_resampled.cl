#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable
#include <pyopencl-complex.h>

////////// IMAGE ----> BUFFER

__kernel void img2d_to_buf_complex(__read_only image2d_t src,
						 __global cfloat_t *dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));
  dest[i+Nx*j] = cfloat_new(val.x,val.y);
}

__kernel void img2d_to_buf_float(__read_only image2d_t src,
						 __global float *dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));
  dest[i+Nx*j] = val.x;
}


__kernel void img3d_to_buf_complex(__read_only image3d_t src,
						 __global cfloat_t *dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float4 val = read_imagef(src,
  						   sampler,
						   (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),1.f*k/(Nz-1.f),0.f));
  dest[i+Nx*j+Nx*Ny*k] = cfloat_new(val.x,val.y);
}

__kernel void img3d_to_buf_float(__read_only image3d_t src,
						 __global float *dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float4 val = read_imagef(src,
  						   sampler,
						   (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),1.f*k/(Nz-1.f),0.f));
  dest[i+Nx*j+Nx*Ny*k] =val.x;
}



////////// IMAGE ----> IMAGE

__kernel void img2d_to_img(__read_only image2d_t src,
						 __write_only image2d_t dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));

  write_imagef(dest,(int2)(i,j),val);
}

__kernel void img3d_to_img(__read_only image3d_t src,
						 __write_only image3d_t dest){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float4 val = read_imagef(src,
  						   sampler,
						   (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),1.f*k/(Nz-1.f),0.f));

  write_imagef(dest,(int4)(i,j,k,0),val);
}
