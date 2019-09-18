
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

#ifndef DTYPE
#define DTYPE float
#endif


__kernel void map_coordinates2(__read_only image2d_t input,
					__global DTYPE* output,
				 __global float* coordinates)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint N = get_global_size(0);

  float y = coordinates[i];
  float x = coordinates[N+i];

  y+=.5f;
  x+=.5f;
  

  float pix = read_imagef(input,sampler, (float2)(x,y)).x;

  output[i] = pix;
}

__kernel void map_coordinates3(__read_only image3d_t input,
					__global DTYPE* output,
				 __global float* coordinates)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint N = get_global_size(0);

  float z = coordinates[i];
  float y = coordinates[N+i];
  float x = coordinates[2*N+i];

  z+=.5f;
  y+=.5f;
  x+=.5f;

  float pix = read_imagef(input,sampler, (float4)(x,y,z,0.f)).x;

  output[i] = pix;
}
