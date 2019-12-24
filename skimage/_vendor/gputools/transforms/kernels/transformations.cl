
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

#ifndef DTYPE
#define DTYPE float
#endif


__kernel void affine(__read_only image3d_t input,
	      			 __global DTYPE* output,
				 __constant float * mat)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  //float x = (mat[0]*i+mat[1]*j+mat[2]*k+mat[3]);
  //float y = (mat[4]*i+mat[5]*j+mat[6]*k+mat[7]);
  //float z = (mat[8]*i+mat[9]*j+mat[10]*k+mat[11]);
  ////ensure correct sampling, see opencl 1.2 specification pg. 329
  //x += 0.5f;
  //y += 0.5f;
  //z += 0.5f;

  float x = i+0.5f;
  float y = j+0.5f;
  float z = k+0.5f;

  float x2 = (mat[8]*z+mat[9]*y+mat[10]*x+mat[11]);
  float y2 = (mat[4]*z+mat[5]*y+mat[6]*x+mat[7]);
  float z2 = (mat[0]*z+mat[1]*y+mat[2]*x+mat[3]);


  float4 coord_norm = (float4)(x2/Nx,y2/Ny,z2/Nz,0.f);

  float pix = read_imagef(input,sampler,coord_norm).x;

  output[i+Nx*j+Nx*Ny*k] = pix;


}

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
