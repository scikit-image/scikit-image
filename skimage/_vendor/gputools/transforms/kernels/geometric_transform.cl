#define  M_PI 3.141592653589793f

#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif

#ifndef SAMPLER_ADDRESS
#define SAMPLER_ADDRESS CLK_ADDRESS_CLAMP
#endif

#ifndef DTYPE
#define DTYPE float
#endif


inline float2 map_coord2(const float2 coord){
  float c1 = coord.x;
  float c0 = coord.y;
  return (float2)(${FUNC2});
}

inline float4 map_coord3(const float4 coord){
  float c2 = coord.x;
  float c1 = coord.y;
  float c0 = coord.z;
  return (float4)(${FUNC3},0.f);
}


__kernel void geometric_transform2(__read_only image2d_t input,
					__global DTYPE* output)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);

  float2 coord = (float2)(i,j);

  coord = map_coord2(coord);
  coord += .5f;
  
  DTYPE pix = READ_IMAGE(input,sampler, coord).x;

  if ((i==10)&&(j==12))
	printf("%d %d %.2f %.2f",i,j, coord.x, coord.y);
  
  output[i+Nx*j] = pix;
}


__kernel void geometric_transform3(__read_only image3d_t input,
					__global DTYPE* output)
{

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE|
      SAMPLER_ADDRESS |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 coord = (float4)(i,j,k,0.f);

  coord = map_coord3(coord);
  coord += .5f;
  
  DTYPE pix = READ_IMAGE(input,sampler, coord).x;

  output[i+Nx*j+Nx*Ny*k] = pix;
}
