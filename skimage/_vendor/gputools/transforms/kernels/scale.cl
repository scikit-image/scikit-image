
#ifndef SAMPLER_FILTER
#define SAMPLER_FILTER CLK_FILTER_LINEAR
#endif




__kernel void scale(__read_only image3d_t input, __global TYPENAME* output)
{

    const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	SAMPLER_FILTER;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  //ensure correct sampling, see opencl 1.2 specification pg. 329
  float x = i + 0.5f;
  float y = j + 0.5f;
  float z = k + 0.5f;


  /*TYPENAME pix = READ_IMAGE(input,sampler,(float4)(1.f*x/(Nx-1.f),
						 1.f*y/(Ny-1.f),
						 1.f*z/(Nz-1.f),0)).x;
    */

  TYPENAME pix = READ_IMAGE(input,sampler,(float4)(1.f*x/Nx,
						 1.f*y/Ny,
						 1.f*z/Nz,0)).x;

  output[i+Nx*j+Nx*Ny*k] = pix;
  

}

// the 1D cubic filter weights
inline float3 _w0(float3 a){
    return (-a*a*a+3*a*a-3*a+1)/6.f;
}
inline float3 _w1(float3 a){
    return (3*a*a*a-6*a*a+4)/6.f;
}
inline float3 _w2(float3 a){
    return (-3*a*a*a+3*a*a+3*a+1)/6.f;
}
inline float3 _w3(float3 a){
    return a*a*a/6.f;
}

__kernel void scale_bicubic(__read_only image3d_t input, __global TYPENAME* output)
{

     const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |
      CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_LINEAR;


    uint i = get_global_id(0);
    uint j = get_global_id(1);
    uint k = get_global_id(2);

    uint Nx = get_global_size(0);
    uint Ny = get_global_size(1);
    uint Nz = get_global_size(2);


    float3 coord = (float3)(i,j,k);

    float3 size = (float3)(Nx,Ny,Nz);

    const float3 coord_tex = (float3)(coord) - 0.5f;
    const float3 coord_int = floor(coord_tex);
	const float3 coord_frac = coord_tex - coord_int;
	float3 w0 = _w0(coord_frac);
	float3 w1 = _w1(coord_frac);
	float3 w2 = _w2(coord_frac);
	float3 w3 = _w3(coord_frac);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;

	const float3 h0 = ((w1 / g0) - 0.5f + coord_int)/size;
	const float3 h1 = ((w3 / g1) + 1.5f + coord_int)/size;



}
