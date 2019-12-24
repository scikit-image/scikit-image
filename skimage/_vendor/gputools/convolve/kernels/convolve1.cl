#ifdef SHORTTYPE
#define READ_IMAGE read_imageui
#define DTYPE short
#else
#define READ_IMAGE read_imagef
#define DTYPE float
#endif

__kernel void convolve1d(__read_only image2d_t input,__global float* h,__global float* output,const int Nx,const int Nhx){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);

  float res = 0.f;

  for (int i = 0; i < Nhx; ++i){
	float dx = -.5f*(Nhx-1)+i;
	res += h[i]*READ_IMAGE(input,sampler,(float2)(i0+dx,0)).x;
  }
  
  output[i0] = res;  

}
