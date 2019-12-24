#ifdef SHORTTYPE
#define READ_IMAGE read_imageui
#define DTYPE short
#else
#define READ_IMAGE read_imagef
#define DTYPE float
#endif

__kernel void convolve2d(__read_only image2d_t input,__global float* h,__global float* output,const int Nx,const int Ny,const int Nhx,const int Nhy){
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);


  float res = 0.f;

  for (int i = 0; i < Nhx; ++i){
	  for (int j = 0; j < Nhy; ++j){

		float dx = -.5f*(Nhx-1)+i;
		float dy = -.5f*(Nhy-1)+j;
		
		res += h[i+Nhx*j]*READ_IMAGE(input,sampler,(float2)(i0+dx,j0+dy)).x;

	  }
  }
  output[i0+j0*Nx] = res;
}


