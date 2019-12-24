
__kernel void bilat2_short(__read_only image2d_t input, __global short* output,const int Nx, const int Ny,const int fSize, const float sigmaX,const float sigmaP)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  float res = 0;
  float sum = 0;
  uint pix0 = read_imageui(input,sampler,(int2)(i,j)).x;
  float SX = 1.f/sigmaX/sigmaX;
  float SP = 1.f/sigmaP/sigmaP;

  for(int k = -fSize;k<=fSize;k++){
    for(int m = -fSize;m<=fSize;m++){

    uint pix1 = read_imageui(input,sampler,(int2)(i+k,j+m)).x;
    float weight = exp(-SX*(k*k+m*m))*
	  exp(-SP*((1.f*pix0-pix1)*(1.f*pix0-pix1)));
    res += pix1*weight;
    sum += weight;

    }
  }

  output[i+Nx*j] = (short)(res/sum);
}

__kernel void bilat2_float(__read_only image2d_t input, __global float* output,const int Nx, const int Ny,const int fSize, const float sigmaX,const float sigmaP)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  float res = 0;
  float sum = 0;
  float pix0 = read_imagef(input,sampler,(int2)(i,j)).x;

  float SX = 1.f/sigmaX/sigmaX;
  float SP = 1.f/sigmaP/sigmaP;
  
  for(int k = -fSize;k<=fSize;k++){
    for(int m = -fSize;m<=fSize;m++){

    float pix1 = read_imagef(input,sampler,(int2)(i+k,j+m)).x;
    float weight = exp(-SX*(k*k+m*m))*
	  exp(-SP*((1.f*pix0-pix1)*(1.f*pix0-pix1)));
    res += pix1*weight;
    sum += weight;

    }
  }

  output[i+Nx*j] = res/sum;
}

