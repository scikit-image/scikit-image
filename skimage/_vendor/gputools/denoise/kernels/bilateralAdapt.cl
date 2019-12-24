
__kernel void run2d(__read_only image2d_t input, __read_only image2d_t sensor, __global short* output,const int Nx, const int Ny,const int fSize, const float sigmaX, const float fac)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  
  const sampler_t samplerSensor = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  float res = 0;
  float sum = 0;
  uint pix0 = read_imageui(input,sampler,(int2)(i,j)).x;

  
  float sigmaP0 = 0.0001f +  read_imageui(sensor,samplerSensor,(int2)(i,j)).x;;

  float sigmaP = sigmaP0;
  
  for(int k = -fSize;k<=fSize;k++){
    for(int m = -fSize;m<=fSize;m++){

    uint pix1 = read_imageui(input,sampler,(int2)(i+k,j+m)).x;
	float sigmaP1 = 0.0001f +  read_imageui(sensor,samplerSensor,(int2)(i+k,j+m)).x;;

	sigmaP = fac*(sigmaP0 + sigmaP1);
    float weight = exp(-1.f/sigmaX/sigmaX*(k*k+m*m))*
	  exp(-1.f/sigmaP/sigmaP*((1.f*pix0-pix1)*(1.f*pix0-pix1)));

	res += pix1*weight;
    sum += weight;

    }
  }

  output[i+Nx*j] = (short)(res/sum);


  
}

