
__kernel void bilat3_float(__read_only image3d_t input, __global float* output,const int Nx, const int Ny,const int fSize, const float sigmaX,const float sigmaP)
{

    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float pix0 = read_imagef(input,sampler,(int4)(i,j,k,0)).x;
  
  float res = 0;
  float sum = 0;


  float SX = 1.f/sigmaX/sigmaX;
  float SP = 1.f/sigmaP/sigmaP;
  

  for(int i2 = -fSize;i2<=fSize;i2++){
	for(int j2 = -fSize;j2<=fSize;j2++){
	  for(int k2 = -fSize;k2<=fSize;k2++){

		float pix1 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;
		float weight = exp(-SX*(i2*i2+j2*j2+k2*k2))*
		  exp(-SP*((1.f*pix0-pix1)*(1.f*pix0-pix1)));
		res += pix1*weight;
		sum += weight;

		res += pix1*weight;
		sum += weight;
	  }
	}
  }
  
  output[i+j*Nx+k*Nx*Ny] = res/sum;

}

