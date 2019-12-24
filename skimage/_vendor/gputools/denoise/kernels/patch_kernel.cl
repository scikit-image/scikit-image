#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void project4(__read_only image2d_t input, __write_only image2d_t output, const int Nx, const int Ny, const int FSIZE)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  // the norming constants of the legendre polynomials 
  uint c0 = (2*FSIZE+1)*(2*FSIZE+1);
  uint c12 = c0*FSIZE*(FSIZE+1)/3;
  uint c3 = c12*FSIZE*(FSIZE+1)/3;

  float d0 = 0,
	d1 = 0,
	d2 = 0,
	d3 = 0;
  
  for (int i2 = -FSIZE; i2 <= FSIZE; ++i2){
	  for (int j2 = -FSIZE; j2 <= FSIZE; ++j2){
		uint pix = read_imageui(input,(int2)(i+i2,j+j2)).x; 

		d0 += 1.f*pix;
		d1 += 1.f*pix*i2;
		d2 += 1.f*pix*j2;
		d3 += 1.f*pix*i2*j2;
		
	  }
  }

  d0 *= 1.f/c0;
  d1 *= 1.f/c12;
  d2 *= 1.f/c12;
  d3 *= 1.f/c3;

  
  write_imagef(output,(int2)(i,j),(float4)(d0,d1,d2,d3));


}



__kernel void nlm2dProject(__read_only image2d_t input,__read_only image2d_t projects, __global short* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  

  uint pix0 = read_imageui(input,sampler,(int2)(i,j)).x;
  
  float res = 0;
  float sum = 0;

  float foo;
  
  uint pix1;
  uint p0, p1;

  float4 proj0 = read_imagef(projects,sampler,(int2)(i,j));

  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

	  float weight = 0;
	  float dist = 0 ;

	  float4 proj1 = read_imagef(projects,sampler,(int2)(i+i1,j+j1));

	  pix1 = read_imageui(input,sampler,(int2)(i+i1,j+j1)).x;

	  
	  dist = length(proj0-proj1)*(2.f*FSIZE+1);
	  weight = exp(-1.f/SIGMA/SIGMA*dist*dist);


	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (short)(res/sum);

  
}


__kernel void nlm2dProjectSensor(__read_only image2d_t input,__read_only image2d_t projects,__read_only image2d_t sensor, __global short* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  

  uint pix0 = read_imageui(input,sampler,(int2)(i,j)).x;
  
  float res = 0;
  float sum = 0;

  float foo;
  
  uint pix1;
  uint p0, p1;

  float4 proj0 = read_imagef(projects,sampler,(int2)(i,j));

  float sigma0 = read_imagef(sensor,sampler,(int2)(i,j)).x;

  float mySigma = SIGMA*(0.00001f+sigma0+sqrt(1.f*pix0));
  
  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

	  float weight = 0;
	  float dist = 0 ;

	  float4 proj1 = read_imagef(projects,sampler,(int2)(i+i1,j+j1));

	  pix1 = read_imageui(input,sampler,(int2)(i+i1,j+j1)).x;

	  
	  dist = length(proj0-proj1)*(2.f*FSIZE+1);
	  weight = exp(-1.f/mySigma/mySigma*dist*dist);


	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (short)(res/sum);

  
}

__kernel void patchDistance(__read_only image2d_t input,__write_only image2d_t output,  const int i0, const int j0,  const int FSIZE)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  
  float dist = 0.f;
  float dist_L1 = 0.f;

  float pix0 = read_imageui(input,sampler,(int2)(i0,j0)).x;


  float pix = read_imageui(input,sampler,(int2)(i,j)).x;

  
  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
	for(int j2 = -FSIZE;j2<=FSIZE;j2++){

	  float p0 = read_imageui(input,sampler,(int2)(i0+i2,j0+j2)).x;
	  float p1 = read_imageui(input,sampler,(int2)(i+i2,j+j2)).x;

	  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
	  //			  dist_L1 += fabs(1.f*p1-1.f*p0);
    }
  }

  
  dist *= 1./(2.f*FSIZE+1.f)/(2.f*FSIZE+1.f);

  write_imagef(output,(int2)(i,j), (float4)(dist,0,0,0));

}
