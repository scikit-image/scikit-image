#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

#ifndef FS
#define FS 3
#endif

#ifndef BS
#define BS 4
#endif /* BS */

#define NPATCH ((2*FS+1)*(2*FS+1))


__kernel void dist(__read_only image2d_t input,__write_only image2d_t output, const int dx,const int dy){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  float pix1  = read_imagef(input,sampler,(int2)(i0,j0)).x;
  float pix2  = read_imagef(input,sampler,(int2)(i0+dx,j0+dy)).x;

  float d = (pix1-pix2);

  d *= d;
	
  write_imagef(output,(int2)(i0,j0),(float4)(d));
  

}


__kernel void convolve(__read_only image2d_t input, __write_only image2d_t output, const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  const int dx = flag & 1;
  const int dy = (flag&2)/2;

  float res = 0.f;

  for (int i = -FS; i <= FS; ++i)
    res += read_imagef(input,sampler,(int2)(i0+dx*i,j0+dy*i)).x;

  write_imagef(output,(int2)(i0,j0),(float4)(res,0,0,0));
  
}

	

__kernel void computePlus(__read_only image2d_t input,__read_only image2d_t distImg,
						  __global float* accBuf,__global float* weightBuf,
						  const int Nx,const int Ny,  const int dx,const int dy, const float sigma){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  float dist  = read_imagef(distImg,sampler,(int2)(i0,j0)).x;

  float pix  = read_imagef(input,sampler,(int2)(i0+dx,j0+dy)).x;

  
  float weight = exp(-1.f*dist/NPATCH/sigma/sigma);

  accBuf[i0+Nx*j0] += (float)(weight*pix);
  weightBuf[i0+Nx*j0] += (float)(weight);


}




__kernel void computeMinus(__read_only image2d_t input,__read_only image2d_t distImg,
						   __global float* accBuf,__global float* weightBuf,
						   const int Nx,const int Ny, const int dx,const int dy, const float sigma){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  float dist  = read_imagef(distImg,sampler,(int2)(i0-dx,j0-dy)).x;

  float pix  = read_imagef(input,sampler,(int2)(i0-dx,j0-dy)).x;

  float Npatch = (2.f*FS+1.f)*(2.f*FS+1.f);
  
  float weight = exp(-1.f*dist/NPATCH/sigma/sigma);

  accBuf[i0+Nx*j0] += (float)(weight*pix);
  weightBuf[i0+Nx*j0] += (float)(weight);
}
