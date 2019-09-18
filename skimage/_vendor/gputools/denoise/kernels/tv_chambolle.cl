#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void div_step(__read_only image3d_t input,__read_only image3d_t pDeriv,__write_only image3d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);
  
  float4 p0 = read_imagef(pDeriv,sampler,(int4)(i0,j0,k0,0));
  float4 px = read_imagef(pDeriv,sampler,(int4)(i0-1,j0,k0,0));
  float4 py = read_imagef(pDeriv,sampler,(int4)(i0,j0-1,k0,0));
  float4 pz = read_imagef(pDeriv,sampler,(int4)(i0,j0,k0-1,0));


  float pix = read_imagef(input,sampler,(int4)(i0,j0,k0,0)).x;

  // compute div 

  float div = 0.f;
  
  div += 1.f*(px.x-p0.x);
  div += 1.f*(py.y-p0.y);
  div += 1.f*(pz.z-p0.z);
  
  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(pix+div,0,0,0));

}



__kernel void grad_step(__read_only image3d_t output,__read_only image3d_t pDeriv,__write_only image3d_t pDeriv2,const float weight){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);
  
  float out0 = read_imagef(output,sampler,(int4)(i0,j0,k0,0)).x;
  float outx = read_imagef(output,sampler,(int4)(i0+1,j0,k0,0)).x;
  float outy = read_imagef(output,sampler,(int4)(i0,j0+1,k0,0)).x;
  float outz = read_imagef(output,sampler,(int4)(i0,j0,k0+1,0)).x;

  // compute gradient
  

  float4 g = (float4)(outx-out0,outy-out0,outz-out0,0);

  float norm = length(g);
  
  norm = 1.f + norm*0.5f / weight;

  float4 p = read_imagef(pDeriv,sampler,(int4)(i0,j0,k0,0));

  float4 res = (p-1.f/6.f*g)/norm;
  
  write_imagef(pDeriv2,(int4)(i0,j0,k0,0),res);

}
