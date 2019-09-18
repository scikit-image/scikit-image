#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void mean_var_2d_float(__read_only image2d_t input,const int xStride,const int Nx,const int Ny,
								__global float * output_mean,__global float *  output_var){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);


  // calculate the mean/variance of the template/data

  float dataMean = 0.f, dataVar = 0.f;
  float tmp;
  
  for (int i = 0; i < Nx; ++i)
	for (int j = 0; j < Ny; ++j){
	  float dx = -.5f*(Nx-1)+i;
	  float dy = -.5f*(Ny-1)+j;
	  tmp = read_imagef(input,sampler,(float2)(i0+dx,j0+dy)).x;
	  dataMean += tmp;
	  dataVar += tmp*tmp;
	  	  
	}

  dataMean *= 1.f/Nx/Ny;
  dataVar = 1.f*dataVar/Nx/Ny - dataMean*dataMean; 

  output_mean[i0+xStride*j0] = dataMean;

  output_var[i0+xStride*j0] = dataVar;

}




__kernel void correlate2d_float(__read_only image2d_t input,__global float* h,const int Nx,const int Ny,__write_only image2d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);


  // calculate the mean/variance of the template/data

  float hMean = 0.f, dataMean = 0.f;
  float hVar = 0.f, dataVar = 0.f;
  float tmp;
  
  for (int i = 0; i < Nx; ++i)
	for (int j = 0; j < Ny; ++j){
	  float dx = -.5f*(Nx-1)+i;
	  float dy = -.5f*(Ny-1)+j;

	  tmp = h[i+Nx*j];
	  
	  hMean += tmp;
	  hVar += tmp*tmp;

	  tmp = read_imagef(input,sampler,(float2)(i0+dx,j0+dy)).x;
	  dataMean += tmp;
	  dataVar += tmp*tmp;
	  
	  
	}

  hMean *= 1.f/Nx/Ny;
  dataMean *= 1.f/Nx/Ny;

  hVar = 1.f*hVar/Nx/Ny - hMean*hMean; 
  dataVar = 1.f*dataVar/Nx/Ny - dataMean*dataMean; 
	
  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
	  for (int j = 0; j < Ny; ++j){

		float dx = -.5f*(Nx-1)+i;
		float dy = -.5f*(Ny-1)+j;
		
		res += (h[i+Nx*j]-hMean)*(read_imagef(input,sampler,(float2)(i0+dx,j0+dy)).x-dataMean);

	  }
  }

  
  // res *= ((dataVar*hVar)>.000000001f)?1.f/sqrt(dataVar*hVar):1.;
  
  write_imagef(output,(int2)(i0,j0),(float4)(res,0,0,0));
  

}


__kernel void correlate2d_short(__read_only image2d_t input,__global float* h,const int Nx,const int Ny,__write_only image2d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);


  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
	  for (int j = 0; j < Ny; ++j){

		float dx = -.5f*(Nx-1)+i;
		float dy = -.5f*(Ny-1)+j;
		
		res += h[i+Nx*j]*read_imageui(input,sampler,(float2)(i0+dx,j0+dy)).x;

	  }
  }
  write_imageui(output,(int2)(i0,j0),(uint4)(res,0,0,0));
  

}




__kernel void correlate3d_float(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,__write_only image3d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
  	  for (int j = 0; j < Ny; ++j){
  		for (int k = 0; k < Nz; ++k){

  		  float dx = -.5f*(Nx-1)+i;
  		  float dy = -.5f*(Ny-1)+j;
  		  float dz = -.5f*(Nz-1)+k;
		
  		  res += h[i+Nx*j+Nx*Ny*k]*read_imagef(input,sampler,(float4)(i0+dx,j0+dy,k0+dz,0)).x;
  		}
  	  }
  }
  
  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(res,0,0,0));
  

}


__kernel void correlate3d_short(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,__write_only image3d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
	  for (int j = 0; j < Ny; ++j){
		for (int k = 0; k < Nz; ++k){

		  float dx = -.5f*(Nx-1)+i;
		  float dy = -.5f*(Ny-1)+j;
		  float dz = -.5f*(Nz-1)+k;
		
		  res += h[i+Nx*j+Nx*Ny*k]*read_imageui(input,sampler,(float4)(i0+dx,j0+dy,k0+dz,0)).x;
		}
	  }
  }
  
  write_imageui(output,(int4)(i0,j0,k0,0),(uint4)(res,0,0,0));
  

}

__kernel void correlate3d_float_buf(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,const int Nx0,const int Ny0,__global float * output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
  	  for (int j = 0; j < Ny; ++j){
  		for (int k = 0; k < Nz; ++k){

  		  float dx = -.5f*(Nx-1)+i;
  		  float dy = -.5f*(Ny-1)+j;
  		  float dz = -.5f*(Nz-1)+k;
		
  		  res += h[i+Nx*j+Nx*Ny*k]*read_imagef(input,sampler,(float4)(i0+dx,j0+dy,k0+dz,0)).x;
  		}
  	  }
  }
  
  output[i0+Nx0*j0+Nx0*Ny0*k0] = res;
  

}


__kernel void correlate3d_short_buf(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,const int Nx0,const int Ny0,__global short * output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);

  

  float res = 0.f;

  for (int i = 0; i < Nx; ++i){
	  for (int j = 0; j < Ny; ++j){
		for (int k = 0; k < Nz; ++k){

		  float dx = -.5f*(Nx-1)+i;
		  float dy = -.5f*(Ny-1)+j;
		  float dz = -.5f*(Nz-1)+k;
		
		  res += h[i+Nx*j+Nx*Ny*k]*read_imageui(input,sampler,(float4)(i0+dx,j0+dy,k0+dz,0)).x;
		}
	  }
  }

  output[i0+Nx0*j0+Nx0*Ny0*k0] = (short)res ;


}


// separable versions

__kernel void correlate_sep2d_float(__read_only image2d_t input, __global float * h, const int N,__write_only image2d_t output, const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  
  const int dx = flag & 1;
  const int dy = (flag&2)/2;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
    res += h[i]*read_imagef(input,sampler,(float2)(i0+dx*j,j0+dy*j)).x;
  }

  write_imagef(output,(int2)(i0,j0),(float4)(res,0,0,0));
  
}


__kernel void correlate_sep2d_short(__read_only image2d_t input, __global float * h, const int N,__write_only image2d_t output, const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);

  
  const int dx = flag & 1;
  const int dy = (flag&2)/2;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
    res += h[i]*read_imageui(input,sampler,(float2)(i0+dx*j,j0+dy*j)).x;
  }

  write_imageui(output,(int2)(i0,j0),(uint4)(res,0,0,0));
  
}



__kernel void correlate_sep3d_float(__read_only image3d_t input, __global float * h, const int N,__write_only image3d_t output,const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  // flag = 4 -> in z axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);
  uint k0 = get_global_id(2);

  const int dx = flag & 1;
  const int dy = (flag&2)/2;
  const int dz = (flag&4)/4;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
	res += h[i]*read_imagef(input,sampler,(float4)(i0+dx*j,j0+dy*j,k0+dz*j,0)).x;
  }

  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(res,0,0,0));
  
}


__kernel void correlate_sep3d_short(__read_only image3d_t input, __global float * h, const int N,__write_only image3d_t output,const int flag){

  // flag = 1 -> in x axis 
  // flag = 2 -> in y axis 
  // flag = 4 -> in z axis 
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  uint j0 = get_global_id(1);
  uint k0 = get_global_id(2);

  const int dx = flag & 1;
  const int dy = (flag&2)/2;
  const int dz = (flag&4)/4;

  float res = 0.f;

  for (int i = 0; i < N; ++i){
	float j = i-.5f*(N-1);
	res += h[i]*read_imageui(input,sampler,(float4)(i0+dx*j,j0+dy*j,k0+dz*j,0)).x;
  }

  write_imageui(output,(int4)(i0,j0,k0,0),(uint4)(res,0,0,0));
  
}








__kernel void foo(__read_only image3d_t input,__global float* h,const int Nx,const int Ny,const int Nz,__write_only image3d_t output){

  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  int i0 = get_global_id(0);
  int j0 = get_global_id(1);
  int k0 = get_global_id(2);


  float res = 0.f;
  write_imagef(output,(int4)(i0,j0,k0,0),(float4)(i0,0,0,0));
  

}
