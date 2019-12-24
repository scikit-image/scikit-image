#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void nlm3_short(__read_only image3d_t input, __global short* output,const int Nx, const int Ny,
					   const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  uint pix0 = read_imageui(input,sampler,(int4)(i,j,k,0)).x;
  
  float res = 0;
  float sum = 0;

  uint pix1;


  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){
  	  for(int k1 = -BSIZE;k1<=BSIZE;k1++){
  		float weight = 0;
  		float dist = 0 ;

  		pix1 = read_imageui(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  		  for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  			for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  			  uint p0 = read_imageui(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  			  uint p1 = read_imageui(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  			  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);
  			}
  		  }
  		}
	  
  		weight = exp(-1.f/SIGMA/SIGMA*dist);
  		res += pix1*weight;
  		sum += weight;

  	  }
  	}
  }
  
  output[i+j*Nx+k*Nx*Ny] = (short)(res/sum);

  
}

__kernel void nlm3_float(__read_only image3d_t input, __global float* output,const int Nx, const int Ny,
					   const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float pix0 = read_imagef(input,sampler,(int4)(i,j,k,0)).x;
  
  float res = 0;
  float sum = 0;

  float pix1;


  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){
  	  for(int k1 = -BSIZE;k1<=BSIZE;k1++){
  		float weight = 0;
  		float dist = 0 ;

  		pix1 = read_imagef(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  		  for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  			for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  			  float p0 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  			  float p1 = read_imagef(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  			  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);
  			}
  		  }
  		}
	  
  		weight = exp(-1.f/SIGMA/SIGMA*dist);
  		res += pix1*weight;
  		sum += weight;

  	  }
  	}
  }
  
  output[i+j*Nx+k*Nx*Ny] = 1.*res/sum;


}



__kernel void nlm3_thresh_short(__read_only image3d_t input, __global short* output,
								const int Nx, const int Ny,
								int FSIZE, int BSIZE,const float SIGMA,
								const float thresh)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float res = 0.f;
  float meanSum = 0.f, sum = 0.f;

  uint pix1;
  float patch_norm = 1./(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);
  
  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
	for(int j2 = -FSIZE;j2<=FSIZE;j2++){
	  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
		pix1 = read_imageui(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

		meanSum = 1.f*pix1;///(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);;
	  }
	}
  }
  meanSum *= 1./(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);

  if (meanSum > thresh){

  	for(int i1 = -BSIZE;i1<=BSIZE;i1++){
  	  for(int j1 = -BSIZE;j1<=BSIZE;j1++){
  		for(int k1 = -BSIZE;k1<=BSIZE;k1++){
  		  float weight = 0;
  		  float dist = 0 ;

  		  pix1 = read_imageui(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  			for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  			  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  				uint p0 = read_imageui(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  				uint p1 = read_imageui(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);
  			  }
  			}
  		  }
	  
  		  weight = exp(-1.f/SIGMA/SIGMA*dist);
  		  res += pix1*weight;
  		  sum += weight;

  		}
  	  }
  	}
  
  	output[i+j*Nx+k*Nx*Ny] = (short)(res/sum);
	// output[i+j*Nx+k*Nx*Ny] = (short)(100);

  }
  else{

	  pix1 = read_imageui(input,sampler,(int4)(i,j,k,0)).x;
  

	  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
		for(int j2 = -FSIZE;j2<=FSIZE;j2++){
		  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
			uint pix2 = read_imageui(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

			 
			float weight = exp(-.1f*(i2*i2+j2*j2+k2*k2)-1.f/SIGMA/SIGMA*(1.f*pix2-pix1)*(1.f*pix2-pix1));
			// float weight = exp(-.1f/SIGMA/SIGMA*(1.f*pix2-pix1)*(1.f*pix2-pix1));

			// float weight = 1.;

			res += pix2*weight;
			sum += weight;
		  }
		}
	  }

	  // res = meanSum;
	  // sum = 1.f;
	  output[i+j*Nx+k*Nx*Ny] = (short)(res/sum);

  }

  
}



__kernel void nlm3_thresh_float(__read_only image3d_t input, __global float* output,
								const int Nx, const int Ny,
								int FSIZE, int BSIZE,const float SIGMA,
								const float thresh)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float res = 0.f;
  float meanSum = 0.f, sum = 0.f;

  float pix1;
  float patch_norm = (1.f+2.f*FSIZE)*(1.f+2.f*FSIZE)*(1.f+2.f*FSIZE);

  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  	for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  	  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  		pix1 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  		meanSum += 1.f*pix1;
  	  }
  	}
  }
  meanSum *= 1./patch_norm;

  if (meanSum > thresh){

  	for(int i1 = -BSIZE;i1<=BSIZE;i1++){
  	  for(int j1 = -BSIZE;j1<=BSIZE;j1++){
  		for(int k1 = -BSIZE;k1<=BSIZE;k1++){
  		  float weight = 0;
  		  float dist = 0 ;

  		  pix1 = read_imagef(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  			for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  			  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  				float p0 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  				float p1 = read_imagef(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
  			  }
  			}
  		  }
	  
  		  weight = exp(-1.f/SIGMA/SIGMA*dist/patch_norm);
  		  res += pix1*weight;
  		  sum += weight;

  		}
  	  }
  	}
  
  	output[i+j*Nx+k*Nx*Ny] = res/sum;

  }
  else{


  	  pix1 = read_imagef(input,sampler,(int4)(i,j,k,0)).x;
  

  	  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  	  	for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  	  	  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  	  		float pix2 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

			 
  	  		float weight = exp(-.1f*(i2*i2+j2*j2+k2*k2)-1.f/SIGMA/SIGMA*(1.f*pix2-pix1)*(1.f*pix2-pix1)/patch_norm);

  	  		res += pix2*weight;
  	  		sum += weight;
  	  	  }
  	  	}
  	  }
	  
  	  output[i+j*Nx+k*Nx*Ny] = res/sum;
	  
	
  }
  
  
}



__kernel void nlm3_thresh_mean_float(__read_only image3d_t input, __global float* output,
								const int Nx, const int Ny,
								int FSIZE, int BSIZE,const float SIGMA,
								const float thresh)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float res = 0.f;
  float meanSum = 0.f, sum = 0.f;

  float pix1;

  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  	for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  	  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  		pix1 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  		meanSum += 1.f*pix1;
  	  }
  	}
  }

  meanSum *= 1./(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);

  if (meanSum > thresh){

  	for(int i1 = -BSIZE;i1<=BSIZE;i1++){
  	  for(int j1 = -BSIZE;j1<=BSIZE;j1++){
  		for(int k1 = -BSIZE;k1<=BSIZE;k1++){
  		  float weight = 0;
  		  float dist = 0 ;

  		  pix1 = read_imagef(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  			for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  			  for(int k2 = -FSIZE;k2<=FSIZE;k2++){
	
  				float p0 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

  				float p1 = read_imagef(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/(1.f+2.f*FSIZE)/(1.f+2.f*FSIZE);
  			  }
  			}
  		  }
	  
  		  weight = exp(-1.f/SIGMA/SIGMA*dist);
  		  res += pix1*weight;
  		  sum += weight;

  		}
  	  }
  	}
  
  	output[i+j*Nx+k*Nx*Ny] = res/sum;

  }
  else{


	output[i+j*Nx+k*Nx*Ny] = meanSum;
	
  }
  
  
}
