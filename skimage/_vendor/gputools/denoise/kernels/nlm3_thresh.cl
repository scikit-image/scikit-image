#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

#ifndef FS
#define FS 2
#endif

#ifndef BS
#define BS 3
#endif /* BS */


#define TWOFS (2*FS+1)


#ifdef FLOAT

#define dtype float
#define outtype float

#define dtype4 float4
#define read_image read_imagef
#define write_image write_imagef

#else

#define dtype uint
#define outtype short

#define dtype4 uint4
#define read_image read_imageui
#define write_image write_imageui

#endif



__kernel void nlm3_thresh(__read_only image3d_t input, __global outtype* output,
								const int Nx, const int Ny,
								const float sigma,
								const float thresh)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);
  

  float res = 0.f;
  float meanSum = 0.f, sum = 0.f;

  dtype pix;

  float patch_norm = 1.f/TWOFS/TWOFS/TWOFS;


  for(int i2 = -FS;i2<=FS;i2++){
	for(int j2 = -FS;j2<=FS;j2++){
	  for(int k2 = -FS;k2<=FS;k2++){
	
		pix = read_image(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;
		meanSum += pix;
	  }
	}
  }
  meanSum *= patch_norm;

  
  if (meanSum > thresh){


  	for(int i1 = -BS;i1<=BS;i1++){
  	  for(int j1 = -BS;j1<=BS;j1++){
  		for(int k1 = -BS;k1<=BS;k1++){
  		  float weight = 0;
  		  float dist = 0 ;

  		  pix = read_image(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
  		  for(int i2 = -FS;i2<=FS;i2++){
  			for(int j2 = -FS;j2<=FS;j2++){
  			  for(int k2 = -FS;k2<=FS;k2++){
	
  				dtype p0 = read_image(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;;

  				dtype p1 = read_image(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

  				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
  			  }
  			}
  		  }
	  
  		  weight = exp(-1.f*dist*patch_norm/sigma/sigma);
  		  res += pix*weight;
  		  sum += weight;

  		}
  	  }
  	}
  
  	output[i+j*Nx+k*Nx*Ny] = (dtype)(res/sum);

  }
  else{

  	  pix = read_image(input,sampler,(int4)(i,j,k,0)).x;
  

  	  for(int i2 = -FS;i2<=FS;i2++){
  		for(int j2 = -FS;j2<=FS;j2++){
  		  for(int k2 = -FS;k2<=FS;k2++){
	
  			dtype pix2 = read_image(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

			dtype dist = (1.f*pix2-pix)*(1.f*pix2-pix);
  			float weight = exp(-.1f*(i2*i2+j2*j2+k2*k2)-dist*patch_norm/sigma);

  			res += pix2*weight;
  			sum += weight;
  		  }
  		}
  	  }

  	  output[i+j*Nx+k*Nx*Ny] = (outtype)(res/sum);

  }

  
}



// __kernel void nlm3_thresh_float(__read_only image3d_t input, __global float* output,
// 								const int Nx, const int Ny,
// 								int FS, int BS,const float SIGMA,
// 								const float thresh)
// {
//   const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

//   uint i = get_global_id(0);
//   uint j = get_global_id(1);
//   uint k = get_global_id(2);
  

//   float res = 0.f;
//   float meanSum = 0.f, sum = 0.f;

//   float pix1;
//   float patch_norm = (1.f+2.f*FS)*(1.f+2.f*FS)*(1.f+2.f*FS);

//   for(int i2 = -FS;i2<=FS;i2++){
//   	for(int j2 = -FS;j2<=FS;j2++){
//   	  for(int k2 = -FS;k2<=FS;k2++){
	
//   		pix1 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

//   		meanSum += 1.f*pix1;
//   	  }
//   	}
//   }
//   meanSum *= 1./patch_norm;

//   if (meanSum > thresh){

//   	for(int i1 = -BS;i1<=BS;i1++){
//   	  for(int j1 = -BS;j1<=BS;j1++){
//   		for(int k1 = -BS;k1<=BS;k1++){
//   		  float weight = 0;
//   		  float dist = 0 ;

//   		  pix1 = read_imagef(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
//   		  for(int i2 = -FS;i2<=FS;i2++){
//   			for(int j2 = -FS;j2<=FS;j2++){
//   			  for(int k2 = -FS;k2<=FS;k2++){
	
//   				float p0 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

//   				float p1 = read_imagef(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

//   				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
//   			  }
//   			}
//   		  }
	  
//   		  weight = exp(-1.f/SIGMA/SIGMA*dist/patch_norm);
//   		  res += pix1*weight;
//   		  sum += weight;

//   		}
//   	  }
//   	}
  
//   	output[i+j*Nx+k*Nx*Ny] = res/sum;

//   }
//   else{


//   	  pix1 = read_imagef(input,sampler,(int4)(i,j,k,0)).x;
  

//   	  for(int i2 = -FS;i2<=FS;i2++){
//   	  	for(int j2 = -FS;j2<=FS;j2++){
//   	  	  for(int k2 = -FS;k2<=FS;k2++){
	
//   	  		float pix2 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

			 
//   	  		float weight = exp(-.1f*(i2*i2+j2*j2+k2*k2)-1.f/SIGMA/SIGMA*(1.f*pix2-pix1)*(1.f*pix2-pix1)/patch_norm);

//   	  		res += pix2*weight;
//   	  		sum += weight;
//   	  	  }
//   	  	}
//   	  }
	  
//   	  output[i+j*Nx+k*Nx*Ny] = res/sum;
	  
	
//   }
  
  
// }



// __kernel void nlm3_thresh_mean_float(__read_only image3d_t input, __global float* output,
// 								const int Nx, const int Ny,
// 								int FS, int BS,const float SIGMA,
// 								const float thresh)
// {
//   const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

//   uint i = get_global_id(0);
//   uint j = get_global_id(1);
//   uint k = get_global_id(2);
  

//   float res = 0.f;
//   float meanSum = 0.f, sum = 0.f;

//   float pix1;

//   for(int i2 = -FS;i2<=FS;i2++){
//   	for(int j2 = -FS;j2<=FS;j2++){
//   	  for(int k2 = -FS;k2<=FS;k2++){
	
//   		pix1 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

//   		meanSum += 1.f*pix1;
//   	  }
//   	}
//   }

//   meanSum *= 1./(1.f+2.f*FS)/(1.f+2.f*FS)/(1.f+2.f*FS);

//   if (meanSum > thresh){

//   	for(int i1 = -BS;i1<=BS;i1++){
//   	  for(int j1 = -BS;j1<=BS;j1++){
//   		for(int k1 = -BS;k1<=BS;k1++){
//   		  float weight = 0;
//   		  float dist = 0 ;

//   		  pix1 = read_imagef(input,sampler,(int4)(i+i1,j+j1,k+k1,0)).x;

	  
//   		  for(int i2 = -FS;i2<=FS;i2++){
//   			for(int j2 = -FS;j2<=FS;j2++){
//   			  for(int k2 = -FS;k2<=FS;k2++){
	
//   				float p0 = read_imagef(input,sampler,(int4)(i+i2,j+j2,k+k2,0)).x;

//   				float p1 = read_imagef(input,sampler,(int4)(i+i1+i2,j+j1+j2,k+k1+k2,0)).x;

//   				dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/(1.f+2.f*FS)/(1.f+2.f*FS);
//   			  }
//   			}
//   		  }
	  
//   		  weight = exp(-1.f/SIGMA/SIGMA*dist);
//   		  res += pix1*weight;
//   		  sum += weight;

//   		}
//   	  }
//   	}
  
//   	output[i+j*Nx+k*Nx*Ny] = res/sum;

//   }
//   else{


// 	output[i+j*Nx+k*Nx*Ny] = meanSum;
	
//   }
  
  
// }
