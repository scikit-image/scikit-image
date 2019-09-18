#pragma OPENCL EXTENSION cl_khr_3d_image_writes : enable

__kernel void run2d_short(__read_only image2d_t input, __global short* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
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

  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

	  float weight = 0;
	  float dist = 0 ;

	  pix1 = read_imageui(input,sampler,(int2)(i+i1,j+j1)).x;

	  
	  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
		for(int j2 = -FSIZE;j2<=FSIZE;j2++){
		  p0 = read_imageui(input,sampler,(int2)(i+i2,j+j2)).x;

		  p1 = read_imageui(input,sampler,(int2)(i+i1+i2,j+j1+j2)).x;

		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
		  
		}
	  }

	  dist *= 1./(2*FSIZE+1)/(2*FSIZE+1);

	  // float distmax = .5f*SIGMA*SIGMA;
	  // dist = max(0.f,dist-distmax);
	  
	  weight = exp(-1.f/SIGMA/SIGMA*dist);
	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (short)(res/sum);

  
}


__kernel void run2d_float(__read_only image2d_t input, __global float* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  

  float pix0 = read_imageui(input,sampler,(int2)(i,j)).x;
  
  float res = 0;
  float sum = 0;

  float foo;
  
  float pix1;
  float p0, p1;

  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

	  float weight = 0;
	  float dist = 0 ;

	  pix1 = read_imagef(input,sampler,(int2)(i+i1,j+j1)).x;

	  
	  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
		for(int j2 = -FSIZE;j2<=FSIZE;j2++){
		  p0 = read_imagef(input,sampler,(int2)(i+i2,j+j2)).x;

		  p1 = read_imagef(input,sampler,(int2)(i+i1+i2,j+j1+j2)).x;

		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
		  
		}
	  }

	  dist *= 1./(2*FSIZE+1)/(2*FSIZE+1);

	  // float distmax = .5f*SIGMA*SIGMA;
	  // dist = max(0.f,dist-distmax);
	  
	  weight = exp(-1.f/SIGMA/SIGMA*dist);
	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (float)(res/sum);

  
}



#ifndef FS
#define FS 3
#endif

__kernel void run2dTest(__read_only image2d_t input, __global short* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
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

  uint pixels0[2*FS+1][2*FS+1];

  for(int i2 = -FS;i2<=FS;i2++){
	for(int j2 = -FS;j2<=FS;j2++){
	  pixels0[i2+FS][j2+FS] = read_imageui(input,sampler,(int2)(i+i2,j+j2)).x;
	}
  }
  
  
  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

	  float weight = 0;
	  float dist = 0 ;
	  
	  pix1 = read_imageui(input,sampler,(int2)(i+i1,j+j1)).x;

	  
	  for(int i2 = -FS;i2<=FS;i2++){
		for(int j2 = -FS;j2<=FS;j2++){
		  p0 = pixels0[i2+FS][j2+FS];
		  p1 = read_imageui(input,sampler,(int2)(i+i1+i2,j+j1+j2)).x;
		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/FS/FS;
		  
		}
	  }

	  // float distmax = .5f*SIGMA*SIGMA;
	  // dist = max(0.f,dist-distmax);
	  
	  weight = exp(-1.f/SIGMA/SIGMA*dist);
	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (short)(res/sum);

  
}



__kernel void run2dBuf(__global short * input, __global short* output,const int Nx, const int Ny, const int FSIZE, const int BSIZE,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i0 = get_global_id(0);
  
  uint i = i0 % Nx;
  uint j = i0 / Nx;
  

  uint pix0 = input[i+j*Nx];
  
  float res = 0;
  float sum = 0;

  float foo;
  
  uint pix1;
  uint p0, p1;

  for(int i1 = -BSIZE;i1<=BSIZE;i1++){
    for(int j1 = -BSIZE;j1<=BSIZE;j1++){

  	  float weight = 0;
  	  float dist = 0 ;


  	  pix1 = input[(i+i1)+(j+j1)*Nx];
	  
  	  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
  		for(int j2 = -FSIZE;j2<=FSIZE;j2++){
  		  p0 = input[(i+i2)+(j+j2)*Nx];
  		  p1 = input[(i+i1+i2)+(j+j1+j2)*Nx];

  		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/FSIZE/FSIZE;
  		}
  	  }

  	  // float distmax = .5f*SIGMA*SIGMA;
  	  // dist = max(0.f,dist-distmax);
	  
  	  weight = exp(-1.f/SIGMA/SIGMA*dist);
  	  res += 1.f*pix1*weight;
  	  sum += weight;
  	}
  }



  //  output[i+j*Nx] = (short)(res/sum);

  output[i0] = (short)(pix0);


  
}




__kernel void run3d(__read_only image3d_t input, __global short* output,const int Nx, const int Ny, const int Nz, const int FSIZE, const int BSIZE,const float SIGMA)
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

			  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/FSIZE/FSIZE;
			}
		  }
		}
		float distmax = .5f*SIGMA*SIGMA;
		// dist = max(0.f,dist-distmax);
	  
		weight = exp(-1.f/SIGMA/SIGMA*dist);
		res += pix1*weight;
		sum += weight;

	  }
	}
  }
  
  output[i+j*Nx+k*Nx*Ny] = (short)(res/sum);


  
}



#ifndef FS
#define FS 3
#endif
#ifndef BS
#define BS 5
#endif

#define AS (BS+FS)

__kernel void run2d_FIXED(__read_only image2d_t input, __global short* output,const int Nx, const int Ny ,const float SIGMA)
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


  uint localPix[2*AS+1][2*AS+1];

  for (int j0 = -AS; j0 <= AS; ++j0){

	for (int i0 = -AS; i0 <= AS; ++i0){
  	  localPix[AS+i0][AS+j0] = read_imageui(input,sampler,(int2)(i+i0,j+j0)).x;
  	}
    
  }
  
  for(int i1 = -BS;i1<=BS;i1++){
    for(int j1 = -BS;j1<=BS;j1++){

	  float weight = 0;
	  float dist = 0 ;

	  pix1 = localPix[AS+i1][AS+j1];

	  
	  for(int i2 = -FS;i2<=FS;i2++){
		for(int j2 = -FS;j2<=FS;j2++){
		  p0 = localPix[AS+i2][AS+j2];
		  
		  p1 = localPix[AS+i2+i1][AS+j2+j1];
		  
		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/FS/FS;
		}
	  }

	  // float distmax = .5f*SIGMA*SIGMA;
	  // dist = max(0.f,dist-distmax);
	  
	  weight = exp(-1.f/SIGMA/SIGMA*dist);
	  res += 1.f*pix1*weight;
	  sum += weight;
	}
  }
  output[i+j*Nx] = (short)(res/sum);

  
}



#ifndef FS
#define FS 3
#endif
#ifndef BS
#define BS 5
#endif
#ifndef GS
#define GS 16
#endif

#define LOCSIZE (2*(BS+FS)+GS)

__kernel void run2d_SHARED(__read_only image2d_t input, __global short* output,const int Nx, const int Ny ,const float SIGMA)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint iLoc = get_local_id(0);
  uint jLoc = get_local_id(1);

  uint i0 = i-iLoc;
  uint j0 = j-jLoc;
  

  __local uint localPix[LOCSIZE][LOCSIZE];

  for (int k = 0; k < LOCSIZE/GS+1; ++k){
	if (iLoc+k*GS<LOCSIZE){
	  for (int m = 0; m < LOCSIZE/GS+1; ++m){
		if (jLoc+m*GS<LOCSIZE){
	
		  localPix[iLoc+k*GS][jLoc+m*GS] = read_imageui(input,sampler,(int2)(i-BS-FS+k*GS,j-BS-FS+m*GS)).x;
		};
	  }
	}
  }


  barrier(CLK_LOCAL_MEM_FENCE);


  float res = 0;
  float sum = 0;

  uint pix1;
  uint p0, p1;


  
  for(int i1 = -BS;i1<=BS;i1++){
    for(int j1 = -BS;j1<=BS;j1++){

  	  float weight = 0;
  	  float dist = 0 ;

  	  pix1 = localPix[iLoc+ BS+FS + i1][jLoc+ BS+FS + j1];
	  
	  
  	  for(int i2 = -FS;i2<=FS;i2++){
  		for(int j2 = -FS;j2<=FS;j2++){
  		  p0 = localPix[iLoc+ BS+FS + i2][jLoc+ BS+FS + j2];
  		  p1 = localPix[iLoc+ BS+FS + i1 + i2][jLoc+ BS+FS +j1 + j2];
	  
		  
  		  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0)/FS/FS;
  		}
  	  }

  	  // float distmax = .5f*SIGMA*SIGMA;
  	  // dist = max(0.f,dist-distmax);
	  
  	  weight = exp(-1.f/SIGMA/SIGMA*dist);
  	  res += 1.f*pix1*weight;
  	  sum += weight;
  	}
  }

  

  output[i+j*Nx] = (short)(res/sum);


  // output[i+j*Nx] = (short)(localPix[iLoc+BS+FS][jLoc+BS+FS]);

  // output[i+j*Nx] = (short)(localPix[(iLoc*iLoc+1049*jLoc) % LOCSIZE][(jLoc*iLoc+1049*jLoc) % LOCSIZE]);


  
}



__kernel void patchDistance(__read_only image2d_t input,__write_only image2d_t output,  const int i0, const int j0,  const int FSIZE)
{
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  
  float dist = 0.f;
  float dist_L1 = 0.f;
    
  uint pix0 = read_imageui(input,sampler,(int2)(i,j)).x;
			
  
  for(int i2 = -FSIZE;i2<=FSIZE;i2++){
	for(int j2 = -FSIZE;j2<=FSIZE;j2++){

	  float p0 = read_imageui(input,sampler,(int2)(i0+i2,j0+j2)).x;
	  float p1 = read_imageui(input,sampler,(int2)(i+i2,j+j2)).x;

	  dist += (1.f*p1-1.f*p0)*(1.f*p1-1.f*p0);
	  //			  dist_L1 += fabs(1.f*p1-1.f*p0);
    }
  }

  dist *= 1./(2*FSIZE+1)/(2*FSIZE+1);

  write_imagef(output,(int2)(i,j), (float4)(dist,0,0,0));

}





