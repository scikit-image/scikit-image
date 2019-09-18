#define FSIZE 2
#define GSIZE 16
#define LSIZE (GSIZE+2*FSIZE)



__kernel void run(__read_only image2d_t input, __global short* output,const int Nx, const int Ny, const float SIGMA)
{
  
  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |	CLK_ADDRESS_CLAMP_TO_EDGE |	CLK_FILTER_NEAREST ;

  
  uint i = get_global_id(0);
  uint j = get_global_id(1);

  uint iGroup = get_local_id(0);
  uint jGroup = get_local_id(1);

  uint index = i+j*Nx;
  uint indexGroup  = iGroup+jGroup*GSIZE;
  
  int2 pos0 = (int2)(i,j);


  const int ASTRIDE = ceil(1.f*LSIZE*LSIZE/GSIZE/GSIZE);

  __local float aLoc[LSIZE*LSIZE];

  for (int k = 0; k < ASTRIDE; ++k){
	uint indexLoc = indexGroup*ASTRIDE+k;
	uint pix = read_imageui(input,sampler,(int2)(k,k)).x;
	if (indexLoc<(LSIZE*LSIZE))
	  aLoc[indexLoc] = 1.f*pix;
		
  }

  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  
  float res = 0;
  float sum = 0;

  for(int k = -FSIZE;k<=FSIZE;k++){
    for(int m = -FSIZE;m<=FSIZE;m++){

	  float pix1 = aLoc[index%(LSIZE*LSIZE)];
	  float weight = exp(-1.f/SIGMA/SIGMA*(k*k+m*m));
	  res += pix1*weight;
	  sum += weight;

    }
  }

  
  
  // const uint LOCSTRIDE = ceil(1.f*LOCSIZE*LOCSIZE/GROUPSIZE/GROUPSIZE);

  // uint4 pix;
  // uint indexGroup = iGroup + jGroup * GROUPSIZE;  

  // uint count = 0;

  // aLoc[0] = 10;
  
  // for (uint k = 0; k < LOCSTRIDE; ++k){
  //   uint indexLoc = k + indexGroup * LOCSTRIDE;
  // 	uint iLoc = indexLoc%LOCSIZE;
  // 	uint jLoc = (indexLoc -i)/LOCSIZE;
	
  // 	if (indexLoc< (LOCSIZE*LOCSIZE)){
  // 	  pos1.x = i-iLoc;
  // 	  pos1.y = j-jLoc;
	  	  
  // 	  pix = read_imageui(input,sampler,pos1);
  // 	  aLoc[indexLoc] = pix.x;
  // 	  count +=1;
  // 	}

  // }

  // barrier(CLK_LOCAL_MEM_FENCE);
  
  // output[index] = aLoc[0];


}
