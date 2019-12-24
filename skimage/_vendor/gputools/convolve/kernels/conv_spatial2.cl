#include <pyopencl-complex.h>

// 2d function

#ifndef ADDRESS_MODE
#define ADDRESS_MODE CLK_ADDRESS_CLAMP
#endif

void kernel mult_inplace(__global cfloat_t *dest, const __global cfloat_t *src){

    uint i = get_global_id(0);
    dest[i] = cfloat_mul(dest[i],src[i]);
}




void kernel fill_patch2(read_only image2d_t src,
							 const int offset_x, const int offset_y,
							 __global cfloat_t *dest, const int offset_dest){


  const sampler_t sampler = ADDRESS_MODE |  CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);


  float val = read_imagef(src,
  						   sampler,
						  (int2)(i+offset_x,j+offset_y)).x;
  
  dest[i+Nx*j+offset_dest] = cfloat_new(val,0.f);

 
}



void kernel fill_patch2_buf( __global float * src,
						   const int src_Nx, const int src_Ny,
						   const int offset_x, const int offset_y,
						   __global cfloat_t *dest, const int offset_dest){

  const sampler_t sampler = ADDRESS_MODE |  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  int i2 = i+offset_x;
  int j2 = j+offset_y;
  

  //clamp to boundary
  float val = ((i2>=0)&&(i2<src_Nx)&&(j2>=0)&&(j2<src_Ny))?src[i2+src_Nx*j2]:0.f;
  
  dest[i+Nx*j+offset_dest] = cfloat_new(val,0.f);

}

void kernel fill_psf_grid2(__global float * src,
						   const int Nx_src,
                            const int offset_x_src,
                            const int offset_y_src,
                             __global cfloat_t *dest,
                             const int Nx_dest,
                             const int Ny_dest,
							 const int offset_x_dest,
							 const int offset_y_dest,
							 const int offset_dest){

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  uint Nx = get_global_size(0);

  int index_src = i+offset_x_src+Nx_src*(j+offset_y_src);
  float val = src[index_src];


  int i_dest = i+offset_x_dest;
  int j_dest = j+offset_y_dest;

  // do a fftshift on the go, as we read very out of order anyway...
  if (i_dest<Nx_dest/2)
    i_dest += Nx_dest/2;
  else
    i_dest = i_dest - Nx_dest/2;

  if (j_dest<Ny_dest/2)
    j_dest += Ny_dest/2;
  else
    j_dest = j_dest - Ny_dest/2;

  int index_dest = i_dest+Nx_dest*j_dest+offset_dest;

  dest[index_dest] = cfloat_new(val,0.f);
}



//2d
void kernel interpolate2( __global cfloat_t * src,
						 __global float * dest,
						 const int x0,
						 const int y0,
						 const int Gx,
						 const int Gy,
						 const int Npatch_x,
						 const int Npatch_y){

  // src are the padded patches
  // dest is the actual img to fill (block by block)
  // the kernels runs over the blocksize
  // x0,y0 are the first dims of the patch buffer to interpolate x0 --> x0+1


  int i = get_global_id(0);
  int j = get_global_id(1);

  //the Nblock sizes
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  // relative coords within image block
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);


  // the coordinates in the image
  int i_im = i+x0*Nx-Nx/2;
  int j_im = j+y0*Ny-Ny/2;

  int index_im = i_im+j_im*Gx*Nx;

    // the index in the patches
  int stride1 = Npatch_x*Npatch_y;
  int stride2 = Npatch_x*Npatch_y*Gx;


  int index11 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2)+
                stride1*(x0-1)+stride2*(y0-1);

  int index12 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2)+
                stride1*x0+stride2*(y0-1);

  int index21 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                stride1*(x0-1)+stride2*y0;

  int index22 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                stride1*x0+stride2*y0;

  
  //interpolation weights

  float a11 = ((x0>0)&&(y0>0))?(1.f-_x)*(1.f-_y):0;
  float a12 = ((x0<Gx)&&(y0>0))?_x*(1.f-_y):0;
  float a21 = ((x0>0)&&(y0<Gy))?(1.f-_x)*_y:0;
  float a22 = ((x0<Gx)&&(y0<Gy))?_x*_y:0;

  // weighted values
  float w11 = ((x0>0)&&(y0>0))?(1.f-_x)*(1.f-_y)*cfloat_abs(src[index11]):0;
  float w12 = ((x0<Gx)&&(y0>0))?_x*(1.f-_y)*cfloat_abs(src[index12]):0;
  float w21 = ((x0>0)&&(y0<Gy))?(1.f-_x)*_y*cfloat_abs(src[index21]):0;
  float w22 = ((x0<Gx)&&(y0<Gy))?_x*_y*cfloat_abs(src[index22]):0;



  if ((i_im>=0)&&(i_im<Nx*Gx)&&(j_im>=0)&&(j_im<Ny*Gy)){

    float nsum  = a11+a12+a21+a22;
    float wsum = w11+w12+w21+w22;
	
    dest[index_im] = wsum/nsum;


	// if ((x0==1) &&(j_im==31)&&(i_im==34))
	//   printf("huhu: %d %d %d %f\n",i_im, j_im, (i+Npatch_x/2), w22);


	
  }
}
