#include <pyopencl-complex.h>

#ifndef ADDRESS_MODE
#define ADDRESS_MODE CLK_ADDRESS_CLAMP
#endif

void kernel mult_inplace(__global cfloat_t *dest, const __global cfloat_t *src){

    uint i = get_global_id(0);
    dest[i] = cfloat_mul(dest[i],src[i]);
}




// 3d function
void kernel fill_patch3(read_only image3d_t src,
							 const int offset_x,
							 const int offset_y,
							 const int offset_z,
							 __global cfloat_t *dest, const int offset_dest){

  const sampler_t sampler = ADDRESS_MODE |  CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  uint Nz = get_global_size(2);

  float val = read_imagef(src,
  						   sampler,
						  (int4)(i+offset_x,j+offset_y,k+offset_z,0)).x;
  
  dest[i+Nx*j+Nx*Ny*k+offset_dest] = cfloat_new(val,0.f);

 
}



void kernel fill_psf_grid3(__global float * src,
						   const int Nx_src,
                            const int Ny_src,
                            const int offset_x_src,
                            const int offset_y_src,
                            const int offset_z_src,
                             __global cfloat_t *dest,
                             const int Nx_dest,
                             const int Ny_dest,
                             const int Nz_dest,
							 const int offset_x_dest,
							 const int offset_y_dest,
							 const int offset_z_dest,
							 const int offset_dest){


  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint k = get_global_id(2);

  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);



  int index_src = i+offset_x_src+Nx_src*(j+offset_y_src)+Nx_src*Ny_src*(k+offset_z_src);
  float val = src[index_src];


  int i_dest = i+offset_x_dest;
  int j_dest = j+offset_y_dest;
  int k_dest = k+offset_z_dest;

  // do a fftshift on the go, as we read very out of order anyway...
  if (i_dest<Nx_dest/2)
    i_dest += Nx_dest/2;
  else
    i_dest = i_dest - Nx_dest/2;

  if (j_dest<Ny_dest/2)
    j_dest += Ny_dest/2;
  else
    j_dest = j_dest - Ny_dest/2;

  if (k_dest<Nz_dest/2)
    k_dest += Nz_dest/2;
  else
    k_dest = k_dest - Nz_dest/2;

  int index_dest = i_dest+Nx_dest*j_dest+Nx_dest*Ny_dest*k_dest+offset_dest;

  dest[index_dest] = cfloat_new(val,0.f);




}

//3d 
void kernel interpolate3( __global cfloat_t * src,
						 __global float * dest,
						 const int x0, const int y0,const int z0,
						 const int Gx,const int Gy,const int Gz,
						  const int Npatch_x,
						  const int Npatch_y,
						  const int Npatch_z){

  const sampler_t sampler = ADDRESS_MODE |  CLK_FILTER_NEAREST;

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  //the Nblock sizes
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  // relative coords within image block
  float _x = 1.f*i/(Nx-1.f);
  float _y = 1.f*j/(Ny-1.f);	  
  float _z = 1.f*k/(Nz-1.f);


  // the coordinates in the image
  int i_im = i+x0*Nx-Nx/2;
  int j_im = j+y0*Ny-Ny/2;
  int k_im = k+z0*Nz-Nz/2;

  int index_im = i_im+j_im*Gx*Nx+k_im*Gx*Gy*Nx*Ny;

  // the index in the patches
  int stride1 = Npatch_x*Npatch_y*Npatch_z;
  int stride2 = Npatch_x*Npatch_y*Npatch_z*Gx;
  int stride3 = Npatch_x*Npatch_y*Npatch_z*Gx*Gy;


  int index111 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*(x0-1)+stride2*(y0-1)+stride3*(z0-1);

  int index112 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*x0+stride2*(y0-1)+stride3*(z0-1);

  int index121 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*(x0-1)+stride2*y0+stride3*(z0-1);

  int index211 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*(x0-1)+stride2*(y0-1)+stride3*z0;

  int index122 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2)+
                stride1*x0+stride2*y0+stride3*(z0-1);

  int index221 = (i+Npatch_x/2)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*(x0-1)+stride2*y0+stride3*z0;

  int index212 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*x0+stride2*(y0-1)+stride3*z0;

  int index222 = (i+Npatch_x/2-Nx)+
                Npatch_x*(j+Npatch_y/2-Ny)+
                Npatch_y*Npatch_x*(k+Npatch_z/2-Nz)+
                stride1*x0+stride2*y0+stride3*z0;




  //interpolation weights

  float a111 = ((x0>0)&&(y0>0)&&(z0>0))?(1.f-_x)*(1.f-_y)*(1.f-_z):0;
  float a112 = ((x0<Gx)&&(y0>0)&&(z0>0))?_x*(1.f-_y)*(1.f-_z):0;
  float a121 = ((x0>0)&&(y0<Gy)&&(z0>0))?(1.f-_x)*_y*(1.f-_z):0;
  float a211 = ((x0>0)&&(y0>0)&&(z0<Gz))?(1.f-_x)*(1.f-_y)*_z:0;
  float a122 = ((x0<Gx)&&(y0<Gy)&&(z0>0))?_x*_y*(1.f-_z):0;
  float a221 = ((x0>0)&&(y0<Gy)&&(z0<Gz))?(1.f-_x)*_y*_z:0;
  float a212 = ((x0<Gx)&&(y0>0)&&(z0<Gz))?_x*(1.f-_y)*_z:0;
  float a222 = ((x0<Gx)&&(y0<Gy)&&(z0<Gz))?_x*_y*_z:0;

  // weighted values
  float w111 = ((x0>0)&&(y0>0)&&(z0>0))?(1.f-_x)*(1.f-_y)*(1.f-_z)*cfloat_abs(src[index111]):0;
  float w112 = ((x0<Gx)&&(y0>0)&&(z0>0))?_x*(1.f-_y)*(1.f-_z)*cfloat_abs(src[index112]):0;
  float w121 = ((x0>0)&&(y0<Gy)&&(z0>0))?(1.f-_x)*_y*(1.f-_z)*cfloat_abs(src[index121]):0;
  float w211 = ((x0>0)&&(y0>0)&&(z0<Gz))?(1.f-_x)*(1.f-_y)*_z*cfloat_abs(src[index211]):0;
  float w122 = ((x0<Gx)&&(y0<Gy)&&(z0>0))?_x*_y*(1.f-_z)*cfloat_abs(src[index122]):0;
  float w221 = ((x0>0)&&(y0<Gy)&&(z0<Gz))?(1.f-_x)*_y*_z*cfloat_abs(src[index221]):0;
  float w212 = ((x0<Gx)&&(y0>0)&&(z0<Gz))?_x*(1.f-_y)*_z*cfloat_abs(src[index212]):0;
  float w222 = ((x0<Gx)&&(y0<Gy)&&(z0<Gz))?_x*_y*_z*cfloat_abs(src[index222]):0;


  if ((i_im>=0)&&(i_im<Nx*Gx)&&(j_im>=0)&&(j_im<Ny*Gy)&&(k_im>=0)&&(k_im<Nz*Gz)){

    float nsum  = a111+a112+a121+a211+a122+a212+a221+a222;;
    float wsum = w111+w112+w121+w211+w122+w212+w221+w222;
    dest[index_im] = wsum/nsum;

    //dest[index_im] = index121;


    //dest[index_im] = wsum;


  }

}
