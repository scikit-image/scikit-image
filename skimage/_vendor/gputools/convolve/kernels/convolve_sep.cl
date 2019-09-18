//2D

__kernel void conv_sep2_x(__global float * input,
						__constant float * h,
						__global float * output,
						  const int Nh){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = 0.f;

  const int h_start = ((i+Nh/2)>=Nx)?i+Nh/2+1-Nx:0;
  const int h_end = ((i-Nh/2)<0)?i+Nh/2+1:Nh;
  const int start = i+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[start-ht+j*Nx];


  output[i+j*Nx] = res;
}

__kernel void conv_sep2_y(__global float * input,
						__constant float * h,
						__global float * output,
						  const int Nh){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float res = 0.f;


  const int h_start = ((j+Nh/2)>=Ny)?j+Nh/2+1-Ny:0;
  const int h_end = ((j-Nh/2)<0)?j+Nh/2+1:Nh;
  const int start = j+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[(start-ht)*Nx+i];

	
  output[i+j*Nx] = res;
}



//3D

__kernel void conv_sep3_x(__global float * input,
						__constant float * h,
						__global float * output,
						  const int Nh){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = 0.f;

  const int h_start = ((i+Nh/2)>=Nx)?i+Nh/2+1-Nx:0;
  const int h_end = ((i-Nh/2)<0)?i+Nh/2+1:Nh;
  const int start = i+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[start-ht+j*Nx+k*Nx*Ny];

	
  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void conv_sep3_y(__global float * input,
						__constant float * h,
						__global float * output,
						  const int Nh){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = 0.f;



  const int h_start = ((j+Nh/2)>=Ny)?j+Nh/2+1-Ny:0;
  const int h_end = ((j-Nh/2)<0)?j+Nh/2+1:Nh;
  const int start = j+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[i+(start-ht)*Nx+k*Nx*Ny];

	
  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void conv_sep3_z(__global float * input,
						__constant float * h,
						__global float * output,
						  const int Nh){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  float res = 0.f;


  const int h_start = ((k+Nh/2)>=Nz)?k+Nh/2+1-Nz:0;
  const int h_end = ((k-Nh/2)<0)?k+Nh/2+1:Nh;
  const int start = k+Nh/2;

  for (int ht = h_start; ht< h_end; ++ht)
    res += h[ht]*input[i+j*Nx+(start-ht)*Nx*Ny];


	
  output[i+j*Nx+k*Nx*Ny] = res;
}
