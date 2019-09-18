

//2D

__kernel void filter_2_x(__global ${DTYPE} * input,
						__global ${DTYPE} * output){

  int i = get_global_id(0);
  int j = get_global_id(1);

  int Nx = get_global_size(0);


  ${DTYPE} res = ${DEFAULT};
  int start = i-${FSIZE_X}/2;


  const int h_start = max(0,${FSIZE_X}/2-i);
  const int h_end = min(${FSIZE_X},Nx-i+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
      ${DTYPE} val = input[start+ht+j*Nx];
	  res = ${FUNC};
	  }

  output[i+j*Nx] = res;
}

__kernel void filter_2_y(__global ${DTYPE} * input,
						__global ${DTYPE} * output){

  int i = get_global_id(0);
  int j = get_global_id(1);
  
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  ${DTYPE} res = ${DEFAULT};

  int start = j-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j);
  const int h_end = min(${FSIZE_Y},Ny-j+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx];
	res = ${FUNC};
	}

  output[i+j*Nx] = res;
}

//3D

__kernel void filter_3_x(__global ${DTYPE} * input,
				    __global ${DTYPE} * output){

					 int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  ${DTYPE} res = ${DEFAULT};

  int start = i-${FSIZE_X}/2;

  const int h_start = max(0,${FSIZE_X}/2-i);
  const int h_end = min(${FSIZE_X},Nx-i+${FSIZE_X}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[start+ht+j*Nx+k*Nx*Ny];
    res = ${FUNC};
	  }

  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void filter_3_y(__global ${DTYPE} * input,
				    __global ${DTYPE} * output){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  ${DTYPE} res = ${DEFAULT};

  int start = j-${FSIZE_Y}/2;

  const int h_start = max(0,${FSIZE_Y}/2-j);
  const int h_end = min(${FSIZE_Y},Ny-j+${FSIZE_Y}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+(start+ht)*Nx+k*Nx*Ny];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = res;
}

__kernel void filter_3_z(__global ${DTYPE} * input,
				    __global ${DTYPE} * output){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  ${DTYPE} res = ${DEFAULT};

  int start = k-${FSIZE_Z}/2;

  const int h_start = max(0,${FSIZE_Z}/2-k);
  const int h_end = min(${FSIZE_Z},Nz-k+${FSIZE_Z}/2);

  for (int ht = h_start; ht< h_end; ++ht){
    ${DTYPE} val = input[i+j*Nx+(start+ht)*Nx*Ny];
	res = ${FUNC};
	}


  output[i+j*Nx+k*Nx*Ny] = res;
}
