// adapted from NVIDIAs CUDA prefix scan implementation 
// https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html

__kernel void scan2d(__global DTYPE *input,__global DTYPE *output,
					 __global DTYPE *sums,
					 __local DTYPE * shared,const int n,
					 const int stride_i, const int stride_j, const int offset_i, const int block_i,const int nblocks, const int nx_real)
{

  const int i = get_local_id(0);
  const int j = get_global_id(1);
  const bool valid_index1 = 2*(i + offset_i)<nx_real;
  const bool valid_index2 = 2*(i + offset_i)+1<nx_real;

  // if (!valid_index)
	// printf("kernel: %d %d\n", 2*(i + offset_i),2*(i + offset_i)+1);

  const int idx_buf1 = 2*(i + offset_i)*stride_i+stride_j*j;
  const int idx_buf2 = (2*(i + offset_i)+1)*stride_i+stride_j*j;

  int off = 1;
  DTYPE initial_val1 = (DTYPE)valid_index1?input[idx_buf1]:(DTYPE)0;
  DTYPE initial_val2 = (DTYPE)valid_index2?input[idx_buf2]:(DTYPE)0;

  shared[2*i] = initial_val1;
  shared[2*i+1] = initial_val2;

  for (int d = n>>1; d > 0; d >>= 1) {
	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < d){
	  int ai = off*(2*i+1)-1;
	  int bi = off*(2*i+2)-1;
	  shared[bi] += shared[ai];
	}

	off *= 2;
  }

  if (i == 0)
	shared[n - 1] = 0;

  for(int d = 1; d < n; d *= 2) {
	off >>= 1;
	barrier(CLK_LOCAL_MEM_FENCE);
	if(i < d){
	  int ai = off*(2*i+1)-1;
	  int bi = off*(2*i+2)-1;
	  DTYPE t  = shared[ai];
	  shared[ai] = shared[bi];
	  shared[bi] += t;
	}
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // inclusive scan: add initial value
  if (valid_index1)
	output[idx_buf1] = shared[2*i] + initial_val1;
  if (valid_index2)
	output[idx_buf2] = shared[2*i+1] + initial_val2;

  if (i==n/2-1)
	sums[block_i+nblocks*j] = shared[n-1]+ initial_val2;
}

__kernel void add_sums2d(__global DTYPE *input,__global DTYPE *output,
						 const int stride_i, const int stride_j,const int nblocks, const int nx_real)
{

  const int iloc = get_local_id(0);
  const int igroup = get_group_id(0);

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const bool valid_index = i<nx_real;

  const int idx_buf = i*stride_i + j*stride_j;

  const DTYPE val = igroup>0?(DTYPE)input[(igroup-1)+nblocks*j]:(DTYPE)0;

  if (valid_index)
	output[idx_buf] += val;

}

__kernel void scan3d(__global DTYPE *input,__global DTYPE *output,
					 __global DTYPE *sums,
					 __local DTYPE * shared,const int n,
					 const int stride_i, const int stride_j, const int stride_k,
					 const int offset_i, const int block_i,const int nblocks, const int ny,const int nx_real)
{

  const int i = get_local_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);

  const bool valid_index1 = 2*(i + offset_i)<nx_real;
  const bool valid_index2 = 2*(i + offset_i)+1<nx_real;

  // if (!valid_index)
	// printf("kernel: %d %d\n", 2*(i + offset_i),2*(i + offset_i)+1);

  const int idx_buf1 = 2*(i + offset_i)*stride_i+stride_j*j+stride_k*k;
  const int idx_buf2 = (2*(i + offset_i)+1)*stride_i+stride_j*j+stride_k*k;

  int off = 1;
  DTYPE initial_val1 = valid_index1?(DTYPE)input[idx_buf1]:(DTYPE)0;
  DTYPE initial_val2 = valid_index2?(DTYPE)input[idx_buf2]:(DTYPE)0;

  shared[2*i] = initial_val1;
  shared[2*i+1] = initial_val2;

  for (int d = n>>1; d > 0; d >>= 1) {
	barrier(CLK_LOCAL_MEM_FENCE);

	if(i < d){
	  int ai = off*(2*i+1)-1;
	  int bi = off*(2*i+2)-1;
	  shared[bi] += shared[ai];
	}

	off *= 2;
  }

  if (i == 0)
	shared[n - 1] = 0;

  for(int d = 1; d < n; d *= 2) {
	off >>= 1;
	barrier(CLK_LOCAL_MEM_FENCE);
	if(i < d){
	  int ai = off*(2*i+1)-1;
	  int bi = off*(2*i+2)-1;
	  DTYPE t  = shared[ai];
	  shared[ai] = shared[bi];
	  shared[bi] += t;
	}
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // inclusive scan: add initial value
  if (valid_index1)
	output[idx_buf1] = shared[2*i] + initial_val1;
  if (valid_index2)
	output[idx_buf2] = shared[2*i+1] + initial_val2;

  if (i==n/2-1)
	sums[block_i+nblocks*j+nblocks*ny*k] = shared[n-1]+ initial_val2;
}

__kernel void add_sums3d(__global DTYPE *input,__global DTYPE *output,
						 const int stride_i, const int stride_j,const int stride_k,const int nblocks, const int ny, const int nx_real)
{

  const int iloc = get_local_id(0);
  const int igroup = get_group_id(0);

  const int i = get_global_id(0);
  const int j = get_global_id(1);
  const int k = get_global_id(2);
  const bool valid_index = i<nx_real;

  const int idx_buf = i*stride_i + j*stride_j +k*stride_k;

  const DTYPE val = igroup>0?(DTYPE)input[(igroup-1)+nblocks*j+nblocks*ny*k]:(DTYPE)0;

  if (valid_index)
	output[idx_buf] += val;

}
