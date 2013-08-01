/*
 * Inplace upsweep (reduce) on a local array [x] of length [m].
 * NB: [m] must be a power of two.
 */
inline void upsweep_pow2(__local int *x, int m) {
  int lid = get_local_id(0);
  int bi = (lid*2)+1;

  int depth = 1 + (int) log2((float)m);
  for (int d=0; d<depth; d++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      int ai = bi - offset;
      x[bi] += x[ai];
    }
  }
}

/*
 * Inplace sweepdown on a local array [x] of length [m].
 * NB: [m] must be a power of two.
 */
inline void sweepdown_pow2(__local int *x, int m) {
  int lid = get_local_id(0);
  int bi = (lid*2)+1;

  int depth = (int) log2((float)m);
  for (int d=depth; d>-1; d--) {
    barrier(CLK_LOCAL_MEM_FENCE);
    int mask = (0x1 << d) - 1;
    if ((lid & mask) == mask) {
      int offset = (0x1 << d);
      int ai = bi - offset;
      int tmp = x[ai];
                x[ai] = x[bi];
                        x[bi] += tmp;
    }
  }
}

/*
 * Inplace scan on a local array [x] of length [m].
 * NB: m must be a power of two.
 */
inline void scan_pow2(__local int *x, int m) {
  int lid = get_local_id(0);
  int lane1 = (lid*2)+1;
  upsweep_pow2(x, m);
  if (lane1 == (m-1)) {
    x[lane1] = 0;
  }
  sweepdown_pow2(x, m);
}

/*
 * Inplace scan on a global array [data] of length [m].
 * We load data into a local array [x] (also of length [m]),
 *   and use a local upsweep and sweepdown.
 * NB: [m] must be a power of two, and
 *     there must be exactly one workgroup of size m/2
 */
__kernel void scan_pow2_wrapper(__global int *data, __local int *x, int m) {
  int gid = get_global_id(0);
  int lane0 = (gid*2);
  int lane1 = (gid*2)+1;

  // load data into local arrays
  x[lane0] = data[lane0];
  x[lane1] = data[lane1];

  // inplace local scan
  scan_pow2(x, m);

  // writeback data
  data[lane0] = x[lane0];
  data[lane1] = x[lane1];
}

__kernel void scan_pad_to_pow2(__global int *data, __local int * x, int n, __global int* length) {
  int gid = get_global_id(0);
  int lane0 = (gid*2);
  int lane1 = (gid*2)+1;
  int m = 2*get_local_size(0);

  x[lane0] = lane0 < n ? data[lane0] : 0;
  x[lane1] = lane1 < n ? data[lane1] : 0;

  upsweep_pow2(x, m);
  if (lane1 == (m-1)) {
	length[0] = x[lane1];
    x[lane1] = 0;
  }
  sweepdown_pow2(x, m);

  if (lane0 < n)
    data[lane0] = x[lane0];
  if (lane1 < n)
    data[lane1] = x[lane1];
}

/*
 * First phase of a multiblock scan.
 *
 * Given a global array [data] of length arbitrary length [n].
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * The last subarray will be padded with 0 if necessary (n < k*m).
 * We use the primitives above to perform a scan operation within each subarray.
 * We store the intermediate reduction of each subarray (following upsweep_pow2) in [part].
 * These partial values can themselves be scanned and fed into [scan_inc_subarrays].
 */
__kernel void scan_subarrays(
  __global int *data, //length [n]
  __local  int *x,    //length [m]
  __global int *part, //length [m]
           int n
#if DEBUG
  , __global int *debug   //length [k*m]
#endif
) {
  // workgroup size
  int wx = get_local_size(0);
  // global identifiers and indexes
  int gid = get_global_id(0);
  int lane0 = (2*gid)  ;
  int lane1 = (2*gid)+1;
  // local identifiers and indexes
  int lid = get_local_id(0);
  int local_lane0 = (2*lid)  ;
  int local_lane1 = (2*lid)+1;
  int grpid = get_group_id(0);
  // list lengths
  int m = wx * 2;
  int k = get_num_groups(0);

  // copy into local data padding elements >= n with 0
  x[local_lane0] = (lane0 < n) ? data[lane0] : 0;
  x[local_lane1] = (lane1 < n) ? data[lane1] : 0;

  // ON EACH SUBARRAY
  // a reduce on each subarray
  upsweep_pow2(x, m);
  // last workitem per workgroup saves last element of each subarray in [part] before zeroing
  if (lid == (wx-1)) {
    part[grpid] = x[local_lane1];
                  x[local_lane1] = 0;
  }
  // a sweepdown on each subarray
  sweepdown_pow2(x, m);

  // copy back to global data
  if (lane0 < n) {
    data[lane0] = x[local_lane0];
  }
  if (lane1 < n) {
    data[lane1] = x[local_lane1];
  }

#if DEBUG
  debug[lane0] = x[local_lane0];
  debug[lane1] = x[local_lane1];
#endif
}

/*
 * Perform the second phase of an inplace exclusive scan on a global array [data] of arbitrary length [n].
 *
 * We assume that we have k workgroups each of size m/2 workitems.
 * Each workgroup handles a subarray of length [m] (where m is a power of two).
 * We sum each element by the sum of the preceding subarrays taken from [part].
 */
__kernel void scan_inc_subarrays(
  __global int *data, //length [n]
  __local  int *x,    //length [m]
  __global int *part, //length [m]
           int n
#if DEBUG
  , __global int *debug   //length [k*m]
#endif
) {
  // global identifiers and indexes
  int gid = get_global_id(0);
  int lane0 = (2*gid)  ;
  int lane1 = (2*gid)+1;
  // local identifiers and indexes
  int lid = get_local_id(0);
  int local_lane0 = (2*lid)  ;
  int local_lane1 = (2*lid)+1;
  int grpid = get_group_id(0);

  // copy into local data padding elements >= n with identity
  x[local_lane0] = (lane0 < n) ? data[lane0] : 0;
  x[local_lane1] = (lane1 < n) ? data[lane1] : 0;

  x[local_lane0] += part[grpid];
  x[local_lane1] += part[grpid];

  // copy back to global data
  if (lane0 < n) {
    data[lane0] = x[local_lane0];
  }
  if (lane1 < n) {
    data[lane1] = x[local_lane1];
  }

#if DEBUG
  debug[lane0] = x[local_lane0];
  debug[lane1] = x[local_lane1];
#endif
}
