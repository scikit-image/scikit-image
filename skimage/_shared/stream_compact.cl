__kernel void compact(
	__global int* in,
	__global int* prefix_sum,
	__global int* out,
	int length
)
{
	int gx = get_global_id(0);

	if (gx > length-1)
		return;

	if (in[gx]) out[prefix_sum[gx]] = gx;
}
