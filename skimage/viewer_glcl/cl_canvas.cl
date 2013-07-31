#define NORM 0.00392156862745098f
#define uint42f4n(c) (float4) (NORM*c.x, NORM*c.y, NORM*c.z, NORM*c.w)
#define RGBA_UI_TO_UI4(c) (uint4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, (c & 0xFF000000) >> 24)

__global kernel void blend_imgui(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__read_only image2d_t input
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > get_image_width(input)-1 || gxy.y > get_image_height(input)-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	uint4 in = read_imageui(input, sampler, gxy);
	float4 out = uint42f4n(in);

	opacity *= out.w;

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void blend_bufui(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__global uint* input,
	int2 dim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > dim.x-1 || gxy.y > dim.y-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	uint in = input[gxy.y*dim.x + gxy.x];
	uint4 in4 = RGBA_UI_TO_UI4(in);
	float4 out = uint42f4n(in4);

	opacity *= out.w;

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void blend_imgf(
	sampler_t sampler,
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	float opacity,
	__read_only image2d_t input
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > get_image_width(input)-1 || gxy.y > get_image_height(input)-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	float4 out = read_imagef(input, sampler, gxy);

	opacity *= out.w;

	write_imagef(rbo_write, gxy, read + (out-read)*opacity);
}

__global kernel void flip(
	__read_only image2d_t rbo_read,
	__write_only image2d_t rbo_write,
	sampler_t sampler
){
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > get_image_width(rbo_read)-1 || gxy.y > get_image_height(rbo_read)-1)
		return;

	float4 read = read_imagef(rbo_read, sampler, gxy);

	gxy.y = get_image_height(rbo_write) - gxy.y;
	write_imagef(rbo_write, gxy, read);
}