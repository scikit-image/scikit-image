#define FALSE 0
#define TRUE 1

#define CL_TRUE -1
#define CL_FALSE 0

#define CL_TRUE_2_TRUE -1

//sqrt(3*255*255) = 441.67295593006372
//sqrt(3*1.0*1.0) = 1.7320508075688772
#ifndef G_NORM(X)
#define G_NORM(X) (1.0f - X/1.7320508075688772f)
#endif

#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, 0)
#define rgba_f2_to_uint(c) (uint) (0xFF << 24 | ((int) (255*c.z)) << 16 | ((int) (255*c.y)) << 8 | (int) (255*c.x))

float norm_length_ui(uint4 vector) {
	float4 f = (float4) (((float) vector.x)/255, ((float) vector.y)/255, ((float) vector.z)/255, ((float) vector.w)/255);

	return length(f);
}

float4 norm_rgba_ui4(uint4 rgba) {
	return (float4) (((float) rgba.x)/255, ((float) rgba.y)/255, ((float) rgba.z)/255, ((float) rgba.w)/255);
}

__kernel void label(
	__global uchar* labels_in,
	__global float* strength_in,
    __global int* points,
    uchar label,
    int n_points,
    __global int* tiles,
    int iteration
) {
   int gx = get_global_id(0);

   if (gx > n_points-1)
       return;

   int gxy = points[gx];

   labels_in[gxy] = label;
   strength_in[gxy] = 1.0;

    tiles[((gxy/IMAGEW)/TILEH)*TILESW + (gxy%IMAGEW)/TILEW] = iteration;
}

__kernel void evolveVonNeumann(
	__global int* tiles_list,
	__global uchar* labels_in,
	__global uchar* labels_out,
	__global float* strength_in,
	__global float* strength_out,
	__global int* has_converge,
	int iteration,
	__global int* tiles,
	__local int* tile_flags, //true if any updates
	__local uchar* s_labels_in,
	__local float* s_strength_in,
//	__local float4* s_img,
	__read_only image2d_t img,
	sampler_t sampler
)
{
	//get tile x,y offset
	int txy = tiles_list[get_group_id(0)];
	int tx = txy%TILESW;
	int ty = txy/TILESW;

	int lx = get_local_id(0);
	int ly = get_local_id(1);

	//image coordinates
	int ix = tx*TILEW + lx;
	int iy = ty*TILEH + ly;
	int ixy = iy*IMAGEW + ix;

	int sw = TILEW + 2;
	int sx = 1 + lx;
	int sy = 1 + ly;
	int sxy = sy*sw + sx;

	s_labels_in[sxy]   = labels_in[ixy];
	s_strength_in[sxy] = strength_in[ixy];
//	s_img[sxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy)));

	int isxy, i_ixy;

	//load padding
	if (ly == 0) { //top
		isxy = sxy - sw;
		i_ixy = ixy - IMAGEW;
		s_strength_in[isxy] = (iy != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
//		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy-1)));
	}
	else if (ly == TILEH-1) { //bottom
		isxy = sxy + sw;
		i_ixy = ixy + IMAGEW;
		s_strength_in[isxy] = (iy != IMAGEH-1) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
//		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy+1)));
	}
	if (lx == 0) { //left
		isxy = sxy - 1;
		i_ixy = ixy - 1;
		s_strength_in[isxy] = (ix != 0) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
//		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy)));
	}
	else if (lx == TILEW-1) { //right
		isxy = sxy + 1;
		i_ixy = ixy + 1;
		s_strength_in[isxy] = (ix != IMAGEW-1) ? strength_in[i_ixy] : 0;
		s_labels_in[isxy]   = labels_in[i_ixy];
//		s_img[isxy]         = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy)));
	}

	if (lx < 5 && ly == 0)
		tile_flags[lx] = FALSE;

	barrier(CLK_LOCAL_MEM_FENCE);

	float4 c = norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy)));
//	float4 c = s_img[sxy];
	uchar label = s_labels_in[sxy];
	float defence = s_strength_in[sxy];

	float4 attack = (float4) (
		G_NORM(distance(c, norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy-1))))) * s_strength_in[sxy-sw],
		G_NORM(distance(c, norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix, iy+1))))) * s_strength_in[sxy+sw],
		G_NORM(distance(c, norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix-1, iy))))) * s_strength_in[sxy-1],
		G_NORM(distance(c, norm_rgba_ui4(read_imageui(img, sampler, (int2) (ix+1, iy))))) * s_strength_in[sxy+1]
	);

	if (attack.x > defence) {
		defence = attack.x;
		label = s_labels_in[sxy-sw];
	}

	if (attack.y > defence) {
		defence = attack.y;
		label = s_labels_in[sxy+sw];
	}

	if (attack.z > defence) {
		defence = attack.z;
		label = s_labels_in[sxy-1];
	}

	if (attack.w > defence) {
		defence = attack.w;
		label = s_labels_in[sxy+1];
	}

	strength_out[ixy] = defence;
	labels_out[ixy] = label;

	if (defence != s_strength_in[sxy] || label != s_labels_in[sxy]) {
		tile_flags[0] = TRUE;

		if (iy != 0 && ly == 0)            tile_flags[1] = TRUE;
		if (iy != IMAGEH && ly == TILEH-1) tile_flags[2] = TRUE;
		if (ix != 0 && lx == 0)            tile_flags[3] = TRUE;
		if (ix != IMAGEW && lx == TILEW-1) tile_flags[4] = TRUE;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (lx == 0 && ly == 0 && tile_flags[0] == TRUE)
		tiles[txy] = iteration;

	if (tile_flags[1]) tiles[txy-TILESW] = iteration;
	if (tile_flags[2]) tiles[txy+TILESW] = iteration;
	if (tile_flags[3]) tiles[txy-1] = iteration;
	if (tile_flags[4]) tiles[txy+1] = iteration;
}