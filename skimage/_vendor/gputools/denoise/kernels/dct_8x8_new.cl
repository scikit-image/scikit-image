/*
 * Copyright (c) 2010, Guoshen Yu <yu@cmap.polytechnique.fr>,
 *                     Guillermo Sapiro <guille@umn.edu>
 * Copyright (C) 2011, Michael Zucchi <notzed@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

//#pragma OPENCL EXTENSION cl_amd_printf : enable
#define d(x)

/* This code implements "DCT image denoising: a simple and effective image 
 * denoising algorithm".
 * 
 * http://www.ipol.im/pub/algo/ys_dct_denoising
 */

#define PATCHSIZE 8

/*
  8x8 DCT basis
 */
constant float DCTbasis[] = {
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	0.35355339059327373085750423342688009142875671386719f,
	//
	0.49039264020161521528962111915461719036102294921875f,
	0.41573480615127261783570133957255166023969650268555f,
	0.27778511650980114433551193542371038347482681274414f,
	0.09754516100806412404189416065491968765854835510254f,
	-0.09754516100806412404189416065491968765854835510254f,
	-0.27778511650980114433551193542371038347482681274414f,
	-0.41573480615127261783570133957255166023969650268555f,
	-0.49039264020161521528962111915461719036102294921875f,
	//
	0.46193976625564336924156805253005586564540863037109f,
	0.19134171618254489088961634024599334225058555603027f,
	-0.19134171618254489088961634024599334225058555603027f,
	-0.46193976625564336924156805253005586564540863037109f,
	-0.46193976625564336924156805253005586564540863037109f,
	-0.19134171618254489088961634024599334225058555603027f,
	0.19134171618254489088961634024599334225058555603027f,
	0.46193976625564336924156805253005586564540863037109f,
	//
	0.41573480615127261783570133957255166023969650268555f,
	-0.09754516100806417955304539191274670884013175964355f,
	-0.49039264020161521528962111915461719036102294921875f,
	-0.27778511650980108882436070416588336229324340820312f,
	0.27778511650980108882436070416588336229324340820312f,
	0.49039264020161521528962111915461719036102294921875f,
	0.09754516100806417955304539191274670884013175964355f,
	-0.41573480615127261783570133957255166023969650268555f,
	//
	0.35355339059327378636865546468470711261034011840820f,
	-0.35355339059327378636865546468470711261034011840820f,
	-0.35355339059327378636865546468470711261034011840820f,
	0.35355339059327378636865546468470711261034011840820f,
	0.35355339059327378636865546468470711261034011840820f,
	-0.35355339059327378636865546468470711261034011840820f,
	-0.35355339059327378636865546468470711261034011840820f,
	0.35355339059327378636865546468470711261034011840820f,
	//
	0.27778511650980114433551193542371038347482681274414f,
	-0.49039264020161532631192358167027123272418975830078f,
	0.09754516100806412404189416065491968765854835510254f,
	0.41573480615127261783570133957255166023969650268555f,
	-0.41573480615127261783570133957255166023969650268555f,
	-0.09754516100806412404189416065491968765854835510254f,
	0.49039264020161532631192358167027123272418975830078f,
	-0.27778511650980114433551193542371038347482681274414f,
	//
	0.19134171618254491864519195587490685284137725830078f,
	-0.46193976625564336924156805253005586564540863037109f,
	0.46193976625564336924156805253005586564540863037109f,
	-0.19134171618254491864519195587490685284137725830078f,
	-0.19134171618254491864519195587490685284137725830078f,
	0.46193976625564336924156805253005586564540863037109f,
	-0.46193976625564336924156805253005586564540863037109f,
	0.19134171618254491864519195587490685284137725830078f,
	//
	0.09754516100806416567525758409828995354473590850830f,
	-0.27778511650980108882436070416588336229324340820312f,
	0.41573480615127267334685257083037868142127990722656f,
	-0.49039264020161521528962111915461719036102294921875f,
	0.49039264020161521528962111915461719036102294921875f,
	-0.41573480615127267334685257083037868142127990722656f,
	0.27778511650980108882436070416588336229324340820312f,
	-0.09754516100806416567525758409828995354473590850830f
};

#if 0
float threshold(float v, float thr) {
	float m = max(fabs(v) - thr, 0.f);
	return  (m / (m+thr)) * v;
}
#else
float threshold(float v, float thr) {
	return fabs(v) < thr ? 0 : v;
}
#endif

/**
 * Perform  8x8 forward dct on data stored in local array
 */

//	dct_forward_threshold(ldata, ly*DFSTRIDE, ly, lx, DFSTRIDE, thr);

void dct_forward_threshold(local float *data, int soff, int doff, int j, int stride, float thr) {
	// rows
	float v = 0;
	for (int i = 0; i < PATCHSIZE; i++) {
		v += data[i + soff] * DCTbasis[j*PATCHSIZE+i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	data[j + soff] = v;
	barrier(CLK_LOCAL_MEM_FENCE);

	// columns
	v = 0;
	for (int i = 0; i < PATCHSIZE; i++) {
		v += data[i * stride + doff] * DCTbasis[j*PATCHSIZE+i];
	}

	v = threshold(v, thr);

	barrier(CLK_LOCAL_MEM_FENCE);
	data[j * stride + doff] = v;
	barrier(CLK_LOCAL_MEM_FENCE);
}

void dct_inverse(local float *data, int soff, int doff, int j, int stride) {
	// rows
	float v = 0;

	for (int i = 0; i < PATCHSIZE; i++) {
		v += data[i + soff] * DCTbasis[i*PATCHSIZE+j];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	data[j + soff] = v;
	barrier(CLK_LOCAL_MEM_FENCE);

	// cols
	v = 0;
	for (int i = 0; i < PATCHSIZE; i++) {
		v += data[i * stride + doff] * DCTbasis[i*PATCHSIZE+j];
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	data[j * stride + doff] = v;
	barrier(CLK_LOCAL_MEM_FENCE);
}

#define DFSTRIDE (9)    //FIXME Why not 8?? 

/**
 * Process a single 8x8 tile using DCT denoising algorithm.
 *
 * Forward DCT in local, threshold, then save.
 */
kernel void
__attribute__((reqd_work_group_size(8, 8, 1)))
dct_denoise(read_only image2d_t src, global float *acc, float thr, int dx, int dy, int astride, int set) {
	const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	local float ldata[8*DFSTRIDE];
	int lx = get_local_id(0);
	int ly = get_local_id(1);
	int x = get_global_id(0) + dx;
	int y = get_global_id(1) + dy;

	float v = read_imagef(src, smp, (int2) { x, y }).s0;

	ldata[lx + ly*DFSTRIDE] = v;
	barrier(CLK_LOCAL_MEM_FENCE);

	dct_forward_threshold(ldata, ly*DFSTRIDE, ly, lx, DFSTRIDE, thr);
	dct_inverse(ldata, ly*DFSTRIDE, ly, lx, DFSTRIDE);

	// accumulate results: always in-range by design
	v = ldata[lx + ly*DFSTRIDE];
	if (set)
	  acc[x + y * astride] = v;
	else
	  acc[x + y * astride] += v;
	
}
/**
 * Fix the accumulation results.
 */
kernel void
__attribute__((reqd_work_group_size(8, 8, 1)))
dct_denoise_normalise(const global float *acc, write_only image2d_t dst, int astride) {
	int x = get_global_id(0) ;
	int y = get_global_id(1) ;
	float v;

	if ((x >= get_image_width(dst)) | (y >= get_image_height(dst)))
		return;

	v = acc[x + y * astride];

	
	
#if 0
	v *= 1.0f / 64.0f;
#else
	float xc = (x < 8) ? x + 1 : x >= get_image_width(dst) - 8 ? get_image_width(dst) - x : 8;
	float yc = (y < 8) ? y + 1 : y >= get_image_height(dst) - 8 ? get_image_height(dst) - y : 8;

	v *= 1.0f / (xc * yc);
#endif
	
	write_imagef(dst, (int2) { x, y }, (float4) v);
}

