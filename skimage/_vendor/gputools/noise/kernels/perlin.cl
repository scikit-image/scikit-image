    // Copyright 2009-2015 Intel Corporation.
//
// The source code, information and material ("Material") contained herein is 
// owned by Intel Corporation or its suppliers or licensors, and title to such 
// Material remains with Intel Corporation or its suppliers or licensors. 
// The Material contains proprietary information of Intel or its suppliers and 
// licensors. The Material is protected by worldwide copyright laws and treaty 
// provisions. No part of the Material may be used, copied, reproduced, modified, 
// published, uploaded, posted, transmitted, distributed or disclosed in any way 
// without Intel's prior express written permission. No license under any patent, 
// copyright or other intellectual property rights in the Material is granted to 
// or conferred upon you, either expressly, by implication, inducement, estoppel 
// or otherwise. Any license under such intellectual property rights must be 
// express and approved by Intel in writing.
//
// *OpenCL(TM) - OpenCL and the OpenCL logo are trademarks of Apple Inc. used by 
// permission by Khronos.
//
// *Third Party trademarks are the property of their respective owners.
//
// Intel and the Intel logo are trademarks of Intel Corporation in the U.S. and/or 
// other countries.
//
// Unless otherwise agreed by Intel in writing, you may not remove or alter this 
// notice or any other notice embedded in Materials by Intel or Intel's suppliers 
// or licensors in any way.




// General note:  Generating random numbers from -1 up to but not including +1

// random shuffling of 0-255
constant	uint	default_perm[256] =  { 
	151, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 99, 37,
	8, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 160, 137, 
	35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 
	134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 
	55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 
	18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 
	250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 
	189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 
	43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 
	97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 
	107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 
	138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 
	140, 36, 103, 30, 227, 47, 16, 58, 69, 17, 209, 76, 132, 187, 45, 127, 
	197, 62, 94, 252, 153, 101, 155, 167, 219, 182, 3, 64, 52, 217, 215, 61, 
	168, 68, 175, 74, 185, 112, 104, 218, 165, 246, 4, 150, 208, 254, 142, 71, 
	230, 220, 105, 92, 145, 235, 249, 14, 41, 239, 156, 180, 226, 89, 203, 117
};

// 16 normalized pair vectors uniform distribution and off-axes.  4 corners randomly chosen gives 2^16 combinations
constant	float2	grads2d[16] = {
									{-0.195090322f, -0.98078528f },
									{-0.555570233f, -0.831469612f },
									{-0.831469612f, -0.555570233f },
									{-0.98078528f, -0.195090322f },
									{-0.98078528f, 0.195090322f },
									{-0.831469612f, 0.555570233f },
									{-0.555570233f, 0.831469612f },
									{-0.195090322f, 0.98078528f },
									{0.195090322f, 0.98078528f },
									{0.555570233f, 0.831469612f },
									{0.831469612f, 0.555570233f },
									{0.98078528f, 0.195090322f },
									{0.98078528f, -0.195090322f },
									{0.831469612f, -0.555570233f },
									{0.555570233f, -0.831469612f },
									{0.195090322f, -0.98078528f }	
									};

constant	char4	grads3d[16] = { 
										{1,1,0,0},{-1,1,0,0},{1,-1,0,0},{-1,-1,0,0},
										{1,0,1,0},{-1,0,1,0},{1,0,-1,0},{-1,0,-1,0},
										{0,1,1,0},{0,-1,1,0},{0,1,-1,0},{0,-1,-1,0},
										{1,1,0,0},{-1,1,0,0},{0,-1,1,0},{0,-1,-1,0}
									};


#if 1

// Wang Hash based RNG
//  Has at least 20 separate cycles, shortest cycle is < 7500 long.  
//  But it yields random looking 2D noise when fed OpenCL work item IDs, 
//  and that short cycle should only be hit for one work item in about 500K.
unsigned int ParallelRNG( unsigned int x )
{
	unsigned int value = x;

	value = (value ^ 61) ^ (value>>16);
	value *= 9;
	value ^= value << 4;
	value *= 0x27d4eb2d;
	value ^= value >> 15;

	return value;
}

#else

// Unix OS RNG - fast, single cycle of all 2^32 numbers, 
//    but not very random looking when used with OpenCL work item IDs.
unsigned int ParallelRNG( unsigned int x )
{
	unsigned int value = x;

    value = 1103515245 * value + 12345;

	return value;
}

#endif




unsigned int ParallelRNG2( unsigned int x,  unsigned int y )
{
	unsigned int value = ParallelRNG(x);

	value = ParallelRNG( y ^ value );

	return value;
}


unsigned int ParallelRNG3( unsigned int x,  unsigned int y,  unsigned int z )
{
	unsigned int value = ParallelRNG(x);

	value = ParallelRNG( y ^ value );

	value = ParallelRNG( z ^ value );

	return value;
}



float
weight_poly3(float weight)
{
	return	weight * weight * (3 - weight * 2);		// Perlin's original interp equation
}


// Ken Perlin's improved ease curve
float
weight_poly5(float weight)
{
	return	weight * weight * weight * (weight * (weight * 6 - 15) + 10);		// Perlin's improved interp equation
}

#define	WEIGHT(w)	weight_poly5(w)




// map 0-255  to -1.0 up to but excluding 1.0 
#define	NORM256(ff)	 ((1.0f / 128.0f) * ((ff)1080 - 128.0f))

// bilinear interpolation
#define	interp(w,b,c)	mix((b),(c),(w))



float
hash_grad_dot2(uint hash, float2 xy)					// 2d gradient look up and dot product with vector
{
	uint indx = hash & 0x0f;

	// look up a unit vector gradient
	float2	grad2 = grads2d[indx];

	return	dot(xy, grad2);
}





//	This calculates the Improved Perlin Noise function once for 2d using default permutation table 
//  Returned result should be between -1.0 and 1.0
float	
Noise_2d( float x, float y )
{ 
	float	X = floor(x);	// lower grid coordinates
	float	Y = floor(y);

	float2	vxy;
	vxy.x = x - X;		// vector from lower grid coordinates
	vxy.y = y - Y;

	float2	vXy = vxy;		vXy.x -= 1.0f;
	float2	vxY = vxy;		vxY.y -= 1.0f;
	float2	vXY = vXy;		vXY.y -= 1.0f;

	int	ux = (int)(X);
	int	uy = (int)(Y);
	int	uX = ux + 1;
	int	uY = uy + 1;

	uint	px    = default_perm[  ux			& 0x0FF ];				// generate permutation grads
	uint	pX    = default_perm[  uX			& 0x0FF ];

	uint	pxy   = default_perm[ (px  + uy)	& 0x0FF ];
	uint	pXy   = default_perm[ (pX  + uy)	& 0x0FF ];
	uint	pxY   = default_perm[ (px  + uY)	& 0x0FF ];
	uint	pXY   = default_perm[ (pX  + uY)	& 0x0FF ];


	float	gxy   = hash_grad_dot2(pxy, vxy);
	float	gXy   = hash_grad_dot2(pXy, vXy);
	float	gxY   = hash_grad_dot2(pxY, vxY);
	float	gXY   = hash_grad_dot2(pXY, vXY);

	float	wx = WEIGHT(vxy.x);
	float	wy = WEIGHT(vxy.y);
	
	return  interp( wy,	interp( wx, gxy, gXy ),	interp( wx, gxY, gXY ));
}




float
hash_grad_dot3(uint hash, float3 xyz)					// 3d gradient look up and dot product with vector
{ 
	uint indx = hash & 0x0f;

	float3	grad3 =  convert_float3( grads3d[indx].xyz );

	return	dot(xyz, grad3); 
}


//	This calculates the Improved Perlin Noise function once for 3d using default permutation table 
//  Returned result should be between -1.0 and 1.0
float	
Noise_3d( float x, float y, float z )
{ 
	float	X = floor(x);	// lower grid coordinates
	float	Y = floor(y);
	float	Z = floor(z);

	float3	vxyz;
	vxyz.x = x - X;		// vector from lower grid coordinates
	vxyz.y = y - Y;
	vxyz.z = z - Z;

	float3	vXyz, vXYz, vXyZ, vxYz, vxYZ, vxyZ, vXYZ;
	vXyz = vxyz;		vXyz.x -= 1.0f;
	vxYz = vxyz;		vxYz.y -= 1.0f;
	vxyZ = vxyz;		vxyZ.z -= 1.0f;

	vXYz = vXyz;	vXYz.y -= 1.0f;
	vXyZ = vXyz;	vXyZ.z -= 1.0f;

	vxYZ = vxYz;	vxYZ.z -= 1.0f;

	vXYZ = vXYz;	vXYZ.z -= 1.0f;

	int	ux = (int)(X);
	int	uy = (int)(Y);
	int	uz = (int)(Z);
	uint	uX = ux + 1;
	uint	uY = uy + 1;
	uint	uZ = uz + 1;

	uint	px    = default_perm[  ux			& 0x0FF ];				// generate permutation grads
	uint	pX    = default_perm[  uX			& 0x0FF ];

	uint	pxy   = default_perm[ (px  + uy)	& 0x0FF ];
	uint	pXy   = default_perm[ (pX  + uy)	& 0x0FF ];
	uint	pxY   = default_perm[ (px  + uY)	& 0x0FF ];
	uint	pXY   = default_perm[ (pX  + uY)	& 0x0FF ];

	uint	pxyz  = default_perm[ (pxy + uz)	& 0x0FF ];
	uint	pXyz  = default_perm[ (pXy + uz)	& 0x0FF ];
	uint	pxYz  = default_perm[ (pxY + uz)	& 0x0FF ];
	uint	pXYz  = default_perm[ (pXY + uz)	& 0x0FF ];
	uint	pxyZ  = default_perm[ (pxy + uZ)	& 0x0FF ];
	uint	pXyZ  = default_perm[ (pXy + uZ)	& 0x0FF ];
	uint	pxYZ  = default_perm[ (pxY + uZ)	& 0x0FF ];
	uint	pXYZ  = default_perm[ (pXY + uZ)	& 0x0FF ];

	float	gxyz  = hash_grad_dot3( pxyz, vxyz );
	float	gXyz  = hash_grad_dot3( pXyz, vXyz );
	float	gxYz  = hash_grad_dot3( pxYz, vxYz );
	float	gXYz  = hash_grad_dot3( pXYz, vXYz );
	float	gxyZ  = hash_grad_dot3( pxyZ, vxyZ );
	float	gXyZ  = hash_grad_dot3( pXyZ, vXyZ );
	float	gxYZ  = hash_grad_dot3( pxYZ, vxYZ );
	float	gXYZ  = hash_grad_dot3( pXYZ, vXYZ );

	float	wx = WEIGHT(vxyz.x);
	float	wy = WEIGHT(vxyz.y);
	float	wz = WEIGHT(vxyz.z);
	
        // interpolate to a single value
	float   result =  
                    interp( wz,
		                    interp( wy,
			                        interp(wx, gxyz, gXyz),
			                        interp(wx, gxYz, gXYz)
			                    ),
		                    interp( wy,
			                        interp(wx, gxyZ, gXyZ),
			                        interp(wx, gxYZ, gXYZ)
			                    )
		                    );

    return result;

}



// Generate "cloud" pattern values using Perlin Noise
float	cloud( float fx, float fy, float fz, float size )
{ 
	float	value = 0.0f;

	while( size >= 1.0f )
	{
		value += size * Noise_3d(fx, fy, fz); 

        size *= 0.5f;
		fx *= 2.0f;
		fy *= 2.0f;
	}
	return value;
}



kernel void
cloud3d(__global float*	output,
	  float				wx,	
	  float				wy,	
	  float				wz	
				)
{
  	int	i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	int	Nx = get_global_size(0);
	int	Ny = get_global_size(0);
	int	Nz = get_global_size(0);
	
	float	fx = (float)i/Nx * wx;	
	float	fy = (float)j/Ny * wy;	
	float	fz = (float)k/Nz * wz;	

	// int	x = get_global_id(0);
	// int y = get_global_id(1);
	// uint offset = x + y *  iImageWidth;

	// float	fx = 2.0f * (float)x * invWidth;
	// float	fy = 2.0f * (float)y * invHeight;
	// float	fz = slice;

	// float	size = (float)iImageWidth;
	// float	value = 0.0f;

	float value = cloud( fx, fy, fz, Nx );


	value *= (float)wx;

	output[i+j*Nx+k*Nx*Ny] = value;

}




kernel void
perlin2d(__global float*	output,
	    float				dx,
		float				dy,
		float				wx,		// 1/width
		float				wy,		// 1/height
		float				off_x,
		float				off_y
				)
{
	int	i = get_global_id(0);
	int j = get_global_id(1);

	int	Nx = get_global_size(0);

	float	fx = off_x+(float)i*dx/wx;
	float	fy = off_y+(float)j*dy/wy;

	float	value;


	value =  Noise_2d(fx, fy); 

	output[i+j*Nx] = value;
}

kernel void
perlin3d(__global float*	output,
		 int offz,
		float				dx,
		float				dy,
		float				dz,
		float				wx,		// 1/width
		float				wy,		// 1/height
		float				wz,
		float				ox,
		float				oy,
		float				oz


				)
{
	int	i = get_global_id(0);
	int j = get_global_id(1);
	int k = get_global_id(2);

	int	Nx = get_global_size(0);
	int	Ny = get_global_size(1);
	
	float	fx = (float)i*dx /wx+ox;
	float	fy = (float)j*dy /wy+oy;
	float	fz = (float)(offz+k)*dz/wz+oz;

	float	value;

	
	value =  Noise_3d(fx, fy, fz); 

	output[i+j*Nx+k*Nx*Ny] = value;
}
