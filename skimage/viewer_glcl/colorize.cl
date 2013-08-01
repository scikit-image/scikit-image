#define NORM 0.00392156862745098f
#define uint42f4n(c) (float4) (NORM*c.x, NORM*c.y, NORM*c.z, NORM*c.w)
#define rgba2f4(c) (float4) (c & 0x000000FF, (c & 0x0000FF00) >> 8, (c & 0x00FF0000) >> 16, (c & 0x00FF0000) >> 24)

float4 HSVtoRGB(float4 HSV)
{
        float4 RGB = (float4)0;
        if (HSV.z != 0)
        {
                float var_h = HSV.x * 6;
                float var_i = (float) ((int) (var_h-0.000001));
                float var_1 = HSV.z * (1.0 - HSV.y);
                float var_2 = HSV.z * (1.0 - HSV.y * (var_h-var_i));
                float var_3 = HSV.z * (1.0 - HSV.y * (1-(var_h-var_i)));
                switch((int)(var_i))
                {
                        case 0: RGB = (float4)(HSV.z, var_3, var_1, HSV.w); break;
                        case 1: RGB = (float4)(var_2, HSV.z, var_1, HSV.w); break;
                        case 2: RGB = (float4)(var_1, HSV.z, var_3, HSV.w); break;
                        case 3: RGB = (float4)(var_1, var_2, HSV.z, HSV.w); break;
                        case 4: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
                        default: RGB = (float4)(HSV.z, var_1, var_2, HSV.w); break;
                }
        }
        RGB.w = HSV.w;
        return (RGB);
}

float4 colorizei(int val, int2 range, int2 hues) {
	if (val < range[0])
		return (float4) (0, 0, 0, 1);
	else if (val > range[1])
		return (float4) (1, 1, 1, 1);

	float normalized = (float) (val-range[0])/(range[1]-range[0]);
	float hue = hues[0] + normalized*(hues[1]-hues[0]);

	float4 hsv = (float4) (hue/360, 1.0, 1.0, 1.0);

	return HSVtoRGB(hsv);
}

float4 colorizef(float x, float2 range, float2 hues, float2 sats, float2 vals) {
	float4 hsv;

	if (x < range[0])
		hsv = (float4) (0, 0, 0, 1);
	else if (x > range[1])
		hsv = (float4) (0, 0, 1, 1);
	else {
		float normalized = (x-range[0])/(range[1]-range[0]);

		hsv = (float4) (
			hues[0] + normalized*(hues[1]-hues[0]),
			sats[0] + normalized*(sats[1]-sats[0]),
			vals[0] + normalized*(vals[1]-vals[0]),
			1
			);
	}

	return HSVtoRGB(hsv);
}

__global kernel void colorize_ui32(
	float2 range,
	float2 hues,
	float2 sats,
	float2 vals,
	__global uint* input,
	__write_only image2d_t output,
	int2 dim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > dim.x-1 || gxy.y > dim.y-1)
		return;

	uint in = input[gxy.y*dim.x + gxy.x];
	float4 out = colorizef((float) in, range, hues, sats, vals);

	write_imagef(output, gxy, out);
}

__global kernel void colorize_i32(
	float2 range,
	float2 hues,
	float2 sats,
	float2 vals,
	__global int* input,
	__write_only image2d_t output,
	int2 dim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > dim.x-1 || gxy.y > dim.y-1)
		return;

	int in = input[gxy.y*dim.x + gxy.x];
	float4 out = colorizef((float) in, range, hues, sats, vals);

	write_imagef(output, gxy, out);
}

__global kernel void colorize_ui8(
	float2 range,
	float2 hues,
	float2 sats,
	float2 vals,
	__global uchar* input,
	__write_only image2d_t output,
	int2 dim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > dim.x-1 || gxy.y > dim.y-1)
		return;

	uchar in = input[gxy.y*dim.x + gxy.x];
	float4 out = colorizef((float) in, range, hues, sats, vals);

	write_imagef(output, gxy, out);
}


__global kernel void colorize_f32(
	float2 range,
	float2 hues,
	float2 sats,
	float2 vals,
	__global float* input,
	__write_only image2d_t output,
	int2 dim
) {
	int2 gxy = (int2) (get_global_id(0), get_global_id(1));

	if (gxy.x > dim.x-1 || gxy.y > dim.y-1)
		return;

	float in = input[gxy.y*dim.x + gxy.x];
	float4 out = colorizef(in, range, hues, sats, vals);

	write_imagef(output, gxy, out);
}
