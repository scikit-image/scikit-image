#include <xmmintrin.h>
#include <malloc.h> 
#include <math.h>

void inline convolve_row_float(const float** src, float* dst, const float* kernel, int width, int kernel_length) {
    int i = 0, k;
    __m128 d4 = _mm_set1_ps(0); //delta
    for( ; i <= width - 16; i += 16) {
        __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;
        for (k = 0; k < kernel_length; k++) {
            __m128 f = _mm_load_ss(kernel+k);
            f = _mm_shuffle_ps(f, f, 0);
            const float* S = src[k] + i;
            __m128 t0 = _mm_loadu_ps(S);
            __m128 t1 = _mm_loadu_ps(S + 4);
            s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));
            t0 = _mm_loadu_ps(S + 8);
            t1 = _mm_loadu_ps(S + 12);
            s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
            s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
        }
        _mm_storeu_ps(dst + i, s0);
        _mm_storeu_ps(dst + i + 4, s1);
        _mm_storeu_ps(dst + i + 8, s2);
        _mm_storeu_ps(dst + i + 12, s3);
    }
    for( ; i <= width - 4; i += 4) {
        __m128 s0 = d4;
        for( k = 0; k < kernel_length; k++ ) {
            __m128 f = _mm_load_ss(kernel+k), t0;
            f = _mm_shuffle_ps(f, f, 0);
            t0 = _mm_loadu_ps(src[k] + i);
            s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
        }
        _mm_storeu_ps(dst + i, s0);
    }
}

void convolve(float* src, float* dst, float* kernel, int width, int height, 
    int kernel_width, int kernel_height, int anchor_x, int anchor_y) {
    int k, y, delta, j, copy_width;
    int length = width * height;
    int filter_length = kernel_width * kernel_height;
    int* offsets = (int*) malloc(filter_length * sizeof(int));
    int* offset_row = (int*) malloc(filter_length * sizeof(int));
    float** buffer = (float**) malloc(filter_length * sizeof(float*));
    float *src_row, *dst_row, *source, *current_row, *iter, *copy_dest, *copy_source;
    float value;
    // buffer width aligned to 4 bytes
    int w_aligned = (int)(ceil(width / 4.0) * 4);
    float* out_buffer = (float*) memalign(16, w_aligned * sizeof(float));
    // anchor as (-1, -1) indicates the middle of kernel
    if ((anchor_x == -1) && (anchor_y == -1)) {
        anchor_x = kernel_width / 2;
        anchor_y = kernel_height / 2;
    }
    for(k = 0; k < filter_length; k++ ) {
        offsets[k] = (k % kernel_width) - anchor_x + (k / kernel_width - anchor_y)*width;
        offset_row[k] = (k / kernel_width - anchor_y);
        buffer[k] = (float*) memalign(16, w_aligned * sizeof(float));
    }
    for (y = 0; y < height; y++) {
        src_row = src + y * width;
        for(k = 0; k < filter_length; k++ ) {
            source = src_row + offsets[k];
            current_row = src_row + offset_row[k]*width;
            // handle y < 0 border
            if (current_row < src) {
                source += (src - current_row);
                current_row = src;
            // handle y > height border
            } else if (current_row >= src + length)  {
                source -= (current_row - (src + length - width));
                current_row = src + length - width;
            }
            // copy source to buffer line
            //memcpy(buffer[k], source, width*sizeof(float));
            if (source < current_row) {
                delta = current_row - source;
                copy_dest = buffer[k] + delta;
                copy_source = current_row;
                copy_width = w_aligned - delta;
                // handle x < 0 border
                if (copy_width > width) {
                    copy_width = width;
                    if (delta + width < w_aligned) {
                        value = *(current_row + width - 1);
                        for (iter = buffer[k] + delta + width; iter < buffer[k] + w_aligned; iter++) { 
                            *iter = value;
                        }
                    }
                }
                if (copy_width < 0) {
                    copy_width = 0;
                    // offset more to the left than the buffer size
                    delta = w_aligned;
                }
                // first row value
                value = *(current_row);
                for (iter = buffer[k]; iter < buffer[k] + delta ; iter++) {
                    *iter = value;
                }
            }
             else {
                delta = source - current_row;
                copy_dest = buffer[k];
                copy_source = source;
                copy_width = width - delta;
                // handle x > width border                
                if (copy_width < 0)
                    copy_width = 0;
                // last row value
                value = *(current_row + width - 1);
                for (iter = buffer[k] + copy_width; iter < buffer[k] + w_aligned; iter++) { 
                    *iter = value;
                }
            }
            if (copy_width > 0) {
                memcpy(copy_dest, copy_source, copy_width*sizeof(float));
            }
        }
        dst_row = dst + y * width;
        convolve_row_float((const float**)buffer, out_buffer, (const float*)kernel, w_aligned, filter_length);
        // copy output buffer row to destination
        memcpy(dst_row, out_buffer, width * sizeof(float));
    }
    // clean up
    free(out_buffer);
    free(offsets);
    free(offset_row);
    for(k = 0; k < filter_length; k++ ) {
        free(buffer[k]);
    }
    free(buffer);
}

//            if (y == 0) {
//                for (j = 0; j < w_aligned; j++)
//                    printf("%.2f ", buffer[k][j]);
//                printf("\n");
//            }

//     if (copy_dest < buffer[k])
//                    printf("XXXX 3\n");                
//                if(copy_dest + copy_width > buffer[k] + w_aligned)
//                    printf("XXXX 4\n");                
//                if(copy_source < src)
//                    printf("XXXX 5\n");                
//                if(copy_source >= src + length) 
//                    printf("XXXX 6 %d %d %d %d %d\n", width, delta, length, current_row-src,current_row + width-1-src);
