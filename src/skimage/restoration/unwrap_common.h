/* Common routines and constants for unwrapping routines */

#ifndef UNWRAP_COMMON_H
#define UNWRAP_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <numpy/random/bitgen.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define PI M_PI
#define TWOPI (2 * M_PI)

#define NOMASK 0
#define MASK 1

// Starting unreliability value for otherwise uninitialized pixels.
// The high value forces initial low reliability.  See Note in unwrap.py.
#define UNRELIABILITY_SENTINEL 9999999.f

// PIXELM information
struct PIXELM {
  int increment;  // No. of 2*pi to add to the pixel to unwrap it
  intptr_t number_of_pixels_in_group;  // No. of pixel in the pixel group
  double value;  // value of the pixel
  double unreliability;
  unsigned char input_mask;  // 0 pixel is masked. NOMASK pixel is not masked
  unsigned char extended_mask;  // 0 pixel is masked. NOMASK pixel is not masked
  int group;  // group No.
  int new_group;
  struct PIXELM *head;  // pointer to the first pixel in the group in the linked
                        // list
  struct PIXELM *last;  // pointer to the last pixel in the group
  struct PIXELM *next;  // pointer to the next pixel in the group
};

typedef struct PIXELM PIXELM;

// the EDGE is the line that connects two pixels.
// if we have S pixels, then we have S horizontal edges and S vertical edges
struct EDGE {
  double reliab;  // reliabilty of the edge and it depends on the two pixels
  PIXELM *pointer_1;  // pointer to the first pixel
  PIXELM *pointer_2;  // pointer to the second pixel
  int increment;  // No. of 2*pi to add to one of the pixels to
  // unwrap it with respect to the second
};

typedef struct EDGE EDGE;

//---------------start quicker_sort algorithm --------------------------------
#define swap(x, y) \
  {                \
    EDGE t;        \
    t = x;         \
    x = y;         \
    y = t;         \
  }
#define order(x, y) \
  if (x.reliab > y.reliab) swap(x, y)
#define o2(x, y) order(x, y)
#define o3(x, y, z) \
  o2(x, y);         \
  o2(x, z);         \
  o2(y, z)

typedef enum {
  yes,
  no
} yes_no;

static inline yes_no find_pivot(EDGE *left, EDGE *right, double *pivot_ptr) {
  EDGE a, b, c, *p;

  a = *left;
  b = *(left + (right - left) / 2);
  c = *right;
  o3(a, b, c);

  if (a.reliab < b.reliab) {
    *pivot_ptr = b.reliab;
    return yes;
  }

  if (b.reliab < c.reliab) {
    *pivot_ptr = c.reliab;
    return yes;
  }

  for (p = left + 1; p <= right; ++p) {
    if (p->reliab != left->reliab) {
      *pivot_ptr = (p->reliab < left->reliab) ? left->reliab : p->reliab;
      return yes;
    }
  }
  return no;
}

static inline EDGE *partition(EDGE *left, EDGE *right, double pivot) {
  while (left <= right) {
    while (left->reliab < pivot) ++left;
    while (right->reliab >= pivot) --right;
    if (left < right) {
      swap(*left, *right);
      ++left;
      --right;
    }
  }
  return left;
}

static inline void quicker_sort(EDGE *left, EDGE *right) {
  EDGE *p;
  double pivot;

  if (find_pivot(left, right, &pivot) == yes) {
    p = partition(left, right, pivot);
    quicker_sort(left, p - 1);
    quicker_sort(p, right);
  }
}
//--------------end quicker_sort algorithm -----------------------------------

// gamma function in the paper
static inline double wrap(double pixel_value) {
  double wrapped_pixel_value;
  if (pixel_value > PI)
    wrapped_pixel_value = pixel_value - TWOPI;
  else if (pixel_value < -PI)
    wrapped_pixel_value = pixel_value + TWOPI;
  else
    wrapped_pixel_value = pixel_value;
  return wrapped_pixel_value;
}

// pixelL_value is the left pixel,  pixelR_value is the right pixel
static inline int find_wrap(double pixelL_value, double pixelR_value) {
  double difference;
  int wrap_value;
  difference = pixelL_value - pixelR_value;

  if (difference > PI)
    wrap_value = -1;
  else if (difference < -PI)
    wrap_value = 1;
  else
    wrap_value = 0;

  return wrap_value;
}

#endif
