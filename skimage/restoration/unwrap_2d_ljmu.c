// 2D phase unwrapping, modified for inclusion in scipy by Gregor Thalhammer
// Original file name: Miguel_2D_unwrapper_with_mask_and_wrap_around_option.c

// This program was written by Munther Gdeisat and Miguel Arevallilo Herraez to
// program the two-dimensional unwrapper
// entitled "Fast two-dimensional phase-unwrapping algorithm based on sorting by
// reliability following a noncontinuous path"
// by  Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor, and Munther
// A. Gdeisat
// published in the Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002.
// This program was written by Munther Gdeisat, Liverpool John Moores
// University, United Kingdom.
// Date 26th August 2007
// The wrapped phase map is assumed to be of floating point data type. The
// resultant unwrapped phase map is also of floating point type.
// The mask is of byte data type.
// When the mask is 255 this means that the pixel is valid
// When the mask is 0 this means that the pixel is invalid (noisy or corrupted
// pixel)
// This program takes into consideration the image wrap around problem
// encountered in MRI imaging.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

#define PI M_PI
#define TWOPI (2 * M_PI)

// TODO: remove global variables
// TODO: make thresholds independent

#define NOMASK 0
#define MASK 1

typedef struct {
  double mod;
  int x_connectivity;
  int y_connectivity;
  int no_of_edges;
} params_t;

// PIXELM information
struct PIXELM {
  int increment;  // No. of 2*pi to add to the pixel to unwrap it
  int number_of_pixels_in_group;  // No. of pixel in the pixel group
  double value;  // value of the pixel
  double reliability;
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

yes_no find_pivot(EDGE *left, EDGE *right, double *pivot_ptr) {
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

EDGE *partition(EDGE *left, EDGE *right, double pivot) {
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

void quicker_sort(EDGE *left, EDGE *right) {
  EDGE *p;
  double pivot;

  if (find_pivot(left, right, &pivot) == yes) {
    p = partition(left, right, pivot);
    quicker_sort(left, p - 1);
    quicker_sort(p, right);
  }
}
//--------------end quicker_sort algorithm -----------------------------------

//--------------------start initialize pixels ----------------------------------
// initialize pixels. See the explination of the pixel class above.
// initially every pixel is assumed to belong to a group consisting of only
// itself
void initialisePIXELs(double *wrapped_image, unsigned char *input_mask,
                      unsigned char *extended_mask, PIXELM *pixel,
                      int image_width, int image_height,
                      char use_seed, unsigned int seed) {
  PIXELM *pixel_pointer = pixel;
  double *wrapped_image_pointer = wrapped_image;
  unsigned char *input_mask_pointer = input_mask;
  unsigned char *extended_mask_pointer = extended_mask;
  int i, j;

  if (use_seed) {
    srand(seed);
  }

  for (i = 0; i < image_height; i++) {
    for (j = 0; j < image_width; j++) {
      pixel_pointer->increment = 0;
      pixel_pointer->number_of_pixels_in_group = 1;
      pixel_pointer->value = *wrapped_image_pointer;
      pixel_pointer->reliability = rand();
      pixel_pointer->input_mask = *input_mask_pointer;
      pixel_pointer->extended_mask = *extended_mask_pointer;
      pixel_pointer->head = pixel_pointer;
      pixel_pointer->last = pixel_pointer;
      pixel_pointer->next = NULL;
      pixel_pointer->new_group = 0;
      pixel_pointer->group = -1;
      pixel_pointer++;
      wrapped_image_pointer++;
      input_mask_pointer++;
      extended_mask_pointer++;
    }
  }
}
//-------------------end initialize pixels -----------

// gamma function in the paper
double wrap(double pixel_value) {
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
int find_wrap(double pixelL_value, double pixelR_value) {
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

void extend_mask(unsigned char *input_mask, unsigned char *extended_mask,
                 int image_width, int image_height, params_t *params) {
  int i, j;
  int image_width_plus_one = image_width + 1;
  int image_width_minus_one = image_width - 1;
  unsigned char *IMP = input_mask + image_width + 1;  // input mask pointer
  unsigned char *EMP = extended_mask + image_width + 1;  // extended mask
                                                         // pointer

  // extend the mask for the image except borders
  for (i = 1; i < image_height - 1; ++i) {
    for (j = 1; j < image_width - 1; ++j) {
      if ((*IMP) == NOMASK && (*(IMP + 1) == NOMASK) &&
          (*(IMP - 1) == NOMASK) && (*(IMP + image_width) == NOMASK) &&
          (*(IMP - image_width) == NOMASK) &&
          (*(IMP - image_width_minus_one) == NOMASK) &&
          (*(IMP - image_width_plus_one) == NOMASK) &&
          (*(IMP + image_width_minus_one) == NOMASK) &&
          (*(IMP + image_width_plus_one) == NOMASK)) {
        *EMP = NOMASK;
      }
      ++EMP;
      ++IMP;
    }
    EMP += 2;
    IMP += 2;
  }

  if (params->x_connectivity == 1) {
    // extend the mask for the right border of the image
    IMP = input_mask + 2 * image_width - 1;
    EMP = extended_mask + 2 * image_width - 1;
    for (i = 1; i < image_height - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + image_width) == NOMASK) &&
          (*(IMP - image_width) == NOMASK) &&
          (*(IMP - image_width - 1) == NOMASK) &&
          (*(IMP - image_width + 1) == NOMASK) &&
          (*(IMP + image_width - 1) == NOMASK) &&
          (*(IMP - 2 * image_width + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP += image_width;
      IMP += image_width;
    }

    // extend the mask for the left border of the image
    IMP = input_mask + image_width;
    EMP = extended_mask + image_width;
    for (i = 1; i < image_height - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + image_width) == NOMASK) &&
          (*(IMP - image_width) == NOMASK) &&
          (*(IMP - image_width + 1) == NOMASK) &&
          (*(IMP + image_width + 1) == NOMASK) &&
          (*(IMP + image_width - 1) == NOMASK) &&
          (*(IMP + 2 * image_width - 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP += image_width;
      IMP += image_width;
    }
  }

  if (params->y_connectivity == 1) {
    // extend the mask for the top border of the image
    IMP = input_mask + 1;
    EMP = extended_mask + 1;
    for (i = 1; i < image_width - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + image_width) == NOMASK) &&
          (*(IMP + image_width * (image_height - 1)) == NOMASK) &&
          (*(IMP + image_width + 1) == NOMASK) &&
          (*(IMP + image_width - 1) == NOMASK) &&
          (*(IMP + image_width * (image_height - 1) - 1) == NOMASK) &&
          (*(IMP + image_width * (image_height - 1) + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP++;
      IMP++;
    }

    // extend the mask for the bottom border of the image
    IMP = input_mask + image_width * (image_height - 1) + 1;
    EMP = extended_mask + image_width * (image_height - 1) + 1;
    for (i = 1; i < image_width - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP - image_width) == NOMASK) &&
          (*(IMP - image_width - 1) == NOMASK) &&
          (*(IMP - image_width + 1) == NOMASK) &&
          (*(IMP - image_width * (image_height - 1)) == NOMASK) &&
          (*(IMP - image_width * (image_height - 1) - 1) == NOMASK) &&
          (*(IMP - image_width * (image_height - 1) + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP++;
      IMP++;
    }
  }
}

void calculate_reliability(double *wrappedImage, PIXELM *pixel, int image_width,
                           int image_height, params_t *params) {
  int image_width_plus_one = image_width + 1;
  int image_width_minus_one = image_width - 1;
  PIXELM *pixel_pointer = pixel + image_width_plus_one;
  double *WIP =
      wrappedImage + image_width_plus_one;  // WIP is the wrapped image pointer
  double H, V, D1, D2;
  int i, j;

  for (i = 1; i < image_height - 1; ++i) {
    for (j = 1; j < image_width - 1; ++j) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - image_width) - *WIP) -
            wrap(*WIP - *(WIP + image_width));
        D1 = wrap(*(WIP - image_width_plus_one) - *WIP) -
             wrap(*WIP - *(WIP + image_width_plus_one));
        D2 = wrap(*(WIP - image_width_minus_one) - *WIP) -
             wrap(*WIP - *(WIP + image_width_minus_one));
        pixel_pointer->reliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }
    pixel_pointer += 2;
    WIP += 2;
  }

  if (params->x_connectivity == 1) {
    // calculating the reliability for the left border of the image
    PIXELM *pixel_pointer = pixel + image_width;
    double *WIP = wrappedImage + image_width;

    for (i = 1; i < image_height - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP + image_width - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - image_width) - *WIP) -
            wrap(*WIP - *(WIP + image_width));
        D1 = wrap(*(WIP - 1) - *WIP) -
             wrap(*WIP - *(WIP + image_width_plus_one));
        D2 = wrap(*(WIP - image_width_minus_one) - *WIP) -
             wrap(*WIP - *(WIP + 2 * image_width - 1));
        pixel_pointer->reliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer += image_width;
      WIP += image_width;
    }

    // calculating the reliability for the right border of the image
    pixel_pointer = pixel + 2 * image_width - 1;
    WIP = wrappedImage + 2 * image_width - 1;

    for (i = 1; i < image_height - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) -
            wrap(*WIP - *(WIP - image_width_minus_one));
        V = wrap(*(WIP - image_width) - *WIP) -
            wrap(*WIP - *(WIP + image_width));
        D1 = wrap(*(WIP - image_width_plus_one) - *WIP) -
             wrap(*WIP - *(WIP + 1));
        D2 = wrap(*(WIP - 2 * image_width - 1) - *WIP) -
             wrap(*WIP - *(WIP + image_width_minus_one));
        pixel_pointer->reliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer += image_width;
      WIP += image_width;
    }
  }

  if (params->y_connectivity == 1) {
    // calculating the reliability for the top border of the image
    PIXELM *pixel_pointer = pixel + 1;
    double *WIP = wrappedImage + 1;

    for (i = 1; i < image_width - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP + image_width * (image_height - 1)) - *WIP) -
            wrap(*WIP - *(WIP + image_width));
        D1 = wrap(*(WIP + image_width * (image_height - 1) - 1) - *WIP) -
             wrap(*WIP - *(WIP + image_width_plus_one));
        D2 = wrap(*(WIP + image_width * (image_height - 1) + 1) - *WIP) -
             wrap(*WIP - *(WIP + image_width_minus_one));
        pixel_pointer->reliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }

    // calculating the reliability for the bottom border of the image
    pixel_pointer = pixel + (image_height - 1) * image_width + 1;
    WIP = wrappedImage + (image_height - 1) * image_width + 1;

    for (i = 1; i < image_width - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - image_width) - *WIP) -
            wrap(*WIP - *(WIP - (image_height - 1) * (image_width)));
        D1 = wrap(*(WIP - image_width_plus_one) - *WIP) -
             wrap(*WIP - *(WIP - (image_height - 1) * (image_width) + 1));
        D2 = wrap(*(WIP - image_width_minus_one) - *WIP) -
             wrap(*WIP - *(WIP - (image_height - 1) * (image_width) - 1));
        pixel_pointer->reliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }
  }
}

// calculate the reliability of the horizontal edges of the image
// it is calculated by adding the reliability of pixel and the relibility of
// its right-hand neighbour
// edge is calculated between a pixel and its next neighbour
void horizontalEDGEs(PIXELM *pixel, EDGE *edge, int image_width,
                     int image_height, params_t *params) {
  int i, j;
  EDGE *edge_pointer = edge;
  PIXELM *pixel_pointer = pixel;
  int no_of_edges = params->no_of_edges;

  for (i = 0; i < image_height; i++) {
    for (j = 0; j < image_width - 1; j++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer + 1)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer + 1);
        edge_pointer->reliab =
            pixel_pointer->reliability + (pixel_pointer + 1)->reliability;
        edge_pointer->increment =
            find_wrap(pixel_pointer->value, (pixel_pointer + 1)->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer++;
    }
    pixel_pointer++;
  }
  // construct edges at the right border of the image
  if (params->x_connectivity == 1) {
    pixel_pointer = pixel + image_width - 1;
    for (i = 0; i < image_height; i++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer - image_width + 1)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer - image_width + 1);
        edge_pointer->reliab = pixel_pointer->reliability +
                               (pixel_pointer - image_width + 1)->reliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value, (pixel_pointer - image_width + 1)->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer += image_width;
    }
  }
  params->no_of_edges = no_of_edges;
}

// calculate the reliability of the vertical edges of the image
// it is calculated by adding the reliability of pixel and the relibility of
// its lower neighbour in the image.
void verticalEDGEs(PIXELM *pixel, EDGE *edge, int image_width, int image_height,
                   params_t *params) {
  int i, j;
  int no_of_edges = params->no_of_edges;
  PIXELM *pixel_pointer = pixel;
  EDGE *edge_pointer = edge + no_of_edges;

  for (i = 0; i < image_height - 1; i++) {
    for (j = 0; j < image_width; j++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer + image_width)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer + image_width);
        edge_pointer->reliab = pixel_pointer->reliability +
                               (pixel_pointer + image_width)->reliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value, (pixel_pointer + image_width)->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer++;
    }  // j loop
  }  // i loop

  // construct edges that connect at the bottom border of the image
  if (params->y_connectivity == 1) {
    pixel_pointer = pixel + image_width * (image_height - 1);
    for (i = 0; i < image_width; i++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer - image_width * (image_height - 1))->input_mask ==
              NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 =
            (pixel_pointer - image_width * (image_height - 1));
        edge_pointer->reliab =
            pixel_pointer->reliability +
            (pixel_pointer - image_width * (image_height - 1))->reliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value,
            (pixel_pointer - image_width * (image_height - 1))->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer++;
    }
  }
  params->no_of_edges = no_of_edges;
}

// gather the pixels of the image into groups
void gatherPIXELs(EDGE *edge, params_t *params) {
  int k;
  PIXELM *PIXEL1;
  PIXELM *PIXEL2;
  PIXELM *group1;
  PIXELM *group2;
  EDGE *pointer_edge = edge;
  int incremento;

  for (k = 0; k < params->no_of_edges; k++) {
    PIXEL1 = pointer_edge->pointer_1;
    PIXEL2 = pointer_edge->pointer_2;

    // PIXELM 1 and PIXELM 2 belong to different groups
    // initially each pixel is a group by it self and one pixel can construct a
    // group
    // no else or else if to this if
    if (PIXEL2->head != PIXEL1->head) {
      // PIXELM 2 is alone in its group
      // merge this pixel with PIXELM 1 group and find the number of 2 pi to add
      // to or subtract to unwrap it
      if ((PIXEL2->next == NULL) && (PIXEL2->head == PIXEL2)) {
        PIXEL1->head->last->next = PIXEL2;
        PIXEL1->head->last = PIXEL2;
        (PIXEL1->head->number_of_pixels_in_group)++;
        PIXEL2->head = PIXEL1->head;
        PIXEL2->increment = PIXEL1->increment - pointer_edge->increment;
      }

      // PIXELM 1 is alone in its group
      // merge this pixel with PIXELM 2 group and find the number of 2 pi to add
      // to or subtract to unwrap it
      else if ((PIXEL1->next == NULL) && (PIXEL1->head == PIXEL1)) {
        PIXEL2->head->last->next = PIXEL1;
        PIXEL2->head->last = PIXEL1;
        (PIXEL2->head->number_of_pixels_in_group)++;
        PIXEL1->head = PIXEL2->head;
        PIXEL1->increment = PIXEL2->increment + pointer_edge->increment;
      }

      // PIXELM 1 and PIXELM 2 both have groups
      else {
        group1 = PIXEL1->head;
        group2 = PIXEL2->head;
        // if the no. of pixels in PIXELM 1 group is larger than the
        // no. of pixels in PIXELM 2 group.  Merge PIXELM 2 group to
        // PIXELM 1 group and find the number of wraps between PIXELM 2
        // group and PIXELM 1 group to unwrap PIXELM 2 group with respect
        // to PIXELM 1 group.  the no. of wraps will be added to PIXELM 2
        // group in the future
        if (group1->number_of_pixels_in_group >
            group2->number_of_pixels_in_group) {
          // merge PIXELM 2 with PIXELM 1 group
          group1->last->next = group2;
          group1->last = group2->last;
          group1->number_of_pixels_in_group =
              group1->number_of_pixels_in_group +
              group2->number_of_pixels_in_group;
          incremento =
              PIXEL1->increment - pointer_edge->increment - PIXEL2->increment;
          // merge the other pixels in PIXELM 2 group to PIXELM 1 group
          while (group2 != NULL) {
            group2->head = group1;
            group2->increment += incremento;
            group2 = group2->next;
          }
        }

        // if the no. of pixels in PIXELM 2 group is larger than the
        // no. of pixels in PIXELM 1 group.  Merge PIXELM 1 group to
        // PIXELM 2 group and find the number of wraps between PIXELM 2
        // group and PIXELM 1 group to unwrap PIXELM 1 group with respect
        // to PIXELM 2 group.  the no. of wraps will be added to PIXELM 1
        // group in the future
        else {
          // merge PIXELM 1 with PIXELM 2 group
          group2->last->next = group1;
          group2->last = group1->last;
          group2->number_of_pixels_in_group =
              group2->number_of_pixels_in_group +
              group1->number_of_pixels_in_group;
          incremento =
              PIXEL2->increment + pointer_edge->increment - PIXEL1->increment;
          // merge the other pixels in PIXELM 2 group to PIXELM 1 group
          while (group1 != NULL) {
            group1->head = group2;
            group1->increment += incremento;
            group1 = group1->next;
          }  // while

        }  // else
      }  // else
    }  // if
    pointer_edge++;
  }
}

// unwrap the image
void unwrapImage(PIXELM *pixel, int image_width, int image_height) {
  int i;
  int image_size = image_width * image_height;
  PIXELM *pixel_pointer = pixel;

  for (i = 0; i < image_size; i++) {
    pixel_pointer->value += TWOPI * (double)(pixel_pointer->increment);
    pixel_pointer++;
  }
}

// set the masked pixels (mask = 0) to the minimum of the unwrapper phase
void maskImage(PIXELM *pixel, unsigned char *input_mask, int image_width,
               int image_height) {
  int image_width_plus_one = image_width + 1;
  int image_height_plus_one = image_height + 1;
  int image_width_minus_one = image_width - 1;
  int image_height_minus_one = image_height - 1;

  PIXELM *pointer_pixel = pixel;
  unsigned char *IMP = input_mask;  // input mask pointer
  double min = DBL_MAX;
  int i;
  int image_size = image_width * image_height;

  // find the minimum of the unwrapped phase
  for (i = 0; i < image_size; i++) {
    if ((pointer_pixel->value < min) && (*IMP == NOMASK))
      min = pointer_pixel->value;

    pointer_pixel++;
    IMP++;
  }

  pointer_pixel = pixel;
  IMP = input_mask;

  // set the masked pixels to minimum
  for (i = 0; i < image_size; i++) {
    if ((*IMP) == MASK) {
      pointer_pixel->value = min;
    }
    pointer_pixel++;
    IMP++;
  }
}

// the input to this unwrapper is an array that contains the wrapped
// phase map.  copy the image on the buffer passed to this unwrapper to
// over-write the unwrapped phase map on the buffer of the wrapped
// phase map.
void returnImage(PIXELM *pixel, double *unwrapped_image, int image_width,
                 int image_height) {
  int i;
  int image_size = image_width * image_height;
  double *unwrapped_image_pointer = unwrapped_image;
  PIXELM *pixel_pointer = pixel;

  for (i = 0; i < image_size; i++) {
    *unwrapped_image_pointer = pixel_pointer->value;
    pixel_pointer++;
    unwrapped_image_pointer++;
  }
}

// the main function of the unwrapper
void unwrap2D(double *wrapped_image, double *UnwrappedImage,
              unsigned char *input_mask, int image_width, int image_height,
              int wrap_around_x, int wrap_around_y,
              char use_seed, unsigned int seed) {
  params_t params = {TWOPI, wrap_around_x, wrap_around_y, 0};
  unsigned char *extended_mask;
  PIXELM *pixel;
  EDGE *edge;
  int image_size = image_height * image_width;
  int No_of_Edges_initially = 2 * image_width * image_height;

  extended_mask = (unsigned char *)calloc(image_size, sizeof(unsigned char));
  pixel = (PIXELM *)calloc(image_size, sizeof(PIXELM));
  edge = (EDGE *)calloc(No_of_Edges_initially, sizeof(EDGE));

  extend_mask(input_mask, extended_mask, image_width, image_height, &params);
  initialisePIXELs(wrapped_image, input_mask, extended_mask, pixel, image_width,
                   image_height, use_seed, seed);
  calculate_reliability(wrapped_image, pixel, image_width, image_height,
                        &params);
  horizontalEDGEs(pixel, edge, image_width, image_height, &params);
  verticalEDGEs(pixel, edge, image_width, image_height, &params);

  if (params.no_of_edges != 0) {
      // sort the EDGEs depending on their reiability. The PIXELs with higher
      // relibility (small value) first
      quicker_sort(edge, edge + params.no_of_edges - 1);
  }
  // gather PIXELs into groups
  gatherPIXELs(edge, &params);

  unwrapImage(pixel, image_width, image_height);
  maskImage(pixel, input_mask, image_width, image_height);

  // copy the image from PIXELM structure to the unwrapped phase array
  // passed to this function
  // TODO: replace by (cython?) function to directly write into numpy array ?
  returnImage(pixel, UnwrappedImage, image_width, image_height);

  free(edge);
  free(pixel);
  free(extended_mask);
}
