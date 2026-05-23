// 2D phase unwrapping, modified for inclusion in scipy by Gregor Thalhammer
// Original file name: Miguel_2D_unwrapper_with_mask_and_wrap_around_option.c

// This program was written by Munther Gdeisat and Miguel Arevallilo Herraez to
// program the two-dimensional unwrapper entitled "Fast two-dimensional
// phase-unwrapping algorithm based on sorting by reliability following a
// noncontinuous path" by  Miguel Arevallilo Herraez, David R. Burton, Michael
// J. Lalor, and Munther A. Gdeisat published in the Journal Applied Optics,
// Vol. 41, No. 35, pp. 7437, 2002.
//
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

#include "unwrap_2d_ljmu.h"

typedef struct {
  double mod;
  int x_connectivity;
  int y_connectivity;
  intptr_t no_of_edges;
} params_t;

//--------------------start initialize pixels ----------------------------------
// Initialize pixels. See the explanation of the pixel class in header above.
// Initially every pixel is assumed to belong to a group consisting of only
// itself.
void initialisePIXELs(double *wrapped_image, unsigned char *input_mask,
                      unsigned char *extended_mask, PIXELM *pixel,
                      intptr_t n_j, intptr_t n_i,
                      bitgen_t* bitgen_state) {
  PIXELM *pixel_pointer = pixel;
  double *wrapped_image_pointer = wrapped_image;
  unsigned char *input_mask_pointer = input_mask;
  unsigned char *extended_mask_pointer = extended_mask;
  intptr_t i, j;

  for (i = 0; i < n_i; i++) {
    for (j = 0; j < n_j; j++) {
      pixel_pointer->increment = 0;
      pixel_pointer->number_of_pixels_in_group = 1;
      pixel_pointer->value = *wrapped_image_pointer;
      // See Note in unwrap.py.
      pixel_pointer->unreliability = UNRELIABILITY_SENTINEL +
          bitgen_state->next_uint32(bitgen_state->state);
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


void extend_mask(unsigned char *input_mask, unsigned char *extended_mask,
                 intptr_t n_j, intptr_t n_i, params_t *params) {
  intptr_t i, j;
  intptr_t n_j_plus_one = n_j + 1;
  intptr_t n_j_minus_one = n_j - 1;
  unsigned char *IMP = input_mask + n_j + 1;  // input mask pointer
  unsigned char *EMP = extended_mask + n_j + 1;  // extended mask
                                                         // pointer

  // extend the mask for the image except borders
  for (i = 1; i < n_i - 1; ++i) {
    for (j = 1; j < n_j - 1; ++j) {
      if ((*IMP) == NOMASK && (*(IMP + 1) == NOMASK) &&
          (*(IMP - 1) == NOMASK) && (*(IMP + n_j) == NOMASK) &&
          (*(IMP - n_j) == NOMASK) &&
          (*(IMP - n_j_minus_one) == NOMASK) &&
          (*(IMP - n_j_plus_one) == NOMASK) &&
          (*(IMP + n_j_minus_one) == NOMASK) &&
          (*(IMP + n_j_plus_one) == NOMASK)) {
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
    IMP = input_mask + 2 * n_j - 1;
    EMP = extended_mask + 2 * n_j - 1;
    for (i = 1; i < n_i - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + n_j) == NOMASK) &&
          (*(IMP - n_j) == NOMASK) &&
          (*(IMP - n_j - 1) == NOMASK) &&
          (*(IMP - n_j + 1) == NOMASK) &&
          (*(IMP + n_j - 1) == NOMASK) &&
          (*(IMP - 2 * n_j + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP += n_j;
      IMP += n_j;
    }

    // extend the mask for the left border of the image
    IMP = input_mask + n_j;
    EMP = extended_mask + n_j;
    for (i = 1; i < n_i - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + n_j) == NOMASK) &&
          (*(IMP - n_j) == NOMASK) &&
          (*(IMP - n_j + 1) == NOMASK) &&
          (*(IMP + n_j + 1) == NOMASK) &&
          (*(IMP + n_j - 1) == NOMASK) &&
          (*(IMP + 2 * n_j - 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP += n_j;
      IMP += n_j;
    }
  }

  if (params->y_connectivity == 1) {
    // extend the mask for the top border of the image
    IMP = input_mask + 1;
    EMP = extended_mask + 1;
    for (i = 1; i < n_j - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP + n_j) == NOMASK) &&
          (*(IMP + n_j * (n_i - 1)) == NOMASK) &&
          (*(IMP + n_j + 1) == NOMASK) &&
          (*(IMP + n_j - 1) == NOMASK) &&
          (*(IMP + n_j * (n_i - 1) - 1) == NOMASK) &&
          (*(IMP + n_j * (n_i - 1) + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP++;
      IMP++;
    }

    // extend the mask for the bottom border of the image
    IMP = input_mask + n_j * (n_i - 1) + 1;
    EMP = extended_mask + n_j * (n_i - 1) + 1;
    for (i = 1; i < n_j - 1; ++i) {
      if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
          (*(IMP + 1) == NOMASK) && (*(IMP - n_j) == NOMASK) &&
          (*(IMP - n_j - 1) == NOMASK) &&
          (*(IMP - n_j + 1) == NOMASK) &&
          (*(IMP - n_j * (n_i - 1)) == NOMASK) &&
          (*(IMP - n_j * (n_i - 1) - 1) == NOMASK) &&
          (*(IMP - n_j * (n_i - 1) + 1) == NOMASK)) {
        *EMP = NOMASK;
      }
      EMP++;
      IMP++;
    }
  }
}

void calculate_unreliability(double *wrappedImage, PIXELM *pixel, intptr_t n_j,
                           intptr_t n_i, params_t *params) {
  intptr_t n_j_plus_one = n_j + 1;
  intptr_t n_j_minus_one = n_j - 1;
  PIXELM *pixel_pointer = pixel + n_j_plus_one;
  double *WIP =
      wrappedImage + n_j_plus_one;  // WIP is the wrapped image pointer
  double H, V, D1, D2;
  intptr_t i, j;

  for (i = 1; i < n_i - 1; ++i) {
    for (j = 1; j < n_j - 1; ++j) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - n_j) - *WIP) -
            wrap(*WIP - *(WIP + n_j));
        D1 = wrap(*(WIP - n_j_plus_one) - *WIP) -
             wrap(*WIP - *(WIP + n_j_plus_one));
        D2 = wrap(*(WIP - n_j_minus_one) - *WIP) -
             wrap(*WIP - *(WIP + n_j_minus_one));
        pixel_pointer->unreliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }
    pixel_pointer += 2;
    WIP += 2;
  }

  if (params->x_connectivity == 1) {
    // calculating the unreliability for the left border of the image
    PIXELM *pixel_pointer = pixel + n_j;
    double *WIP = wrappedImage + n_j;

    for (i = 1; i < n_i - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP + n_j - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - n_j) - *WIP) -
            wrap(*WIP - *(WIP + n_j));
        D1 = wrap(*(WIP - 1) - *WIP) -
             wrap(*WIP - *(WIP + n_j_plus_one));
        D2 = wrap(*(WIP - n_j_minus_one) - *WIP) -
             wrap(*WIP - *(WIP + 2 * n_j - 1));
        pixel_pointer->unreliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer += n_j;
      WIP += n_j;
    }

    // calculating the unreliability for the right border of the image
    pixel_pointer = pixel + 2 * n_j - 1;
    WIP = wrappedImage + 2 * n_j - 1;

    for (i = 1; i < n_i - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) -
            wrap(*WIP - *(WIP - n_j_minus_one));
        V = wrap(*(WIP - n_j) - *WIP) -
            wrap(*WIP - *(WIP + n_j));
        D1 = wrap(*(WIP - n_j_plus_one) - *WIP) -
             wrap(*WIP - *(WIP + 1));
        D2 = wrap(*(WIP - 2 * n_j - 1) - *WIP) -
             wrap(*WIP - *(WIP + n_j_minus_one));
        pixel_pointer->unreliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer += n_j;
      WIP += n_j;
    }
  }

  if (params->y_connectivity == 1) {
    // calculating the unreliability for the top border of the image
    PIXELM *pixel_pointer = pixel + 1;
    double *WIP = wrappedImage + 1;

    for (i = 1; i < n_j - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP + n_j * (n_i - 1)) - *WIP) -
            wrap(*WIP - *(WIP + n_j));
        D1 = wrap(*(WIP + n_j * (n_i - 1) - 1) - *WIP) -
             wrap(*WIP - *(WIP + n_j_plus_one));
        D2 = wrap(*(WIP + n_j * (n_i - 1) + 1) - *WIP) -
             wrap(*WIP - *(WIP + n_j_minus_one));
        pixel_pointer->unreliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }

    // calculating the unreliability for the bottom border of the image
    pixel_pointer = pixel + (n_i - 1) * n_j + 1;
    WIP = wrappedImage + (n_i - 1) * n_j + 1;

    for (i = 1; i < n_j - 1; ++i) {
      if (pixel_pointer->extended_mask == NOMASK) {
        H = wrap(*(WIP - 1) - *WIP) - wrap(*WIP - *(WIP + 1));
        V = wrap(*(WIP - n_j) - *WIP) -
            wrap(*WIP - *(WIP - (n_i - 1) * (n_j)));
        D1 = wrap(*(WIP - n_j_plus_one) - *WIP) -
             wrap(*WIP - *(WIP - (n_i - 1) * (n_j) + 1));
        D2 = wrap(*(WIP - n_j_minus_one) - *WIP) -
             wrap(*WIP - *(WIP - (n_i - 1) * (n_j) - 1));
        pixel_pointer->unreliability = H * H + V * V + D1 * D1 + D2 * D2;
      }
      pixel_pointer++;
      WIP++;
    }
  }
}

// calculate the unreliability of the horizontal edges of the image
// it is calculated by adding the unreliability of pixel and the relibility of
// its right-hand neighbour
// edge is calculated between a pixel and its next neighbour
void horizontalEDGEs(PIXELM *pixel, EDGE *edge, intptr_t n_j,
                     intptr_t n_i, params_t *params) {
  intptr_t i, j;
  EDGE *edge_pointer = edge;
  PIXELM *pixel_pointer = pixel;
  intptr_t no_of_edges = params->no_of_edges;

  for (i = 0; i < n_i; i++) {
    for (j = 0; j < n_j - 1; j++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer + 1)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer + 1);
        edge_pointer->reliab =
            pixel_pointer->unreliability + (pixel_pointer + 1)->unreliability;
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
    pixel_pointer = pixel + n_j - 1;
    for (i = 0; i < n_i; i++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer - n_j + 1)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer - n_j + 1);
        edge_pointer->reliab = pixel_pointer->unreliability +
                               (pixel_pointer - n_j + 1)->unreliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value, (pixel_pointer - n_j + 1)->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer += n_j;
    }
  }
  params->no_of_edges = no_of_edges;
}

// calculate the unreliability of the vertical edges of the image
// it is calculated by adding the unreliability of pixel and the relibility of
// its lower neighbour in the image.
void verticalEDGEs(PIXELM *pixel, EDGE *edge, intptr_t n_j, intptr_t n_i,
                   params_t *params) {
  intptr_t i, j;
  intptr_t no_of_edges = params->no_of_edges;
  PIXELM *pixel_pointer = pixel;
  EDGE *edge_pointer = edge + no_of_edges;

  for (i = 0; i < n_i - 1; i++) {
    for (j = 0; j < n_j; j++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer + n_j)->input_mask == NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 = (pixel_pointer + n_j);
        edge_pointer->reliab = pixel_pointer->unreliability +
                               (pixel_pointer + n_j)->unreliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value, (pixel_pointer + n_j)->value);
        edge_pointer++;
        no_of_edges++;
      }
      pixel_pointer++;
    }  // j loop
  }  // i loop

  // construct edges that connect at the bottom border of the image
  if (params->y_connectivity == 1) {
    pixel_pointer = pixel + n_j * (n_i - 1);
    for (i = 0; i < n_j; i++) {
      if (pixel_pointer->input_mask == NOMASK &&
          (pixel_pointer - n_j * (n_i - 1))->input_mask ==
              NOMASK) {
        edge_pointer->pointer_1 = pixel_pointer;
        edge_pointer->pointer_2 =
            (pixel_pointer - n_j * (n_i - 1));
        edge_pointer->reliab =
            pixel_pointer->unreliability +
            (pixel_pointer - n_j * (n_i - 1))->unreliability;
        edge_pointer->increment = find_wrap(
            pixel_pointer->value,
            (pixel_pointer - n_j * (n_i - 1))->value);
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
  intptr_t k;
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
void unwrapImage(PIXELM *pixel, intptr_t n_j, intptr_t n_i) {
  intptr_t i;
  intptr_t image_size = n_j * n_i;
  PIXELM *pixel_pointer = pixel;

  for (i = 0; i < image_size; i++) {
    pixel_pointer->value += TWOPI * (double)(pixel_pointer->increment);
    pixel_pointer++;
  }
}

// set the masked pixels (mask = 0) to the minimum of the unwrapper phase
void maskImage(PIXELM *pixel, unsigned char *input_mask, intptr_t n_j,
               intptr_t n_i) {
  PIXELM *pointer_pixel = pixel;
  unsigned char *IMP = input_mask;  // input mask pointer
  double min = DBL_MAX;
  intptr_t i;
  intptr_t image_size = n_j * n_i;

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
void returnImage(PIXELM *pixel, double *unwrapped_image, intptr_t n_j,
                 intptr_t n_i) {
  intptr_t i;
  intptr_t image_size = n_j * n_i;
  double *unwrapped_image_pointer = unwrapped_image;
  PIXELM *pixel_pointer = pixel;

  for (i = 0; i < image_size; i++) {
    *unwrapped_image_pointer = pixel_pointer->value;
    pixel_pointer++;
    unwrapped_image_pointer++;
  }
}

// the main function of the unwrapper
int unwrap2D(double *wrapped_image, double *UnwrappedImage,
        unsigned char *input_mask, intptr_t n_j, intptr_t n_i,
        int wrap_around_j, int wrap_around_i,
        bitgen_t* bitgen_state) {
  params_t params = {TWOPI, wrap_around_j, wrap_around_i, 0};
  unsigned char *extended_mask;
  PIXELM *pixel;
  EDGE *edge;
  intptr_t image_size = n_i * n_j;
  intptr_t No_of_Edges_initially = 2 * n_j * n_i;

  pixel = (PIXELM *)calloc(image_size, sizeof(PIXELM));
  if (pixel == NULL) {
      return 1;
  }
  extended_mask = (unsigned char *)calloc(image_size, sizeof(unsigned char));
  if (extended_mask == NULL) {
      free(pixel);
      return 1;
  }
  edge = (EDGE *)calloc(No_of_Edges_initially, sizeof(EDGE));
  if (edge == NULL) {
      free(pixel);
      free(extended_mask);
      return 1;
  }

  extend_mask(input_mask, extended_mask, n_j, n_i, &params);
  initialisePIXELs(wrapped_image, input_mask, extended_mask, pixel, n_j,
                   n_i, bitgen_state);
  calculate_unreliability(wrapped_image, pixel, n_j, n_i,
                        &params);
  horizontalEDGEs(pixel, edge, n_j, n_i, &params);
  verticalEDGEs(pixel, edge, n_j, n_i, &params);

  if (params.no_of_edges != 0) {
      // sort the EDGEs depending on their reiability. The PIXELs with higher
      // relibility (small value) first
      quicker_sort(edge, edge + params.no_of_edges - 1);
  }
  // gather PIXELs into groups
  gatherPIXELs(edge, &params);

  unwrapImage(pixel, n_j, n_i);
  maskImage(pixel, input_mask, n_j, n_i);

  // copy the image from PIXELM structure to the unwrapped phase array
  // passed to this function
  // TODO: replace by (cython?) function to directly write into numpy array ?
  returnImage(pixel, UnwrappedImage, n_j, n_i);

  free(edge);
  free(pixel);
  free(extended_mask);

  return 0;
}
