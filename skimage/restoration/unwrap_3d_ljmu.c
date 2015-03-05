// 3D phase unwrapping, modified for inclusion in scipy by Gregor Thalhammer
// Original file name: Hussein_3D_unwrapper_with_mask_and_wrap_around_option.c

// This program was written by Hussein Abdul-Rahman and Munther Gdeisat to
// program the three-dimensional phase unwrapper
// entitled "Fast three-dimensional phase-unwrapping algorithm based on sorting
// by
// reliability following a noncontinuous path"
// by  Hussein Abdul-Rahman, Munther A. Gdeisat, David R. Burton, and Michael J.
// Lalor,
// published in the Proceedings of SPIE -
// The International Society for Optical Engineering, Vol. 5856, No. 1, 2005,
// pp. 32-40
// This program was written by Munther Gdeisat, Liverpool John Moores
// University, United Kingdom.
// Date 31st August 2007
// The wrapped phase volume is assumed to be of floating point data type. The
// resultant unwrapped phase volume is also of floating point type.
// Read the data from the file frame by frame
// The mask is of byte data type.
// When the mask is 255 this means that the voxel is valid
// When the mask is 0 this means that the voxel is invalid (noisy or corrupted
// voxel)
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

#define NOMASK 0
#define MASK 1

typedef struct {
  double mod;
  int x_connectivity;
  int y_connectivity;
  int z_connectivity;
  int no_of_edges;
} params_t;

// VOXELM information
struct VOXELM {
  int increment;  // No. of 2*pi to add to the voxel to unwrap it
  int number_of_voxels_in_group;  // No. of voxel in the voxel group
  double value;  // value of the voxel
  double reliability;
  unsigned char input_mask;  // MASK voxel is masked. NOMASK voxel is not masked
  unsigned char extended_mask;  // MASK voxel is masked. NOMASK voxel is not
                                // masked
  int group;  // group No.
  int new_group;
  struct VOXELM *head;  // pointer to the first voxel in the group in the linked
                        // list
  struct VOXELM *last;  // pointer to the last voxel in the group
  struct VOXELM *next;  // pointer to the next voxel in the group
};

typedef struct VOXELM VOXELM;

// the EDGE is the line that connects two voxels.
// if we have S voxels, then we have S horizontal edges and S vertical edges
struct EDGE {
  double reliab;  // reliabilty of the edge and it depends on the two voxels
  VOXELM *pointer_1;  // pointer to the first voxel
  VOXELM *pointer_2;  // pointer to the second voxel
  int increment;  // No. of 2*pi to add to one of the
  // voxels to unwrap it with respect to
  // the second
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

//--------------------start initialize voxels ----------------------------------
// initiale voxels. See the explanation of the voxel class above.
// initially every voxel is assumed to belong to a group consisting of only
// itself
void initialiseVOXELs(double *WrappedVolume, unsigned char *input_mask,
                      unsigned char *extended_mask, VOXELM *voxel,
                      int volume_width, int volume_height, int volume_depth,
                      char use_seed, unsigned int seed) {
  VOXELM *voxel_pointer = voxel;
  double *wrapped_volume_pointer = WrappedVolume;
  unsigned char *input_mask_pointer = input_mask;
  unsigned char *extended_mask_pointer = extended_mask;
  int n, i, j;

  if (use_seed) {
    srand(seed);
  }

  for (n = 0; n < volume_depth; n++) {
    for (i = 0; i < volume_height; i++) {
      for (j = 0; j < volume_width; j++) {
        voxel_pointer->increment = 0;
        voxel_pointer->number_of_voxels_in_group = 1;
        voxel_pointer->value = *wrapped_volume_pointer;
        voxel_pointer->reliability = rand();
        voxel_pointer->input_mask = *input_mask_pointer;
        voxel_pointer->extended_mask = *extended_mask_pointer;
        voxel_pointer->head = voxel_pointer;
        voxel_pointer->last = voxel_pointer;
        voxel_pointer->next = NULL;
        voxel_pointer->new_group = 0;
        voxel_pointer->group = -1;
        voxel_pointer++;
        wrapped_volume_pointer++;
        input_mask_pointer++;
        extended_mask_pointer++;
      }
    }
  }
}
//-------------------end initialize voxels -----------

// gamma function in the paper
double wrap(double voxel_value) {
  double wrapped_voxel_value;
  if (voxel_value > PI)
    wrapped_voxel_value = voxel_value - TWOPI;
  else if (voxel_value < -PI)
    wrapped_voxel_value = voxel_value + TWOPI;
  else
    wrapped_voxel_value = voxel_value;
  return wrapped_voxel_value;
}

// voxelL_value is the left voxel,  voxelR_value is the right voxel
int find_wrap(double voxelL_value, double voxelR_value) {
  double difference;
  int wrap_value;
  difference = voxelL_value - voxelR_value;

  if (difference > PI)
    wrap_value = -1;
  else if (difference < -PI)
    wrap_value = 1;
  else
    wrap_value = 0;

  return wrap_value;
}

void extend_mask(unsigned char *input_mask, unsigned char *extended_mask,
                 int volume_width, int volume_height, int volume_depth,
                 params_t *params) {
  int n, i, j;
  int vw = volume_width, vh = volume_height, vd = volume_depth;
  int fs = volume_width * volume_height;  // frame size
  int frame_size = volume_width * volume_height;
  int volume_size = volume_width * volume_height * volume_depth;  // volume size
  int vs = volume_size;
  unsigned char *IMP =
      input_mask + frame_size + volume_width + 1;  // input mask pointer
  unsigned char *EMP =
      extended_mask + frame_size + volume_width + 1;  // extended mask pointer

  // extend the mask for the volume except borders
  for (n = 1; n < volume_depth - 1; n++) {
    for (i = 1; i < volume_height - 1; i++) {
      for (j = 1; j < volume_width - 1; j++) {
        if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
            (*(IMP + 1) == NOMASK) && (*(IMP + vw) == NOMASK) &&
            (*(IMP + vw - 1) == NOMASK) && (*(IMP + vw + 1) == NOMASK) &&
            (*(IMP - vw) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP - vw + 1) == NOMASK) && (*(IMP + fs) == NOMASK) &&
            (*(IMP + fs - 1) == NOMASK) && (*(IMP + fs + 1) == NOMASK) &&
            (*(IMP + fs - vw) == NOMASK) && (*(IMP + fs - vw - 1) == NOMASK) &&
            (*(IMP + fs - vw + 1) == NOMASK) && (*(IMP + fs + vw) == NOMASK) &&
            (*(IMP + fs + vw - 1) == NOMASK) &&
            (*(IMP + fs + vw + 1) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP - fs - 1) == NOMASK) && (*(IMP - fs + 1) == NOMASK) &&
            (*(IMP - fs - vw) == NOMASK) && (*(IMP - fs - vw - 1) == NOMASK) &&
            (*(IMP - fs - vw + 1) == NOMASK) && (*(IMP - fs + vw) == NOMASK) &&
            (*(IMP - fs + vw - 1) == NOMASK) &&
            (*(IMP - fs + vw + 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        ++EMP;
        ++IMP;
      }
      EMP += 2;
      IMP += 2;
    }
    EMP += 2 * volume_width;
    IMP += 2 * volume_width;
  }

  if (params->x_connectivity == 1) {
    // extend the mask to the front side of the phase volume
    IMP = input_mask + frame_size + volume_width;  // input mask pointer
    EMP = extended_mask + frame_size + volume_width;  // extended mask pointer
    for (n = 1; n < volume_depth - 1; n++) {
      for (i = 1; i < volume_height - 1; i++) {
        if ((*IMP) == NOMASK && (*(IMP + vw - 1) == NOMASK) &&
            (*(IMP + 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP + vw) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP + fs) == NOMASK) && (*(IMP - 1) == NOMASK) &&
            (*(IMP + vw + 1) == NOMASK) && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP + 2 * vw - 1) == NOMASK) && (*(IMP - fs - 1) == NOMASK) &&
            (*(IMP + fs + vw + 1) == NOMASK) && (*(IMP - fs - vw) == NOMASK) &&
            (*(IMP + fs + vw) == NOMASK) && (*(IMP - fs - vw + 1) == NOMASK) &&
            (*(IMP + fs + 2 * vw - 1) == NOMASK) &&
            (*(IMP - fs + vw - 1) == NOMASK) && (*(IMP + fs + 1) == NOMASK) &&
            (*(IMP - fs + 1) == NOMASK) && (*(IMP + fs + vw - 1) == NOMASK) &&
            (*(IMP - fs + 2 * vw - 1) == NOMASK) &&
            (*(IMP + fs - vw + 1) == NOMASK) && (*(IMP - fs + vw) == NOMASK) &&
            (*(IMP + fs - vw) == NOMASK) && (*(IMP - fs + vw + 1) == NOMASK) &&
            (*(IMP + fs - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP += vw;
        IMP += vw;
      }
      EMP += 2 * vw;
      IMP += 2 * vw;
    }

    // extend the mask to the rear side of the phase volume
    IMP = input_mask + frame_size + 2 * volume_width - 1;  // input mask pointer
    EMP = extended_mask + frame_size + 2 * volume_width -
          1;  // extended mask pointer
    for (n = 1; n < volume_depth - 1; n++) {
      for (i = 1; i < volume_height - 1; i++) {
        if ((*IMP) == NOMASK && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP - 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP + vw) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP + fs) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP + 1) == NOMASK) && (*(IMP + vw - 1) == NOMASK) &&
            (*(IMP - 2 * vw + 1) == NOMASK) &&
            (*(IMP - fs - vw - 1) == NOMASK) && (*(IMP + fs + 1) == NOMASK) &&
            (*(IMP - fs - 2 * vw + 1) == NOMASK) &&
            (*(IMP + fs + vw - 1) == NOMASK) && (*(IMP - fs - 1) == NOMASK) &&
            (*(IMP + fs - vw + 1) == NOMASK) &&
            (*(IMP - fs - vw + 1) == NOMASK) && (*(IMP + fs - 1) == NOMASK) &&
            (*(IMP - fs - vw) == NOMASK) && (*(IMP + fs + vw) == NOMASK) &&
            (*(IMP - fs + vw - 1) == NOMASK) &&
            (*(IMP + fs - 2 * vw + 1) == NOMASK) &&
            (*(IMP - fs + vw) == NOMASK) && (*(IMP + fs - vw) == NOMASK) &&
            (*(IMP - fs + 1) == NOMASK) && (*(IMP + fs - vw - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP += vw;
        IMP += vw;
      }
      EMP += 2 * vw;
      IMP += 2 * vw;
    }
  }

  if (params->y_connectivity == 1) {
    // extend the mask to the left side of the phase volume
    IMP = input_mask + frame_size + 1;
    EMP = extended_mask + frame_size + 1;
    for (n = 1; n < volume_depth - 1; n++) {
      for (j = 1; j < volume_width - 1; j++) {
        if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
            (*(IMP + 1) == NOMASK) && (*(IMP + fs - vw) == NOMASK) &&
            (*(IMP + vw) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP + fs) == NOMASK) && (*(IMP + fs - vw - 1) == NOMASK) &&
            (*(IMP + vw + 1) == NOMASK) && (*(IMP + fs - vw + 1) == NOMASK) &&
            (*(IMP + vw - 1) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP + fs + vw + 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP + fs + vw) == NOMASK) && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP + fs + vw - 1) == NOMASK) && (*(IMP - fs - 1) == NOMASK) &&
            (*(IMP + fs + 1) == NOMASK) && (*(IMP - fs + 1) == NOMASK) &&
            (*(IMP + fs - 1) == NOMASK) && (*(IMP - fs + vw - 1) == NOMASK) &&
            (*(IMP + 2 * fs - vw + 1) == NOMASK) &&
            (*(IMP - fs + vw) == NOMASK) && (*(IMP + 2 * fs - vw) == NOMASK) &&
            (*(IMP - fs + vw + 1) == NOMASK) &&
            (*(IMP + 2 * fs - vw - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP++;
        IMP++;
      }
      EMP += fs - vw + 2;
      IMP += fs - vw + 2;
    }

    // extend the mask to the right side of the phase volume
    IMP = input_mask + 2 * frame_size - volume_width + 1;
    EMP = extended_mask + 2 * frame_size - volume_width + 1;
    for (n = 1; n < volume_depth - 1; n++) {
      for (j = 1; j < volume_width - 1; j++) {
        if ((*IMP) == NOMASK && (*(IMP + 1) == NOMASK) &&
            (*(IMP - 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP - fs + vw) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP + fs) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP - fs + vw + 1) == NOMASK) && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP - fs + vw - 1) == NOMASK) &&
            (*(IMP - fs - vw - 1) == NOMASK) && (*(IMP + vw + 1) == NOMASK) &&
            (*(IMP - fs - vw + 1) == NOMASK) && (*(IMP + vw - 1) == NOMASK) &&
            (*(IMP - fs - vw) == NOMASK) && (*(IMP + vw) == NOMASK) &&
            (*(IMP - fs - 1) == NOMASK) && (*(IMP + fs + 1) == NOMASK) &&
            (*(IMP - fs + 1) == NOMASK) && (*(IMP + fs - 1) == NOMASK) &&
            (*(IMP - 2 * fs + vw - 1) == NOMASK) &&
            (*(IMP + fs - vw + 1) == NOMASK) &&
            (*(IMP - 2 * fs + vw) == NOMASK) && (*(IMP + fs - vw) == NOMASK) &&
            (*(IMP - 2 * fs + vw + 1) == NOMASK) &&
            (*(IMP + fs - vw - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP++;
        IMP++;
      }
      EMP += fs - vw + 2;
      IMP += fs - vw + 2;
    }
  }

  if (params->z_connectivity == 1) {
    // extend the mask to the bottom side of the phase volume
    IMP = input_mask + volume_width + 1;
    EMP = extended_mask + volume_width + 1;
    for (i = 1; i < volume_height - 1; ++i) {
      for (j = 1; j < volume_width - 1; ++j) {
        if ((*IMP) == NOMASK && (*(IMP - 1) == NOMASK) &&
            (*(IMP + 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP + vw) == NOMASK) && (*(IMP + fs) == NOMASK) &&
            (*(IMP + vs - fs) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP + vw + 1) == NOMASK) && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP + vw - 1) == NOMASK) &&
            (*(IMP + vs - fs - vw - 1) == NOMASK) &&
            (*(IMP + fs + vw + 1) == NOMASK) &&
            (*(IMP + vs - fs - vw) == NOMASK) && (*(IMP + fs + vw) == NOMASK) &&
            (*(IMP + vs - fs - vw + 1) == NOMASK) &&
            (*(IMP + fs + vw - 1) == NOMASK) &&
            (*(IMP + vs - fs - 1) == NOMASK) && (*(IMP + fs + 1) == NOMASK) &&
            (*(IMP + vs - fs + 1) == NOMASK) && (*(IMP + fs - 1) == NOMASK) &&
            (*(IMP + vs - fs + vw - 1) == NOMASK) &&
            (*(IMP + fs - vw + 1) == NOMASK) &&
            (*(IMP + vs - fs + vw) == NOMASK) && (*(IMP + fs - vw) == NOMASK) &&
            (*(IMP + vs - fs + vw + 1) == NOMASK) &&
            (*(IMP + fs - vw - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP++;
        IMP++;
      }
      EMP += 2;
      IMP += 2;
    }

    // extend the mask to the top side of the phase volume
    IMP = input_mask + volume_size - frame_size + volume_width + 1;
    EMP = extended_mask + volume_size - frame_size + volume_width + 1;
    for (i = 1; i < volume_height - 1; ++i) {
      for (j = 1; j < volume_width - 1; ++j) {
        if ((*IMP) == NOMASK && (*(IMP + 1) == NOMASK) &&
            (*(IMP - 1) == NOMASK) && (*(IMP - vw) == NOMASK) &&
            (*(IMP - fs + vw) == NOMASK) && (*(IMP - fs) == NOMASK) &&
            (*(IMP - vs + fs) == NOMASK) && (*(IMP - vw - 1) == NOMASK) &&
            (*(IMP + vw + 1) == NOMASK) && (*(IMP - vw + 1) == NOMASK) &&
            (*(IMP + vw - 1) == NOMASK) && (*(IMP - fs - vw - 1) == NOMASK) &&
            (*(IMP - vs + fs + vw + 1) == NOMASK) &&
            (*(IMP - fs - vw + 1) == NOMASK) &&
            (*(IMP - vs + fs + vw - 1) == NOMASK) &&
            (*(IMP - fs - vw) == NOMASK) && (*(IMP - vs + fs + vw) == NOMASK) &&
            (*(IMP - fs - 1) == NOMASK) && (*(IMP - vs + fs + 1) == NOMASK) &&
            (*(IMP - fs + 1) == NOMASK) && (*(IMP - vs + fs - 1) == NOMASK) &&
            (*(IMP - fs + vw - 1) == NOMASK) &&
            (*(IMP - vs + fs - vw + 1) == NOMASK) &&
            (*(IMP - fs + vw) == NOMASK) && (*(IMP - vs + fs - vw) == NOMASK) &&
            (*(IMP - fs + vw + 1) == NOMASK) &&
            (*(IMP - vs + fs - vw - 1) == NOMASK)) {
          *EMP = NOMASK;
        }
        EMP++;
        IMP++;
      }
      EMP += 2;
      IMP += 2;
    }
  }
}

void calculate_reliability(double *wrappedVolume, VOXELM *voxel,
                           int volume_width, int volume_height,
                           int volume_depth, params_t *params) {
  int frame_size = volume_width * volume_height;
  int volume_size = volume_width * volume_height * volume_depth;
  VOXELM *voxel_pointer;
  double H, V, N, D1, D2, D3, D4, D5, D6, D7, D8, D9, D10;
  double *WVP;
  int n, i, j;

  WVP = wrappedVolume + frame_size + volume_width + 1;
  voxel_pointer = voxel + frame_size + volume_width + 1;
  for (n = 1; n < volume_depth - 1; n++) {
    for (i = 1; i < volume_height - 1; i++) {
      for (j = 1; j < volume_width - 1; j++) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP - 1) - *WVP) - wrap(*WVP - *(WVP + 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP + frame_size));
          D1 = wrap(*(WVP - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width + 1));
          D2 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width - 1));
          D3 = wrap(*(WVP - frame_size - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width + 1));
          D4 = wrap(*(WVP - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width));
          D5 = wrap(*(WVP - frame_size - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width - 1));
          D6 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D7 = wrap(*(WVP - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 1));
          D8 = wrap(*(WVP - frame_size + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width + 1));
          D9 = wrap(*(WVP - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width));
          D10 = wrap(*(WVP - frame_size + volume_width + 1) - *WVP) -
                wrap(*WVP - *(WVP + frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer++;
        WVP++;
      }
      voxel_pointer += 2;
      WVP += 2;
    }
    voxel_pointer += 2 * volume_width;
    WVP += 2 * volume_width;
  }

  if (params->x_connectivity == 1) {
    // calculating reliability for the front side of the phase volume...add
    // volume_width
    WVP = wrappedVolume + frame_size + volume_width;
    voxel_pointer = voxel + frame_size + volume_width;
    for (n = 1; n < volume_depth - 1; ++n) {
      for (i = 1; i < volume_height - 1; ++i) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP + volume_width - 1) - *WVP) - wrap(*WVP - *(WVP + 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP + frame_size));
          D1 = wrap(*(WVP - 1) - *WVP) - wrap(*WVP - *(WVP + volume_width + 1));
          D2 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + 2 * volume_width - 1));
          D3 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width + 1));
          D4 = wrap(*(WVP - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width));
          D5 = wrap(*(WVP - frame_size - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 2 * volume_width - 1));
          D6 = wrap(*(WVP - frame_size + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D7 = wrap(*(WVP - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width - 1));
          D8 = wrap(*(WVP - frame_size + 2 * volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width + 1));
          D9 = wrap(*(WVP - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width));
          D10 = wrap(*(WVP - frame_size + volume_width + 1) - *WVP) -
                wrap(*WVP - *(WVP + frame_size - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer += volume_width;
        WVP += volume_width;
      }
      voxel_pointer += 2 * volume_width;
      WVP += 2 * volume_width;
    }

    // calculating reliability for the rear side of the phase volume.....
    // subtract volume_width
    WVP = wrappedVolume + frame_size + 2 * volume_width - 1;
    voxel_pointer = voxel + frame_size + 2 * volume_width - 1;
    for (n = 1; n < volume_depth - 1; ++n) {
      for (i = 1; i < volume_height - 1; ++i) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP - volume_width + 1) - *WVP) - wrap(*WVP - *(WVP - 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP + frame_size));
          D1 = wrap(*(WVP - volume_width - 1) - *WVP) - wrap(*WVP - *(WVP + 1));
          D2 = wrap(*(WVP + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP - 2 * volume_width + 1));
          D3 = wrap(*(WVP - frame_size - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D4 = wrap(*(WVP - frame_size - 2 * volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width - 1));
          D5 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width + 1));
          D6 = wrap(*(WVP - frame_size - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 1));
          D7 = wrap(*(WVP - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width));
          D8 = wrap(*(WVP - frame_size + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 2 * volume_width + 1));
          D9 = wrap(*(WVP - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width));
          D10 = wrap(*(WVP - frame_size + 1) - *WVP) -
                wrap(*WVP - *(WVP + frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer += volume_width;
        WVP += volume_width;
      }
      voxel_pointer += 2 * volume_width;
      WVP += 2 * volume_width;
    }
  }

  if (params->y_connectivity == 1) {
    // calculating reliability for the left side of the phase volume...add
    // frame_size
    WVP = wrappedVolume + frame_size + 1;
    voxel_pointer = voxel + frame_size + 1;
    for (n = 1; n < volume_depth - 1; ++n) {
      for (j = 1; j < volume_width - 1; ++j) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP - 1) - *WVP) - wrap(*WVP - *(WVP + 1));
          V = wrap(*(WVP + frame_size - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP + frame_size));
          D1 = wrap(*(WVP + frame_size - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width + 1));
          D2 = wrap(*(WVP + frame_size - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width - 1));
          D3 = wrap(*(WVP - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width + 1));
          D4 = wrap(*(WVP - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width));
          D5 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width - 1));
          D6 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D7 = wrap(*(WVP - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 1));
          D8 = wrap(*(WVP - frame_size + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + 2 * frame_size - volume_width + 1));
          D9 = wrap(*(WVP - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + 2 * frame_size - volume_width));
          D10 = wrap(*(WVP - frame_size + volume_width + 1) - *WVP) -
                wrap(*WVP - *(WVP + 2 * frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer++;
        WVP++;
      }
      voxel_pointer += frame_size - volume_width + 2;
      WVP += frame_size - volume_width + 2;
    }

    // calculating reliability for the right side of the phase volume...subtract
    // frame_size
    WVP = wrappedVolume + 2 * frame_size - volume_width + 1;
    voxel_pointer = voxel + 2 * frame_size - volume_width + 1;
    for (n = 1; n < volume_depth - 1; ++n) {
      for (j = 1; j < volume_width - 1; ++j) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP + 1) - *WVP) - wrap(*WVP - *(WVP - 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP - frame_size + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP + frame_size));
          D1 = wrap(*(WVP - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP - frame_size + volume_width + 1));
          D2 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP - frame_size + volume_width - 1));
          D3 = wrap(*(WVP - frame_size - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width + 1));
          D4 = wrap(*(WVP - frame_size - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width - 1));
          D5 = wrap(*(WVP - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + volume_width));
          D6 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D7 = wrap(*(WVP - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 1));
          D8 = wrap(*(WVP - 2 * frame_size + volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width + 1));
          D9 = wrap(*(WVP - 2 * frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width));
          D10 = wrap(*(WVP - 2 * frame_size + volume_width + 1) - *WVP) -
                wrap(*WVP - *(WVP + frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer++;
        WVP++;
      }
      voxel_pointer += frame_size - volume_width + 2;
      WVP += frame_size - volume_width + 2;
    }
  }

  if (params->z_connectivity == 1) {
    // calculating reliability for the bottom side of the phase volume...add
    // volume_size
    WVP = wrappedVolume + volume_width + 1;
    voxel_pointer = voxel + volume_width + 1;
    for (i = 1; i < volume_height - 1; ++i) {
      for (j = 1; j < volume_width - 1; ++j) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP - 1) - *WVP) - wrap(*WVP - *(WVP + 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP + frame_size) - *WVP) -
              wrap(*WVP - *(WVP + volume_size - frame_size));
          D1 = wrap(*(WVP - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width + 1));
          D2 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width - 1));
          D3 = wrap(*(WVP + volume_size - frame_size - volume_width - 1) -
                    *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width + 1));
          D4 = wrap(*(WVP + volume_size - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width));
          D5 = wrap(*(WVP + volume_size - frame_size - volume_width + 1) -
                    *WVP) -
               wrap(*WVP - *(WVP + frame_size + volume_width - 1));
          D6 = wrap(*(WVP + volume_size - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size + 1));
          D7 = wrap(*(WVP + volume_size - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - 1));
          D8 = wrap(*(WVP + volume_size - frame_size + volume_width - 1) -
                    *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width + 1));
          D9 = wrap(*(WVP + volume_size - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP + frame_size - volume_width));
          D10 = wrap(*(WVP + volume_size - frame_size + volume_width + 1) -
                     *WVP) -
                wrap(*WVP - *(WVP + frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer++;
        WVP++;
      }
      voxel_pointer += 2;
      WVP += 2;
    }

    // calculating reliability for the top side of the phase volume...subtract
    // volume_size
    WVP = wrappedVolume + volume_size - frame_size + volume_width + 1;
    voxel_pointer = voxel + volume_size - frame_size + volume_width + 1;
    for (i = 1; i < volume_height - 1; ++i) {
      for (j = 1; j < volume_width - 1; ++j) {
        if (voxel_pointer->extended_mask == NOMASK) {
          H = wrap(*(WVP + 1) - *WVP) - wrap(*WVP - *(WVP - 1));
          V = wrap(*(WVP - volume_width) - *WVP) -
              wrap(*WVP - *(WVP + volume_width));
          N = wrap(*(WVP - frame_size) - *WVP) -
              wrap(*WVP - *(WVP - volume_size + frame_size));
          D1 = wrap(*(WVP - volume_width - 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width + 1));
          D2 = wrap(*(WVP - volume_width + 1) - *WVP) -
               wrap(*WVP - *(WVP + volume_width - 1));
          D3 =
              wrap(*(WVP - frame_size - volume_width - 1) - *WVP) -
              wrap(*WVP - *(WVP - volume_size + frame_size + volume_width + 1));
          D4 =
              wrap(*(WVP - frame_size - volume_width + 1) - *WVP) -
              wrap(*WVP - *(WVP - volume_size + frame_size + volume_width - 1));
          D5 = wrap(*(WVP - frame_size - volume_width) - *WVP) -
               wrap(*WVP - *(WVP - volume_size + frame_size + volume_width));
          D6 = wrap(*(WVP - frame_size - 1) - *WVP) -
               wrap(*WVP - *(WVP - volume_size + frame_size + 1));
          D7 = wrap(*(WVP - frame_size + 1) - *WVP) -
               wrap(*WVP - *(WVP - volume_size + frame_size - 1));
          D8 =
              wrap(*(WVP - frame_size + volume_width - 1) - *WVP) -
              wrap(*WVP - *(WVP - volume_size + frame_size - volume_width + 1));
          D9 = wrap(*(WVP - frame_size + volume_width) - *WVP) -
               wrap(*WVP - *(WVP - volume_size + frame_size - volume_width));
          D10 =
              wrap(*(WVP - frame_size + volume_width + 1) - *WVP) -
              wrap(*WVP - *(WVP - volume_size + frame_size - volume_width - 1));
          voxel_pointer->reliability =
              H * H + V * V + N * N + D1 * D1 + D2 * D2 + D3 * D3 + D4 * D4 +
              D5 * D5 + D6 * D6 + D7 * D7 + D8 * D8 + D9 * D9 + D10 * D10;
        }
        voxel_pointer++;
        WVP++;
      }
      voxel_pointer += 2;
      WVP += 2;
    }
  }
}

// calculate the reliability of the horizontal edges of the volume.  it
// is calculated by adding the reliability of voxel and the relibility
// of its right neighbour. edge is calculated between a voxel and its
// next neighbour
void horizontalEDGEs(VOXELM *voxel, EDGE *edge, int volume_width,
                     int volume_height, int volume_depth, params_t *params) {
  int n, i, j;
  EDGE *edge_pointer = edge;
  VOXELM *voxel_pointer = voxel;
  int no_of_edges = params->no_of_edges;

  for (n = 0; n < volume_depth; n++) {
    for (i = 0; i < volume_height; i++) {
      for (j = 0; j < volume_width - 1; j++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer + 1)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer + 1);
          edge_pointer->reliab =
              voxel_pointer->reliability + (voxel_pointer + 1)->reliability;
          edge_pointer->increment =
              find_wrap(voxel_pointer->value, (voxel_pointer + 1)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer++;
      }
      voxel_pointer++;
    }
  }
  if (params->x_connectivity == 1) {
    voxel_pointer = voxel + volume_width - 1;
    for (n = 0; n < volume_depth; n++) {
      for (i = 0; i < volume_height; i++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer - volume_width + 1)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer - volume_width + 1);
          edge_pointer->reliab =
              voxel_pointer->reliability +
              (voxel_pointer - volume_width + 1)->reliability;
          edge_pointer->increment = find_wrap(
              voxel_pointer->value, (voxel_pointer - volume_width + 1)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer += volume_width;
      }
    }
  }
  params->no_of_edges = no_of_edges;
}

void verticalEDGEs(VOXELM *voxel, EDGE *edge, int volume_width,
                   int volume_height, int volume_depth, params_t *params) {
  int n, i, j;
  int no_of_edges = params->no_of_edges;
  VOXELM *voxel_pointer = voxel;
  EDGE *edge_pointer = edge + no_of_edges;
  int frame_size = volume_width * volume_height;
  int next_voxel = frame_size - volume_width;

  for (n = 0; n < volume_depth; n++) {
    for (i = 0; i < volume_height - 1; i++) {
      for (j = 0; j < volume_width; j++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer + volume_width)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer + volume_width);
          edge_pointer->reliab = voxel_pointer->reliability +
                                 (voxel_pointer + volume_width)->reliability;
          edge_pointer->increment = find_wrap(
              voxel_pointer->value, (voxel_pointer + volume_width)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer++;
      }
    }
    voxel_pointer += volume_width;
  }

  if (params->y_connectivity == 1) {
    voxel_pointer = voxel + frame_size - volume_width;
    for (n = 0; n < volume_depth; n++) {
      for (i = 0; i < volume_width; i++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer - next_voxel)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer - next_voxel);
          edge_pointer->reliab = voxel_pointer->reliability +
                                 (voxel_pointer - next_voxel)->reliability;
          edge_pointer->increment = find_wrap(
              voxel_pointer->value, (voxel_pointer - next_voxel)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer++;
      }
      voxel_pointer += next_voxel;
    }
  }
  params->no_of_edges = no_of_edges;
}

void normalEDGEs(VOXELM *voxel, EDGE *edge, int volume_width, int volume_height,
                 int volume_depth, params_t *params) {
  int n, i, j;
  int no_of_edges = params->no_of_edges;
  int frame_size = volume_width * volume_height;
  int volume_size = volume_width * volume_height * volume_depth;
  VOXELM *voxel_pointer = voxel;
  EDGE *edge_pointer = edge + no_of_edges;
  int next_voxel = volume_size - frame_size;

  for (n = 0; n < volume_depth - 1; n++) {
    for (i = 0; i < volume_height; i++) {
      for (j = 0; j < volume_width; j++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer + frame_size)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer + frame_size);
          edge_pointer->reliab = voxel_pointer->reliability +
                                 (voxel_pointer + frame_size)->reliability;
          edge_pointer->increment = find_wrap(
              voxel_pointer->value, (voxel_pointer + frame_size)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer++;
      }
    }
  }

  if (params->z_connectivity == 1) {
    voxel_pointer = voxel + next_voxel;
    for (i = 0; i < volume_height; i++) {
      for (j = 0; j < volume_width; j++) {
        if (voxel_pointer->input_mask == NOMASK &&
            (voxel_pointer - next_voxel)->input_mask == NOMASK) {
          edge_pointer->pointer_1 = voxel_pointer;
          edge_pointer->pointer_2 = (voxel_pointer - next_voxel);
          edge_pointer->reliab = voxel_pointer->reliability +
                                 (voxel_pointer - next_voxel)->reliability;
          edge_pointer->increment = find_wrap(
              voxel_pointer->value, (voxel_pointer - next_voxel)->value);
          edge_pointer++;
          no_of_edges++;
        }
        voxel_pointer++;
      }
    }
  }
  params->no_of_edges = no_of_edges;
}

// gather the voxels of the volume into groups
void gatherVOXELs(EDGE *edge, params_t *params) {
  int k;
  VOXELM *VOXEL1;
  VOXELM *VOXEL2;
  VOXELM *group1;
  VOXELM *group2;
  EDGE *pointer_edge = edge;
  int incremento;

  for (k = 0; k < params->no_of_edges; k++) {
    VOXEL1 = pointer_edge->pointer_1;
    VOXEL2 = pointer_edge->pointer_2;

    // VOXELM 1 and VOXELM 2 belong to different groups
    // initially each voxel is in a group by itself and one voxel can construct
    // a group
    // no else or else if to this if
    if (VOXEL2->head != VOXEL1->head) {
      // VOXELM 2 is alone in its group
      // merge this voxel with VOXELM 1 group and find the number of 2 pi to add
      // to or subtract to unwrap it
      if ((VOXEL2->next == NULL) && (VOXEL2->head == VOXEL2)) {
        VOXEL1->head->last->next = VOXEL2;
        VOXEL1->head->last = VOXEL2;
        (VOXEL1->head->number_of_voxels_in_group)++;
        VOXEL2->head = VOXEL1->head;
        VOXEL2->increment = VOXEL1->increment - pointer_edge->increment;
      }

      // VOXELM 1 is alone in its group
      // merge this voxel with VOXELM 2 group and find the number of 2 pi to add
      // to or subtract to unwrap it
      else if ((VOXEL1->next == NULL) && (VOXEL1->head == VOXEL1)) {
        VOXEL2->head->last->next = VOXEL1;
        VOXEL2->head->last = VOXEL1;
        (VOXEL2->head->number_of_voxels_in_group)++;
        VOXEL1->head = VOXEL2->head;
        VOXEL1->increment = VOXEL2->increment + pointer_edge->increment;
      }

      // VOXELM 1 and VOXELM 2 both have groups
      else {
        group1 = VOXEL1->head;
        group2 = VOXEL2->head;
        // if the no. of voxels in VOXELM 1 group is larger than the no. of
        // voxels
        // in VOXELM 2 group.   Merge VOXELM 2 group to VOXELM 1 group
        // and find the number of wraps between VOXELM 2 group and VOXELM 1
        // group
        // to unwrap VOXELM 2 group with respect to VOXELM 1 group.
        // the no. of wraps will be added to VOXELM 2 grop in the future
        if (group1->number_of_voxels_in_group >
            group2->number_of_voxels_in_group) {
          // merge VOXELM 2 with VOXELM 1 group
          group1->last->next = group2;
          group1->last = group2->last;
          group1->number_of_voxels_in_group =
              group1->number_of_voxels_in_group +
              group2->number_of_voxels_in_group;
          incremento =
              VOXEL1->increment - pointer_edge->increment - VOXEL2->increment;
          // merge the other voxels in VOXELM 2 group to VOXELM 1 group
          while (group2 != NULL) {
            group2->head = group1;
            group2->increment += incremento;
            group2 = group2->next;
          }
        }

        // if the no. of voxels in VOXELM 2 group is larger than the no. of
        // voxels
        // in VOXELM 1 group.   Merge VOXELM 1 group to VOXELM 2 group
        // and find the number of wraps between VOXELM 2 group and VOXELM 1
        // group
        // to unwrap VOXELM 1 group with respect to VOXELM 2 group.
        // the no. of wraps will be added to VOXELM 1 grop in the future
        else {
          // merge VOXELM 1 with VOXELM 2 group
          group2->last->next = group1;
          group2->last = group1->last;
          group2->number_of_voxels_in_group =
              group2->number_of_voxels_in_group +
              group1->number_of_voxels_in_group;
          incremento =
              VOXEL2->increment + pointer_edge->increment - VOXEL1->increment;
          // merge the other voxels in VOXELM 2 group to VOXELM 1 group
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

// unwrap the volume
void unwrapVolume(VOXELM *voxel, int volume_width, int volume_height,
                  int volume_depth) {
  int i;
  int volume_size = volume_width * volume_height * volume_depth;
  VOXELM *voxel_pointer = voxel;

  for (i = 0; i < volume_size; i++) {
    voxel_pointer->value += TWOPI * (double)(voxel_pointer->increment);
    voxel_pointer++;
  }
}

// set the masked voxels (mask = 0) to the minimum of the unwrapper phase
void maskVolume(VOXELM *voxel, unsigned char *input_mask, int volume_width,
                int volume_height, int volume_depth) {
  int volume_width_plus_one = volume_width + 1;
  int volume_height_plus_one = volume_height + 1;
  int volume_width_minus_one = volume_width - 1;
  int volume_height_minus_one = volume_height - 1;

  VOXELM *pointer_voxel = voxel;
  unsigned char *IMP = input_mask;  // input mask pointer
  double min = DBL_MAX;
  int i, j;
  int volume_size = volume_width * volume_height * volume_depth;

  // find the minimum of the unwrapped phase
  for (i = 0; i < volume_size; i++) {
    if ((pointer_voxel->value < min) && (*IMP == NOMASK))
      min = pointer_voxel->value;

    pointer_voxel++;
    IMP++;
  }

  pointer_voxel = voxel;
  IMP = input_mask;

  // set the masked voxels to minimum
  for (i = 0; i < volume_size; i++) {
    if ((*IMP) == MASK) {
      pointer_voxel->value = min;
    }
    pointer_voxel++;
    IMP++;
  }
}

// the input to this unwrapper is an array that contains the wrapped
// phase map.  copy the volume on the buffer passed to this unwrapper
// to over-write the unwrapped phase map on the buffer of the wrapped
// phase map.
void returnVolume(VOXELM *voxel, double *unwrappedVolume, int volume_width,
                  int volume_height, int volume_depth) {
  int i;
  int volume_size = volume_width * volume_height * volume_depth;
  double *unwrappedVolume_pointer = unwrappedVolume;
  VOXELM *voxel_pointer = voxel;

  for (i = 0; i < volume_size; i++) {
    *unwrappedVolume_pointer = voxel_pointer->value;
    voxel_pointer++;
    unwrappedVolume_pointer++;
  }
}

// the main function of the unwrapper
void unwrap3D(double *wrapped_volume, double *unwrapped_volume,
              unsigned char *input_mask, int volume_width, int volume_height,
              int volume_depth, int wrap_around_x, int wrap_around_y,
              int wrap_around_z, char use_seed, unsigned int seed) {
  params_t params = {TWOPI, wrap_around_x, wrap_around_y, wrap_around_z, 0};
  unsigned char *extended_mask;
  VOXELM *voxel;
  EDGE *edge;
  int volume_size = volume_height * volume_width * volume_depth;
  int No_of_Edges_initially = 3 * volume_width * volume_height * volume_depth;

  extended_mask = (unsigned char *)calloc(volume_size, sizeof(unsigned char));
  voxel = (VOXELM *)calloc(volume_size, sizeof(VOXELM));
  edge = (EDGE *)calloc(No_of_Edges_initially, sizeof(EDGE));
  ;

  extend_mask(input_mask, extended_mask, volume_width, volume_height,
              volume_depth, &params);
  initialiseVOXELs(wrapped_volume, input_mask, extended_mask, voxel,
                   volume_width, volume_height, volume_depth, use_seed, seed);
  calculate_reliability(wrapped_volume, voxel, volume_width, volume_height,
                        volume_depth, &params);
  horizontalEDGEs(voxel, edge, volume_width, volume_height, volume_depth,
                  &params);
  verticalEDGEs(voxel, edge, volume_width, volume_height, volume_depth,
                &params);
  normalEDGEs(voxel, edge, volume_width, volume_height, volume_depth, &params);

  if (params.no_of_edges != 0) {
      // sort the EDGEs depending on their reiability. The VOXELs with higher
      // relibility (small value) first
      quicker_sort(edge, edge + params.no_of_edges - 1);
  }

  // gather VOXELs into groups
  gatherVOXELs(edge, &params);

  unwrapVolume(voxel, volume_width, volume_height, volume_depth);
  maskVolume(voxel, input_mask, volume_width, volume_height, volume_depth);

  // copy the volume from VOXELM structure to the unwrapped phase array passed
  // to this function
  returnVolume(voxel, unwrapped_volume, volume_width, volume_height,
               volume_depth);

  free(edge);
  free(voxel);
  free(extended_mask);
}
