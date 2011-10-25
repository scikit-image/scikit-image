/* Lear's GIST implementation, version 1.0, (c) INRIA 2009, Licence: GPL */



#ifndef STANDALONE_IMAGE_H
#define STANDALONE_IMAGE_H

/****************************************************************************
 * Image structures 
 */



typedef struct {
  int width,height;     /* dimensions */
  int stride;		/* width of image in memory */

  float *data;		/* data in latin reading order */
} image_t;

typedef struct {
  int width,height;     /* here, stride = width */
  
  float *c1;		/* R */
  float *c2;		/* G */
  float *c3;		/* B */
  
} color_image_t;

image_t *image_new(int width, int height);

image_t *image_cpy(image_t *src);

void image_delete(image_t *image);


color_image_t *color_image_new(int width, int height);

color_image_t *color_image_cpy(color_image_t *src);

void color_image_delete(color_image_t *image);


typedef struct {
  int size;			/* Number of images in the list */
  int alloc_size;		/* Number of allocated images */

  image_t **data;		/* List of images */

} image_list_t;


image_list_t *image_list_new(void);

void image_list_append(image_list_t *list, image_t *image);

void image_list_delete(image_list_t *list);


#endif
