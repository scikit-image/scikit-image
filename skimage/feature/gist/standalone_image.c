/* Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */
#include <stdlib.h>
#include <string.h>


#include "standalone_image.h"

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))
#define NEW(type) NEWA(type,1)

image_t *image_new(int width, int height) {
  image_t *im=NEW(image_t);
  im->width=im->stride=width;
  im->height=height;
  int wh=width*height;
  im->data=NEWA(float,wh);
  return im;
}

image_t *image_cpy(image_t *src) {
  image_t *im=image_new(src->width,src->height); 
  memcpy(im->data,src->data,sizeof(*(src->data))*src->width*src->height);
  return im;
}

void image_delete(image_t *im) {
  free(im->data);
  free(im);
}


color_image_t *color_image_new(int width, int height) {
  color_image_t *im=NEW(color_image_t);
  im->width=width;
  im->height=height;
  int wh=width*height;
  im->c1=NEWA(float,wh);
  im->c2=NEWA(float,wh);
  im->c3=NEWA(float,wh);
  return im;
}

color_image_t *color_image_cpy(color_image_t *src) {
  color_image_t *im=color_image_new(src->width,src->height); 
  memcpy(im->c1,src->c1,sizeof(*(src->c1))*src->width*src->height);
  memcpy(im->c2,src->c2,sizeof(*(src->c1))*src->width*src->height);
  memcpy(im->c3,src->c3,sizeof(*(src->c1))*src->width*src->height);
  return im;
}

void color_image_delete(color_image_t *im) {
  free(im->c1);
  free(im->c2);
  free(im->c3);
  free(im);
}



image_list_t *image_list_new(void) {
  image_list_t *list=NEW(image_list_t);
  list->size=list->alloc_size=0;
  list->data=NULL;
  return list;
}

void image_list_append(image_list_t *list, image_t *image) {
  if(list->size==list->alloc_size) {
    list->alloc_size=(list->alloc_size+1)*3/2;
    list->data=realloc(list->data,sizeof(*list->data)*list->alloc_size);
  }
  list->data[list->size++]=image;
}

void image_list_delete(image_list_t *list) {
  int i;

  for(i=0;i<list->size;i++) 
    image_delete(list->data[i]);
  free(list->data);
  free(list);
}

