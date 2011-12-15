/* Lear's GIST implementation, version 1.1, (c) INRIA 2009, Licence: PSFL */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "gist.h"



static color_image_t *load_ppm(const char *fname) {
  FILE *f=fopen(fname,"r");
  if(!f) {
    perror("could not open infile");
    exit(1);
  }
  int px,width,height,maxval;
  if(fscanf(f,"P%d %d %d %d",&px,&width,&height,&maxval)!=4 || 
     maxval!=255 || (px!=6 && px!=5)) {
    fprintf(stderr,"Error: input not a raw PGM/PPM with maxval 255\n");
    exit(1);    
  }
  fgetc(f); /* eat the newline */
  color_image_t *im=color_image_new(width,height);

  int i;
  for(i=0;i<width*height;i++) {
    im->c1[i]=fgetc(f);
    if(px==6) {
      im->c2[i]=fgetc(f);
      im->c3[i]=fgetc(f);    
    } else {
      im->c2[i]=im->c1[i];
      im->c3[i]=im->c1[i];   
    }
  }
  
  fclose(f);
  return im;
}


static void usage(void) {
  fprintf(stderr,"compute_gist options... [infilename]\n"
          "infile is a PPM raw file\n"
          "options:\n"
          "[-nblocks nb] use a grid of nb*nb cells (default 4)\n"
          "[-orientationsPerScale o_1,..,o_n] use n scales and compute o_i orientations for scale i\n"
          );
  
  exit(1);
}



int main(int argc,char **args) {
  
  const char *infilename="/dev/stdin";
  int nblocks=4;
  int n_scale=3;
  int orientations_per_scale[50]={8,8,4};
  

  while(*++args) {
    const char *a=*args;
    
    if(!strcmp(a,"-h")) usage();
    else if(!strcmp(a,"-nblocks")) {
      if(!sscanf(*++args,"%d",&nblocks)) {
        fprintf(stderr,"could not parse %s argument",a); 
        usage();
      }
    } else if(!strcmp(a,"-orientationsPerScale")) {
      char *c;
      n_scale=0;
      for(c=strtok(*++args,",");c;c=strtok(NULL,",")) {
        if(!sscanf(c,"%d",&orientations_per_scale[n_scale++])) {
          fprintf(stderr,"could not parse %s argument",a); 
          usage();         
        }
      }
    } else {
      infilename=a;
    }
  }

  color_image_t *im=load_ppm(infilename);
  
  float *desc=color_gist_scaletab(im,nblocks,n_scale,orientations_per_scale);

  int i;
  
  int descsize=0;
  /* compute descriptor size */
  for(i=0;i<n_scale;i++) 
    descsize+=nblocks*nblocks*orientations_per_scale[i];

  descsize*=3; /* color */

  /* print descriptor */
  for(i=0;i<descsize;i++) 
    printf("%.4f ",desc[i]);

  printf("\n");
  
  free(desc);

  color_image_delete(im);

  return 0; 
}
