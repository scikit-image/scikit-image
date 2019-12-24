// median finding via wirths method
// http://www.gentoogeek.org/files/median/medians__1_d_8h.html#a3eb4543b5522a9ff6feba692cab5bc7a

// 2D

__kernel void median_2(__global ${DTYPE} * input,
						__global ${DTYPE} * output){

  int x = get_global_id(0);
  int y = get_global_id(1);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  ${DTYPE} a[${FSIZE_Y}*${FSIZE_X}];

  for (int m = 0; m < ${FSIZE_Y}; ++m) {
	for (int n = 0; n < ${FSIZE_X}; ++n) {
		
	  int x2 = x+n-(${FSIZE_X}/2);
	  int y2 = y+m-(${FSIZE_Y}/2);
		
	  bool inside = ((x2>=0)&&(x2<Nx)&&(y2>=0)&&(y2<Ny));
		

	  a[n+${FSIZE_X}*m] = inside?input[x2+y2*Nx]:(${DTYPE})(${CVAL});


	}
	  
  }


  // find median via Wirth
  
  int k = (${FSIZE_X}*${FSIZE_Y})/2;
  int n = (${FSIZE_X}*${FSIZE_Y});
  int i,j,l,m;
  
  ${DTYPE} val;

  l=0 ; m=n-1 ;
  while (l<m) {
  	val=a[k] ;
  	i=l ;
  	j=m ;
  	do {
  	  while (a[i]<val) i++ ;
  	  while (val<a[j]) j-- ;
  	  if (i<=j) {
  		${DTYPE} tmp = a[i];
  		a[i] = a[j];
  		a[j] = tmp;
		
  		i++ ; j-- ;
  	  }
  	} while (i<=j) ;
  	if (j<k) l=i ;
  	if (k<i) m=j ;


  }
  
  output[x+y*Nx] = a[k];
  

}


__kernel void median_3(__global ${DTYPE} * input,
						__global ${DTYPE} * output){

  int x = get_global_id(0);
  int y = get_global_id(1);
  int z = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);



  ${DTYPE} a[${FSIZE_Z}*${FSIZE_Y}*${FSIZE_X}];

  for (int p = 0; p < ${FSIZE_Z}; ++p) {
	for (int m = 0; m < ${FSIZE_Y}; ++m) {
	  for (int n = 0; n < ${FSIZE_X}; ++n) {
		
	  int x2 = x+n-(${FSIZE_X}/2);
	  int y2 = y+m-(${FSIZE_Y}/2);
	  int z2 = z+p-(${FSIZE_Z}/2);
		
	  bool inside = ((x2>=0)&&(x2<Nx)&&(y2>=0)&&(y2<Ny)&&(z2>=0)&&(z2<Nz));

	  a[n+${FSIZE_X}*m+${FSIZE_X}*${FSIZE_Y}*p] = inside?input[x2+y2*Nx+z2*Nx*Ny]:(${DTYPE})(${CVAL});
	  }
	}
  }


  // find median via Wirth
  
  int k = (${FSIZE_X}*${FSIZE_Y}*${FSIZE_Z})/2;
  int n = (${FSIZE_X}*${FSIZE_Y}*${FSIZE_Z});
  int i,j,l,m;
  
  ${DTYPE} val;

  l=0 ; m=n-1 ;
  while (l<m) {
  	val=a[k] ;
  	i=l ;
  	j=m ;
  	do {
  	  while (a[i]<val) i++ ;
  	  while (val<a[j]) j-- ;
  	  if (i<=j) {
  		${DTYPE} tmp = a[i];
  		a[i] = a[j];
  		a[j] = tmp;
		
  		i++ ; j-- ;
  	  }
  	} while (i<=j) ;
  	if (j<k) l=i ;
  	if (k<i) m=j ;
  }
  
  output[x+y*Nx+z*Nx*Ny] = a[k];

}
