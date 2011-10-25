# Lear's GIST implementation, version 1.0, (c) INRIA 2009, Licence: GPL


all: compute_gist

WFFTINC=
WFFTLIB=
# WFFTINC=-I/scratch/adonis/mordelet/install/include
# WFFTLIB=-L/scratch/adonis/mordelet/install/lib


# gist.c: ../../descriptors/gist.c 
#	cp $< $@


gist.o: gist.c gist.h standalone_image.h
	gcc -c -Wall -g $< $(WFFTINC) -DUSE_GIST -DSTANDALONE_GIST

standalone_image.o: standalone_image.c standalone_image.h
	gcc -c -Wall -g $< 

compute_gist: compute_gist.c gist.o standalone_image.o
	gcc -Wall -g -o $@ $^ $(WFFTLIB) -lfftw3f



clean: 
	rm -f *.o compute_gist

