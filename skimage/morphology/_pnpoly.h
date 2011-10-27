/* `pnpoly` is from

   http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html

   Copyright (c) 1970-2003, Wm. Randolph Franklin

   Permission is hereby granted, free of charge, to any person
   obtaining a copy of this software and associated documentation
   files (the "Software"), to deal in the Software without
   restriction, including without limitation the rights to use, copy,
   modify, merge, publish, distribute, sublicense, and/or sell copies
   of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimers.
    2. Redistributions in binary form must reproduce the above
       copyright notice in the documentation and/or other materials
       provided with the distribution.
    3. The name of W. Randolph Franklin may not be used to endorse or
       promote products derived from this Software without specific
       prior written permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE. */

#ifdef __cplusplus
extern "C" {
#endif

unsigned char pnpoly(int nr_verts, double *xp, double *yp, double x, double y)
{
    int i, j;
    unsigned char c = 0;
    for (i = 0, j = nr_verts-1; i < nr_verts; j = i++) {
        if ((((yp[i]<=y) && (y<yp[j])) ||
             ((yp[j]<=y) && (y<yp[i]))) &&
            (x < (xp[j] - xp[i]) * (y - yp[i]) / (yp[j] - yp[i]) + xp[i]))
	    
	    c = !c;
    }
    return c;
}

void npnpoly(int nr_verts, double *xp, double *yp,
	     int nr_points, double *x, double *y,
	     unsigned char *result)
/*
 * For N provided points, calculate whether they are in
 * the polygon defined by vertices *xp, *yp.
 *
 * nr_verts : number of vertices
 * *xp, *yp : x and y coordinates of vertices
 * nr_points : number of data points provided
 * *x, *y : data points
 */
{
    unsigned char n = 0;
    for (n = 0; n < nr_points; n++) {
	result[n] = pnpoly(nr_verts, xp, yp, x[n], y[n]);
    }
}

#ifdef __cplusplus
}
#endif
