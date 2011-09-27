/* cpmorphology.c - C routines for morphology processing
 *
 * CellProfiler is distributed under the GNU General Public License,
 * but this file is licensed under the more permissive BSD license.
 * See the accompanying file LICENSE for details.
 *
 * Copyright (c) 2003-2009 Massachusetts Institute of Technology
 * Copyright (c) 2009-2011 Broad Institute
 * All rights reserved.
 *
 * Please see the AUTHORS file for credits.
 *
 * Website: http://www.cellprofiler.org
 *
 *
 * cpmaximum - find the maximum value for an array within
 *             the context of a structuring element.
 *             arg1 - the float64 ndarray image
 *             arg2 - the boolean ndarray structuring element
 *             arg3 - the offset of the origin of the structuring element (a 2-tuple)
 */
#include <stdlib.h>
#include <math.h>
#include <Python.h>
#include <numpy/arrayobject.h>

static void to_stdout(const char *text)
{
     PySys_WriteStdout(text);     
}

static PyObject *
cpmaximum(PyObject *self, PyObject *args)
{
     PyObject *image     = NULL; /* the image ndarray */
     PyObject *cimage    = NULL; /* a contiguous version of the array */
     PyObject *structure = NULL; /* the structure ndarray */
     PyObject *cstructure= NULL; /* a contiguous version of the structure */
     PyObject *im_shape  = NULL; /* the image shape sequence */
     PyObject *str_shape = NULL; /* the structure shape sequence */
     PyObject *offset    = NULL; /* 2-tuple giving the x and y offset of the origin of the structuring element */
     long     height     = 0;    /* the image height */
     long     width      = 0;    /* the image width */
     long     strheight  = 0;    /* the structuring element height */
     long     strwidth   = 0;    /* the structuring element width */
     PyObject *output    = NULL; /* the output ndarray */
     double   *imdata    = NULL; /* the image data */
     char     *strdata   = NULL; /* the structure data (really boolean) */
     double   *out_data  = NULL; /* the output data */
     const char *error   = NULL;
     PyObject **shapes[] = { &im_shape, &im_shape, &str_shape, &str_shape };
     long     *slots[]   = { &width, &height, &strwidth, &strheight };
     int      indices[]  = { 0,1,0,1 };
     int      i          = 0;
     int      j          = 0;
     int      k          = 0;
     int      l          = 0;
     npy_intp dims[2];
     long     xoff       = 0;
     long     yoff       = 0;

     image = PySequence_GetItem(args,0);
     if (! image) {
          error = "Failed to get image from arguments";
          goto exit;
     }
     cimage = PyArray_ContiguousFromAny(image, NPY_DOUBLE,2,2);
     if (! cimage) {
          error = "Failed to make a contiguous array from first argument";
          goto exit;
     }
     structure = PySequence_GetItem(args,1);
     if (! structure) {
          error = "Failed to get structuring element from arguments";
          goto exit;
     }
     cstructure = PyArray_ContiguousFromAny(structure, NPY_BOOL,2,2);
     if (! cstructure) {
          error = "Failed to make a contiguous array from second argument";
          goto exit;
     }
     im_shape = PyObject_GetAttrString(cimage,"shape");
     if (! im_shape) {
          error = "Failed to get image.shape";
          goto exit;
     }
     if (! PyTuple_Check(im_shape)) {
          error = "image.shape not a tuple";
          goto exit;
     }
     if (PyTuple_Size(im_shape) != 2) {
          error = "image is not 2D";
          goto exit;
     }

     str_shape = PyObject_GetAttrString(cstructure,"shape");
     if (! str_shape) {
          error = "Failed to get structure.shape";
          goto exit;
     }
     if (! PyTuple_Check(str_shape)) {
          error = "structure.shape not a tuple";
          goto exit;
     }
     if (PyTuple_Size(str_shape) != 2) {
          error = "structure is not 2D";
          goto exit;
     }
     for (i=0;i<4;i++) {
          PyObject *obDim = PyTuple_GetItem(*shapes[i],indices[i]);
          *(slots[i]) = PyInt_AsLong(obDim);
          if (PyErr_Occurred()) {
               error = "Array shape is not a tuple of integers";
               goto exit;
          }
     }
     imdata = (double *)PyArray_DATA(cimage);
     if (! imdata) {
          error = "Failed to get image data";
          goto exit;
     }
     strdata = (char *)PyArray_DATA(cstructure);
     if (! strdata) {
          error = "Failed to get structure data";
          goto exit;
     }
     offset = PySequence_GetItem(args,2);
     if (! offset) {
          error = "Failed to get offset into structure from args";
          goto exit;
     }
     if (! PyTuple_Check(offset)) {
          error = "offset is not a tuple";
          goto exit;
     } else {
          PyObject *temp = PyTuple_GetItem(offset,0);
          if (! temp) {
               error = "Failed to get x offset from tuple";
               goto exit;
          }
          xoff = PyInt_AsLong(temp);
          if (PyErr_Occurred()) {
               error = "Offset X is not an integer";
               goto exit;
          }
          temp = PyTuple_GetItem(offset,1);
          if (! temp) {
               error = "Failed to get y offset from tuple";
               goto exit;
          }
          yoff = PyInt_AsLong(temp);
          if (PyErr_Occurred()) {
               error = "Offset Y is not an integer";
               goto exit;
          }
     }
     
     dims[0] = width;
     dims[1] = height;
     output = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
     if (! output) {
          error = "Failed to create output array";
          goto exit;
     }
     out_data = (double *)PyArray_DATA(output);
     memcpy(out_data,imdata,height*width*sizeof(double));

     for (j=0;j<height;j++) {
          for (i=0;i<width;i++,imdata++) {
               char *strptr = strdata;
               double value = - 1000000; /* was INFINITY but Unix doesn't like */
               if (i-xoff < 0 ||
                   j-yoff < 0 ||
                   i+strwidth-xoff >= width ||
                   j+strheight-yoff >= height) {
                    /* A corner case (literally) - check limit as we go */
                    for (l=-yoff;l<strheight-yoff;l++) {
                         double *imdata_ptr = imdata + l*width-xoff;
                         for (k=-xoff;k<strwidth-xoff;k++,imdata_ptr++) {
                              double data;
                              if (! *strptr++) continue;
                              if (i+k < 0) continue;
                              if (i+k >= width) continue;
                              if (j+l < 0) continue;
                              if (j+l >= width) continue;
                              data = *imdata_ptr;
                              if (data > value)
                                   value = data;
                         }
                    }
               } else {
                    /* Don't worry about corners and go faster */
                    for (l=-yoff;l<strheight-yoff;l++) {
                         double *imdata_ptr = imdata + l*width-xoff;
                         for (k=-xoff;k<strwidth-xoff;k++,imdata_ptr++) {
                              double data;
                              if (! *strptr++) continue;
                              data = *imdata_ptr;
                              if (data > value)
                                   value = data;
                         }
                    }
               }
               *out_data++ = value;
          }
     }
     

  exit:
     if (image) {
          Py_DECREF(image);
     }
     if (cimage) {
          Py_DECREF(cimage);
     }
     if (structure) {
          Py_DECREF(structure);
     }
     if (cstructure) {
          Py_DECREF(cstructure);
     }
     if (im_shape) {
          Py_DECREF(im_shape);
     }
     if (str_shape) {
          Py_DECREF(str_shape);
     }
     if (offset) {
          Py_DECREF(offset);
     }

     if (error) {
          if (output) {
               Py_DECREF(output);
          }
          output = PyString_FromString(error);
          if (! output) {
               Py_RETURN_NONE;
          }
     }
     return output;
}

static PyMethodDef methods[] = {
    {"cpmaximum", (PyCFunction)cpmaximum, METH_VARARGS, NULL},
    { NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC init_cpmorphology(void)
{
    Py_InitModule("_cpmorphology", methods);
    import_array();
}

