#include <Python.h>
#include "numpy/arrayobject.h"

static char _find_contours_doc[] =
"This module defines C helper functions for find_contours";


static char iterate_and_store_doc[] =
"iterate_and_store(array, level, vertex_connect_high)\n\
\n\
Iterate across the given array in a marching-squares fashion, looking for\n\
segments that cross 'level'. If such a segment is found, its coordinates are\n\
added to a growing list of segments, which is returned by the function.\n\
if vertex_connect_high is nonzero, high-values pixels are considered to be\n\
face+vertex connected into objects; otherwise low-valued pixels are.";


// Nasty macros to inline the inner loop of interpolating the position of
// the contour and add an appropriate tuple to the output list.
// These macros define blocks of code that can only be used within the inner
// loop of 'iterate_and_store' because they use variables therefrom.

#define GET_FRACTION(from_value, to_value) \
  ((level - from_value) / (to_value - from_value))

#define TOP { \
  output0 = coords[0]; \
  output1 = coords[1] + GET_FRACTION(*ul_ptr, *ur_ptr); \
}

#define BOTTOM { \
  output0 = coords[0] + 1; \
  output1 = coords[1] + GET_FRACTION(*ll_ptr, *lr_ptr); \
}

#define LEFT { \
  output0 = coords[0] + GET_FRACTION(*ul_ptr, *ll_ptr); \
  output1 = coords[1]; \
}

#define RIGHT { \
  output0 = coords[0] + GET_FRACTION(*ur_ptr, *lr_ptr); \
  output1 = coords[1] + 1; \
}

#define ADD_TUPLE { \
  PyObject* tuple = Py_BuildValue("(dd)", output0, output1); \
  if (!tuple) { \
    Py_DECREF(double_array); \
    Py_DECREF(arc_list); \
    return NULL; \
  } \
  char res = PyList_Append(arc_list, tuple); \
        Py_DECREF(tuple); \
        if (res < 0) { \
    Py_DECREF(double_array); \
    Py_DECREF(arc_list); \
                return NULL; \
        } \
}

#define ADD_SEGMENT(START, END) { \
  double output0, output1; \
  START \
  ADD_TUPLE \
        END \
        ADD_TUPLE \
}

static PyObject*
iterate_and_store(PyObject *self, PyObject *args)
{
        PyObject* array;
        double level;
        int vertex_connect_high;
        if (!PyArg_ParseTuple(args, "Odi:iterate_and_store", &array, &level,
          &vertex_connect_high)) {
                return NULL;
        }

  PyObject* double_array = PyArray_FromAny(array,
    PyArray_DescrFromType(NPY_DOUBLE), 2, 2, NPY_CONTIGUOUS | NPY_ALIGNED,
      NULL);
  if (!double_array) {
    return NULL;
  }

  npy_intp *dims = PyArray_DIMS(double_array);
  if (dims[0] < 2 || dims[1] < 2) {
    Py_DECREF(double_array);
    PyErr_SetString(PyExc_ValueError, "Input array must be at least 2x2.");
    return NULL;
  }

  // The plan is to iterate a 2x2 square across the input array. This means
  // that the upper-left corner of the square needs to iterate across a
  // sub-array that's one-less-large in each direction (so that the square
  // never steps out of bounds). The square is represented by four pointers:
  // ul, ur, ll, and lr (for 'upper left', etc.). We also maintain the current
  // 2D coordinates for the position of the upper-left pointer. Note that we
  // ensured that the array is of type 'double' and is C-contiguous (last
  // index varies the fastest).

  // Current coords start at 0,0.
  npy_intp coords[2] = {0,0};
  // Precompute the size of the array minus 2 in each direction, so we'll know
  // when to update the coordinates and double-increment the square pointers
  // so that the upper-left pointer never visits the last column.
  npy_intp dims_m2[2];
  dims_m2[0] = dims[0] - 2;
  dims_m2[1] = dims[1] - 2;
  // Calculate the number of iterations we'll need
  npy_intp num_square_steps = (dims[0] - 1) * (dims[1] - 1);
  // and set up the square pointers.
  double* ul_ptr = PyArray_DATA(double_array);
  double* ur_ptr = ul_ptr + 1;
  double* ll_ptr = ul_ptr + dims[1];
  double* lr_ptr = ll_ptr + 1;

  // make a list to hold the returned coordinates
  PyObject* arc_list = PyList_New(0);
  if (!arc_list) {
    Py_DECREF(double_array);
    return NULL;
  }
  while(num_square_steps--) {
    // There are sixteen different possible square types, diagramed below.
    // A + indicates that the vertex is above the contour value, and a -
    // indicates that the vertex is below or equal to the contour value.
    // The vertices of each square are:
    // ul ur
    // ll lr
    // and can be treated as a binary value with the bits in that order. Thus
    // each square case can be numbered:
    //  0--   1+-   2-+   3++   4--   5+-   6-+   7++
    //   --    --    --    --    +-    +-    +-    +-
    //
    //  8--   9+-  10-+  11++  12--  13+-  14-+  15++
    //   -+    -+    -+    -+    ++    ++    ++    ++
    //
    // The position of the line segment that cuts through (or doesn't, in case
    // 0 and 15) each square is clear, except in cases  6 and 9. In this case,
    // where the segments are placed is determined by vertex_connect_high.
    // If vertex_connect_high is false, then lines like \\ are drawn
    // through square 6, and lines like // are drawn through square 9.
    // Otherwise, the situation is reversed.
    // Finally, recall that we draw the lines so that (moving from tail to
    // head) the lower-valued pixels are on the left of the line. So, for
    // example, case 1 entails a line slanting from the middle of the top of
    // the square to the middle of the left side of the square.

    unsigned char square_case = 0;
    if ((*ul_ptr) > level) square_case += 1;
    if ((*ur_ptr) > level) square_case += 2;
    if ((*ll_ptr) > level) square_case += 4;
    if ((*lr_ptr) > level) square_case += 8;

    switch(square_case)
      {
      case 0: // no line
        break;
      case 1:  // top to left
        ADD_SEGMENT(TOP, LEFT);
        break;
      case 2: // right to top
        ADD_SEGMENT(RIGHT, TOP);
        break;
      case 3: // right to left
        ADD_SEGMENT(RIGHT, LEFT);
        break;
      case 4: // left to bottom
        ADD_SEGMENT(LEFT, BOTTOM);
        break;
      case 5: // top to bottom
        ADD_SEGMENT(TOP, BOTTOM);
        break;
      case 6:
        if (vertex_connect_high)
          {
          // left to top
          ADD_SEGMENT(LEFT, TOP);
          // right to bottom
          ADD_SEGMENT(RIGHT, BOTTOM);
          }
        else
          {
          // right to top
          ADD_SEGMENT(RIGHT, TOP);
          // left to bottom
          ADD_SEGMENT(LEFT, BOTTOM);
          }
        break;
      case 7: // right to bottom
        ADD_SEGMENT(RIGHT, BOTTOM);
        break;
      case 8: // bottom to right
        ADD_SEGMENT(BOTTOM, RIGHT);
        break;
      case 9:
        if (vertex_connect_high)
          {
          // top to right
          ADD_SEGMENT(TOP, RIGHT);
          // bottom to left
          ADD_SEGMENT(BOTTOM, LEFT);
          }
        else
          {
          // top to left
          ADD_SEGMENT(TOP, LEFT);
          // bottom to right
          ADD_SEGMENT(BOTTOM, RIGHT);
          }
        break;
      case 10: // bottom to top
        ADD_SEGMENT(BOTTOM, TOP);
        break;
      case 11: // bottom to left
        ADD_SEGMENT(BOTTOM, LEFT);
        break;
      case 12: // left to right
        ADD_SEGMENT(LEFT, RIGHT);
        break;
      case 13: // top to right
        ADD_SEGMENT(TOP, RIGHT);
        break;
      case 14: // left to top
        ADD_SEGMENT(LEFT, TOP);
        break;
      case 15: // no line
        break;
      } // switch square_case

    if (coords[1] < dims_m2[1]) {
      coords[1]++;
    } else {
      coords[1] = 0;
      coords[0]++;
      // Double-increment pointers to advance them to the next row, since
      // we're skipping the last column, as far as ul_ptr is concerned.
      ul_ptr++; ur_ptr++; ll_ptr++; lr_ptr++;
    }
    ul_ptr++; ur_ptr++; ll_ptr++; lr_ptr++;
  } // iteration

  // get rid of the double array reference that we own
  Py_DECREF(double_array);
  return arc_list;
}


static PyMethodDef _find_contours_methods[] = {
        {"iterate_and_store", iterate_and_store, METH_VARARGS, iterate_and_store_doc},
        {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
init_find_contours(void)
{
        Py_InitModule3("_find_contours", _find_contours_methods, _find_contours_doc);
        import_array();
}
