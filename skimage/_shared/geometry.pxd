cdef unsigned char point_in_polygon(ssize_t nr_verts, double *xp, double *yp,
                                    double x, double y)

cdef void points_in_polygon(ssize_t nr_verts, double *xp, double *yp,
                            ssize_t nr_points, double *x, double *y,
                            unsigned char *result)
