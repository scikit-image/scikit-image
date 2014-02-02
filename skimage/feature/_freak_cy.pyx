#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False



def _get_mean_intensity(double[:, ::1] integral_img, Py_ssize_t keypoint_x,
                        Py_ssize_t keypoint_y, double[:, ::1] rotated_pattern,
                        double[:] radii_stretched, double pattern_scale,
                        double[:] pattern_intensities):

    cdef Py_ssize_t i
    cdef double x_p, y_p, t

    for i in range(len(rotated_pattern)):
        x_p = keypoint_x + rotated_pattern[i, 0]
        y_p = keypoint_y + rotated_pattern[i, 1]
        t = radii_stretched[i] * pattern_scale
        pattern_intensities[i] = (integral_img[<int>(x_p + t), <int>(y_p + t)]
                                  + integral_img[<int>(x_p - t - 1), <int>(y_p - t)]
                                  - integral_img[<int>(x_p - t - 1), <int>(y_p + t)]
                                  - integral_img[<int>(x_p + t), <int>(y_p - t - 1)])
    return pattern_intensities

