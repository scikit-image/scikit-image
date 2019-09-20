#pragma once

#include "snakes2d_operators.h"
#include "snakes2d_operators_bidir.h"
#include "snakes3d_operators.h"
#include "snakes3d_operators_bidir.h"
#include <algorithm>
#include <cassert>
#include <vector>

#include <algorithm>

#if defined(_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;
inline omp_int_t omp_get_thread_num() { return 0; }
inline omp_int_t omp_get_num_threads() { return 1; }
#endif

typedef struct point3d
{
  int x, y, z;
} point3d;

typedef struct point2d
{
  int x, y;
} point2d;

bool is_inside(int t_xi, int t_L) { return (t_xi >= 0) && (t_xi < t_L); }
bool is_central(int t_xi, int t_L) { return (t_xi > 0) && (t_xi < t_L - 1); }

bool SId_2d_borders(int xi, int yi, uint8_t *levelset, int nx, int ny)
{
  // All diagonal elements for the borders are zero
  // only non-diagonal masks are relevant
  int const stride_x = 1;
  int const stride_y = nx;
  int const index = xi + stride_y * yi;

  if (is_central(xi, nx))
  {
    // mask along x is valid
    return SId_2d_1(levelset, index, stride_x, stride_y);
  }
  if (is_central(yi, ny))
  {
    // mask along y is valid
    return SId_2d_3(levelset, index, stride_x, stride_y);
  }
  return false;
}

bool SId_3d_borders(int xi, int yi, int zi, uint8_t *levelset, int nx, int ny, int nz)
{
  // All diagonal elements for the borders are zero
  // only non-diagonal masks are relevant
  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;
  int const index = xi + stride_y * yi + stride_z * zi;

  if (is_central(xi, nx) && is_central(yi, ny))
  {
    // mask along x is valid
    return SId_3d_2(levelset, index, stride_x, stride_y, stride_z);
  }
  if (is_central(yi, ny) && is_central(zi, nz))
  {
    // mask along y is valid
    return SId_3d_0(levelset, index, stride_x, stride_y, stride_z);
  }
  if (is_central(zi, nz) && is_central(xi, nx))
  {
    // mask along z is valid
    return SId_3d_1(levelset, index, stride_x, stride_y, stride_z);
  }
  return false;
}

bool ISd_2d_borders(int xi, int yi, uint8_t *levelset, int nx, int ny)
{
  // I defined a bidirectional macro for the dilations,
  // outside coordinates are mapped back to INDEX, i.e. constant
  // boundary conditions with value = levelset[index]
  int stride_x = 1;
  int stride_y = nx;
  int nstride_x = 1;
  int nstride_y = nx;
  int const index = xi + stride_y * yi;

  if (xi == 0)
  {
    nstride_x = 0;
  }
  if (yi == 0)
  {
    nstride_y = 0;
  }
  if (xi == nx - 1)
  {
    stride_x = 0;
  }
  if (yi == ny - 1)
  {
    stride_y = 0;
  }
  return ISd_2d_any_bidir(levelset, index, nstride_x, stride_x, nstride_y, stride_y);
}
bool ISd_3d_borders(int xi, int yi, int zi, uint8_t *levelset, int nx, int ny, int nz)
{
  // I defined a bidirectional macro for the dilations,
  // outside coordinates are mapped back to INDEX, i.e. constant
  // boundary conditions with value = levelset[index]
  int stride_x = 1;
  int stride_y = nx;
  int stride_z = nx * ny;
  int nstride_x = 1;
  int nstride_y = nx;
  int nstride_z = nx * ny;
  int const index = xi + stride_y * yi + stride_z * zi;

  if (xi == 0)
  {
    nstride_x = 0;
  }
  if (yi == 0)
  {
    nstride_y = 0;
  }
  if (zi == 0)
  {
    nstride_z = 0;
  }
  if (xi == nx - 1)
  {
    stride_x = 0;
  }
  if (yi == ny - 1)
  {
    stride_y = 0;
  }
  if (zi == nz - 1)
  {
    stride_z = 0;
  }
  // printf("ISd_3d_borders (%d,%d,%d)/(%d,%d,%d)\n", xi, yi, zi, nx, ny, nz);
  // printf("strides (%d,%d,%d)/(%d,%d,%d)\n", nstride_x, nstride_y, nstride_z,
  // stride_x,
  //       stride_y, stride_z);

  return ISd_3d_any_bidir(levelset, index, nstride_x, stride_x, nstride_y, stride_y,
                          nstride_z, stride_z);
}

template <unsigned int N>
double masked_average(double *image, uint8_t *mask, int size)
{
  double cumulative_sum = 0;
  int counter = 0;
  for (int i = 0; i < size; i++)
  {
    if (mask[i] == N)
    {
      cumulative_sum += image[i];
      counter++;
    }
  }
  if (counter == 0)
  {
    return {0};
  }
  return cumulative_sum / counter;
}

template <class T>
void reduce(std::vector<T> *v1, int begin, int end)
{
  if (end - begin == 1)
    return;
  int pivot = (begin + end) / 2;
  // not uspported in MSVC
  //#pragma omp task
  reduce(v1, begin, pivot);
  reduce(v1, pivot, end);
  // not uspported in MSVC
  // #pragma omp taskwait
  v1[begin].insert(v1[begin].end(), v1[pivot].begin(), v1[pivot].end());
}

namespace pysnakes2d
{

bool is_edge(uint8_t *levelset, point2d point, int nx, int ny)
{
  int const xi = point.x;
  int const yi = point.y;

  int const stride_x = 1;
  int const stride_y = nx;

  // Edges are only inside image
  if (xi < 0 || yi < 0 || xi > nx - 1 || yi > ny - 1)
  {
    return false;
  }

  int const index = xi + stride_y * yi;
  // Define border as a not 1 valued voxel with an 6 connected active neighbour
  if (levelset[index] == 1)
  {
    return false;
  }

  int const index_left = index - (xi > 0) * stride_x;
  int const index_right = index + (xi < (nx - 1)) * stride_x;
  int const index_down = index - (yi > 0) * stride_y;
  int const index_up = index + (yi < (ny - 1)) * stride_y;
  if (levelset[index_left] == 1 || levelset[index_right] == 1 ||
      levelset[index_down] == 1 || levelset[index_up] == 1)
  {
    return true;
  }

  return false;
}

void update_edge(uint8_t *levelset, long *counter, std::vector<point2d> &edge_points,
                 int nx, int ny)
{

  counter[nx * ny] += 1;
  int current_iteration = counter[nx * ny];

  int const stride_x = 1;
  int const stride_y = nx;

  std::vector<point2d> new_edge;
  for (const point2d &point : edge_points)
  {
    int const index = point.x + stride_y * point.y;
    if (counter[index] != current_iteration)
    {
      counter[index] = current_iteration;
      if (is_edge(levelset, point, nx, ny))
      {
        new_edge.push_back(point);
      }
    }
  }
  edge_points = new_edge;
}

void check_and_add_edges(std::vector<point2d> &edge_points,
                         std::vector<point2d> &unchecked_points, uint8_t *levelset,
                         long *counter, int nx, int ny)
{

  // counter[nx*ny*nz] += 1;
  int current_iteration = counter[nx * ny];

  int const stride_x = 1;
  int const stride_y = nx;

  for (const point2d &p : unchecked_points)
  {
    for (int yi = p.y - 1; yi < p.y + 2; yi++)
    {
      if (is_inside(yi, ny))
      {
        for (int xi = p.x - 1; xi < p.x + 2; xi++)
        {
          if (is_inside(xi, nx))
          {
            int index = xi + stride_y * yi;
            if (counter[index] != current_iteration)
            {
              counter[index] = current_iteration;

              int index_left = index - (xi > 0) * stride_x;
              int index_right = index + (xi < (nx - 1)) * stride_x;
              int index_down = index - (yi > 0) * stride_y;
              int index_up = index + (yi < (ny - 1)) * stride_y;

              if (levelset[index] != 1 &&
                  (levelset[index_left] == 1 || levelset[index_right] == 1 ||
                   levelset[index_down] == 1 || levelset[index_up] == 1))
              {
                point2d edge = {xi, yi};
                edge_points.push_back(edge);
              }
            }
          }
        }
      }
    }
  }
  return;
}
} // namespace pysnakes2d

void evolve_edge_2d(double *image, uint8_t *levelset, long *counter,
                    std::vector<point2d> &edge_points, int nx, int ny, double lambda1,
                    double lambda2)
{

  int const stride_x = 1;
  int const stride_y = nx;

  double c0 = masked_average<0>(image, levelset, nx * ny);
  double c1 = masked_average<1>(image, levelset, nx * ny);

  counter[nx * ny] += 1;
  int current_iteration = counter[nx * ny];

  std::vector<point2d> changed_add, changed_remove;
  std::vector<point2d> *changed_add_p, *changed_remove_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_add_p = new std::vector<point2d>[omp_get_num_threads()];
      changed_remove_p = new std::vector<point2d>[omp_get_num_threads()];
    }

    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);
#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point2d p = edge_points[pi];
      for (int yi = p.y - 1; yi < p.y + 2; yi++)
      {
        if (is_central(yi, ny))
        {
          for (int xi = p.x - 1; xi < p.x + 2; xi++)
          {
            if (is_central(xi, nx))
            {
              int const index = xi + stride_y * yi;

              if (counter[index] != current_iteration)
              {
                // possible race
                counter[index] = current_iteration;

                bool gx = levelset[index + stride_x] != levelset[index - stride_x];
                bool gy = levelset[index + stride_y] != levelset[index - stride_y];

                bool abs_grad = (gx || gy);
                double value = image[index];
                double aux =
                    abs_grad
                        ? (lambda1 * pow(value - c1, 2) - lambda2 * pow(value - c0, 2))
                        : 0.0;

                if (aux < 0 && levelset[index] == 0)
                {
                  point2d point = {xi, yi};
                  changed_add_p[omp_get_thread_num()].push_back(point);
                }
                if (aux > 0 && levelset[index] == 1)
                {
                  point2d point = {xi, yi};
                  changed_remove_p[omp_get_thread_num()].push_back(point);
                }
              }
            }
          }
        }
      }
    }
#pragma omp single
    {
      reduce(changed_add_p, 0, omp_get_num_threads());
      reduce(changed_remove_p, 0, omp_get_num_threads());
    }
  }
  changed_add = changed_add_p[0], changed_remove = changed_remove_p[0];
  delete[] changed_add_p;
  delete[] changed_remove_p;

  for (const point2d &point : changed_add)
  {
    levelset[point.x + stride_y * point.y] = 1;
  }
  for (const point2d &point : changed_remove)
  {
    levelset[point.x + stride_y * point.y] = 0;
  }

  pysnakes2d::update_edge(levelset, counter, edge_points, nx, ny);
  pysnakes2d::check_and_add_edges(edge_points, changed_remove, levelset, counter, nx,
                                  ny);
  pysnakes2d::check_and_add_edges(edge_points, changed_add, levelset, counter, nx, ny);
  return;
}

std::vector<point2d> get_edge_list_2d(uint8_t *levelset, int nx, int ny)
{

  int const stride_x = 1;
  int const stride_y = nx;

  std::vector<point2d> retval;
  for (int yi = 0; yi < ny; yi++)
  {
    for (int xi = 0; xi < nx; xi++)
    {

      int index = xi + stride_y * yi;
      int index_left = index - (xi > 0) * stride_x;
      int index_right = index + (xi < (nx - 1)) * stride_x;
      int index_down = index - (yi > 0) * stride_y;
      int index_up = index + (yi < (ny - 1)) * stride_y;
      if (levelset[index] != 1 &&
          (levelset[index_left] == 1 || levelset[index_right] == 1 ||
           levelset[index_down] == 1 || levelset[index_up] == 1))
      {
        point2d edge = {xi, yi};
        retval.push_back(edge);
      }
    }
  }

  return retval;
}

void fast_marching_erosion_2d(std::vector<point2d> &edge_points, uint8_t *levelset,
                              long *counter, int nx, int ny)
{

  int const stride_x = 1;
  int const stride_y = nx;

  counter[nx * ny] += 1;
  int current_iteration = counter[nx * ny];

  std::vector<point2d> changed;
  std::vector<point2d> *changed_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_p = new std::vector<point2d>[omp_get_num_threads()];
    }

    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);

#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point2d p = edge_points[pi];
      for (int yi = p.y - 1; yi < p.y + 2; yi++)
      {
        if (is_inside(yi, ny))
        {
          for (int xi = p.x - 1; xi < p.x + 2; xi++)
          {
            if (is_inside(xi, nx))
            {
              int const index = xi + stride_y * yi;
              if (counter[index] != current_iteration)
              {
                counter[index] = current_iteration;
                // // apply erosion only on 1
                if (levelset[index] == 1)
                {

                  if (is_central(xi, nx) && is_central(yi, ny))
                  {
                    // normal case
                    if (SId_2d_any(levelset, index, stride_x, stride_y) == 0)
                    {
                      point2d point = {xi, yi};
                      changed_p[omp_get_thread_num()].push_back(point);
                    }
                  }
                  else if (SId_2d_borders(xi, yi, levelset, nx, ny) == 0)
                  {
                    point2d point = {xi, yi};
                    changed_p[omp_get_thread_num()].push_back(point);
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp single
    {
      reduce(changed_p, 0, omp_get_num_threads());
    }
  }
  changed = changed_p[0];
  delete[] changed_p;

  if (changed.size() == 0)
    return;

  for (const point2d &point : changed)
  {
    levelset[point.x + stride_y * point.y] = 0;
  }

  pysnakes2d::update_edge(levelset, counter, edge_points, nx, ny);
  pysnakes2d::check_and_add_edges(edge_points, changed, levelset, counter, nx, ny);
  return;
}

void fast_marching_dilation_2d(std::vector<point2d> &edge_points, uint8_t *levelset,
                               long *counter, int nx, int ny)
{
  int const stride_x = 1;
  int const stride_y = nx;

  counter[nx * ny] += 1;
  int current_iteration = counter[nx * ny];

  std::vector<point2d> changed;
  std::vector<point2d> *changed_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_p = new std::vector<point2d>[omp_get_num_threads()];
    }

    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);
#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point2d p = edge_points[pi];
      for (int yi = p.y - 1; yi < p.y + 2; yi++)
      {
        if (is_inside(yi, ny))
        {
          for (int xi = p.x - 1; xi < p.x + 2; xi++)
          {
            if (is_inside(xi, nx))
            {
              int const index = xi + stride_y * yi;
              if (counter[index] != current_iteration)
              {
                counter[index] = current_iteration;
                // apply dilation only on 0
                if (levelset[index] == 0)
                {
                  if (is_central(xi, nx) && is_central(yi, ny))
                  {
                    // normal case
                    if (ISd_2d_any(levelset, index, stride_x, stride_y) == 1)
                    {
                      point2d point = {xi, yi};
                      changed_p[omp_get_thread_num()].push_back(point);
                    }
                  }
                  else if (ISd_2d_borders(xi, yi, levelset, nx, ny) == 1)
                  {
                    point2d point = {xi, yi};
                    changed_p[omp_get_thread_num()].push_back(point);
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp single
    {
      reduce(changed_p, 0, omp_get_num_threads());
    }
  }
  changed = changed_p[0];
  delete[] changed_p;

  if (changed.size() == 0)
    return;

  for (const point2d &point : changed)
  {
    levelset[point.x + stride_y * point.y] = 1;
  }

  pysnakes2d::update_edge(levelset, counter, edge_points, nx, ny);
  pysnakes2d::check_and_add_edges(edge_points, changed, levelset, counter, nx, ny);
  return;
}

namespace pysnakes3d
{

bool is_edge(uint8_t *levelset, point3d point, int nx, int ny, int nz)
{
  int const xi = point.x;
  int const yi = point.y;
  int const zi = point.z;

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  int const index = xi + stride_y * yi + stride_z * zi;

  // Keep level sed padded from border
  if (xi < 0 || yi < 0 || zi < 0 || xi > nx - 1 || yi > ny - 1 || zi > nz - 1)
  {
    return false;
  }

  // Define border as a not 1 valued voxel with an 6 connected active neighbour
  if (levelset[index] == 1)
  {
    return false;
  }
  int const index_x0 = index - (xi > 0) * stride_x;
  int const index_x1 = index + (xi < (nx - 1)) * stride_x;
  int const index_y0 = index - (yi > 0) * stride_y;
  int const index_y1 = index + (yi < (ny - 1)) * stride_y;
  int const index_z0 = index - (zi > 0) * stride_z;
  int const index_z1 = index + (zi < (nz - 1)) * stride_z;
  if ((levelset[index_x0] == 1) || (levelset[index_x1] == 1) ||
      (levelset[index_y0] == 1) || (levelset[index_y1] == 1) ||
      (levelset[index_z0] == 1) || (levelset[index_z1] == 1))
  {
    return true;
  }

  return false;
}

void update_edge(uint8_t *levelset, long *counter, std::vector<point3d> &edge_points,
                 int nx, int ny, int nz)
{
  counter[nx * ny * nz] += 1;
  int current_iteration = counter[nx * ny * nz];

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  std::vector<point3d> new_edge;
  new_edge.reserve(edge_points.size());

  for (const point3d &point : edge_points)
  {
    int const index = point.x + stride_y * point.y + stride_z * point.z;
    if (counter[index] != current_iteration)
    {
      counter[index] = current_iteration;
      if (is_edge(levelset, point, nx, ny, nz))
      {
        new_edge.push_back(point);
      }
    }
  }
  edge_points = new_edge;
}

void check_and_add_edges(std::vector<point3d> &edge_points,
                         std::vector<point3d> &unchecked_points, uint8_t *levelset,
                         long *counter, int nx, int ny, int nz)
{

  counter[nx * ny * nz] += 1;
  int current_iteration = counter[nx * ny * nz];

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  if (edge_points.capacity() < edge_points.size() + unchecked_points.size())
  {
    edge_points.reserve(edge_points.size() + unchecked_points.size());
  }

  for (const point3d &p : unchecked_points)
  {
    for (int zi = p.z - 1; zi < p.z + 2; zi++)
    {
      if (is_inside(zi, nz))
      {
        for (int yi = p.y - 1; yi < p.y + 2; yi++)
        {
          if (is_inside(yi, ny))
          {
            for (int xi = p.x - 1; xi < p.x + 2; xi++)
            {
              if (is_inside(xi, nx))
              {
                int const index = xi + stride_y * yi + stride_z * zi;
                if (counter[index] != current_iteration)
                {
                  counter[index] = current_iteration;

                  int const index_x0 = index - (xi > 0) * stride_x;
                  int const index_x1 = index + (xi < (nx - 1)) * stride_x;
                  int const index_y0 = index - (yi > 0) * stride_y;
                  int const index_y1 = index + (yi < (ny - 1)) * stride_y;
                  int const index_z0 = index - (zi > 0) * stride_z;
                  int const index_z1 = index + (zi < (nz - 1)) * stride_z;
                  if (levelset[index] != 1 &&
                      (levelset[index_x0] == 1 || levelset[index_x1] == 1 ||
                       levelset[index_y0] == 1 || levelset[index_y1] == 1 ||
                       levelset[index_z0] == 1 || levelset[index_z1] == 1))
                  {

                    point3d edge = {xi, yi, zi};
                    edge_points.push_back(edge);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return;
}

} // namespace pysnakes3d

void evolve_edge_3d(double *image, uint8_t *levelset, long *counter,
                    std::vector<point3d> &edge_points, int nx, int ny, int nz,
                    double lambda1, double lambda2)
{

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  double c0 = masked_average<0>(image, levelset, nx * ny * nz);
  double c1 = masked_average<1>(image, levelset, nx * ny * nz);

  counter[nx * ny * nz] += 1;
  int current_iteration = counter[nx * ny * nz];

  std::vector<point3d> changed_add, changed_remove;
  std::vector<point3d> *changed_add_p, *changed_remove_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_add_p = new std::vector<point3d>[omp_get_num_threads()];
      changed_remove_p = new std::vector<point3d>[omp_get_num_threads()];
    }

    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);
#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point3d p = edge_points[pi];
      for (int zi = p.z - 1; zi < p.z + 2; zi++)
      {
        if (is_central(zi, nz))
        {
          for (int yi = p.y - 1; yi < p.y + 2; yi++)
          {
            if (is_central(yi, ny))
            {
              for (int xi = p.x - 1; xi < p.x + 2; xi++)
              {
                if (is_central(xi, nx))
                {
                  int const index = xi + stride_y * yi + stride_z * zi;

                  if (counter[index] != current_iteration)
                  {
                    // possible race gets cleared up at at check_and_add_edges
                    counter[index] = current_iteration;

                    bool gx = levelset[index + stride_x] != levelset[index - stride_x];
                    bool gy = levelset[index + stride_y] != levelset[index - stride_y];
                    bool gz = levelset[index + stride_z] != levelset[index - stride_z];

                    double abs_grad = (gx || gy || gz);
                    double value = image[index];
                    double aux = abs_grad ? (lambda1 * pow(value - c1, 2) -
                                             lambda2 * pow(value - c0, 2))
                                          : 0.0;

                    if (aux < 0 && levelset[index] == 0)
                    {
                      point3d point = {xi, yi, zi};
                      changed_add_p[omp_get_thread_num()].push_back(point);
                    }
                    if (aux > 0 && levelset[index] == 1)
                    {
                      point3d point = {xi, yi, zi};
                      changed_remove_p[omp_get_thread_num()].push_back(point);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#pragma omp single
    {
      reduce(changed_add_p, 0, omp_get_num_threads());
      reduce(changed_remove_p, 0, omp_get_num_threads());
    }
  }
  changed_add = changed_add_p[0], changed_remove = changed_remove_p[0];
  delete[] changed_add_p;
  delete[] changed_remove_p;

  for (const point3d &point : changed_add)
  {
    levelset[point.x + stride_y * point.y + stride_z * point.z] = 1;
  }
  for (const point3d &point : changed_remove)
  {
    levelset[point.x + stride_y * point.y + stride_z * point.z] = 0;
  }

  pysnakes3d::update_edge(levelset, counter, edge_points, nx, ny, nz);
  pysnakes3d::check_and_add_edges(edge_points, changed_remove, levelset, counter, nx,
                                  ny, nz);
  pysnakes3d::check_and_add_edges(edge_points, changed_add, levelset, counter, nx, ny,
                                  nz);
  return;
}

void fast_marching_erosion_3d(std::vector<point3d> &edge_points, uint8_t *levelset,
                              long *counter, int nx, int ny, int nz)
{

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  counter[nx * ny * nz] += 1;
  int current_iteration = counter[nx * ny * nz];

  std::vector<point3d> changed;
  std::vector<point3d> *changed_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_p = new std::vector<point3d>[omp_get_num_threads()];
    }

    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);
#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point3d point = edge_points[pi];
      for (int k = -1; k < 2; k++)
      {
        int zi = point.z + k;
        if (is_inside(zi, nz))
        {
          for (int j = -1; j < 2; j++)
          {
            int yi = point.y + j;
            if (is_inside(yi, ny))
            {
              for (int i = -1; i < 2; i++)
              {
                int xi = point.x + i;
                if (is_inside(xi, nx))
                {
                  int const index = xi + stride_y * yi + stride_z * zi;
                  if (counter[index] != current_iteration)
                  {
                    counter[index] = current_iteration;
                    // // apply erosion only on 1
                    if (levelset[index] == 1)
                    {
                      if (is_central(xi, nx) && is_central(yi, ny) &&
                          is_central(zi, nz))
                      {
                        // normal case
                        if (SId_3d_any(levelset, index, stride_x, stride_y, stride_z) ==
                            0)
                        {
                          point3d point = {xi, yi, zi};
                          changed_p[omp_get_thread_num()].push_back(point);
                        }
                      }
                      else if (SId_3d_borders(xi, yi, zi, levelset, nx, ny, nz) ==
                               0)
                      {
                        point3d point = {xi, yi, zi};
                        changed_p[omp_get_thread_num()].push_back(point);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

#pragma omp single
    {
      reduce(changed_p, 0, omp_get_num_threads());
    }
  }
  changed = changed_p[0];
  delete[] changed_p;

  if (changed.size() == 0)
    return;

  for (const point3d &point : changed)
  {
    levelset[point.x + stride_y * point.y + stride_z * point.z] = 0;
  }

  pysnakes3d::update_edge(levelset, counter, edge_points, nx, ny, nz);
  pysnakes3d::check_and_add_edges(edge_points, changed, levelset, counter, nx, ny, nz);
  return;
}

void fast_marching_dilation_3d(std::vector<point3d> &edge_points, uint8_t *levelset,
                               long *counter, int nx, int ny, int nz)
{
  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  counter[nx * ny * nz] += 1;
  int current_iteration = counter[nx * ny * nz];

  std::vector<point3d> changed;
  std::vector<point3d> *changed_p;

#pragma omp parallel
  {
#pragma omp single
    {
      changed_p = new std::vector<point3d>[omp_get_num_threads()];
    }
    // cast to int to be compatible with OpenMP 2.0
    assert(edge_points.size() < INT_MAX / 2);
#pragma omp for
    for (int pi = 0; pi < static_cast<int>(edge_points.size()); pi++)
    {
      point3d p = edge_points[pi];
      for (int zi = p.z - 1; zi < p.z + 2; zi++)
      {
        if (is_inside(zi, nz))
        {
          for (int yi = p.y - 1; yi < p.y + 2; yi++)
          {
            if (is_inside(yi, ny))
            {
              for (int xi = p.x - 1; xi < p.x + 2; xi++)
              {
                if (is_inside(xi, nx))
                {

                  int const index = xi + stride_y * yi + stride_z * zi;
                  if (counter[index] != current_iteration)
                  {
                    counter[index] = current_iteration;

                    // apply dilation only on 0
                    if (levelset[index] == 0)
                    {
                      if (is_central(xi, nx) && is_central(yi, ny) &&
                          is_central(zi, nz))
                      {

                        // normal case
                        if (ISd_3d_any(levelset, index, stride_x, stride_y, stride_z) ==
                            1)
                        {
                          point3d point = {xi, yi, zi};
                          changed_p[omp_get_thread_num()].push_back(point);
                        }
                      }
                      else if (ISd_3d_borders(xi, yi, zi, levelset, nx, ny, nz) ==
                               1)
                      {
                        point3d point = {xi, yi, zi};
                        changed_p[omp_get_thread_num()].push_back(point);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
#pragma omp single
    {
      reduce(changed_p, 0, omp_get_num_threads());
    }
  }
  changed = changed_p[0];
  delete[] changed_p;

  if (changed.size() == 0)
    return;

  for (const point3d &point : changed)
  {
    levelset[point.x + stride_y * point.y + stride_z * point.z] = 1;
  }

  pysnakes3d::update_edge(levelset, counter, edge_points, nx, ny, nz);
  pysnakes3d::check_and_add_edges(edge_points, changed, levelset, counter, nx, ny, nz);
  return;
}

std::vector<point3d> get_edge_list_3d(uint8_t *levelset, int nx, int ny, int nz)
{

  int const stride_x = 1;
  int const stride_y = nx;
  int const stride_z = nx * ny;

  std::vector<point3d> retval;
  for (int zi = 0; zi < nz; zi++)
  {
    for (int yi = 0; yi < ny; yi++)
    {
      for (int xi = 0; xi < nx; xi++)
      {
        int index = xi + stride_y * yi + stride_z * zi;
        int const index_x0 = index - (xi > 0) * stride_x;
        int const index_x1 = index + (xi < (nx - 1)) * stride_x;
        int const index_y0 = index - (yi > 0) * stride_y;
        int const index_y1 = index + (yi < (ny - 1)) * stride_y;
        int const index_z0 = index - (zi > 0) * stride_z;
        int const index_z1 = index + (zi < (nz - 1)) * stride_z;
        if (levelset[index] != 1 &&
            (levelset[index_x0] == 1 || levelset[index_x1] == 1 ||
             levelset[index_y0] == 1 || levelset[index_y1] == 1 ||
             levelset[index_z0] == 1 || levelset[index_z1] == 1))
        {
          point3d edge = {xi, yi, zi};
          retval.push_back(edge);
        }
      }
    }
  }
  return retval;
}

struct sortfunc
{
  bool operator()(const point2d &left, const point2d &right) const
  {
    if (left.y != right.y)
    {
      return left.y < right.y;
    }
    return left.x < right.x;
  }
  bool operator()(const point3d &left, const point3d &right) const
  {
    if (left.z != right.z)
    {
      return left.z < right.z;
    }
    if (left.y != right.y)
    {
      return left.y < right.y;
    }
    return left.x < right.x;
  }
};

void sort_edge2d(std::vector<point2d> &edge)
{
  std::sort(edge.begin(), edge.end(), sortfunc());

  return;
}
void sort_edge3d(std::vector<point3d> &edge)
{
  std::sort(edge.begin(), edge.end(), sortfunc());

  return;
}
