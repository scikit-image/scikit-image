"""
Image measurement tools. A collection of functions for image measurement.

Functions
---------
find_contours : Find iso-valued contours in a 2D array for a given level value.
regionprops : Calculate various properties of labeled image regions.
regionprops_table: Return measurements for image region properties in tabular form.
perimeter : Calculate perimeter of objects in the label image.
perimeter_crofton : Approximate perimeter of objects in the label image using Crofton formula.
euler_number : Calculate Euler characteristic of labeled image regions.
approximate_polygon : Approximate a polygonal curve(s) for a given contour or binary image.
subdivide_polygon : Subdivide a polygon into smaller chunks.
LineModelND : A class to fit a straight n-dimensional line to a set of data points.
CircleModel : A class to fit a circular model to a set of data points.
EllipseModel : A class to fit an elliptical model to a set of data points.
ransac : An implementation of the RANSAC algorithm for robust linear estimation.
block_reduce : Reduce the image size by block averaging.
moments : Calculate raw (uncentered) moments of the distribution.
moments_central : Calculate central moments of the distribution.
moments_coords : Calculate moments about the origin of the image.
moments_coords_central : Calculate central moments about the center of mass of the image.
moments_normalized : Calculate normalized central moments of the distribution.
moments_hu : Calculate Hu moments of the distribution.
inertia_tensor : Calculate inertia tensor of the input image.
inertia_tensor_eigvals : Calculate eigenvalues of the inertia tensor.
marching_cubes : Generate a triangular mesh for a given isosurface using the marching cubes algorithm.
mesh_surface_area : Compute total surface area of the triangular mesh.
profile_line : Profile intensity along a directed line segment inside the image.
label : Label connected regions of an integer array.
points_in_poly : Check if points fall within a polygon.
grid_points_in_poly : Find grid points falling within a polygon.
shannon_entropy : Calculate the Shannon entropy of an array-like sequence.
blur_effect : Apply a blur effect to an image.
pearson_corr_coeff : Compute Pearson's correlation coefficient between two channels.
manders_coloc_coeff : Compute Mander's coefficients for colocalization analysis.
manders_overlap_coeff : Compute Mander's overlap coefficients for colocalization analysis.
intersection_coeff : Compute intersection coefficient for colocalization analysis.

"""


from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes, mesh_surface_area
from ._regionprops import (regionprops, perimeter,
                           perimeter_crofton, euler_number, regionprops_table)
from ._polygon import approximate_polygon, subdivide_polygon
from .pnpoly import (points_in_poly, grid_points_in_poly)
from ._moments import (moments, moments_central, moments_coords,
                       moments_coords_central, moments_normalized, centroid,
                       moments_hu, inertia_tensor, inertia_tensor_eigvals)
from .profile import profile_line
from .fit import LineModelND, CircleModel, EllipseModel, ransac
from .block import block_reduce
from ._label import label
from .entropy import shannon_entropy
from ._blur_effect import blur_effect
from ._colocalization import (pearson_corr_coeff, manders_coloc_coeff,
                              manders_overlap_coeff, intersection_coeff)

__all__ = ['find_contours',
           'regionprops',
           'regionprops_table',
           'perimeter',
           'perimeter_crofton',
           'euler_number',
           'approximate_polygon',
           'subdivide_polygon',
           'LineModelND',
           'CircleModel',
           'EllipseModel',
           'ransac',
           'block_reduce',
           'moments',
           'moments_central',
           'moments_coords',
           'moments_coords_central',
           'moments_normalized',
           'moments_hu',
           'inertia_tensor',
           'inertia_tensor_eigvals',
           'marching_cubes',
           'mesh_surface_area',
           'profile_line',
           'label',
           'points_in_poly',
           'grid_points_in_poly',
           'shannon_entropy',
           'blur_effect',
           'pearson_corr_coeff',
           'manders_coloc_coeff',
           'manders_overlap_coeff',
           'intersection_coeff',
           ]
