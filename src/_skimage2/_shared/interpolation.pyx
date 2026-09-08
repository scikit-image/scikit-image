from .interpolation cimport coord_map


def coord_map_py(dim, coord, mode):
    """Python wrapper for `interpolation.coord_map`."""
    mode_c = ord(mode[0].upper())
    return coord_map(dim, coord, mode_c)
