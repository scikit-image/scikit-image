from ._hog import hog
from .texture import greycomatrix, greycoprops, local_binary_pattern
from .peak import peak_local_max
from .corner import (corner_kitchen_rosenfeld, corner_harris, corner_shi_tomasi,
                     corner_foerstner)
from .corner_cy import corner_moravec
from .template import match_template
