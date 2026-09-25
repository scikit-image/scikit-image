"""
This script uses SymPy to generate the analytical equations used to transform
raw moments into central moments in ``skimage/measure/_moments_analytical.py``
"""

from sympy import symbols, binomial, Sum
from sympy import IndexedBase, Idx
from sympy.printing.pycode import pycode

# from sympy import init_printing, pprint
# init_printing(use_unicode=True)

# Define a moments matrix, M
M = IndexedBase('M')

ndim = 3
order = 3
if ndim == 2:
    # symbols for the centroid components
    c_x, c_y = symbols('c_x c_y')
    # indices into the moments matrix, M
    i, j = symbols('i j', cls=Idx)

    # centroid
    cx = M[1, 0] / M[0, 0]
    cy = M[0, 1] / M[0, 0]

    # Loop over all moments of order <= `order`.
    print(f"Generating expressions for {ndim}D moments of order <= {order}")
    for p in range(0, order + 1):
        for q in range(0, order + 1):
            if p + q > order:
                continue
            # print(f"order={p + q}: p={p}, q={q}")

            # This expression is given at
            # https://en.wikipedia.org/wiki/Image_moment#Central_moments
            expr = Sum(
                (
                    binomial(p, i)
                    * binomial(q, j)
                    * (-cx) ** (p - i)
                    * (-cy) ** (q - j)
                    * M[i, j]
                ),
                (i, 0, p),
                (j, 0, q),
            ).doit()

            # substitute back in the c_x and c_y symbols
            expr = expr.subs(M[1, 0] / M[0, 0], c_x)
            expr = expr.subs(M[0, 1] / M[0, 0], c_y)

            # print python code for generation of the central moment
            python_code = f"moments_central[{p}, {q}] = {pycode(expr)}\n"
            # replace symbol names with corresponding python variable names
            python_code = python_code.replace('c_', 'c')
            python_code = python_code.replace('M[', 'm[')
            print(python_code)

elif ndim == 3:
    # symbols for the centroid components
    c_x, c_y, c_z = symbols('c_x c_y c_z')
    # indices into the moments matrix, M
    i, j, k = symbols('i j k', cls=Idx)

    # centroid
    cx = M[1, 0, 0] / M[0, 0, 0]
    cy = M[0, 1, 0] / M[0, 0, 0]
    cz = M[0, 0, 1] / M[0, 0, 0]

    # Loop over all moments of order <= `order`.
    print(f"Generating expressions for {ndim}D moments of order <= {order}")
    for p in range(0, order + 1):
        for q in range(0, order + 1):
            for r in range(0, order + 1):
                if p + q + r > order:
                    continue
                # print(f"order={p + q + r}: p={p}, q={q}, r={r}")

                # This expression is a 3D extension of the 2D equation given at
                # https://en.wikipedia.org/wiki/Image_moment#Central_moments
                expr = Sum(
                    (
                        binomial(p, i)
                        * binomial(q, j)
                        * binomial(r, k)
                        * (-cx) ** (p - i)
                        * (-cy) ** (q - j)
                        * (-cz) ** (r - k)
                        * M[i, j, k]
                    ),
                    (i, 0, p),
                    (j, 0, q),
                    (k, 0, r),
                ).doit()

                # substitute back in c_x, c_y, c_z
                expr = expr.subs(M[1, 0, 0] / M[0, 0, 0], c_x)
                expr = expr.subs(M[0, 1, 0] / M[0, 0, 0], c_y)
                expr = expr.subs(M[0, 0, 1] / M[0, 0, 0], c_z)

                # print python code for generation of the central moment

                python_code = f"moments_central[{p}, {q}, {r}] = {pycode(expr)}\n"
                # replace symbol names with corresponding python variable names
                python_code = python_code.replace('c_', 'c')
                python_code = python_code.replace('M[', 'm[')
                print(python_code)
