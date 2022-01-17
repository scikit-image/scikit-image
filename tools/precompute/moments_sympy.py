"""
This script uses SymPy to generate the analytical equations used to transform
raw moments into central moments in skimage/measure/_moments_analytical.py
"""

from sympy import symbols, binomial, Sum
from sympy import IndexedBase, Idx
from sympy import init_printing, pprint
from sympy.printing.pycode import pycode

# Define a moments matrix, m
m = IndexedBase('m')

ndim = 3
if ndim == 2:
    # symbols for the centroid componets
    cx, cy, cz = symbols('cx cy')
    # indices into the moments matrix, m
    i, j = symbols('i j', cls=Idx)

    # centroid
    cx = m[1, 0] / m[0, 0]
    cy = m[0, 1] / m[0, 0]

    # loop over all moments with order <= `order`.
    order = 3
    print(f"Generating expressions for {ndim}D moments with order <= {order}")
    for p in range(0, order + 1):
        for q in range(0, order + 1):
            if p + q > order:
                continue
            # print(f"order={p + q}: p={p}, q={q}")

            # This expression is given at
            # https://en.wikipedia.org/wiki/Image_moment#Central_moments
            expr = Sum(
                (
                    binomial(p, i) * binomial(q, j) *
                    (-cx)**(p - i) * (-cy)**(q - j) *
                    m[i, j]
                ),
                (i, 0, p), (j, 0, q)
            ).doit()

            # substitute back in the cx and cy symbols
            expr = expr.subs(m[1, 0]/m[0, 0], cx)
            expr = expr.subs(m[0, 1]/m[0, 0], cy)

            # print python code for generation of the central moment
            print(f"m[{p}, {q}] = {pycode(expr)}")
            print("\n")

elif ndim == 3:

    # symbols for the centroid componets
    cx, cy, cz = symbols('cx cy cz')
    # indices into the moments matrix, m
    i, j, k = symbols('i j k', cls=Idx)

    # centroid
    cx = m[1, 0, 0] / m[0, 0, 0]
    cy = m[0, 1, 0] / m[0, 0, 0]
    cz = m[0, 0, 1] / m[0, 0, 0]

    # loop over all moments with order <= `order`.
    order = 3

    print(f"Generating expressions for {ndim}D moments with order <= {order}")
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
                        binomial(p, i) * binomial(q, j) * binomial(r, k) *
                        (-cx)**(p - i) * (-cy)**(q - j) * (-cz)**(r - k) *
                        m[i, j, k]
                    ),
                    (i, 0, p), (j, 0, q), (k, 0, r)
                ).doit()

                # substitute back in cx, cy, cz
                expr = expr.subs(m[1, 0, 0]/m[0, 0, 0], cx)
                expr = expr.subs(m[0, 1, 0]/m[0, 0, 0], cy)
                expr = expr.subs(m[0, 0, 1]/m[0, 0, 0], cz)

                # print python code for generation of the central moment
                print(f"m[{p}, {q}, {r}] = {pycode(expr)}")
                print("\n")
