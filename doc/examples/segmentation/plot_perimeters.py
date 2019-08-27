"""
=========================
Different perimeters
=========================

This example shows the uncertainty in evaluating perimeters.
Take a square, rotate it, and evaluate the perimeters.

"""
from skimage.measure import perimeter
from skimage.measure import crofton_perimeter
from skimage.transform import rotate
import matplotlib.pyplot as plt
import numpy as np


square = np.zeros((100, 100));
square [40:60, 40:60] = 1;

for n in [4,6]:        
    p =[];
    angles = range(90);
    for i in angles:

        rotated_square=rotate(square, i, order=0); # nearest neighbor
        p.append(perimeter(rotated_square, n));
    plt.plot(angles, p);

for d in [2,4]:
    p =[];
    angles = range(90);
    for i in angles:

        rotated_square=rotate(square, i, order=0); # nearest neighbor
        p.append(crofton_perimeter(rotated_square, n));
    plt.plot(angles, p);
        
plt.xlabel('Rotation angle');
plt.ylabel('Perimeter of the rotated square')
plt.legend(['N4 perimeter', 'N8 perimeter', 'Crofton 2 directions', 'Crofton 4 directions'])
plt.show();