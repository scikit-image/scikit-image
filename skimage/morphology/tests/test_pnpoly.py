import numpy as np

from skimage.morphology._pnpoly import points_inside_poly

class test_poly():
    def test_square(self):
        v = np.array([[0, 0],
                      [0, 1],
                      [1, 1],
                      [1, 0]])
        assert(points_inside_poly([[0.5, 0.5]], v)[0])
        assert(not points_inside_poly([[-0.1, 0.1]], v)[0])

    def test_triangle(self):
        v = np.array([[0, 0],
                      [1, 0],
                      [0.5, 0.75]])
        assert(points_inside_poly([[0.5, 0.7]], v)[0])
        assert(not points_inside_poly([[0.5, 0.76]], v)[0])
        assert(not points_inside_poly([[0.7, 0.5]], v)[0])

    def test_type(self):
        assert(points_inside_poly([[0, 0]], [[0, 0]]).dtype == np.bool)

if __name__ == "__main__":
    np.testing.run_module_suite()
