from numpy.testing import *
import numpy as np
import scikits.image.filter as F
from scikits.image.backend import use_backend
import scikits

class BackendTester():
    def test_all_backends(self):
        for backend in scikits.image.backends.list:
            if backend == "default": 
                continue
            for function_name in dir(self):
                if function_name.startswith("test") and function_name != "test_all_backends":
                    yield (getattr(self, function_name), backend)
                    
class TestSobel(BackendTester):
    def test_00_00_zeros(self, backend=None):
        """Sobel on an array of all zeros"""
        result = F.sobel(np.zeros((10, 10), dtype=np.float32), backend=backend)
        assert (np.all(result == 0))
    
    def test_01_01_horizontal(self, backend=None):
        """Sobel on an edge should be a horizontal line"""
        i, j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(np.float32)
        result = F.sobel(image, backend=backend)
        assert (np.all(result[np.abs(i) > 1] == 0))
    
    def test_01_02_vertical(self, backend=None):
        """Sobel on a vertical edge should be a vertical line"""
        i,j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(np.float32)
        result = F.sobel(image, backend=backend)
        assert (np.all(result[np.abs(j) > 1] == 0))


class TestHSobel():        
    def _test_00_00_zeros(self):
        """Horizontal sobel on an array of all zeros"""
        result = F.sobel(np.zeros((10, 10)), axis=0)
        assert (np.all(result == 0))

    def _test_01_01_horizontal(self):
        """Horizontal Sobel on an edge should be a horizontal line"""
        i,j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.sobel(image, axis=0)
        assert (np.all(result[np.abs(i) > 1] == 0))
    
    def _test_01_02_vertical(self):
        """Horizontal Sobel on a vertical edge should be zero"""
        i,j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.sobel(image, axis=0)
        assert (np.all(result == 0))   


class TestVSobel():
    def _test_00_00_zeros(self):
        """Vertical sobel on an array of all zeros"""
        result = F.sobel(np.zeros((10, 10)), axis=1)
        assert (np.all(result == 0))
    
    def test_01_01_vertical(self):
        """Vertical Sobel on an edge should be a vertical line"""
        i,j = np.mgrid[-5:6, -5:6]
        image = (j >= 0).astype(float)
        result = F.sobel(image, axis=1)
        # Fudge the eroded points
        assert (np.all(result[np.abs(j) > 1] == 0))
    
    def test_01_02_horizontal(self):
        """Vertical Sobel on a horizontal edge should be zero"""
        i,j = np.mgrid[-5:6, -5:6]
        image = (i >= 0).astype(float)
        result = F.sobel(image, axis=1)
        eps = .000001
        assert (np.all(np.abs(result) < eps))       


#class TestPrewitt():
#    def test_00_00_zeros(self):
#        """Prewitt on an array of all zeros"""
#        result = F.prewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
#        assert (np.all(result == 0))
#    
#    def test_00_01_mask(self):
#        """Prewitt on a masked array should be zero"""
#        np.random.seed(0)
#        result = F.prewitt(np.random.uniform(size=(10, 10)), 
#                          np.zeros((10, 10), bool))
#        eps = .000001
#        assert (np.all(np.abs(result) < eps))

#    def test_01_01_horizontal(self):
#        """Prewitt on an edge should be a horizontal line"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (i >= 0).astype(float)
#        result = F.prewitt(image)
#        # Fudge the eroded points
#        i[np.abs(j) == 5] = 10000
#        eps = .000001
#        assert (np.all(result[i == 0] == 1))
#        assert (np.all(np.abs(result[np.abs(i) > 1]) < eps))
#    
#    def test_01_02_vertical(self):
#        """Prewitt on a vertical edge should be a vertical line"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (j >= 0).astype(float)
#        result = F.prewitt(image)
#        eps = .000001
#        j[np.abs(i)==5] = 10000
#        assert (np.all(result[j == 0] == 1))
#        assert (np.all(np.abs(result[np.abs(j) > 1]) < eps))

#class TestHPrewitt():
#    def test_00_00_zeros(self):
#        """Horizontal sobel on an array of all zeros"""
#        result = F.hprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
#        assert (np.all(result == 0))
#    
#    def test_00_01_mask(self):
#        """Horizontal prewitt on a masked array should be zero"""
#        np.random.seed(0)
#        result = F.hprewitt(np.random.uniform(size=(10, 10)), 
#                          np.zeros((10, 10), bool))
#        eps = .000001
#        assert (np.all(np.abs(result) < eps))

#    def test_01_01_horizontal(self):
#        """Horizontal prewitt on an edge should be a horizontal line"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (i >= 0).astype(float)
#        result = F.hprewitt(image)
#        # Fudge the eroded points
#        i[np.abs(j) == 5] = 10000
#        eps = .000001
#        assert (np.all(result[i == 0] == 1))
#        assert (np.all(np.abs(result[np.abs(i) > 1]) < eps))
#    
#    def test_01_02_vertical(self):
#        """Horizontal prewitt on a vertical edge should be zero"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (j >= 0).astype(float)
#        result = F.hprewitt(image)
#        eps = .000001
#        assert (np.all(np.abs(result) < eps))

#class TestVPrewitt():
#    def test_00_00_zeros(self):
#        """Vertical prewitt on an array of all zeros"""
#        result = F.vprewitt(np.zeros((10, 10)), np.ones((10, 10), bool))
#        assert (np.all(result == 0))
#    
#    def test_00_01_mask(self):
#        """Vertical prewitt on a masked array should be zero"""
#        np.random.seed(0)
#        result = F.vprewitt(np.random.uniform(size=(10, 10)), 
#                          np.zeros((10, 10), bool))
#        assert (np.all(result == 0))

#    def test_01_01_vertical(self):
#        """Vertical prewitt on an edge should be a vertical line"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (j >= 0).astype(float)
#        result = F.vprewitt(image)
#        # Fudge the eroded points
#        j[np.abs(i) == 5] = 10000
#        assert (np.all(result[j == 0] == 1))
#        eps = .000001
#        assert (np.all(np.abs(result[np.abs(j) > 1]) < eps))
#    
#    def test_01_02_horizontal(self):
#        """Vertical prewitt on a horizontal edge should be zero"""
#        i,j = np.mgrid[-5:6, -5:6]
#        image = (i >= 0).astype(float)
#        result = F.vprewitt(image)
#        eps = .000001
#        assert (np.all(np.abs(result) < eps))
        
        
if __name__ == "__main__":
    run_module_suite()
