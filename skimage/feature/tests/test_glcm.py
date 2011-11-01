import numpy as np
from skimage.feature import glcm

class TestGLCM():
    def test_output_angles(self):
        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)        
        result = glcm(image, [1], [0, np.pi/2], 4)
        assert result.shape == (4, 4, 1, 2)
        expected1 = np.array([[2, 2, 1, 0],
                             [0, 2, 0, 0],
                             [0, 0, 3, 1],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected1)
        expected2 = np.array([[3, 0, 2, 0],
                             [0, 2, 2, 0],
                             [0, 0, 1, 2],
                             [0, 0, 0, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 1], expected2)        
    
    def test_output_symmetric_1(self):
        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)        
        result = glcm(image, [1], [np.pi/2], 4, symmetric=True)
        assert result.shape == (4, 4, 1, 1)
        expected = np.array([[6, 0, 2, 0],
                             [0, 4, 2, 0],
                             [2, 2, 2, 2],
                             [0, 0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)    

    def test_result_symmetric_2(self):
        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)        
        result = glcm(image, [1], [0], 4, symmetric=True)[:, :, 0, 0]
        np.testing.assert_array_equal(result, result.transpose())

    def test_output_distance(self):
        image = np.array([[0, 0, 0, 0],
                          [1, 0, 0, 1],
                          [2, 0, 0, 2],
                          [3, 0, 0, 3]], dtype=np.uint8)        
        result = glcm(image, [3], [0], 4, symmetric=False)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)           

    def test_output_combo(self):
        image = np.array([[0],
                          [1],
                          [2],
                          [3]], dtype=np.uint8)
        result = glcm(image, [1, 2], [0, np.pi/2], 4)
        assert result.shape == (4, 4, 2, 2)
        
        z = np.zeros((4, 4), dtype=np.uint32)
        e1 = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]], dtype=np.uint32)
        e2 = np.array([[0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]], dtype=np.uint32)
        
        np.testing.assert_array_equal(result[:, :, 0, 0], z)
        np.testing.assert_array_equal(result[:, :, 1, 0], z)
        np.testing.assert_array_equal(result[:, :, 0, 1], e1)
        np.testing.assert_array_equal(result[:, :, 1, 1], e2)

    def test_normed(self):
        image = np.array([[0, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 2, 2, 2],
                          [2, 2, 3, 3]], dtype=np.uint8)        
        result = glcm(image, [1], [0], 4, normal=True)
        np.testing.assert_almost_equal(result.sum(), 1.0)
    
    
if __name__ == '__main__':
    np.testing.run_module_suite()
    