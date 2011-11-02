import numpy as np
from skimage.feature import glcm, compute_glcm_prop

class TestGLCM():
    def setup(self):
        self.image = np.array([[0, 0, 1, 1],
                               [0, 0, 1, 1],
                               [0, 2, 2, 2],
                               [2, 2, 3, 3]], dtype=np.uint8)   
        
    def test_output_angles(self):
        result = glcm(self.image, [1], [0, np.pi/2], 4)
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
        result = glcm(self.image, [1], [np.pi/2], 4, symmetric=True)
        assert result.shape == (4, 4, 1, 1)
        expected = np.array([[6, 0, 2, 0],
                             [0, 4, 2, 0],
                             [2, 2, 2, 2],
                             [0, 0, 2, 0]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)    

    def test_result_symmetric_2(self):    
        result = glcm(self.image, [1], [0], 4, symmetric=True)[:, :, 0, 0]
        np.testing.assert_array_equal(result, result.transpose())

    def test_output_distance(self):
        im = np.array([[0, 0, 0, 0],
                       [1, 0, 0, 1],
                       [2, 0, 0, 2],
                       [3, 0, 0, 3]], dtype=np.uint8)          
        result = glcm(im, [3], [0], 4, symmetric=False)
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]], dtype=np.uint32)
        np.testing.assert_array_equal(result[:, :, 0, 0], expected)

    def test_output_combo(self):
        im = np.array([[0],
                       [1],
                       [2],
                       [3]], dtype=np.uint8)
        result = glcm(im, [1, 2], [0, np.pi/2], 4)
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

    def test_output_empty(self):
        result = glcm(self.image, [10], [0], 4)
        np.testing.assert_array_equal(result[:, :, 0, 0], 
                                      np.zeros((4, 4), dtype=np.uint32))  
        result = glcm(self.image, [10], [0], 4, normed=True)
        np.testing.assert_array_equal(result[:, :, 0, 0], 
                                      np.zeros((4, 4), dtype=np.uint32))      

    def test_normed(self):     
        result = glcm(self.image, [1, 2, 3], [0, np.pi/2, np.pi], 4, 
                      normed=True)
        for d in range(result.shape[2]):
            for a in range(result.shape[3]):
                np.testing.assert_almost_equal(result[:, :, d, a].sum(), 
                                               1.0)
    
    def test_contrast(self):
        result = glcm(self.image, [1], [0], 4, 
                      normed=True, symmetric=True)
        result = np.round(result, 3)
        contrast = compute_glcm_prop(result, 'contrast')
        np.testing.assert_almost_equal(contrast[0, 0], 0.586)
    
    def test_dissimilarity(self):
        result = glcm(self.image, [1], [0], 4, 
                      normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = compute_glcm_prop(result, 'dissimilarity')
        np.testing.assert_almost_equal(dissimilarity[0, 0], 0.418)

    def test_dissimilarity_2(self):
        result = glcm(self.image, [1], [np.pi/2], 4, 
                      normed=True, symmetric=True)
        result = np.round(result, 3)
        dissimilarity = compute_glcm_prop(result, 'dissimilarity')[0, 0]
        np.testing.assert_almost_equal(dissimilarity, 0.664)

    def test_invalid_property(self):
        result = glcm(self.image, [1], [0], 4)
        np.testing.assert_raises(ValueError, compute_glcm_prop, 
                                 result, 'ABC')
    
    def test_homogeneity(self):
        result = glcm(self.image, [1], [0], 4, normed=True, symmetric=True)
        homogeneity = compute_glcm_prop(result, 'homogeneity')[0, 0]
        np.testing.assert_almost_equal(homogeneity, 0.80833333)

    def test_energy(self):
        result = glcm(self.image, [1], [0], 4, normed=True, symmetric=True)
        energy = compute_glcm_prop(result, 'energy')[0, 0]
        np.testing.assert_almost_equal(energy, 0.38188131)
    
    def test_correlation(self):
        result = glcm(self.image, [1], [0], 4, normed=True, symmetric=True)
        energy = compute_glcm_prop(result, 'correlation')[0, 0]
        np.testing.assert_almost_equal(energy, 0.71953255)
    
    def test_uniform_properties(self):
        im = np.ones((4, 4), dtype=np.uint8)
        result = glcm(im, [1, 2], [0, np.pi/2], 4, normed=True, 
                      symmetric=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 
                     'energy', 'correlation']:
            compute_glcm_prop(result, prop)

if __name__ == '__main__':
    np.testing.run_module_suite()
    