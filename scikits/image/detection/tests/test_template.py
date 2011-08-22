import os.path
import numpy as np
from numpy.testing import *
from scikits.image import data_dir
from scikits.image.detection import *
from numpy.random import randn

def test_template():
    size = 100
    image = np.zeros((400, 400), dtype=np.float32)
    target = np.tri(size) + np.tri(size)[::-1]
    target = target.astype(np.float32)
    target_positions = [(50, 50), (200, 200)]
    for x, y in target_positions:
        image[x:x+size, y:y+size] = target
    image += randn(400, 400)*2
    
    for method in ["norm-corr", "norm-coeff"]:
        result = match_template(image, target, method=method)
        delta = 5
        found_positions = []
        # find the targets
        for i in range(50):
            index = np.argmax(result)        
            y, x = np.unravel_index(index, result.shape)
            if not found_positions:
                found_positions.append((x, y))
            for position in found_positions:
                distance = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
                if distance > delta:
                    found_positions.append((x, y))
            result[y, x] = 0
            if len(found_positions) == len(target_positions):
                break

        for x, y in target_positions:
            print x, y
            found = False
            for position in found_positions:
                distance = np.sqrt((x - position[0]) ** 2 + (y - position[1]) ** 2)
                if distance < delta:
                    found = True
            assert found
    
if __name__ == "__main__":
    run_module_suite()
