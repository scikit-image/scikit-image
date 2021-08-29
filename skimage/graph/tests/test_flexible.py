import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy import fft
import skimage.graph.mcp as mcp
from skimage import data, img_as_float

from skimage._shared.testing import assert_array_equal

a = np.ones((8, 8), dtype=np.float32)
a[1::2] *= 2.0

src = img_as_float(data.camera()[128:256, 128:256])
src = src + 0.05 * np.random.standard_normal(src.shape)
target = np.roll(src, (15, -10), axis=(0, 1))
target = target + 0.05 * np.random.standard_normal(target.shape)
src_freq = fft.fftn(src)
target_freq = fft.fftn(target)

# current implementation
image_product1 = src_freq * target_freq.conj()
cross_correlation1 = fft.ifftn(image_product1)

# fixed implementation
image_product = image_product1 / np.abs(image_product1)
cross_correlation = fft.ifftn(image_product)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(np.abs(cross_correlation1), cmap=plt.cm.gray)
axes[0].set_title('Existing Implementation')
axes[1].imshow(np.abs(cross_correlation), cmap=plt.cm.gray)
axes[1].set_title('Proposed Implementation')
for ax in axes:
    ax.set_axis_off()
plt.tight_layout()
plt.show()


class FlexibleMCP(mcp.MCP_Flexible):
    """ Simple MCP subclass that allows the front to travel 
    a certain distance from the seed point, and uses a constant
    cost factor that is independent of the cost array.
    """
    
    def _reset(self):
        mcp.MCP_Flexible._reset(self)
        self._distance = np.zeros((8, 8), dtype=np.float32).ravel()
    
    def goal_reached(self, index, cumcost):
        if self._distance[index] > 4:
            return 2
        else:
            return 0
    
    def travel_cost(self, index, new_index, offset_length):
        return 1.0  # fixed cost
    
    def examine_neighbor(self, index, new_index, offset_length):
        pass  # We do not test this
        
    def update_node(self, index, new_index, offset_length):
        self._distance[new_index] = self._distance[index] + 1


def test_flexible():
    # Create MCP and do a traceback
    mcp = FlexibleMCP(a)
    costs, traceback = mcp.find_costs([(0, 0)])
    
    # Check that inner part is correct. This basically
    # tests whether travel_cost works.
    assert_array_equal(costs[:4, :4], [[1, 2, 3, 4],
                                       [2, 2, 3, 4],
                                       [3, 3, 3, 4],
                                       [4, 4, 4, 4]])
    
    # Test that the algorithm stopped at the right distance.
    # Note that some of the costs are filled in but not yet frozen,
    # so we take a bit of margin
    assert np.all(costs[-2:, :] == np.inf)
    assert np.all(costs[:, -2:] == np.inf)
