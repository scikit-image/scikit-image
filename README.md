## Loading TIFF Stacks

Use `scikit-image` to load multi-page TIFF files for microgravity experiments:

```python
from skimage import io

stack = io.imread('microgravity_stack.tif')
print(stack.shape)
```
