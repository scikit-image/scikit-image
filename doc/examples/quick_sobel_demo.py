"""
Quick Sobel edge detection demo
--------------------------------

This example shows how to load a sample image,
apply the Sobel filter, and save the result.
"""

from skimage import data, filters, io

# Load sample image
image = data.coins()

# Apply Sobel edge detection
edges = filters.sobel(image)

# Save output
io.imsave("coins_edges.png", edges)

print("Saved coins_edges.png")
