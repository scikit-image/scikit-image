from skimage import graph, data, io, segmentation, color
from matplotlib import pyplot as plt


img = data.coffee()

labels = segmentation.slic(img, compactness=30, n_segments=400)

g = graph.rag_mean_color(img, labels)

out = graph.rag.rag_draw(labels, g, img)
plt.figure()
plt.title("RAG with all edges shown in green.")
plt.imshow(out)

out = graph.rag.rag_draw(labels, g, img, high_color=(1,0,0), thresh=30)
plt.figure()
plt.title("RAG with edge weights less than 30, color mapped between green and red.")
plt.imshow(out)

plt.show()
