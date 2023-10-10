"""
Utilities for graph-based image segmentation, object detection and 
network analysis.
This module provides functions for graph-based analysis of images and segmentations,
e.g. path finding and cost evaluation, merging and cutting of graphs, or finding
the closest centrality.

"""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
