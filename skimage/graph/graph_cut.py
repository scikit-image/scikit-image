try:
    import networkx as nx
except ImportError:
    import warnings
    warnings.warn('"cut_threshold" requires networkx')
import numpy as np
import _ncut
import _ncut_cy
from scipy.sparse import linalg
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence as ANC
from scipy.sparse.linalg.eigen.arpack.arpack import ArpackError as APE

def cut_threshold(labels, rag, thresh):
    """Combine regions seperated by weight less than threshold.

    Given an image's labels and its RAG, output new labels by
    combining regions whose nodes are seperated by a weight less
    than the given threshold.

    Parameters
    ----------
    labels : ndarray
        The array of labels.
    rag : RAG
        The region adjacency graph.
    thresh : float
        The threshold. Regions connected by edges with smaller weights are
        combined.

    Returns
    -------
    out : ndarray
        The new labelled array.

    Examples
    --------
    >>> from skimage import data, graph, segmentation
    >>> img = data.lena()
    >>> labels = segmentation.slic(img)
    >>> rag = graph.rag_mean_color(img, labels)
    >>> new_labels = graph.cut_threshold(labels, rag, 10)

    References
    ----------
    .. [1] Alain Tremeau and Philippe Colantoni
           "Regions Adjacency Graph Applied To Color Image Segmentation"
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.11.5274

    """
    # Because deleting edges while iterating through them produces an error.
    to_remove = [(x, y) for x, y, d in rag.edges_iter(data=True)
                 if d['weight'] >= thresh]
    rag.remove_edges_from(to_remove)

    comps = nx.connected_components(rag)

    # We construct an array which can map old labels to the new ones.
    # All the labels within a connected component are assigned to a single
    # label in the output.
    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(comps):
        for node in nodes:
            for label in rag.node[node]['labels']:
                map_array[label] = i

    return map_array[labels]


def cut_n(labels, rag, thresh):

    _ncut_relabel(rag,thresh)
    
    from_ = range(labels.max()+1)
    to = [ rag.node[x]['ncut label']  for x in from_ ]
    map_array = np.array(to)
    
    return map_array[labels]

def _ncut_relabel(rag, cut_thresh = 0.0001):
    d, w = _ncut.DW_matrix(rag)
    error = False

    try:
        m = w.shape[0]
        vals,vectors = linalg.eigsh(d-w,M=d,which='SM',k = min(100,m-2))
    except ANC as e:
        vals = e.eigenvalues
        vectors = e.eigenvectors
        if len(vals) == 0:
            error = True
    except ValueError:
        error = True
    except APE:
        error = True
    
    if not error :
        vals,vectors = np.real(vals), np.real(vectors)
        index2 = _ncut_cy.argmin2(vals)

        ev = np.real(vectors[:,index2])
        ev = _ncut.norml(ev)

        mcut = np.inf
        thresh = None
        for t in np.arange(0,1,0.1):
            mask = ev > t
            cost = _ncut.ncut_cost(mask,d,w)
            if cost < mcut :
                mcut = cost
                thresh = t

        if ( mcut < cut_thresh ):
            mask = ev > thresh

            nodes1 = [ n for i,n in enumerate(rag.nodes()) if mask[i]]
            nodes2 = [ n for i,n in enumerate(rag.nodes()) if not mask[i]]

            sub1 = rag.subgraph(nodes1)
            sub2 = rag.subgraph(nodes2)

            _ncut_relabel(sub1,cut_thresh)
            _ncut_relabel(sub2, cut_thresh)
            return

    node = rag.nodes()[0]
    new_label = rag.node[node]['labels'][0]
    for n in rag.nodes():
        rag.node[n]['ncut label'] = new_label

    
