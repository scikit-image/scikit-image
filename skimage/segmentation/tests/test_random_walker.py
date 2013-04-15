import numpy as np
from skimage.segmentation import random_walker


def make_2d_syntheticdata(lx, ly=None):
    if ly is None:
        ly = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly)) + 0.1 * np.random.randn(lx, ly)
    small_l = int(lx / 5)
    data[lx / 2 - small_l:lx / 2 + small_l,
         ly / 2 - small_l:ly / 2 + small_l] = 1
    data[lx / 2 - small_l + 1:lx / 2 + small_l - 1,
         ly / 2 - small_l + 1:ly / 2 + small_l - 1] = \
                        0.1 * np.random.randn(2 * small_l - 2, 2 * small_l - 2)
    data[lx / 2 - small_l, ly / 2 - small_l / 8:ly / 2 + small_l / 8] = 0
    seeds = np.zeros_like(data)
    seeds[lx / 5, ly / 5] = 1
    seeds[lx / 2 + small_l / 4, ly / 2 - small_l / 4] = 2
    return data, seeds


def make_3d_syntheticdata(lx, ly=None, lz=None):
    if ly is None:
        ly = lx
    if lz is None:
        lz = lx
    np.random.seed(1234)
    data = np.zeros((lx, ly, lz)) + 0.1 * np.random.randn(lx, ly, lz)
    small_l = int(lx / 5)
    data[lx / 2 - small_l:lx / 2 + small_l,
         ly / 2 - small_l:ly / 2 + small_l,
         lz / 2 - small_l:lz / 2 + small_l] = 1
    data[lx / 2 - small_l + 1:lx / 2 + small_l - 1,
         ly / 2 - small_l + 1:ly / 2 + small_l - 1,
         lz / 2 - small_l + 1:lz / 2 + small_l - 1] = 0
    # make a hole
    hole_size = np.max([1, small_l / 8])
    data[lx / 2 - small_l,
         ly / 2 - hole_size:ly / 2 + hole_size,
         lz / 2 - hole_size:lz / 2 + hole_size] = 0
    seeds = np.zeros_like(data)
    seeds[lx / 5, ly / 5, lz / 5] = 1
    seeds[lx / 2 + small_l / 4, ly / 2 - small_l / 4, lz / 2 - small_l / 4] = 2
    return data, seeds


def test_2d_bf():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_bf = random_walker(data, labels, beta=90, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob_bf = random_walker(data, labels, beta=90, mode='bf',
                                 return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    # Now test with more than two labels
    labels[55, 80] = 3
    full_prob_bf = random_walker(data, labels, beta=90, mode='bf',
                                 return_full_prob=True)
    assert (full_prob_bf[1, 25:45, 40:60] >=
            full_prob_bf[0, 25:45, 40:60]).all()
    assert len(full_prob_bf) == 3
    assert data.shape == labels.shape


def test_2d_cg():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_cg = random_walker(data, labels, beta=90, mode='cg')
    assert (labels_cg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob = random_walker(data, labels, beta=90, mode='cg',
                              return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg


def test_2d_cg_mg():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels_cg_mg = random_walker(data, labels, beta=90, mode='cg_mg')
    assert (labels_cg_mg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    full_prob = random_walker(data, labels, beta=90, mode='cg_mg',
                              return_full_prob=True)
    assert (full_prob[1, 25:45, 40:60] >=
            full_prob[0, 25:45, 40:60]).all()
    assert data.shape == labels.shape
    return data, labels_cg_mg


def test_types():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = 255 * (data - data.min()) / (data.max() - data.min())
    data = data.astype(np.uint8)
    labels_cg_mg = random_walker(data, labels, beta=90, mode='cg_mg')
    assert (labels_cg_mg[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels_cg_mg


def test_reorder_labels():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels[labels == 2] = 4
    labels_bf = random_walker(data, labels, beta=90, mode='bf')
    assert (labels_bf[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels_bf


def test_2d_inactive():
    lx = 70
    ly = 100
    data, labels = make_2d_syntheticdata(lx, ly)
    labels[10:20, 10:20] = -1
    labels[46:50, 33:38] = -2
    labels = random_walker(data, labels, beta=90)
    assert (labels.reshape((lx, ly))[25:45, 40:60] == 2).all()
    assert data.shape == labels.shape
    return data, labels


def test_3d():
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    labels = random_walker(data, labels, mode='cg')
    assert (labels.reshape(data.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data.shape == labels.shape
    return data, labels


def test_3d_inactive():
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    old_labels = np.copy(labels)
    labels[5:25, 26:29, 26:29] = -1
    after_labels = np.copy(labels)
    labels = random_walker(data, labels, mode='cg')
    assert (labels.reshape(data.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data.shape == labels.shape
    return data, labels, old_labels, after_labels


def test_multispectral_2d():
    lx, ly = 70, 100
    data, labels = make_2d_syntheticdata(lx, ly)
    data = data[..., np.newaxis].repeat(2, axis=-1)  # Expect identical output
    multi_labels = random_walker(data, labels, mode='cg', multichannel=True)
    assert data[..., 0].shape == labels.shape
    single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[25:45, 40:60] == 2).all()
    assert data[..., 0].shape == labels.shape
    return data, multi_labels, single_labels, labels


def test_multispectral_3d():
    n = 30
    lx, ly, lz = n, n, n
    data, labels = make_3d_syntheticdata(lx, ly, lz)
    data = data[..., np.newaxis].repeat(2, axis=-1)  # Expect identical output
    multi_labels = random_walker(data, labels, mode='cg', multichannel=True)
    assert data[..., 0].shape == labels.shape
    single_labels = random_walker(data[..., 0], labels, mode='cg')
    assert (multi_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert (single_labels.reshape(labels.shape)[13:17, 13:17, 13:17] == 2).all()
    assert data[..., 0].shape == labels.shape
    return data, multi_labels, single_labels, labels


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
