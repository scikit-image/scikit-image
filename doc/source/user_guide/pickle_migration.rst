.. _pickle-migration:

Pickling across scikit-image 0.25 and 0.26+
===========================================

Pickles store import paths, not live objects. After the internal megamove that
places implementations under ``_skimage2`` while keeping the public ``skimage``
namespace, pickles created with scikit-image 0.25.x may fail to load unless the
stable public paths are restored.

Loading older pickles
---------------------

For pickles created with scikit-image 0.25.x (or other releases that stored
``skimage.*`` paths), use the compatibility helpers exported from the top-level
``skimage`` namespace:

.. code-block:: python

    import skimage as ski

    with open("region.pkl", "rb") as f:
        region = ski.pickle_load(f)

    blob = open("region.pkl", "rb").read()
    region = ski.pickle_loads(blob)

These helpers resolve registered public paths to the current ``_skimage2``
implementations. When a shim module still re-exports the registered type,
standard :func:`pickle.loads` also works.

Creating new pickles
--------------------

Registered types are pickled with stable ``skimage.*`` paths instead of
``_skimage2.*``. You can use the standard library:

.. code-block:: python

    import pickle
    from skimage.measure import regionprops

    region = regionprops(label_image)[0]
    blob = pickle.dumps(region)

For explicit control over the compatibility pickler, use:

.. code-block:: python

    import skimage as ski

    blob = ski.pickle_dumps(region)

Currently registered types include ``RegionProperties``, geometric transform
classes, ``ThinPlateSplineTransform``, and ``ImageCollection`` /
``MultiImage``. Run ``python tools/audit_pickle_surface.py`` in the source tree
to list the registered paths maintained in :mod:`skimage._pickle_compat`.

Limitations
-----------

- Pickles referencing removed or renamed API may still fail and need a version-
  specific alias added to the registry.
- ``joblib`` and ``cloudpickle`` may bypass the registry; report failures
  separately if you rely on those serializers.
- Only curated public types are registered. Private symbols and most functions
  are out of scope unless a real unpickling failure is reported.

Maintainers
-----------

When adding a stateful public class that users may pickle, register it in
:mod:`skimage._pickle_compat`, re-export it from the shim module if needed, and
extend ``tests/skimage/test_pickle_compat.py``. Use
``tools/audit_pickle_surface.py --strict`` in CI to catch missing registrations.
