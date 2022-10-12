# On building with Meson

```
pip install meson ninja cython pythran
```

```
meson build --prefix=$PWD/build
```

```
ninja -C build
```

```
meson install -C build
```

The `skimage/morphology/skeletonize_3d.pyx.in`
file needs to be built into a pyx file, with its
own build command, using Tempita (see Scipy).

That pyx file will appear in the _build_ directory, and can be built from there.

If that file had to import local `*.pyx` files (it
does not) then the dependencies would need to make
sure that the relevant pyx files are already
copied into the build directory by defining build dependencies (see `_cython_tree` in the Scipy meson build files).
