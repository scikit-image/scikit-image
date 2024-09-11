# scikit-image: Image processing in Python

[![Image.sc forum](https://img.shields.io/badge/dynamic/json.svg?label=forum&url=https%3A%2F%2Fforum.image.sc%2Ftags%2Fscikit-image.json&query=%24.topic_list.tags.0.topic_count&colorB=brightgreen&suffix=%20topics&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAYAAAAfSC3RAAABPklEQVR42m3SyyqFURTA8Y2BER0TDyExZ+aSPIKUlPIITFzKeQWXwhBlQrmFgUzMMFLKZeguBu5y+//17dP3nc5vuPdee6299gohUYYaDGOyyACq4JmQVoFujOMR77hNfOAGM+hBOQqB9TjHD36xhAa04RCuuXeKOvwHVWIKL9jCK2bRiV284QgL8MwEjAneeo9VNOEaBhzALGtoRy02cIcWhE34jj5YxgW+E5Z4iTPkMYpPLCNY3hdOYEfNbKYdmNngZ1jyEzw7h7AIb3fRTQ95OAZ6yQpGYHMMtOTgouktYwxuXsHgWLLl+4x++Kx1FJrjLTagA77bTPvYgw1rRqY56e+w7GNYsqX6JfPwi7aR+Y5SA+BXtKIRfkfJAYgj14tpOF6+I46c4/cAM3UhM3JxyKsxiOIhH0IO6SH/A1Kb1WBeUjbkAAAAAElFTkSuQmCC)](https://forum.image.sc/tags/scikit-image)
[![Contributor forum](https://img.shields.io/badge/Scientific_Python-Contributor_forum-brightgreen?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAMAAAAoLQ9TAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAHgUExURQAAAAD/AEC/QFygOFygOFOYNFygOFygOFyhOF2iLlygN1ygOFygOFyfOFmmM1ygOFmbN1ygOF2gO2WmT4C6mp3O5ZzM4YO7n53N5J3N5J3N5J3N5J3N5J3N5J3N5J3N5JXE1H2vsSNkOSRlO3+xtJ7L3p3N5H2wspm/vsHX026YfIqrkvP29Nzo5MHX1Jm/vX2ws////////////9Le1f///////xFUIZy3pP///6zDsh5dLW2Xd0p9Vh9eLhtbKglOGRpbKSJgMEyAWFicNlugOEGGLVKXNFOXNFygOFSYNEWKLjl/KS51JU+TMlWZNVyfOFCVM0qPMDd9KB5lHkKHLUWKLzuBKlGVM1CUMzh+KT+EKzB1JU6TMjR6JzuAKjJ4Jzd8KClvIz2DK12hOnOwdFidNjF3JRphHBlgHBFZGCFoH1SZNZ3N5JbI0V2hO1meNxBXF1KWNFygOXGvcH23j5zM4JXIzozCuInAsIW8oytvNzh5UZTHzJbI0JLGyDZ2Wjt6YJrL4JvL4dLf1uHp4+ju6e/z8Pz9/P////P29LzPwKS+qpKwmX+jiGmTc0B2TRdZJzZvRGmUc8rZzsXVyABIEQRLFfv8/Nnk3J25pDlxRmCNa0h8VCllNwZMFvbtL40AAABEdFJOUwAAAUDN97eKWQMusP4jBacG08vBWBnx8hiNjNnY+vnx8/L29vHy8MPw8ez59efx8MNmBtLRI+XzLBrQ0StbtfP+87FZrUzRuAAAAAFiS0dEMdnbHXIAAAAHdElNRQfoCQsVDSxxSaUKAAAA+ElEQVQY0y2PVVdCURQGPxMbu7tbwe5Gt9iiYne77e7WY+ex5a+6Lt55mTWPAwCwtbN3cNQ4QcXZxbW6xlBb56Zx97C2luqNxobGpmYiLQBPr5bWtnYydXR2mc3eAGx8unt6+/oHBoeGR0bHfAE///EJw+TU9MzsHBEFBCKI5xeIFpdMtLyyurYejBDmjc2t7Z3dvf0DZg5FGFs5PPp3OCKY+TgyKjomNu6EmeORkJiUnHJ6dn5xKVLT0jMyoRPi6vrm9u7+4fHp+UXooM8SKq9SvmXrgZxcpd4/Pr+kzMtXZgoKv39+LVJKWVSs/paUlpVbKiqrlPgDvJVMjOyXg/EAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjQtMDktMTFUMjE6MTM6MzErMDA6MDCZCUdBAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI0LTA5LTExVDIxOjEzOjMxKzAwOjAw6FT//QAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNC0wOS0xMVQyMToxMzo0NCswMDowMOe8+JwAAAAASUVORK5CYII=)](https://discuss.scientific-python.org/c/contributor/skimage)
[![Project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://skimage.zulipchat.com)

[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![SPEC 1 — Lazy Loading of Submodules and Functions](https://img.shields.io/badge/SPEC-1-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0001/)
[![SPEC 4 — Using and Creating Nightly Wheels](https://img.shields.io/badge/SPEC-4-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0004/)
[![SPEC 6 — Keys to the Castle](https://img.shields.io/badge/SPEC-6-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0006/)
[![SPEC 7 — Seeding pseudo-random number generation](https://img.shields.io/badge/SPEC-7-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0007/)
[![SPEC 8 — Securing the Release Process](https://img.shields.io/badge/SPEC-8-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0008/)

- **Website (including documentation):** [https://scikit-image.org/](https://scikit-image.org)
- **Documentation:** [https://scikit-image.org/docs/stable/](https://scikit-image.org/docs/stable/)
- **User forum:** [https://forum.image.sc/tag/scikit-image](https://forum.image.sc/tag/scikit-image)
- **Developer forum:** [https://discuss.scientific-python.org/c/contributor/skimage](https://discuss.scientific-python.org/c/contributor/skimage)
- **Source:** [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image)

## Installation

- **pip:** `pip install scikit-image`
- **conda:** `conda install -c conda-forge scikit-image`

Also see [installing `scikit-image`](https://github.com/scikit-image/scikit-image/blob/main/INSTALL.rst).

## License

See [LICENSE.txt](https://github.com/scikit-image/scikit-image/blob/main/LICENSE.txt).

## Citation

If you find this project useful, please cite:

> Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
> François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle
> Gouillart, Tony Yu, and the scikit-image contributors.
> _scikit-image: Image processing in Python_. PeerJ 2:e453 (2014)
> https://doi.org/10.7717/peerj.453

(See "Cite this repository" in sidebar for BiBTeX.)
