# scikit-image: Image processing in Python
[![Stackoverflow](https://img.shields.io/badge/stackoverflow-Ask%20questions-blue.svg)](https://stackoverflow.com/questions/tagged/scikit-image)
[![Gitter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/scikit-image/scikit-image?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
[![codecov.io](https://codecov.io/github/scikit-image/scikit-image/coverage.svg?branch=master)](https://codecov.io/github/scikit-image/scikit-image?branch=master)

- **Website (including documentation):** [http://scikit-image.org/](http://scikit-image.org)
- **Mailing list:** [https://mail.python.org/mailman/listinfo/scikit-image](https://mail.python.org/mailman/listinfo/scikit-image)
- **Source:** [https://github.com/scikit-image/scikit-image](https://github.com/scikit-image/scikit-image)

## Installation from binaries

- **Debian/Ubuntu:** ``sudo apt-get install python-skimage``
- **OSX:** ``pip install scikit-image``
- **Anaconda:** ``conda install -c conda-forge scikit-image``
- **Windows:** Download [Windows binaries](http://www.lfd.uci.edu/~gohlke/pythonlibs/#scikit-image)

Also see [installing ``scikit-image``](INSTALL.rst).

## Installation from source

Install dependencies using:

```
pip install -r requirements.txt
```

Then, install scikit-image using:

```
$ pip install .
```

If you plan to develop the package, you may run it directly from source:

```
$ pip install -e .  # Do this once to add package to Python path
```

Every time you modify Cython files, also run:

```
$ python setup.py build_ext -i  # Build binary extensions
```

## License (Modified BSD)

Copyright (C) 2011, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

## Citation

If you find this project useful, please cite:

> Stéfan van der Walt, Johannes L. Schönberger, Juan Nunez-Iglesias,
> François Boulogne, Joshua D. Warner, Neil Yager, Emmanuelle
> Gouillart, Tony Yu, and the scikit-image contributors.
> *scikit-image: Image processing in Python*. PeerJ 2:e453 (2014)
> http://dx.doi.org/10.7717/peerj.453
