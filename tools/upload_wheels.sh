#!/bin/bash
# Script to download / check and upload scikit_image wheels for release
if [ "`which twine`" == "" ]; then
    echo "twine not on path; need to pip install twine?"
    exit 1
fi
if [ "`which wheel-uploader`" == "" ]; then
    echo "wheel-uploader not on path; see https://github.com/MacPython/terryfy"
    exit 1
fi
SK_VERSION=`git describe --tags`
if [ "${SK_VERSION:0:1}" != 'v' ]; then
    echo "scikit image version $SK_VERSION does not start with 'v'"
    exit 1
fi
echo "Trying download / upload of version ${SK_VERSION:1}"
wheel-uploader -v scikit_image "${SK_VERSION:1}"
wheel-uploader -v scikit_image -t manylinux1 "${SK_VERSION:1}"
wheel-uploader -v scikit_image -t win "${SK_VERSION:1}"
