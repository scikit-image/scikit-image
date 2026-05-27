#!/usr/bin/env bash

# Build libomp for MacOS
# Used in pyproject.toml [tool.cibuildwheel.macos]
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
build_dir=${DIR}/libs_build
LLVM_VERSION=${LLVM_VERSION:-19.1.1}
INSTALL_CMD=${INSTALL_CMD:-"sudo make install"}

echo MACOSX_DEPLOYMENT_TARGET $MACOSX_DEPLOYMENT_TARGET

git clone --depth 1 --branch llvmorg-${LLVM_VERSION} https://github.com/llvm/llvm-project
pushd llvm-project/openmp
mkdir build
cd build
cmake -DCMAKE_C_COMPILER=${CC:-clang} -DCMAKE_CXX_COMPILER=${CXX:-clang++} ${CMAKE_FLAGS:-} ..
make ${MAKE_FLAGS:-}
${INSTALL_CMD}

popd
rm -rf llvm-project
