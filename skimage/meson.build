# Platform detection
is_windows = host_machine.system() == 'windows'
is_mingw = is_windows and cc.get_id() == 'gcc'

cython_c_args = []
if is_windows
  # For mingw-w64, link statically against the UCRT.
  gcc_link_args = ['-lucrt', '-static']
  if is_mingw
    add_project_link_arguments(gcc_link_args, language: ['c', 'cpp'])
    # Force gcc to float64 long doubles for compatibility with MSVC
    # builds, for C only.
    add_project_arguments('-mlong-double-64', language: 'c')
    # Make fprintf("%zd") work (see https://github.com/rgommers/scipy/issues/118)
    add_project_arguments('-D__USE_MINGW_ANSI_STDIO=1', language: ['c', 'cpp'])
    # Manual add of MS_WIN64 macro when not using MSVC.
    # https://bugs.python.org/issue28267
    bitness = run_command(
      '_build_utils/gcc_build_bitness.py',
      check: true
    ).stdout().strip()
    if bitness == '64'
      add_project_arguments('-DMS_WIN64', language: ['c', 'cpp'])
    endif
    # Silence warnings emitted by PyOS_snprintf for (%zd), see
    # https://github.com/rgommers/scipy/issues/118.
    # Use as c_args for extensions containing Cython code
    cython_c_args += ['-Wno-format-extra-args', '-Wno-format']
  endif
endif

# When cross-compiling skimage, the compiler needs access to NumPy and Pythran
# headers for the host platform (where the package will actually run). These
# headers may be incompatible with any corresponding headers that might be
# installed on the build system (where the compilation is performed). To make
# sure that the compiler finds the right headers, paths can be configured in
# the 'properties' section of a Meson cross file:
#
#   [properties]
#   numpy-include-dir = '/path/to/host/numpy/includes'
#   pythran-include-dir = '/path/to/host/pythran/includes'
#
# If a cross file is not provided or does not specify either of these
# properties, fall back to running Python on the build system to query NumPy or
# Pythran directly for the appropriate paths. This will detect appropriate
# paths for native builds. (This might even work for certain build/host cross
# combinations, but don't rely on that.)
#
# For more information about cross compilation in Meson, including a definition
# of "build" and "host" in this context, refer to
#
#     https://mesonbuild.com/Cross-compilation.html

# NumPy include directory
incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
if incdir_numpy == 'not-given'
  # If not specified, try to query NumPy from the build python
  incdir_numpy = run_command(py3,
    [
      '-c',
      'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
    ],
    check: true
  ).stdout().strip()
endif

# Don't use the deprecated NumPy C API. Define this to a fixed version instead of
# NPY_API_VERSION in order not to break compilation for released SciPy versions
# when NumPy introduces a new deprecation. Use in a meson.build file::
#
#   py3.extension_module('_name',
#     'source_fname',
#     numpy_nodepr_api)
#
numpy_nodepr_api = '-DNPY_NO_DEPRECATED_API=NPY_1_23_API_VERSION'

np_dep = declare_dependency(
  include_directories: incdir_numpy,
  compile_args: numpy_nodepr_api
)

cc = meson.get_compiler('c')

# Pythran include directory
incdir_pythran = meson.get_external_property('pythran-include-dir', 'not-given')
if incdir_pythran == 'not-given'
  # If not specified, try to query Pythran from the build python
  incdir_pythran = run_command(py3,
    [
      '-c',
      'import os; os.chdir(".."); import pythran; print(os.path.dirname(pythran.__file__));'
    ],
    check: true
  ).stdout().strip()
endif

inc_pythran = include_directories(incdir_pythran)

# Pythran build flags
cpp_args_pythran = [
  '-DENABLE_PYTHON_MODULE',
  '-D__PYTHRAN__=3',
  '-DPYTHRAN_BLAS_NONE'
]

# Deal with M_PI & friends; add `use_math_defines` to c_args
# Cython doesn't always get this correctly itself
# explicitly add the define as a compiler flag for Cython-generated code.
if is_windows
  use_math_defines = ['-D_USE_MATH_DEFINES']
else
  use_math_defines = []
endif

python_sources = [
  '__init__.py',
  '__init__.pyi',
  'conftest.py',
  'py.typed'
]

py3.install_sources(
  python_sources,
  pure: false,
  subdir: 'skimage'
)

cython_cli = find_program('_build_utils/cythoner.py')
cy = meson.get_compiler('cython')

cython_args = ['@INPUT@', '@OUTPUT@']
if cy.version().version_compare('>=3.1.0')
  cython_args += ['-Xfreethreading_compatible=True']
endif

cython_cpp_args = cython_args + ['--cplus']

cython_gen = generator(cython_cli,
  arguments : cython_args,
  output : '@BASENAME@.c')

cython_gen_cpp = generator(cython_cli,
  arguments : ['@INPUT@', '@OUTPUT@', '--cplus'],
  output : '@BASENAME@.cpp')

c_undefined_ok = ['-Wno-maybe-uninitialized']

# Suppress warning for deprecated Numpy API.
# (Suppress warning messages emitted by #warning directives).
# Replace with numpy_nodepr_api after Cython 3.0 is out
_cpp_Wno_cpp = cpp.get_supported_arguments('-Wno-cpp')
cython_c_args += [_cpp_Wno_cpp, use_math_defines]
cython_cpp_args = cython_c_args

subdir('_shared')
subdir('_vendored')
subdir('color')
subdir('data')
subdir('draw')
subdir('exposure')
subdir('feature')
subdir('filters')
subdir('future')
subdir('graph')
subdir('io')
subdir('measure')
subdir('metrics')
subdir('morphology')
subdir('registration')
subdir('restoration')
subdir('segmentation')
subdir('transform')
subdir('util')
