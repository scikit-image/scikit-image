import os


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]

    return requires


dependencies = {
    dep: parse_requirements_file("requirements/" + dep + ".txt")
    for dep in ["build", "data", "default", "developer", "docs", "optional", "test"]
}

def list_deps(target):
    return f"""{newline.join(f'    "{dep}",' for dep in dependencies[target])}"""

filename = "pyproject.toml"
newline = "\n"
project = """[project]
name = "scikit-image"
license = {file = "LICENSE.txt"}
description = "Image processing in Python"
maintainers = [
    {name = "scikit-image developers", email="skimage-core@discuss.scientific-python.org"},
]
requires-python = ">=3.8"
readme = "README.md"
classifiers = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: C',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3 :: Only',
    "Topic :: Software Development :: Libraries",
    'Topic :: Scientific/Engineering',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
]
dynamic = ['version']
"""
project_dependencies = f"""dependencies = [
    # TODO: update to "pin-compatible" once possible, see
    # https://github.com/FFY00/meson-python/issues/29
{list_deps("default")}
]
"""

project_optional_dependencies = f"""
[project.optional-dependencies]
data = [
{list_deps("data")}
]
developer = [
{list_deps("developer")}
]
docs = [
{list_deps("docs")}
]
optional = [
{list_deps("optional")}
]
test = [
{list_deps("test")}
]
"""

project_footer = """
[project.urls]
homepage = "https://scikit-image.org"
documentation = "https://scikit-image.org/docs/stable"
source = "https://github.com/scikit-image/scikit-image"
download = "https://pypi.org/project/scikit-image/#files"
tracker = "https://github.com/scikit-image/scikit-image/issues"
"""

build = """
[build-system]
build-backend = "mesonpy"
requires = [
    "meson-python @ git+https://github.com/mesonbuild/meson-python@babdf709618fbeeaf59634bc2b4331ca5f7924ce",
    # `wheel` is needed for non-isolated builds, given that `meson-python`
    # doesn't list it as a runtime requirement (at least in 0.10.0)
    # See https://github.com/FFY00/meson-python/blob/main/pyproject.toml#L4
    "wheel",
    "setuptools<=65.5",
    "packaging",
    "Cython>=0.29.24",
    "pythran",
    "lazy_loader>=0.1",

    # We follow scipy for much of these pinnings
    # https://github.com/scipy/scipy/blob/main/pyproject.toml

    # On Windows we need to avoid 1.21.6, 1.22.0 and 1.22.1 because they were
    # built with vc142. 1.22.3 is the first version that has 32-bit Windows
    # wheels *and* was built with vc141. So use that:
    "numpy==1.22.3; python_version=='3.10' and platform_system=='Windows' and platform_python_implementation != 'PyPy'",

    # default numpy requirements
    "numpy==1.21.1; python_version=='3.8' and platform_python_implementation != 'PyPy'",
    "numpy==1.21.1; python_version=='3.9' and platform_python_implementation != 'PyPy'",
    "numpy==1.21.6; python_version=='3.10' and platform_system != 'Windows' and platform_python_implementation != 'PyPy'",
    "numpy==1.23.3; python_version=='3.11' and platform_python_implementation != 'PyPy'",

    # For Python versions which aren't yet officially supported,
    # we specify an unpinned NumPy which allows source distributions
    # to be used and allows wheels to be used as soon as they
    # become available.
    "numpy; python_version>='3.12'",
    "numpy; python_version>='3.8' and platform_python_implementation=='PyPy'",
]

[tool.devpy]
package = 'skimage'

[tool.devpy.commands]
"Build" = ["devpy.build", "devpy.test"]
"Environments" = ["devpy.shell", "devpy.ipython", "devpy.python"]
"Documentation" = [".devpy/cmds.py:docs"]
"Metrics" = [".devpy/cmds.py:asv", ".devpy/cmds.py:coverage"]
"""

# os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, "w") as file:
    file.write(project)
    file.write(project_dependencies)
    file.write(project_optional_dependencies)
    file.write(project_footer)
    file.write(build)
