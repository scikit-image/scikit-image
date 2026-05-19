#!/usr/bin/env python3

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path


PORTED_SDIR = '_skimage2'
UNPORTED_SDIR = 'skimage'

MESON_TEMPLATE = '''\
python_sources = [
{mod_list}
]

py3.install_sources(
  python_sources,
  pure: false,          # Will be installed next to binaries
  subdir: '{mod_path}'  # Folder relative to site-packages to install to
)
'''


def _parts_past_src(path):
    parts = path.parts
    src_parts_i = [i for i, p in enumerate(parts) if p == 'src']
    if not src_parts_i:
        raise ValueError(f'No `src` string in parts: {path}')
    return parts[src_parts_i[-1] + 1 :]


def fname2mod_name(path):
    return '.'.join(_parts_past_src(path.with_suffix('')))


def shadow_imports(input_dir, output_dir):
    directories = set()
    for mod_fname in input_dir.glob('**/*.py*'):
        rel_in = mod_fname.relative_to(input_dir)
        mod_in = fname2mod_name(mod_fname)
        in_suffix = rel_in.suffix
        out_fname = output_dir / rel_in.with_suffix(
            '.py' if in_suffix == '.pyx' else in_suffix
        )
        parent = out_fname.parent
        directories.add(parent)
        if not parent.exists():
            parent.mkdir(parents=True)
        if mod_fname.stem == '__init__':
            out_fname.write_text(mod_fname.read_text())
        else:
            out_fname.write_text(f'''\
from {mod_in} import *
from {mod_in} import __doc__  # noqa: F401
''')
    for parent in directories:
        mod_list = '  ' + '\n  '.join(
            f"  '{str(p.name)}'," for p in parent.glob('*.py*')
        )
        mod_path = '/'.join(_parts_past_src(parent))
        meson_build = MESON_TEMPLATE.format(mod_list=mod_list, mod_path=mod_path)
        (parent / 'meson.build').write_text(meson_build)


def get_parser():
    parser = ArgumentParser(
        description=__doc__,  # Usage from docstring
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('--output-dir', help='output directory')
    return parser


def args2dirs(args):
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise ValueError(f'Input directory {str(input_dir)} is not a directory')
    if args.output_dir is None:
        parts = list(input_dir.parts)
        if PORTED_SDIR not in parts:
            raise ValueError(f'`{PORTED_SDIR}` not in path parts')
        output_dir = Path(*[UNPORTED_SDIR if p == PORTED_SDIR else p for p in parts])
    else:
        output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, output_dir


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_dir, output_dir = args2dirs(args)
    # Might consider:
    # * Doing test copy / git src move
    # * Modifying test imports.
    # * Modifying meson.build files for moved sources.
    # * Adding copied source module to top-level meson.build files and
    #   __init__.pyi
    # * Replacing `skimage` imports in docstrings.
    shadow_imports(input_dir, output_dir)


if __name__ == '__main__':
    main()
