#!/usr/bin/env python3
"""Download every one of scikit-image's CDN-hosted example datasets.

For offline/air-gapped builds or Linux distribution packaging: pre-populates
a local cache of every data file that scikit-image's public ``skimage.data``
functions can fetch, verifying each file's sha256 hash against the registry
in ``src/_skimage2/data/_registry.py``.

Depends only on ``pooch`` -- scikit-image itself does not need to be
installed to run this script (the registry is parsed directly out of its
source file, not imported).

Note: this does not download files used only internally by scikit-image's
test suite (files with no entry in ``registry_urls``); those are fetched
directly from the scikit-image GitHub repository by the test runner and are
not needed to use the public ``skimage.data`` API.
"""

import argparse
import ast
import concurrent.futures
import sys
from pathlib import Path

import pooch

_HERE = Path(__file__).resolve().parent
_REGISTRY_PATH = _HERE.parent / 'src' / '_skimage2' / 'data' / '_registry.py'


def _load_registry():
    """Parse ``registry``/``registry_urls`` out of ``_registry.py``.

    Only evaluates simple top-level ``name = <expr>`` assignments (building
    up a namespace of just those names as it goes, e.g. so an f-string like
    ``registry_urls``'s values can reference an already-assigned URL
    constant) -- any other statement (imports, function/class defs, etc.) is
    skipped rather than executed, so this keeps working even if the registry
    module gains code unrelated to the two dicts this script needs.
    """
    source = _REGISTRY_PATH.read_text()
    tree = ast.parse(source, filename=str(_REGISTRY_PATH))

    namespace = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        expr = ast.fix_missing_locations(ast.Expression(node.value))
        code = compile(expr, str(_REGISTRY_PATH), 'eval')
        namespace[target.id] = eval(code, {'__builtins__': {}}, namespace)  # noqa: S307

    return namespace['registry'], namespace['registry_urls']


def _download_one(data_filename, url, expected_hash, dest_dir, force):
    dest_path = dest_dir / data_filename
    if force:
        dest_path.unlink(missing_ok=True)
    already_correct = dest_path.exists() and pooch.file_hash(dest_path) == expected_hash

    try:
        pooch.retrieve(
            url,
            known_hash=f'sha256:{expected_hash}',
            fname=data_filename,
            path=dest_dir,
            progressbar=False,
        )
    except Exception as err:  # noqa: BLE001 -- pooch raises a mix of exception types
        return data_filename, 'error', str(err)
    return data_filename, 'cached' if already_correct else 'downloaded', None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dest',
        required=True,
        type=Path,
        help='Directory to download data files into '
        '(mirrors the layout skimage.data expects).',
    )
    parser.add_argument(
        '--jobs', type=int, default=8, help='Number of parallel downloads (default: 8).'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download files even if a correctly-hashed copy already exists.',
    )
    args = parser.parse_args(argv)

    registry, registry_urls = _load_registry()
    skipped = sorted(set(registry) - set(registry_urls))
    missing_hash = sorted(set(registry_urls) - set(registry))
    if missing_hash:
        print(
            f'WARNING: {len(missing_hash)} registry_urls entr'
            f'{"y" if len(missing_hash) == 1 else "ies"} with no matching hash '
            f'in registry, skipping: {", ".join(missing_hash)}',
            file=sys.stderr,
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    # Pre-create every subdirectory the registry needs up front: pooch's own
    # directory creation isn't safe against concurrent calls for the same
    # path, so leaving this to the parallel downloads below races and
    # intermittently raises FileExistsError.
    for key in registry_urls:
        if key in registry:
            (args.dest / key).parent.mkdir(parents=True, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [
            pool.submit(_download_one, key, url, registry[key], args.dest, args.force)
            for key, url in registry_urls.items()
            if key in registry
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    downloaded = [r for r in results if r[1] == 'downloaded']
    cached = [r for r in results if r[1] == 'cached']
    errors = [r for r in results if r[1] == 'error']

    print(
        f'{len(downloaded)} downloaded, {len(cached)} already cached, {len(errors)} failed'
    )
    if skipped:
        print(
            f'{len(skipped)} test-only fixtures skipped (not part of the public '
            'skimage.data API; fetched by the test suite from GitHub directly)'
        )
    for key, _status, err in errors:
        print(f'  FAILED {key}: {err}', file=sys.stderr)

    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
