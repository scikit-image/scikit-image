"""Inspect API of a given package or submodule.

This script is able to iterate the member tree of a given Python module. The discovered
information can be exported into two CSV files (--csv option): "<module>_tree.csv"
collects general information on the discovered API tree, "<module>_params.csv" collects
information on the parameters and returns of discovered callables.
"""


import typing as ty
import logging
import tempfile
import argparse
import functools
import csv
from pathlib import Path

import griffe
import griffe.docstrings
from griffe.dataclasses import Object
from griffe.exceptions import AliasResolutionError


logger = logging.getLogger(__name__)


def tree_to_csv(path: Path, root_obj: Object):
    """Export general node information into a CSV file."""

    def to_table(obj: Object):
        row = {
            "path": obj.path,
            "name": obj.name,
            "depth": obj.path.count("."),
            "is_public": all(not x.startswith("_") for x in obj.path.split(".")),
            "type": obj.kind.name.lower() if not obj.is_alias else "alias",
            "alias_target": ""
        }
        if obj.is_alias:
            try:
                row["alias_target"] = obj.target_path
            except AliasResolutionError:
                row["alias_target"] = "extern"
            return [row]

        return [row]

    with path.open("w") as file:
        writer = None
        for obj in walk(root_obj):
            table = to_table(obj)
            if writer is None:
                writer = csv.DictWriter(file, table[0].keys())
                writer.writeheader()
            writer.writerows(table)


def params_to_csv(path: Path, root_obj: Object):
    """Export information on parameters and returns into a CSV file."""

    def to_table(obj: Object):
        from numpydoc.docscrape import NumpyDocString

        print(obj.path)
        docstring = NumpyDocString(obj.docstring.value if obj.docstring else "")

        table = []
        for position, param in enumerate(obj.parameters):
            for docstring_param in docstring["Parameters"]:
                if docstring_param.name == param.name:
                    ds_type = docstring_param.type
                    break
            else:
                ds_type = ""
            row = {
                "path": obj.path,
                "is_public": all(not x.startswith("_") for x in obj.path.split(".")),
                "kind": param.kind.name,
                "position": position,
                "name": param.name,
                "docstring_type": ds_type,
                "annotation": "",
                "default": "" if param.default is None else param.default,
            }
            table.append(row)

        # Extract returns
        for i, docstring_return in enumerate(docstring["Returns"]):
            return_spec = {
                "path": obj.path,
                "kind": "return",
                "position": i,
                "name": docstring_return.name,
                "docstring_type": docstring_return.type,
                "annotation": "",
                "default": "",
            }
            table.append(return_spec)

        return table

    with path.open("w") as file:
        writer = None
        for obj in walk(root_obj):
            if obj.is_alias:
                continue
            if not hasattr(obj, "parameters"):
                continue
            table = to_table(obj)
            if not table:
                continue
            if writer is None:
                writer = csv.DictWriter(file, table[0].keys())
                writer.writeheader()
            writer.writerows(table)


def cli(func: ty.Callable) -> ty.Callable:
    """Pass command line options to wrapped function if called without args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("module_name", metavar="module")
    parser.add_argument(
        "--csv", dest="export_csv", action="store_true", help="export results to CSV"
    )
    parser.add_argument(
        "--private", action="store_true", help="include private members"
    )
    parser.add_argument(
        "--max-depth", type=int, help="maximal recursion depth, no limit if not given"
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        default=0,
        action="count",
        help="increase verbosity of the script and log",
    )

    @functools.wraps(func)
    def _cli(*args, **kwargs):
        if not args and not kwargs:
            kwargs = vars(parser.parse_args())
        return func(*args, **kwargs)

    return _cli


def walk(obj: Object, _visited=None):
    if _visited is None:
        _visited = set()

    yield obj
    if obj in _visited:
        return
    else:
        _visited.add(obj)
    try:
        members = obj.members.values()
    except AliasResolutionError:
        members = []
    for member in members:
        yield from walk(member, _visited=_visited)


def format_griffe_obj(obj: Object):
    return f"{obj.path} <{obj.__class__.__name__}>"


@cli
def main(
    module_name: str, export_csv: bool, private: bool, max_depth: int | None, verbosity: int
):
    """Create an API table for a given Python package."""
    log_file = Path(tempfile.gettempdir()) / (Path(__file__).name + ".log")
    print(f"Logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO if verbosity < 1 else logging.DEBUG,
    )

    loader = griffe.loader.GriffeLoader()
    module = loader.load_module(module_name)
    unresolved, iterations = loader.resolve_aliases(implicit=True,
                                                    external=False)
    if unresolved:
        logger.info(
            f"{len(unresolved)} aliases were still unresolved after {iterations} iterations")
    else:
        logger.info(f"All aliases were resolved after {iterations} iterations")

    count = 0
    print("\nMember tree:")
    for obj in walk(module):
        print("  " * obj.path.count(".") + format_griffe_obj(obj))
        count += 1
    print(f"\nFound {count} objects in tree")

    if export_csv:
        tree_csv = Path.cwd() / f"{module_name}_tree.csv"
        params_csv = Path.cwd() / f"{module_name}_params.csv"
        tree_to_csv(tree_csv, module)
        params_to_csv(params_csv, module)
        print(f"\nCreated\n{tree_csv}\n{params_csv}")


if __name__ == "__main__":
    main()
