"""Create an API table for a given Python package."""

import logging
import importlib
import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path

import click


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class Entry:
    source_path: str
    discovery_paths: list
    obj: object

    @property
    def is_public(self):
        return len(self.public_discovery_paths) > 0

    @property
    def sorted_discovery_paths(self):
        paths = sorted(self.discovery_paths, key=lambda x: x.count("."))
        return paths

    @property
    def public_discovery_paths(self):
        paths = [path for path in self.sorted_discovery_paths if "._" not in path]
        if len(paths) > 1:
            logger.warning(
                "object at %r with more than 1 public discovery path: %r",
                self.source_path,
                paths,
            )
        return paths

    @property
    def type_name(self):
        if inspect.isclass(self.obj):
            return "class"
        else:
            return type(self.obj).__name__


def _source_path(obj):
    obj_module = inspect.getmodule(obj)
    if obj_module is None:
        return None
    path = f"{obj_module.__name__}."
    try:
        path += obj.__qualname__
    except AttributeError:
        try:
            path += obj.__name__
        except AttributeError:
            return None
    return path


def visit_api(table, *, discovery_path, obj, parent=None):
    obj_module = inspect.getmodule(obj)

    source_path = _source_path(obj)
    if source_path is None:
        source_path = _source_path(parent)
        if source_path is None:
            logger.info(
                "stopping API discovery at %r, cannot determine path to source",
                discovery_path,
            )
            return
        source_path += discovery_path.split(".")[-1]

    if source_path in table:
        logger.info(
            "stopping API discovery at %r, already know %r", discovery_path, source_path
        )
        table[source_path].discovery_paths.append(discovery_path)
        return  # Already visited, avoids endless recursion

    table[source_path] = Entry(
        source_path=source_path, discovery_paths=[discovery_path], obj=obj
    )

    # Skip member discovery if module or source_path is unknown
    if obj_module is None or source_path is None:
        return

    for member_name, member in inspect.getmembers(obj):
        member_discovery_path = f"{discovery_path}.{member_name}"

        if inspect.iscode(member):
            logger.debug(
                "stopping API discovery at %r, is code object", member_discovery_path
            )
            continue  # Skip code objects

        member_module = inspect.getmodule(member)
        if member_module is None:
            if inspect.isbuiltin(member) or inspect.ismethodwrapper(member):
                logger.debug(
                    "stopping API discovery at %r, builtin or method wrapper %r",
                    member_discovery_path,
                    member,
                )
                continue  # Silently method wrappers

        elif obj_module.__name__ not in member_module.__name__:
            logger.debug(
                "stopping API discovery at %r, in external package %r",
                member_discovery_path,
                member_module.__name__,
            )
            continue  # Skip members defined outside inspected package

        visit_api(table, discovery_path=member_discovery_path, obj=member, parent=obj)


def print_public_api(table):
    entries = [entry for entry in table.values() if entry.is_public]
    for entry in sorted(entries, key=lambda e: e.public_discovery_paths[0]):
        print(entry.sorted_discovery_paths[0], f"({entry.type_name})")

    print(f"{len(entries)} discovered objects in public API")


def print_full_api(table, blacklist=None):
    if blacklist is None:
        blacklist = []

    entries = table
    for black in blacklist:
        entries = {k: v for k, v in entries.items() if black not in k}

    for entry in sorted(entries.values(), key=lambda e: e.sorted_discovery_paths[0]):
        print(entry.sorted_discovery_paths[0], f"({entry.type_name})")

    print(f"{len(table)} discovered objects in total")


@click.command()
@click.argument("module")
def main(module):
    """Create an API table for a given Python package."""
    log_file = Path(tempfile.gettempdir()) / (Path(__file__).name + ".log")
    print(f"logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
    )

    table = {}
    api_root = importlib.import_module(module)
    visit_api(table, obj=api_root, discovery_path=module)

    print_full_api(table)


if __name__ == "__main__":
    main()
