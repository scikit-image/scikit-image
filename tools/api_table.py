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

    def __str__(self):
        return f"{self.sorted_discovery_paths[0]} ({self.type_name})"

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
        return type(self.obj).__name__


def visit_api(table, *, discovery_path, obj):
    obj_module = inspect.getmodule(obj)

    source_path = f"{obj_module.__name__}."
    try:
        source_path += obj.__qualname__
    except AttributeError:
        try:
            source_path += obj.__name__
        except AttributeError:
            logger.warning(
                "stopping API discovery at %r, cannot determine source path, "
                "no __qualname__ or __name__ attributes",
                discovery_path,
            )
            pass

    if source_path in table:
        logger.info(
            "stopping API discovery at %r, already know %r", discovery_path, source_path
        )
        table[source_path].discovery_paths.append(discovery_path)
        return  # Already visited, avoids endless recursion

    table[source_path] = Entry(
        source_path=source_path, discovery_paths=[discovery_path], obj=obj
    )

    for member_name, member in inspect.getmembers(obj):
        member_discovery_path = f"{discovery_path}.{member_name}"

        if inspect.iscode(member):
            logger.debug(
                "stopping API discovery at %r, is code object", member_discovery_path
            )
            continue  # Skip code objects

        member_module = inspect.getmodule(member)
        if member_module is None:
            # logger.debug(
            #     "stopping API discovery at %r, cannot find its module, "
            #     "probably a builtin",
            #     member_discovery_path,
            # )
            continue  # Skip built-ins
        if obj_module.__name__ not in member_module.__name__:
            logger.debug(
                "stopping API discovery at %r, in external package %r",
                member_discovery_path,
                member_module.__name__,
            )
            continue  # Skip members defined outside inspected package

        visit_api(table, obj=member, discovery_path=member_discovery_path)


def print_public_api(table):
    public_entries = [entry for entry in table.values() if entry.is_public]

    for entry in sorted(public_entries, key=lambda e: e.public_discovery_paths[0]):
        print(entry)

    print(f"{len(public_entries)} objects in public API")


@click.command()
@click.argument("module")
def main(module):
    """Create an API table for a given Python package."""
    log_file = Path(tempfile.gettempdir()) / (Path(__file__).name + ".log")
    print(f"logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.DEBUG,
    )

    table = {}
    api_root = importlib.import_module(module)
    visit_api(table, obj=api_root, discovery_path=module)

    print_public_api(table)


if __name__ == "__main__":
    main()
