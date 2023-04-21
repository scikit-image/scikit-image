"""Create an API table for a given Python package."""

import logging
import importlib
import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

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


def _unwrap(obj):
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    return obj


def _is_discover_loop(parents, node):
    for a, b in combinations(parents + (node,), 2):
        if a is b:
            return True
    return False


def visit_api(table, *, discovery_path, node, _parents=(), _rec_depth=0):
    if _is_discover_loop(_parents, node):
        logger.warning(
            "stopping API discovery at %r, detected discover loop", discovery_path
        )
        return

    obj_module = inspect.getmodule(node)

    source_path = _source_path(node)
    if source_path is None:
        source_path = _source_path(_parents[-1])
        if source_path is None:
            logger.warning(
                "stopping API discovery at %r, cannot determine path to source",
                discovery_path,
            )
            return
        # Fallback to parent's source path + member name (last part of discovery_path)
        name = discovery_path.split(".")[-1]
        source_path = f"{source_path}.{name}"

    if source_path in table:
        table[source_path].discovery_paths.append(discovery_path)
    else:
        table[source_path] = Entry(
            source_path=source_path, discovery_paths=[discovery_path], obj=node
        )

    if obj_module is None or source_path is None:
        return  # Skip member discovery if module or source_path is unknown

    for member_name, member in inspect.getmembers(node):
        member_discovery_path = f"{discovery_path}.{member_name}"

        if inspect.iscode(member):
            logger.debug(
                "stopping API discovery at %r, is code object", member_discovery_path
            )
            continue

        member_module = inspect.getmodule(member)
        if member_module is None:
            if inspect.isbuiltin(member) or inspect.ismethodwrapper(member):
                logger.debug(
                    "stopping API discovery at %r, builtin or method wrapper %r",
                    member_discovery_path,
                    member,
                )
                continue

        elif obj_module.__name__ not in member_module.__name__:
            logger.debug(
                "stopping API discovery at %r, in external package %r",
                member_discovery_path,
                member_module.__name__,
            )
            continue

        visit_api(
            table,
            discovery_path=member_discovery_path,
            node=member,
            _parents=_parents + (node,),
            _rec_depth=_rec_depth + 1,
        )


def print_table(table: dict[str, Entry]):
    for key in sorted(table.keys()):
        entry = table[key]
        print(f"\n{entry.source_path} ({entry.type_name})")
        for path in entry.sorted_discovery_paths:
            print("    " + path)
    print(f"\n{len(table)} discovered objects in total")


def print_public_api(table: dict[str, Entry]):
    print()
    public = {k: v for k, v in table.items() if v.is_public}
    for entry in sorted(public.values(), key=lambda e: e.public_discovery_paths[0]):
        print(f"{entry.public_discovery_paths[0]} ({entry.type_name})")
        for path in entry.public_discovery_paths[1:]:
            print("    " + path)
    print(f"\n{len(public)} discovered objects in total")


@click.command()
@click.argument("module")
@click.option("--private", is_flag=True)
def main(module, private):
    """Create an API table for a given Python package."""
    log_file = Path(tempfile.gettempdir()) / (Path(__file__).name + ".log")
    print(f"logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
    )

    table: dict[str, Entry] = {}
    api_root = importlib.import_module(module)
    visit_api(table, node=api_root, discovery_path=module)

    if private:
        print_table(table)
    else:
        print_public_api(table)


if __name__ == "__main__":
    main()
