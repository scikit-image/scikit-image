"""Inspect API of a given package or submodule.

This script is able to iterate the member tree of a given Python module. The discovered
information can be exported into two CSV files (--csv option): "<module>_tree.csv"
collects general information on the discovered API tree, "<module>_params.csv" collects
information on the parameters and returns of discovered callables.
"""


import typing as ty
import logging
import importlib
import inspect
import tempfile
import types
import argparse
import functools
import csv
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class Node:
    """An object in the discovered member tree at runtime.

    This object represents a node in the member tree as discovered by `MemberWalker`.

    Parameters
    ----------
    member_path : str
        The dotted path under which the node reachable from the root object by
        recursively following its members.
    obj : Any
        The object that's found at the node.
    module : types.ModuleType, optional
        The module in which the node is found. Identical with `obj` if it is a module.
    parent : ApiNode, optional
        The parent node of this one which is one step higher in the member tree.
    """

    member_path: str
    obj: ty.Any
    module: types.ModuleType | None
    parent: "Node | None"

    @property
    def nodes(self) -> "list[Node]":
        """Chain of nodes from the root node to this one."""
        nodes_ = []
        node = self
        while node is not None:
            nodes_.insert(0, node)
            node = node.parent
        return nodes_

    @property
    def parents(self) -> "list[Node]":
        """Chain of nodes from the root node to this ones parent."""
        return self.nodes[:-1]

    @property
    def name(self) -> str:
        return self.member_path.split(".")[-1]

    @property
    def source_path(self) -> str:
        """The path to the object's source."""
        if inspect.ismodule(self.obj):
            return self.module.__name__

        if self.module is not None:
            path = f"{self.module.__name__}."
            try:
                path += self.obj.__qualname__
            except AttributeError:
                path += getattr(self.obj, "__name__", self.name)
        else:
            path = f"{self.parent.source_path}.{self.name}"
        return path

    @property
    def signature(self) -> ty.Union[inspect.Signature, None]:
        """The signature of `obj` if it exists."""
        try:
            return inspect.signature(self.obj, follow_wrapped=False)
        except (ValueError, TypeError):
            return None

    @property
    def parameter_count(self) -> int:
        """Number of parameters in the signature.

        0 if `obj` has no signature.
        """
        signature = self.signature
        if signature:
            return len(signature.parameters)
        else:
            return 0

    @property
    def return_count(self) -> int:
        """Number of return values that are documented in the docstring.

        0 if none are documented in `obj`'s docstring.
        """
        from numpydoc.docscrape import get_doc_object

        docstring = get_doc_object(self.obj)
        return_spec = docstring["Returns"]
        return len(return_spec)

    @property
    def depth(self) -> int:
        """The depth of the node in the import tree."""
        return self.member_path.count(".")

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.member_path!r}>"

    def __str__(self) -> str:
        s = self.member_path
        if self.signature:
            s += str(self.signature)
        s += f" {{{type(self.obj).__name__}}}"
        return s

    def to_table(self) -> list[dict[str, ty.Any]]:
        """Export general node information in a table like format."""
        table = {
            "source_path": self.source_path,
            "member_path (unique)": self.member_path,
            "name": self.name,
            "import_depth": self.depth,
            "type": type(self.obj).__name__,
            "param_count": self.parameter_count,
            "return_count": self.return_count,
        }
        return [table]

    def to_param_table(self) -> list[dict[str, ty.Any]]:
        """Export information on parameters and returns in a table like format."""
        from numpydoc.docscrape import get_doc_object

        table = []
        docstring = get_doc_object(self.obj)

        def replace_empty_sig_value(x):
            return "" if x is inspect.Signature.empty else x

        # Extract parameters
        try:
            parameters = self.signature.parameters
        except AttributeError:
            parameters = {}

        for i, (name, param) in enumerate(parameters.items()):
            for docstring_param in docstring["Parameters"]:
                if docstring_param.name == name:
                    ds_type = docstring_param.type
                    break
            else:
                logger.warning(
                    "parameter %r is missing in docstring for %r",
                    name,
                    self.member_path,
                )
                ds_type = ""

            parameter_spec = {
                "source_path": self.source_path,
                "kind": f"parameter ({param.kind})",
                "position": i,
                "name": name,
                "docstring_type": ds_type,
                "annotation": replace_empty_sig_value(param.annotation),
                "default": replace_empty_sig_value(param.default),
            }
            table.append(parameter_spec)

        # Extract returns
        for i, docstring_return in enumerate(docstring["Returns"]):
            return_spec = {
                "source_path": self.source_path,
                "kind": "return",
                "position": i,
                "name": docstring_return.name,
                "docstring_type": docstring_return.type,
                "annotation": "",
                "default": "",
            }
            table.append(return_spec)

        return table


def _unwrap_callable(obj):
    """"""
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    return obj


def _has_duplicate(objects: ty.Iterable) -> bool:
    """Return True if any two objects in a sequence are identical."""
    for a, b in combinations(objects, 2):
        if a is b:
            return True
    return False


class MemberWalker:
    """Walk the member tree of a given Python object.

    This class implements a generator that follows members of a given root recursively
    until certain conditions are met. Each member is yielded as an `ApiNode`.

    Parameters
    ----------
    root_path : str

    root_obj : ty.Any
        The root object of the iterated tree.
    max_depth : int, optional
        The maximal depth to which the import tree is followed. E.g. 2 will include
        `foo.bar.baz` but ignore its members.

    Examples
    --------
    >>> import importlib
    >>> name = "argparse"
    >>> module = importlib.import_module(name)
    >>> walker = MemberWalker(root_path=name, root_obj=module)
    >>> iterable = iter(walker)
    >>> next(iterable)
    <ApiNode 'argparse'>
    >>> len(list(walker))
    535
    """

    ignored_types = (
        types.CodeType,
        types.BuiltinFunctionType,
        types.BuiltinMethodType,
        types.WrapperDescriptorType,
        types.MethodWrapperType,
        types.MethodDescriptorType,
        types.ClassMethodDescriptorType,
    )

    def __init__(self, root_path: str, root_obj: object, *, max_depth: int = None):
        node = Node(
            member_path=root_path,
            obj=root_obj,
            parent=None,
            module=inspect.getmodule(root_obj),
        )
        # Instance is readonly to ensure that the recursion doesn't set and rely on
        # instance attributes
        object.__setattr__(self, "_root_node", node)
        object.__setattr__(self, "max_depth", max_depth)

    def __iter__(self) -> ty.Generator:
        return self._discover(self._root_node)

    def __setattr__(self, key, value):
        # Make iterator readonly to ensure that the recursion doesn't set and rely on
        # instance attributes
        raise AttributeError(f"cannot assign to readonly attribute {key}")

    def _discover(self, node: Node, *, _rec_depth=0) -> ty.Generator:
        """Yield the given node, its members and recurse up to `max_depth`."""
        if self.max_depth is not None and self.max_depth < _rec_depth:
            logger.debug(
                "stopping API discovery, reached max depth %r at %r",
                self.max_depth,
                node.member_path,
            )
            return

        if self._stop_before_yield(node):
            return
        yield node
        if self._stop_before_members(node):
            return

        for member_name, member in inspect.getmembers(node.obj):
            if self._stop_before_recursion(node, member, member_name):
                continue
            sub_node = Node(
                member_path=f"{node.member_path}.{member_name}",
                obj=member,
                module=inspect.getmodule(member),
                parent=node,
            )
            yield from self._discover(
                sub_node,
                _rec_depth=_rec_depth + 1,
            )

    def _stop_before_yield(self, node: Node) -> bool:
        parent = node.parent

        if node.module is None:
            if parent and inspect.isfunction(parent.obj):
                logger.debug(
                    "stopping API discovery at %r, is function attribute",
                    node.member_path,
                )
                return True

        if (
            parent
            and node.module
            and parent.module
            and parent.module.__name__ not in node.module.__name__
        ):
            logger.debug(
                "stopping API discovery at %r, in external package %r",
                node.member_path,
                node.module.__name__,
            )
            return True

        if _has_duplicate(map(lambda n: n.obj, node.nodes)):
            logger.warning(
                "stopping API discovery at %r, detected discover loop", node.member_path
            )
            return True

        return False

    def _stop_before_members(self, node: Node) -> bool:
        if node.module is None:
            logger.debug(
                "stopping API discovery at %r, not inspecting attribute members",
                node.member_path,
            )
            return True
        return False

    def _stop_before_recursion(
        self, node: Node, member: object, member_name: str
    ) -> bool:
        if type(member) in self.ignored_types:
            # Stop silently for these ignored types
            return True

        return False


class PublicMemberWalker(MemberWalker):
    """Walk the public member tree of a given Python object.

    Similar to `ApiWalker` but stop at private members. Private members are those whose
    name is prefixed with at least one underscore.
    """

    def _stop_before_recursion(
        self, node: Node, member: object, member_name: str
    ) -> bool:
        if member_name.startswith("_"):
            return True
        return super()._stop_before_recursion(node, member, member_name)


def warn_inconsistent__all__(nodes: ty.Iterable[Node]) -> None:
    """Warn about inconsistencies in API declared by ``__all__`` and import paths.

    Compares the public API as defined by the `__all__` attribute of modules and the
    API as defined by the import path.
    """
    # Build set of API paths as defined by __all__
    api_defined_by__all__ = set()
    for node in nodes:
        if inspect.ismodule(node.obj):
            if not hasattr(node.obj, "__all__"):
                logger.warning("public module %r without __all__", node.member_path)
                continue
            api_defined_by__all__.update(
                f"{node.member_path}.{name}" for name in node.obj.__all__
            )

    # Iterate API that is reachable via public imports
    for node in nodes:
        if inspect.ismodule(node.obj):
            continue
        if node.member_path in api_defined_by__all__:
            api_defined_by__all__.remove(node.member_path)
        elif node.parent and inspect.ismodule(node.parent.obj):
            logger.warning("public object %r not included in __all__", node.member_path)

    for member_path in api_defined_by__all__:
        logger.warning(
            "public object %r in __all__ was not found in source", member_path
        )


def tree_to_csv(path: Path, nodes: ty.Iterable[Node]):
    """Export general node information into a CSV file."""
    with path.open("w") as file:
        writer = None
        for node in nodes:
            table = node.to_table()
            if writer is None:
                writer = csv.DictWriter(file, table[0].keys())
                writer.writeheader()
            writer.writerows(table)


def params_to_csv(path: Path, nodes: ty.Iterable[Node]):
    """Export information on parameters and returns into a CSV file."""
    already_documented = set()
    with path.open("w") as file:
        writer = None
        for node in nodes:
            if node.source_path in already_documented:
                continue
            already_documented.add(node.source_path)
            table: list[dict] = node.to_param_table()
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
    parser.add_argument("module")
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


@cli
def main(
    module: str, export_csv: bool, private: bool, max_depth: int | None, verbosity: int
):
    """Create an API table for a given Python package."""
    log_file = Path(tempfile.gettempdir()) / (Path(__file__).name + ".log")
    print(f"Logging to {log_file}")
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO if verbosity < 1 else logging.DEBUG,
    )

    api_root = importlib.import_module(module)
    if private:
        walker = MemberWalker(root_path=module, root_obj=api_root, max_depth=max_depth)
    else:
        walker = PublicMemberWalker(
            root_path=module, root_obj=api_root, max_depth=max_depth
        )

    nodes = []
    print("\nMember tree:")
    for node in walker:
        print("  " * node.depth + str(node))
        nodes.append(node)
    print(f"\nFound {len(nodes)} nodes in tree")

    if not private:
        warn_inconsistent__all__(nodes)

    if export_csv:
        tree_csv = Path.cwd() / f"{module}_tree.csv"
        params_csv = Path.cwd() / f"{module}_params.csv"
        tree_to_csv(tree_csv, nodes)
        params_to_csv(params_csv, nodes)
        print(f"\nCreated\n{tree_csv}\n{params_csv}")


if __name__ == "__main__":
    main()
