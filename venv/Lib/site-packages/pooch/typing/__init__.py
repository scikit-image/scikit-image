# Copyright (c) 2018 The Pooch Developers.
# Distributed under the terms of the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
#
# This code is part of the Fatiando a Terra project (https://www.fatiando.org)
#
"""
Custom classes for type annotations

This module provides additional `PEP 484 <https://peps.python.org/pep-0484/>`_
type aliases used in ``pooch``'s codebase.
"""

import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    Union,
)

# Import Pooch only if TYPE_CHECKING is true to avoid circular loops at runtime
if TYPE_CHECKING:
    from .. import Pooch


__all__ = [
    "Action",
    "Downloader",
    "PathType",
    "PathInputType",
    "ParsedURL",
    "Processor",
]


Action = Literal["download", "fetch", "update"]
PathType = Union[str, os.PathLike]
PathInputType = Union[PathType, list[PathType], tuple[PathType]]
Processor = Callable[[str, Action, Optional["Pooch"]], Any]


class Downloader(Protocol):
    """
    Class used to define the type definition for the downloader function.
    """

    # pylint: disable=too-few-public-methods
    def __call__(  # noqa: E704
        self,
        fname: str,
        action: Optional[PathType],
        pooch: Optional["Pooch"],
        *,
        check_only: Optional[bool] = None,
    ) -> Any: ...


class ParsedURL(TypedDict):
    """
    Type for a dictionary generated after parsing a URL.

    The dictionary contains three keys: protocol, netloc and path.
    """

    protocol: str
    netloc: str
    path: str
