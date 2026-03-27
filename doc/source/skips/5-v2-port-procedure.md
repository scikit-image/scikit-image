(skip5-v2-port_procedure)=

# SKIP 5 — Skimage2 porting procedure

:Author: Matthew Brett
:Status: Draft
:Type: Standards Track
:Created: 2026-03-27
:Resolved: <null>
:Resolution: <null>
:Version effective: <null>

## Abstract

We describe the general procedure for porting the code in `skimage` to the Skimage2 namespace.

## Motivation and Scope

We have already decided to have three namespaces:

- `skimage` (housing the Skimage1 API)
- `_skimage2` (housing the evolving Skimage2 API)
- `skimage2` (imports `_skimage2` and adds FutureWarning. We won't discuss that further here.

`skimage` can import directly from `_skimage2`, but `_skimage2` cannot import directly from `skimage`; if it does need `skimage` routines, it must do local (deferred, inline) imports, inside functions or methods, to avoid circular imports.

We have also agreed that we should _copy_ tests from `skimage` (e.g. `tests/skimage/transform/tests` to `tests/skimage2/transform/tests` as we port.

At some point, we will need a full Skimage2 implementation in `_skimage2`, with a full set of tests exercising that namespace. There will be a few functions and modules that will stay in `skimage`, such as everything in `skimage/future`, but otherwise, all or nearly all code in `skimage`

This section describes the need for the proposed change. It should
describe the existing problem, who it affects, what it is trying to
solve, and why. This section should explicitly address the scope of and
key requirements for the proposed change.

## Detailed description

This section should provide a detailed description of the proposed
change. It should include examples of how the new functionality would be
used, intended use-cases, and pseudocode illustrating its use.

## Related Work

This section should list relevant and/or similar technologies, possibly
in other libraries. It does not need to be comprehensive, just list the
major examples of prior and relevant art.

## Implementation

This section lists the major steps required to implement the SKIP. Where
possible, it should be noted where one step is dependent on another, and
which steps may be optionally omitted. Where it makes sense, each step
should include a link related pull requests as the implementation
progresses.

Any pull requests or developmt branches containing work on this SKIP
should be linked to from here. (A SKIP does not need to be implemented
in a single pull request if it makes sense to implement it in discrete
phases).

## Backward compatibility

This section describes the ways in which the SKIP breaks backward
compatibility.

## Alternatives

If there were any alternative solutions to solving the same problem,
they should be discussed here, along with a justification for the chosen
approach.

## Discussion

This section may just be a bullet list including links to any
discussions regarding the SKIP, but could also contain additional
comments about that discussion:

- This includes links to discussion forum threads or relevant GitHub
  discussions.

## References and Footnotes

All SKIPs should be declared as dedicated to the public domain with the
CC0 license[^1], as in [Copyright](copyright-section), below, with attribution
encouraged with CC0+BY [^2].

(copyright-section)=

## Copyright

This document is dedicated to the public domain with the Creative
Commons CC0 license[^3]. Attribution to this source is encouraged where
appropriate, as per CC0+BY[^4].

[^1]:
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^2]: <https://dancohen.org/2013/11/26/cc0-by/>

[^3]:
    CC0 1.0 Universal (CC0 1.0) Public Domain Dedication,
    <https://creativecommons.org/publicdomain/zero/1.0/>

[^4]: <https://dancohen.org/2013/11/26/cc0-by/>
