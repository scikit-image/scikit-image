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

### Background

We have already decided to have three namespaces:

- `skimage` (housing the Skimage1 API)
- `_skimage2` (housing the evolving Skimage2 API)
- `skimage2` (imports `_skimage2` and adds FutureWarning. We won't discuss that further here.

`skimage` can import directly from `_skimage2`, but `_skimage2` cannot import directly from `skimage`; if it does need `skimage` routines, it must do local (deferred, inline) imports, inside functions or methods, to avoid circular imports.

We have also agreed that we should _copy_ tests from `skimage` (e.g. `tests/skimage/transform/tests` to `tests/skimage2/transform/tests`) as we port.

At some point, we will need a full Skimage2 implementation in `_skimage2`, with a full set of tests exercising that namespace. There will be a few functions and modules that will stay in `skimage`, such as everything in `skimage/future`, but otherwise, all or nearly all code in `skimage` will move in some form to `_skimage2`.

### Big-bang or bit-by-bit

There are two potential approaches to this problem.

#### Big-bang

One implementation we will call "big-bang". Here we move all the current
`skimage` implementations, that will have versions in Skimage2, into the
`_skimage` namespace. We import the `_skimage2` implementations back into the
`skimage` namespace, while preserving the current `skimage` wrappers for
already ported code.

We also copy all tests from `tests/skimage` to `tests/_skimage2`

For more details in the procedure, see the [detailed description
section](detailed-description) below.

This leaves us with a complete `_skimage` namespace, but where we have yet to
complete the API and other changes in that namespace.

After this, all Skimage2 changes take place in the `src/_skimage2` and
`tests/_skimage2` trees.

We could also call this approach "move-and-edit".

#### Bit-by-bit

This is our approach at time of writing. For each Skimage2-related change, we
copy the implementation of the relevant functions (etc) to the `src/_skmage2`
tree, along with the relevant tests, and make suitable wrappers that import and modify that implementation in the `src/skimage` tree.

Thus the `_skimage2` namespace fills up gradually, as we do the migration of the API and other code.

### Benefits and disbenefits from the approaches

First let's start with the general ideas, and then get down to specifics.

#### Bit-by-bit as migration log

One benefit for the bit-by-bit approach is, if we do the changes in
a particular way, then we can reasonably believe that all the code /APIs in
`src/_skimage2` is at least something like the code / APIs in the eventual
first Skimage2 release.

The "particular way" is one where we only ever move functions to `_skimage2` where we are confident that the code will continue to be about the same, and with the same API, as for Skimage2. That is, in moving any code, we assert that this code is Skimage2 ready.

The idea here is that we not only make partial Skimage2 changes, but we do all
likely Skimage2 changes. In that way, we can use the code in `_skimage2` (as
compared to that in `skimage`) as a record of what changes we have done in the
migration. Call this the migration-log function of the bit-by-bit approach.

#### Bit-by-bit as partial guarantee of future API

With the big-bang approach, we can't initially tell users to start using
`_skimage2` code, with the expectation that the code will, in fact, have the API of Skimage2 — because, at first, the code will be a still changing version of `skimage`.

Of course, we could keep the `_skimage2` directory under wraps until it is mostly complete. Or we could advertise only a few parts of the `_skimage2` / `skimage2` namespace.

#### Migration in practice

Let's first consider the bit-by-bit approach, and some fake module `fake` in
`src/skimage/fakepkg/fake.py`. Let's assume the tests are in a single test
module: `tests/skimage/fakepkg/tests/test_fake.py`.

The `fake` module might look something like this:

```python
def foo(a, b, c):
    "No Skimage2 changes yet proposed"
    return bar(baz(a, b), c)

def bar(d, f=None):
    "Planned Skimage2 API changes"
    return baz(a ** 2, f)

def baz(g, h)
    return g * 2 + h
```

`test_fake.py` has:

```python
from _skimage.fakepkg.fake import foo, bar, baz

def test_foo():
    # No planned Skimage2 change.
    assert foo(1, 2, 3)

def test_bar():
    # Planned Skimage2 change of behavior.
    assert bar(1)
    assert bar(1, 2) == 3

def test_baz():
    # No planned Skimage2 change of behavior.
    assert baz(1, 2) == 4
```

Let us say we want to do a PR to change the default value of `bar` from None
to 10.

With the bit-by-bit approach, we have some work to do. First we have to
identify what will go in the PR. On analysis we do need to move over the
`bar` function, and the `baz` function, but we don't need the `foo` function. So we edit `src/_skimage2/fakepkg/fake.py` to have only:

```python
def bar(d, f=10):
    "Planned Skimage2 API changed"
    return baz(a ** 2, f)

def baz(g, h)
    return g * 2 + h
```

Leaving `src/skimage/fakepkg/fake.py`:

```python
from _skimage2.fakepkg.fake import bar, baz

@migration_warning('Default value of f changed in skimage2 from None to 10')
def bar(d, f=None):
    return bar(d, f)

def foo(a, b, c):
    "No Skimage2 changes yet proposed"
    return bar(baz(a, b), c)
```

We also might want to split up the test function:

`tests/_skimage2/fakepkg/tests/test_fake.py`:

```python
from _skimage2.fakepkg.fake import bar, baz

def test_bar():
    # Planned Skimage2 change of behavior.
    assert bar(1)
    assert bar(1, 2) == 3

def test_baz():
    # No planned Skimage2 change of behavior.
    assert baz(1, 2) == 4
```

`tests/_skimage2/fakepkg/tests/test_fake.py`:

```python
from _skimage.fakepkg.fake import foo

def test_foo():
    # No planned Skimage2 change.
    assert foo(1, 2, 3)
```

There are a few problems here.

- We were tempted to first — analyze what code was using what, and then split
  up files, increasing work.
- We had to think about where our imports are coming from.
- We are left, towards the end of the porting process, of pulling the bits of
  the file still in `skimage` back into the code in `_skimage2`.
- If we want to maintain the migration-log benefit of bit-by-bit, we have to
  think about which functions are fully `_skimage2`-ready, and which are not.
  For example, will we be changing `baz` or `foo` in Skimage2? This adds extra
  complexity to the port process, and review.

In contrast, it is more straightforward to do the port with the big-bang
approach. We already have copied tests
(`tests/_skimage2/fakepkg/tests/test_fake.py`, so we just modify that
appropriately, leaving the `skimage` version intact. We don't have to split up the `fake.py` function, or think about which functions are fully Skimage2-ready, we concentrate on the changes of interest to us. We do, of course, have to write suitable wrappers or alternative implementations in the `skimage` namespace, as before.

### On the migration-log idea

Given there are development costs for the bit-by-bit implementation, how great are the benefits, in particular, for the migration-log idea.

The question has to be asked in relation to:

- The problems we are expecting from incomplete migration.
- Any alternatives we could use to track migration.
- Our default position on change-by-default compared to stay-same-by-default.

What problems are we predicting from migration? There could be many, but
I suspect the big ones will be where we have attempted to harmonize an API,
but have missed a function — for example, for coordinate systems or
in-function image scaling.

Are these well-dealt with by making checks for these part of each PR process?
Or are they better dealt with by screening the whole `_skimage2` codebase for
instances, for example with AI, and dealing with those? We will likely find
it easier to whole-code-base screening and changes when all the code is in one
`_skimage2` tree, rather than broken up into `_skimage2` and `skimage`.

We do have alternative methods of tracking the ports. For example, we could
maintain a checklist like that in
<https://github.com/scikit-image/scikit-image/wiki/API-changes-for-skimage2>.
We could make it part of the PR process, maintaining that list. I suppose one
could argue that the partially filled `_skimage2` is a better log, but it is,
at least, not as readable.

Lastly — one could argue that, by forcing reviewers to think about whether all Skimage2 changes have been done, we can make sure we don't forget to make changes that occur to us as we port — but this begs the question of whether we should go looking for Skimage2 changes, or treat the default as "leave-as-is", on the basis that the greater the volume of changes, the higher the risk that Skimage1 users will not in fact migrate.

Is the agreed (and debated, and updated) checklist page a better record of the changes we want to make? One that might serve as a brake on lower-value changes? And can we reassure ourselves we are ready for Skimage2 by careful review of the whole code-base (in `_skimage2`) when we are nearer release?

(detailed-description)=

## Detailed description

This is just a sketch. For now I'll put some LLM agent instructions, but we can think about how we would do this in practice.

Draft LLM instructions:

> The directory `src/_skimage2` contains code implementing the new version
> 2 API of the scikit-image API. The directory `src/skimage` contains the code
> implementing version 1 of the API. There are matching tests in
> `tests/skimage2` and `tests/skimage` respectively. Notice that, for code that
> has already reached `src/_skimage2`, there are matching wrappers in the `src/skimage` tree that imports the `_skimage2` code and implements the old version 1 API using the new version 2 code.
>
> Analyze these directory trees. For every function or class that is not yet implemented in `_skimage2`, copy the implementation to `_skimage2` and import that function or class, now in `_skimage2`, back into the `skimage` namespace, with the same name and module path. Copy all the tests not present in `tests/skimage` to matching positions in `tests/skimage2`.
>
> First run all the tests in `tests/skimage` to make sure you have correctly
> imported all functions from `_skimage2` to `skimage`, and fix accordingly. Then run all the tests in `tests/skimage2` and fix accordingly. Make sure that you can assert that a) no functions or classes are now missing from `_skimage2` what were in `skimage` and b) these functions and classes have all been correctly imported back into `skimage`, and c) that all tests for `skimage` are running and that d) all tests for `_skimage2` are running, and the tests for `_skimage2` are a superset of those for `skimage`.

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
