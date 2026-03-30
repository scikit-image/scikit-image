(skip5-v2-port_procedure)=

# SKIP 5 — scikit-image v2 porting procedure

:Author: Matthew Brett
:Status: Draft
:Type: Standards Track
:Created: 2026-03-27
:Resolved: <null>
:Resolution: <null>
:Version effective: <null>

## Abstract

We describe the general procedure for porting the code in `skimage` to the scikit-image v2 namespace.

## Motivation and Scope

### Background

#### Decisions agreed before this Skip

* We will have three namespaces:

  - `skimage` (housing the Skimage1 API)
  - `_skimage2` (housing the evolving scikit-image v2 API)
  - `skimage2` (imports `_skimage2` and adds FutureWarning. We won't discuss that further here.

  For rationale for this structure, see [^import-structure].

* We agreed that we should _copy_ tests from `skimage` (e.g.
  `tests/skimage/transform/tests` to `tests/skimage2/transform/tests`) as we
  port.

[^import-structure]: `skimage` can import directly from `_skimage2`, but `_skimage2` cannot import
  directly from `skimage`; if it does need `skimage` routines, it must do local
  (deferred, inline) imports, inside functions or methods, to avoid circular
  imports.

  This structure is to allow `skimage` to import from the scikit-image v2
  namespace without triggering a warning.  Conversely, `skimage2` exists only
  to trigger a `FutureWarning` for any imports from `_skimage2`.

#### Endpoint

At some point, we will need a full scikit-image v2 implementation in
`_skimage2`, with a full set of tests exercising that namespace. There will be
a few functions and modules that will stay in `skimage`, such as everything in
`skimage/future`, but otherwise, all or nearly all code in `skimage` will move
in some form to `_skimage2`.

### Porting strategies

There are two potential approaches to this problem, that we will call *big-bang* and *bit-by-bit*.

#### Big-bang

One implementation we will call "big-bang". Here we move all the current
`skimage` implementations, that will have versions in scikit-image v2, into the
`_skimage2` namespace. We import the `_skimage2` implementations back into the
`skimage` namespace, while preserving the current `skimage` wrappers for
already ported code.

We also copy all tests from `tests/skimage` to `tests/_skimage2`

For more details in the procedure, see the [detailed description
section](detailed-description) below.

This leaves us with a complete `_skimage2` namespace, but where we have yet to
complete the API and other changes in that namespace.

After this, all scikit-image v2 changes take place in the `src/_skimage2` and
`tests/_skimage2` trees.

We could also call this approach "move-and-edit".

#### Bit-by-bit

This is our approach at time of writing. For each scikit-image v2-related change, we
move the implementation of the relevant functions (etc) to the `src/_skmage2`
tree, along with the relevant tests, and make suitable wrappers that import and modify that implementation in the `src/skimage` tree.

Thus the `_skimage2` namespace fills up gradually, as we do the migration of the API and other code.

### Benefits and disbenefits from the approaches

First let's start with the general ideas, and then get down to specifics.

#### Bit-by-bit as migration log

We can add an extra constraint to the bit-by-bit approach, which is to ensure
that we only ever move functions to `_skimage2` where we are confident that the
code will continue to be about the same, and with the same API, as for
scikit-image v2. That is, in moving any code, we assert that this code is
scikit-image v2 ready.

Call this variant - *bit-by-bit-and-certify*, or BBBC for short.

If we use BBBC then we can reasonably believe that all the code /APIs in
`src/_skimage2` is at least something like the code / APIs in the eventual
first scikit-image v2 release.

The idea here is that, as we submit PRs, we not only make partial scikit-image
v2 changes, but we do all likely scikit-image v2 changes. In that way, we can
use the code in `_skimage2` (as compared to that in `skimage`) as a record of
what changes we have done in the migration. Call this the migration-log
function of the bit-by-bit approach.

With the big-bang approach, we can't initially tell users to start using
`_skimage2` code, with the expectation that the code will, in fact, have the
API of scikit-image v2 — because, at first, the code will be a still changing
version of `skimage`.

Of course, we could keep the `_skimage2` directory under wraps until it is
mostly complete. Or we could advertise only a few parts of the `_skimage2`
/ `skimage2` namespace.

#### Migration in practice

Let's first consider the bit-by-bit approach, and some fake module `fake` in
`src/skimage/fakepkg/fake.py`. Let's assume the tests are in a single test
module: `tests/skimage/fakepkg/tests/test_fake.py`.

The `fake` module might look something like this:

```python
def foo(a, b, c):
    "No scikit-image v2 changes yet proposed"
    return bar(baz(a, b), c)

def bar(d, f=None):
    "Planned scikit-image v2 API changes"
    return baz(d ** 2, f)

def baz(g, h)
    return g * 2 + h
```

`test_fake.py` has:

```python
from skimage.fakepkg.fake import foo, bar, baz

def test_foo():
    # No planned scikit-image v2 change.
    assert foo(1, 2, 3)

def test_bar():
    # Planned scikit-image v2 change of behavior.
    assert bar(1)
    assert bar(1, 2) == 3

def test_baz():
    # No planned scikit-image v2 change of behavior.
    assert baz(1, 2) == 4
```

Let us say we want to do a PR to change the default value of `bar` above from None
to 10.

With the BBBC approach, and this example, we will have to *disentangle* the
ported functions and their tests, to work out what will go into the `_skimage2`
namespace.  Call this — the *disentangle problem*.

Disentangling involves identifying what will go in the PR. On analysis we do
need to move over the `bar` function, and the `baz` function, but we don't need
the `foo` function.  Or perhaps, perhaps we know the `foo` function is due for
more scikit-image v2 changes, that we don't want to do at the moment.   So we
edit `src/_skimage2/fakepkg/fake.py` to have only:

```python
def bar(d, f=10):
    "Planned scikit-image v2 API changed"
    return baz(d ** 2, f)

def baz(g, h)
    return g * 2 + h
```

This leaves `src/skimage/fakepkg/fake.py` behind:

```python
from _skimage2.fakepkg.fake import bar2, baz

@migration_warning('Default value of f changed in skimage2 from None to 10')
def bar(d, f=None):
    return bar2(d, f)

def foo(a, b, c):
    "No scikit-image v2 changes yet proposed"
    return bar(baz(a, b), c)
```

We also might want to split up the test function:

`tests/_skimage2/fakepkg/tests/test_fake.py`:

```python
from _skimage2.fakepkg.fake import bar, baz

def test_bar():
    # Planned scikit-image v2 change of behavior.
    assert bar(1)
    assert bar(1, 2) == 3

def test_baz():
    # No planned scikit-image v2 change of behavior.
    assert baz(1, 2) == 4
```

`tests/_skimage2/fakepkg/tests/test_fake.py`:

```python
from _skimage.fakepkg.fake import foo

def test_foo():
    # No planned scikit-image v2 change.
    assert foo(1, 2, 3)
```

There are a few problems here.

- The Certify aspect of the BBBC process requires to analyze what code was
  using what, and then split up files, increasing work.  We do not need to do
  this analysis or splitting of files for the Big-Bang procedure.
- We had to think about where our imports are coming from.
- We are left, towards the end of the porting process, with the task of pulling
  the bits of the file still in `skimage` back into the code in `_skimage2`.
- If we want to maintain the migration-log benefit of bit-by-bit, we have to
  think about which functions are fully `_skimage2`-ready, and which are not.
  For example, will we be changing `baz` or `foo` in scikit-image v2? This adds extra
  complexity to the port process, and review.

In contrast, it is more straightforward to do the port with the big-bang
approach. We already have copied all the tests, in one big-bang step.  So we already have a complete
`tests/_skimage2/fakepkg/tests/test_fake.py`.  We just modify that
appropriately, leaving the `skimage` version intact. We don't have to split up
the `fake.py` function, or think about which functions are fully scikit-image
v2-ready, we concentrate on the changes of interest to us. We do, of course,
have to write suitable wrappers or alternative implementations in the `skimage`
namespace, as before.

### On the migration-log idea

Gwiven there are development costs for the bit-by-bit implementation, how great are the benefits, in particular, for the migration-log idea.

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

#### Altnerative ways of tracking migration

We should in any case, maintain a checklist like that in
<https://github.com/scikit-image/scikit-image/wiki/API-changes-for-skimage2>.
We could make it part of the PR process, maintaining that list.

Note that, no matter what approach we take, any changed APIs in `_skimage2`
*must* have wrappers or alternative implmentations in `skimage`.

If we use the big-bang approach, these wrappers and implementations provide
a full migration log, because any function that is, as yet, unmodified, will be
directly imported and not modified in the `skimage` tree.

Specifically, and for example, we can see what has been ported by looking at
the `skmage/transform/__init__.py` module, as this will have something on the lines of:

```python
from _skimage2.transform import *
import _skimage2.transform as sk2t

# Any API-change wrappers below.
@migration_warning('Default value of f changed in skimage2 from None to 10')
def somefunction(d, f=None):
    # Some processing
    return sk2t.somefunction(d, f)

# And so on.
```

Lastly, I wonder whether the Certify requirement in the BBBC approach, will
bias us to choose to make scikit-image v2 changes that we would not do (and
should not do) without the Certify apporach. I suspect we'll find ourselves
trying to think of any possible change we could make for scikit-image v2 in
every PR, and I suspect that will lead to us making more changes, some of which
will prove to 50-50 calls that probably aren't worth the disruption. This is
a speculation that the bit-by-bit-certify process will mean that the default
moves further towards if-in-doubt-change, and that is not desirable.

Is the agreed (and debated, and updated) checklist page a better record of the
changes we want to make? One that might serve as a brake on lower-value
changes? And can we reassure ourselves we are ready for scikit-image v2 by
careful review of the whole code-base (in `_skimage2`) when we are nearer
release?

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

Any pull requests or development branches containing work on this SKIP
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
