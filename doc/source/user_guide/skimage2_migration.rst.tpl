.. -*- mode: rst -*-
.. vi: filetype=rst

.. _skimage2-migration:

*****************************************
Migration guide: from skimage to skimage2
*****************************************

.. hint::

    This document is a work in progress and still subject to change.

Scikit-image is preparing to release version 2.0 as a new package: ``skimage2``.
Alongside skimage2, we will release version 1.0.0. Versions 1.x will be using
the current API. Versions 1.1.x will throw a ``FutureWarning`` upon import, as
a means to notify users that they should either upgrade to skimage2 or pin to
version 1.0.x.

We have undertaken this to make some long-outstanding, backward-incomptible
changes to the scikit-image API. Most changes were difficult or impossible to
make using deprecations alone. To honor the Hinsen principle (that is, never
change results silently unless to fix a bug), we introduce a new package,
which gives users an explicit way of upgrading. Users also have the option to
use the two versions side-by-side while they do so.

You can find a more detailed description of our motivation and discussion
leading up to this in :doc:`SKIP 4 <../skips/4-transition-to-v2>`.

.. _enable-skimage2-warnings:

Enable skimage2-related warnings
================================

Even before skimage2 is released, you may enable skimage2-related warnings to
prepare for code changes early on. Run the following `warnings filter
<https://docs.python.org/3/library/warnings.html#the-warnings-filter>`__ before
you use scikit-image in your code:

.. code-block:: python

    import warnings
    import skimage as ski
    warnings.filterwarnings(action="default", category=ski.util.PendingSkimage2Change)

This will raise a warning in code that needs to be modified to continue
functioning with the new, skimage2 API.

Updating existing code
======================

When switching to the new ``skimage2`` namespace, some code will need to be updated to continue working the way it did before.

.. note::

    For a while, you will be able to use ``skimage`` and ``skimage2`` (the 2.0
    API) side-by-side, to facilitate porting. The new API may, for the same
    function call, return different results—e.g., because of a change in
    a keyword argument default value. By importing functionality from
    ``skimage2``, you explicitly opt in to the new behavior.

{% macro format_label(title) -%}
sk2adv-{{ title | replace('.', '-') | replace('_', '-') }}
{%- endmacro %}

{#- Format an advice section, pops `title` from `advice_map` implicitly! #}
{%- macro format_advice_section(title, ul_char='-') -%}
.. _{{ format_label(title) }}:

``{{ title }}``
{{ ul_char * (title | length + 4) }}

{# Consume item, calling script checks if dict is emptied -#}
{{ advice_map.pop(title) }}
{%- endmacro %}

{#- Format "gray functions" manually #}
.. _sk2adv-gray-funcs:

Grayscale morphological operators in `skimage.morphology`
---------------------------------------------------------

The following functions are deprecated in favor of counterparts in `skimage2.morphology`:

{% for name in advice_map['gray_funcs'] %}
- :ref:`{{ format_label(name) }}`
{% endfor %}

The new counterparts behave differently in the following ways:

- All functions now default to ``mode='ignore'`` (was ``mode='reflect'``).
- Additionally, ``skimage2.morphology.dilation``, ``skimage2.morphology.closing``,
  and ``skimage2.morphology.black_tophat`` now also mirror the footprint (invert
  its order in each dimension). Note this only impacts behavior for asymmetric
  footprints.

.. admonition:: Background for changes
    :class: note dropdown

    The new behavior ensures that, with default parameters, ``closing`` and
    ``opening`` are `extensive and anti-extensive
    <https://en.wikipedia.org/wiki/Mathematical_morphology#Properties_of_the_basic_operators>`__
    respectively. This change also aligns the behavior for asymmetric
    footprints with SciPy's ``scipy.ndimage.grey_*`` functions.

    Refer to
    `gh-6665 <https://github.com/scikit-image/scikit-image/issues/6665>`__,
    `gh-6676 <https://github.com/scikit-image/scikit-image/issues/6676>`__,
    `gh-8046 <https://github.com/scikit-image/scikit-image/pull/8046>`__, and
    `gh-8060 <https://github.com/scikit-image/scikit-image/pull/8060>`__ for
    more details.

{% for name in advice_map.pop('gray_funcs') %}
{{ format_advice_section(title=name, ul_char='^') }}
{% endfor -%}


Removal of ``skimage.future``
-----------------------------

There will be no ``future`` submodule in ``skimage2``.  If you are using
modules in current ``skimage.future``, please either vendor the
``skimage.future`` code in your own code-base, or use other libraries.  If you
are making heavy use of ``skimage.future`` routines, please feel free to raise
an issue at the `scikit-image issues
<https://github.com/scikit-image/scikit-image/issues>`__ page to ask us to
port the function you want to use.

{% for name in advice_map.pop('future_funcs') %}
{{ format_advice_section(title=name, ul_char='^') }}
{% endfor -%}

{# Note that we need to have used any supporting dictionaries such as
   'future_funcs' above by this point, otherwise they will be pulled out and
   used in the clause below. #}

{#- Iterate over and format remaining advice #}
{%- for name in advice_map.keys() | sort %}
{{ format_advice_section(title=name) }}
{% endfor %}

Deprecations prior to skimage2
==============================

We have already introduced a number of changes and deprecations to our API.
These are part of the API cleanup for skimage2 but are not breaking.
You will simply notice these as the classical deprecation warnings that you are already used to.
We list them here, because updating your code to the new API will make it easier to transition to skimage2.

*To be defined.*
