# Plugin for docstub
# Loaded via pyproject.toml::[tools.docstub.plugin_paths]

from docstub.plugins import hookimpl


GRAMMAR = """
%import doctype (start, WS_INLINE, dtype_to_expand, qualname)

// Override form for union of dtypes
%override dtype_to_expand: "in {" qualname ("," qualname)+ "}"

%ignore WS_INLINE
"""


@hookimpl
def docstub_load_doctype_grammar():
    return GRAMMAR
