## scikit-image WASM wheels local index

This is a stub directory that stores the Pyodide/WASM wheels built as a part of
the CircleCI doc build process, which are then indexed by JupyterLite and shipped
to become available in the JupyterLite environment for installation with the
Python (Pyodide) kernel. When running locally, a user can drop wheels here to
test them in the JupyterLite environment, and install them using `piplite` via the
`%pip install` magic command. Note that in some cases, you might need to embed the
version of the package you want to install based on its wheel filename so that it
is picked up by `piplite` from the local index rather than preferring other indices
such as PyPI (pure Python packages) or jsDelivr (what Pyodide uses).
