# Building docs

To build the HTML documentation, run:

```sh
spin docs
```

Output is in `scikit-image/doc/build/html/`. Add the `--clean`
flag to build from scratch, deleting any cached output.

## Gallery

The example gallery is built using
[Sphinx-Gallery](https://sphinx-gallery.github.io).
Refer to their documentation for complete usage instructions, and also
to existing examples in `doc/examples`.

Gallery examples should have a maximum figure width of 8 inches.
You can also [change a gallery entry's thumbnail](https://sphinx-gallery.github.io/stable/configuration.html#choosing-thumbnail).

## Fixing Warnings

- "citation not found: R###" There is probably an underscore after a
  reference in the first line of a docstring (e.g. [1]\_). Use this
  method to find the source file: \$ cd doc/build; grep -rin R####
- "Duplicate citation R###, other instance in..."" There is probably a
  [2] without a [1] in one of the docstrings
- Make sure to use pre-sphinxification paths to images (not the
  \_images directory)
