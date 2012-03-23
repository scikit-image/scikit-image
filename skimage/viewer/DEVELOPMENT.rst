
Organization
============

`scikits-loupe` is organized into the following subpackages:

utils
   Basic utilities shared by other subpackages.
widgets
   Basic interactive tools used to interact with viewers. For example a slider
   widget to adjust the value of a parameter.
viewers
   Image and image collection viewers.
plugins
   Tools that can be attached to a viewer for image analysis and manipulation.
   For example, a tool to adjust the contrast of an image.

The order of this list is important: a subpackage can import from any
subpackage preceding it in this list, but not those after it. For example,
a function in `viewers` can use functions in `utils` and `widgets` but not in
`plugins`.


Todo
====

- Add interface (menu?) for connecting a Plugin to an ImageWindow.
- Add check for image collections in `imshow` and divert to `CollectionViewer`.
- Add `CompareViewer` for comparing two images side-by-side.
- Add commit/revert image buttons (a la skivi)
- Add to `io` stack button (a la skivi)


Open Questions
==============

- Is `ImageViewer.show` the best way to start the mainloop or would a normal
  function be better?
- Rename `CollectionViewer` to `StackViewer`?

