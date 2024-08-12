"""Custom Sphinx extensions for scikit-image's docs.

Have a look at the `setup` function to see what kind of functionality is added.
"""

import re
from pathlib import Path

from sphinx.util import logging
from sphinx.directives.other import TocTree


logger = logging.getLogger(__name__)


def natural_sort_key(item):
    """Transform entries into tuples that can be sorted in natural order [1]_.

    This can be passed to the "key" argument of Python's `sorted` function.

    Parameters
    ----------
    item :
        Item to generate the key from. `str` is called on this item before generating
        the key.

    Returns
    -------
    key : tuple[str or int]
        Key to sort by.

    Examples
    --------
    >>> natural_sort_key("release_notes_2.rst")
    ('release_notes_', 2, '.rst')
    >>> natural_sort_key("release_notes_10.rst")
    ('release_notes_', 10, '.rst')
    >>> sorted(["10.b", "2.c", "100.a"], key=natural_sort_key)
    ['2.c', '10.b', '100.a']

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Natural_sort_order
    """
    splitted = re.split(r"(\d+)", str(item))
    key = tuple(int(x) if x.isdigit() else x for x in splitted)
    return key


class NaturalSortedTocTree(TocTree):
    """Directive that sorts all TOC entries in natural order by their file names.

    Behaves similar to Sphinx's default ``toctree`` directive. The ``reversed`` option
    is respected, though the given order of entries (or globbed entries) is ignored.
    """

    def parse_content(self, toctree):
        ret = super().parse_content(toctree)
        reverse = 'reversed' in self.options
        toctree['entries'] = sorted(
            toctree['entries'], key=natural_sort_key, reverse=reverse
        )
        return ret


RANDOM_JS_TEMPLATE = '''\

   function insert_gallery() {
       var images = {{IMAGES}};
       var links = {{LINKS}};

       ix = Math.floor(Math.random() * images.length);
       document.write(
'{{GALLERY_DIV}}'.replace('IMG', images[ix]).replace('URL', links[ix])
       );

       console.log('{{GALLERY_DIV}}'.replace('IMG', images[ix]).replace('URL', links[ix]));
   };

'''


GALLERY_DIV = '''\
<div class="gallery_image">
      <a href="URL"><img src="IMG"/></a>
</div>\
'''


def write_random_js(app, exception):
    """Generate a javascript snippet that links to a random gallery example."""
    if app.builder.format != "html":
        logger.debug(
            "[skimage_extensions] skipping generation of random.js for non-html build"
        )
        return

    build_dir = Path(app.outdir)
    random_js_path = Path(app.outdir) / "_static/random.js"

    image_urls = []
    tutorial_urls = []
    url_root = "https://scikit-image.org/docs/dev/"
    examples = build_dir.rglob("auto_examples/**/plot_*.html")
    for example in examples:
        image_name = f"sphx_glr_{example.stem}_001.png"
        if not (build_dir / "_images" / image_name).exists():
            continue
        image_url = f'{url_root}_images/{image_name}'
        tutorial_url = f'{url_root}{example.relative_to(build_dir)}'
        image_urls.append(image_url)
        tutorial_urls.append(tutorial_url)

    if tutorial_urls == 0:
        logger.error(
            "[skimage_extensions] did not find any gallery examples while creating %s",
            random_js_path,
        )
        return

    content = RANDOM_JS_TEMPLATE.replace('{{IMAGES}}', str(image_urls))
    content = content.replace('{{LINKS}}', str(tutorial_urls))
    content = content.replace('{{GALLERY_DIV}}', ''.join(GALLERY_DIV.split('\n')))

    random_js_path.parent.mkdir(parents=True, exist_ok=True)
    with open(random_js_path, 'w') as file:
        file.write(content)
    logger.info(
        "[skimage_extensions] created %s with %i possible targets",
        random_js_path,
        len(tutorial_urls),
    )


def setup(app):
    app.add_directive('naturalsortedtoctree', NaturalSortedTocTree)
    app.connect('build-finished', write_random_js)
    return {'parallel_read_safe': True}
