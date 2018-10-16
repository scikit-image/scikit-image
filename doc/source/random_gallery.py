# Generate a javascript snippet that links to a random gallery example

import os
import glob

base = os.path.abspath(os.path.dirname(__file__))
example_dir = os.path.join(base, 'auto_examples')
js_fn = os.path.join(base, '_static/random.js')

javascript = '''\

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

gallery_div = '''\
<div class="gallery_image">
      <a href="URL"><img src="IMG"/></a>
</div>\
'''

examples = glob.glob(os.path.join(example_dir, '**/plot_*.py'), recursive=True)

images, links = [], []
image_url = 'https://scikit-image.org/docs/dev/_images/'
link_url = 'https://scikit-image.org/docs/dev/auto_examples/'

for example_path in examples:
    example_path = os.path.relpath(example_path, example_dir)
    example_path, ext = os.path.splitext(example_path)

    image_path, image_file = example_path.rsplit('/', maxsplit=1)
    image_file = 'sphx_glr_' + image_file + '_001.png'

    images.append(image_url + image_file)
    links.append(link_url + example_path + '.html')

javascript = javascript.replace('{{IMAGES}}', str(images))
javascript = javascript.replace('{{LINKS}}', str(links))
javascript = javascript.replace('{{GALLERY_DIV}}', ''.join(gallery_div.split('\n')))

f = open(js_fn, 'w')
f.write(javascript)
f.close()
