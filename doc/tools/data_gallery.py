from skimage import data, io
import os.path
import jinja2


if not os.path.exists("../source/data_gallery/images"):
    os.makedirs("../source/data_gallery/images")



image_list = []
for name in data.__all__:
    if name != 'load':
        filename = '../source/data_gallery/images/' + name + '.png'
        if not os.path.isfile(filename):
            image = getattr(data, name)()
            io.imsave(filename, image)

        desc = getattr(data, name).__doc__.split('.')[0] + '.'
        desc = desc.replace('"', '\\"')
        image_list.append(
            {'name': name + '.png',
             'title': name,
             'desc': desc})

loader = jinja2.FileSystemLoader(searchpath="../source/data_gallery")
env = jinja2.Environment(loader=loader)
template = env.get_template("gallery_template.html")

output = template.render(images=image_list)
with open('../source/data_gallery/gallery.html', 'w') as f:
    f.write(output)
