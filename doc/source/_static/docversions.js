
function insert_version_links() {
    var labels = ['0.5dev', '0.4', '0.3'];
    var links = ['http://scikits-image.org/docs/dev/index.html',
                 'http://scikits-image.org/docs/0.4/index.html',
                 'http://scikits-image.org/docs/0.3/index.html'];

    document.write('<ul class="versions">\n');

    for (i = 0; i < labels.length; i++){
        document.write('<li> <a href="URL">skimage VERSION</a> </li>\n'
                        .replace('VERSION', labels[i])
                        .replace('URL', links[i]));
    }
    document.write('</ul>\n');
}

