function insert_version_links() {
    var labels = ['dev', '0.4', '0.3'];

    document.write('<ul class="versions">\n');

    for (i = 0; i < labels.length; i++){
        document.write('<li> <a href="URL">skimage VERSION</a> </li>\n'
                        .replace('VERSION', labels[i])
                        .replace('URL', 'http://scikits-image.org/docs/' + labels[i]));
    }
    document.write('</ul>\n');
}
