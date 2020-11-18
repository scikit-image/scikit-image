var versions = ['dev', '0.17.x', '0.16.x', '0.15.x', '0.14.x', '0.13.x', '0.12.x', '0.11.x', '0.10.x', '0.9.x', '0.8.0', '0.7.0', '0.6', '0.5', '0.4', '0.3'];

function insert_version_links() {
    for (i = 0; i < versions.length; i++){
        open_list = '<li>'
        if (typeof(DOCUMENTATION_OPTIONS) !== 'undefined') {
            if ((DOCUMENTATION_OPTIONS['VERSION'] == versions[i]) ||
                (DOCUMENTATION_OPTIONS['VERSION'].match(/dev$/) && (i == 0))) {
                open_list = '<li id="current">'
            }
        }
        document.write(open_list);
        document.write('<a href="URL">skimage VERSION</a> </li>\n'
                        .replace('VERSION', versions[i])
                        .replace('URL', 'https://scikit-image.org/docs/' + versions[i]));
    }
}

function stable_version() {
    return versions[1];
}
