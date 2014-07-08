$(document).ready(function () {
    // create backup of div containing code
    var backup = $('.highlight-python'),
        editor;

    // hide Run, only visible when code edited
    // hide loading animation, visible, when code running
    $('#runcode').hide();
    $('#loading').hide();

    // edit button
    $('#editcode').bind('click', function () {
        // fetch code url
        var code_url = $('a.download.internal:first').attr('href'),
        // fetch height of div which showed the code
            code_height = $('.highlight-python').height();

        // fetch code and insert into editor
        $.get(code_url, function (data, status) {
            if (status === "success") {
                // replace div with editor
                $('.highlight-python').replaceWith('<div id="editor"></div>');

                editor = ace.edit("editor");

                $('#editor').height(code_height);
                // place curson at end to prevent entire code being selected 
                // after using setValue (which is a feature)
                editor.setValue(data, 1);

                // editor.setTheme("ace/theme/monokai");
                editor.getSession().setMode("ace/mode/python");

                // edit successful, show Run button
                $('#editcode').hide();
                $('#runcode').show();
            }
        });
    });

    function codetoJSON(code) {
        return JSON.stringify({'data': code});
    }

    function handleoutput(output) {
        output = output.result;
        var imagemeta = 'data:image/png;base64,',
        // output is a key, value pair of filename: uuencoded content
        // output = JSON.stringify(output)
        // TODO: it loads the last generated image into the outputimage tag
        // that needs to be changed

        // example images are first children, within a div of class section
            example_images = $('.section > img'),
        // index for iterating through example images
            i = 0,
            key,
            image;
        for (key in output) {
            image = output[key];
            image = imagemeta + image;
            // more images generated than in initial example
            if(i >= example_images.length){
                $('.section > img:last')
                    .clone()
                    .attr('src', image)
                    .insertAfter('.section > img:last');
            } else {
                // console.log(example_images[i]);
                example_images[i].src = image;
                // example_images[i].attr('src', image);
                i = i + 1;
            }
        }
    }

    $.support.cors = true;
    $('#runcode').bind('click', function () {
        // debug
        // console.log('detect click');

        // add animation, hide Run to prevent duplicate runs
        $('#loading').show();
        $(this).hide();

        var code = editor.getValue(),
        // console.log(code);
            jcode = codetoJSON(code);
        // console.log(jcode);
        $.ajax({
            type: 'POST',
            // Provide correct Content-Type, so that Flask will know how to process it.
            contentType: 'application/json',
            // Encode your data as JSON.
            data: jcode,
            // This is the type of data you're expecting back from the server.
            dataType: 'json',
            url: 'http://198.206.133.45:5000/runcode',
            success: function (e) {
                // remove animation, show Run
                $('#loading').hide();
                $('#runcode').show();
                handleoutput(e);
            }
        });
    });

    // revert back to example inside div
    $('#reload').bind('click', function () {
        $('div#editor').replaceWith(backup);
        // hide Run, only visible when code edited
        $('#runcode').hide();
        $('#loading').hide();
        $('#editcode').show();
    });
});