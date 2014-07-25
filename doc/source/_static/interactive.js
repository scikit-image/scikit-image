$(document).ready(function () {
    $.support.cors = true;

    // create backup of div containing code
    var backup = $('.highlight-python'),
        editor;

    // hide Run, only visible when code edited
    // hide loading animation, visible, when code running
    $('#runcode').hide();
    $('#loading').hide();
    $('#error-message').hide();
    $('#success-message').hide();
    $('.all-output').hide();


    // edit button
    $('#editcode').bind('click', function () {
        // store scroll position to prevent jumping of scroll bar
        var temp_scroll = $(window).scrollTop(),
        // fetch code url
            code_url = $('a.download.internal:first').attr('href'),
        // fetch height of div which showed the code
            code_height = $('.highlight-python').height();

        // fetch code and insert into editor
        $.get(code_url, function (data, status) {
            if (status === "success") {
                // replace div with editor
                $('.highlight-python').replaceWith('<div id="editor"></div>');

                editor = ace.edit("editor");

                editor.on('change', function () {
                    var doc = editor.getSession().getDocument(),
                        // line height varies with zoom level and font size
                        // correct way to find height is using the renderer
                        line_height = editor.renderer.lineHeight;
                    code_height = line_height * doc.getLength() + 'px';
                    $('#editor').height(code_height);
                    editor.resize();
                });

                // place curson at end to prevent entire code being selected 
                // after using setValue (which is a feature)
                editor.setValue(data, 1);

                // editor.setTheme("ace/theme/monokai");
                editor.getSession().setMode("ace/mode/python");

                // restore scroll bar position after adding editor
                $(window).scrollTop(temp_scroll);

                // edit successful, show Run button
                $('#editcode').hide();
                $('#runcode').show();

                // execute code on pressing 'Shift+Enter'
                editor.commands.addCommand({
                    name: 'execute_code',
                    bindKey: {win: 'Shift-Enter'},
                    exec: function (editor) {
                        runcode();
                    },
                    readOnly: true // false if this command should not apply in readOnly mode
                });
            }
        });
    });

    function codetoJSON(code) {
        return JSON.stringify({'data': code});
    }

    function handleoutput(output) {
        var output_images = output.result,
            stdout = output.stdout,
            stderr = output.stderr,
            imagemeta = 'data:image/png;base64,',
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
        for (key in output_images) {
            image = output_images[key];
            image = imagemeta + image;
            // more images generated than in initial example
            // here we replace the original images present
            // if (i >= example_images.length) {
            //     $('.section > img:last')
            //         .clone()
            //         .attr('src', image)
            //         .insertAfter('.section > img:last');
            // } else {
            //     // console.log(example_images[i]);
            //     example_images[i].src = image;
            //     // example_images[i].attr('src', image);
            //     i = i + 1;
            // }

            // this stacks images below the editor
            if (i === 0) {
                $('.section > img:first')
                    .clone()
                    .attr('src', image)
                    .addClass('output_image')
                    .insertAfter('#editor');
                    i = i + 1;
            } else {
                $('.section > img.output_image:last')
                    .clone()
                    .attr('src', image)
                    .addClass('output_image')
                    .insertAfter('.section > img.output_image:last');
            }
        }
        if (stdout === "") {
            $('.stdout-group, #stdout').hide();
        } else {
            $('.stdout-group').show();
            $('#stdout').html(stdout).show();
        }

        if (stderr === "") {
            $('.stderr-group, #stderr').hide();
        } else {
            $('.stderr-group').show();
            $('#stderr').html(stderr).show();
        }
        $('.all-output').show();
    }

    function runcode() {
        // debug
        // console.log('detect click');

        // add animation, hide Run to prevent duplicate runs
        $('#loading').show();
        // hide message from previous Run
        $('#error-message').hide();
        $('#success-message').hide();
        $('.all-output').hide();

        $('#runcode').hide();

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
                // TODO: Refactor to something like reset
                $('#loading').hide();
                $('#runcode').show();
                handleoutput(e);
                // suggest number of images received
                if ($.isEmptyObject(e.result)) {
                    num_images = 0;
                } else {
                    num_images = Object.keys(e.result).length;
                }
                $('#success-message').html("Success: Received " + num_images + " image(s)").show();
            },
            error: function (jqxhr, text_status, error_thrown) {
                // TODO: Refactor to something like reset
                $('#loading').hide();
                $('#runcode').show();

                error_code = jqxhr.status;
                error_text = jqxhr.statusText;
                $('#error-message').html(error_text + ' ' + error_code);
                $('#error-message').show();
            }
        });
    }

    $('#runcode').bind('click', runcode);

    // revert back to example inside div
    $('#reload').bind('click', function () {
        $('div#editor').replaceWith(backup);
        // hide Run, only visible when code edited
        $('#runcode').hide();
        $('#loading').hide();
        $('#editcode').show();
        $('#error-message').hide();
        $('#success-message').hide();
        $('.all-output').hide();
    });
});