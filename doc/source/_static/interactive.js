$(document).ready(function () {
    $.support.cors = true;

    // create backup of div containing code
    var backup,
        encodedcode = $(".tobehidden"),
        code_running = false,
        editor,
        snippet,
        snippet_index;

    // Create Base64 Object
    var Base64={_keyStr:"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=",encode:function(e){var t="";var n,r,i,s,o,u,a;var f=0;e=Base64._utf8_encode(e);while(f<e.length){n=e.charCodeAt(f++);r=e.charCodeAt(f++);i=e.charCodeAt(f++);s=n>>2;o=(n&3)<<4|r>>4;u=(r&15)<<2|i>>6;a=i&63;if(isNaN(r)){u=a=64}else if(isNaN(i)){a=64}t=t+this._keyStr.charAt(s)+this._keyStr.charAt(o)+this._keyStr.charAt(u)+this._keyStr.charAt(a)}return t},decode:function(e){var t="";var n,r,i;var s,o,u,a;var f=0;e=e.replace(/[^A-Za-z0-9\+\/\=]/g,"");while(f<e.length){s=this._keyStr.indexOf(e.charAt(f++));o=this._keyStr.indexOf(e.charAt(f++));u=this._keyStr.indexOf(e.charAt(f++));a=this._keyStr.indexOf(e.charAt(f++));n=s<<2|o>>4;r=(o&15)<<4|u>>2;i=(u&3)<<6|a;t=t+String.fromCharCode(n);if(u!=64){t=t+String.fromCharCode(r)}if(a!=64){t=t+String.fromCharCode(i)}}t=Base64._utf8_decode(t);return t},_utf8_encode:function(e){e=e.replace(/\r\n/g,"\n");var t="";for(var n=0;n<e.length;n++){var r=e.charCodeAt(n);if(r<128){t+=String.fromCharCode(r)}else if(r>127&&r<2048){t+=String.fromCharCode(r>>6|192);t+=String.fromCharCode(r&63|128)}else{t+=String.fromCharCode(r>>12|224);t+=String.fromCharCode(r>>6&63|128);t+=String.fromCharCode(r&63|128)}}return t},_utf8_decode:function(e){var t="";var n=0;var r=c1=c2=0;while(n<e.length){r=e.charCodeAt(n);if(r<128){t+=String.fromCharCode(r);n++}else if(r>191&&r<224){c2=e.charCodeAt(n+1);t+=String.fromCharCode((r&31)<<6|c2&63);n+=2}else{c2=e.charCodeAt(n+1);c3=e.charCodeAt(n+2);t+=String.fromCharCode((r&15)<<12|(c2&63)<<6|c3&63);n+=3}}return t}}

    var clear_images = "\nplt.close()\n";

    // hide Run, only visible when code edited
    // hide loading animation, visible, when code running
    $('#runcode').hide();
    $('#loading').hide();
    $('#error-message').hide();
    $('#success-message').hide();
    $('.all-output').hide();
    $('.tobehidden').hide();


    function editcode (snippet) {

        editor = ace.edit("editor");

        // fetch height of div which showed the code
        code_height = $(this).height();

        editor.on('change', function () {
            var doc = editor.getSession().getDocument(),
                // line height varies with zoom level and font size
                // correct way to find height is using the renderer
                line_height = editor.renderer.lineHeight;
            code_height = line_height * doc.getLength() + 'px';
            $('#editor').height(code_height);
            editor.resize();
        });

        // place cursor at end to prevent entire code being selected
        // after using setValue (which is a feature)
        editor.setValue(snippet, 1);

        // editor.setTheme("ace/theme/monokai");
        editor.getSession().setMode("ace/mode/python");

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

        // store scroll position to prevent jumping of scroll bar
        var temp_scroll = $(window).scrollTop();

        // restore scroll bar position after adding editor
        $(window).scrollTop(temp_scroll);
    }

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
        // example_images = $('.section > img'),
        // index for iterating through example images
            i = 0,
            key,
            image,
            timestamp;

        console.log(output);
        console.log(output.stdout);

        if (!output.result.hasOwnProperty('busy')) {
            for (key in output_images) {
                // if it is not the timestamp go ahead and add as an image
                if (key.indexOf('timestamp') == -1) {
                    image = output_images[key];
                    image = imagemeta + image;
                    timestamp = output_images[key+'timestamp'];
                    console.log(timestamp);
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
                            // image creation timstamp
                            // .attr('title', timestamp)
                            .addClass('output_image')
                            .insertAfter('#run_btn');
                            i = i + 1;
                    } else {
                        $('.section > img.output_image:last')
                            .clone()
                            .attr('src', image)
                            // .attr('title', timestamp)
                            .addClass('output_image')
                            .insertAfter('.section > img.output_image:last');
                    }
                }
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

    function getcode() {
        var resulting_code = '',
            enc_snippet;
        for(var i=0; i<snippet_index; i++) {
            enc_snippet = encodedcode.eq(i).html();
            enc_snippet = Base64.decode(enc_snippet);
            resulting_code = resulting_code + enc_snippet;
            resulting_code = resulting_code + clear_images;
        }
        resulting_code = resulting_code + editor.getValue();
        return resulting_code;
    }

    function runcode() {
        if (!code_running) {
            code_running = true;
            // debug
            // console.log('detect click');

            // add animation, hide Run to prevent duplicate runs
            $('#loading').show();
            // hide message from previous Run
            $('#error-message').hide();
            $('#success-message').hide();
            $('.all-output').hide();
            // get rid of output images from previous run
            $('img.output_image').remove();

            $('#runcode').hide();

            var code = getcode(),
            // console.log(code);
                jcode = codetoJSON(code);
                // get editor-bg
                editor_color = $('.ace-tm').css('background-color');
                readonly_editor_color = '#F5F5F5';

            // disable editing when code is run
            editor.setReadOnly(true);
            $('.ace-tm').css('background-color', readonly_editor_color);

            // console.log(jcode);
            $.ajax({
                type: 'POST',
                // Provide correct Content-Type, so that Flask will know how to process it.
                contentType: 'application/json',
                // Encode your data as JSON.
                data: jcode,
                // This is the type of data you're expecting back from the server.
                dataType: 'json',
                url: 'http://ci.scipy.org:8000/runcode',
                success: function (e) {
                    // enable editing after response
                    editor.setReadOnly(false);
                    $('.ace-tm').css('background-color', editor_color);

                    // remove animation, show Run
                    // TODO: Refactor to something like reset
                    $('#loading').hide();
                    $('#runcode').show();
                    handleoutput(e);
                    // suggest number of images received
                    if ($.isEmptyObject(e.result)) {
                        num_images = 0;
                    } else {
                        // half of the keys are timestamps of images in the other half
                        num_images = Object.keys(e.result).length/2;
                    }
                    if (e.result.hasOwnProperty('busy')) {
                        $('#success-message').html("Server Busy, try again later!").show();
                    } else {
                        $('#success-message').html("Success: Received " + num_images + " image(s) at " + e.timestamp + " UTC -5").show();
                    }
                    code_running = false;
                },
                error: function (jqxhr, text_status, error_thrown) {
                    // enable editing after response
                    editor.setReadOnly(false);
                    $('.ace-tm').css('background-color', editor_color);

                    // TODO: Refactor to something like reset
                    $('#loading').hide();
                    $('#runcode').show();

                    error_code = jqxhr.status;
                    error_text = jqxhr.statusText;
                    $('#error-message').html(error_text + ' ' + error_code);
                    $('#error-message').show();
                    code_running = false;
                }
            });
        } else {
            console.log('wait for response..');
        }
    }

    function reload () {
        $('div#editor').replaceWith(backup);
        $('div.highlight-python').unbind('click');
        // replacing messes up event handlers so need to bind again
        $('div.highlight-python').bind('click', function (){
            // replace div with editor
            backup = $(this);

            snippet_index = $(this).closest('div.highlight-python').index('div.highlight-python'),
            snippet = encodedcode.eq(snippet_index).html();

            $(this).replaceWith('<div id="editor"></div>');

            console.log(snippet_index);
            snippet = Base64.decode(snippet);
            editcode(snippet);
        });
        // hide Run, only visible when code edited
        $('#runcode').hide();
        $('#loading').hide();
        $('#editcode').show();
        $('.all-output').hide();
    }

    // edit button fetches code from the URL
    $('#editcode').bind('click', function () {
        // fetch code url
        var code_url = $('a.download.internal:first').attr('href');

        $.get(code_url, function (data, status) {
            if (status === "success") {
                editcode(data);
            }
        });
    });

    // clicking on the snippet fetches only the snippet for editing
    $('div.highlight-python').bind('click', function (){
        // replace div with editor
        backup = $(this);

        snippet_index = $(this).closest('div.highlight-python').index('div.highlight-python'),
        snippet = encodedcode.eq(snippet_index).html();

        $(this).replaceWith('<div id="editor"></div>');
        console.log(snippet_index);

        snippet = Base64.decode(snippet);
        editcode(snippet);
    });

    $('#runcode').bind('click', runcode);
    // revert back to example inside div
    $('#reload').bind('click', reload);

    $(document).keyup(function(e) {
        if (e.keyCode == 27) {
            reload();
        }   // esc
    })
});
