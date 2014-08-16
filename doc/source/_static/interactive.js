$(document).ready(function () {
    $.support.cors = true;

    // create backup of div containing code
    var backup = [],
        editor = [],
        encodedcode = $(".tobehidden"),
        code_running = false;

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

    $('div.highlight-python').each(function (index) {
        $(this).data('index', index);
        backup[index] = $(this);
    });

    $('.editcode').each(function (index) {
        $(this).data('index', index);
    });

    // inspired from a JSfiddle
    String.format = function() {
        // The string containing the format items (e.g. "{0}")
        // will and always has to be the first argument.
        var theString = arguments[0];

        // start with the second argument (i = 1)
        for (var i = 1; i < arguments.length; i++) {
            // "gm" = RegEx options for Global search (more than one instance)
            // and for Multiline search
            var regEx = new RegExp("\\{" + (i - 1) + "\\}", "gm");
            theString = theString.replace(regEx, arguments[i]);
        }

        return theString;
    }


    function editcode (snippet, snippet_index, code_height) {

        var editor_name = String.format("editor{0}", snippet_index);
        editor[snippet_index] = ace.edit(editor_name);

        editor[snippet_index].on('change', function () {
            var doc = editor[snippet_index].getSession().getDocument(),
                // line height varies with zoom level and font size
                // correct way to find height is using the renderer
                line_height = editor[snippet_index].renderer.lineHeight;
            code_height = line_height * doc.getLength() + 'px';
            $('#' + editor_name).height(code_height);
            editor[snippet_index].resize();
        });

        // place cursor at end to prevent entire code being selected
        // after using setValue (which is a feature)
        editor[snippet_index].setValue(snippet, 1);

        // editor.setTheme("ace/theme/monokai");
        editor[snippet_index].getSession().setMode("ace/mode/python");

        // edit successful, show Run button
        $('.editcode').eq(snippet_index).hide();
        $('#runcode').show();

        // execute code on pressing 'Shift+Enter'
        editor[snippet_index].commands.addCommand({
            name: 'execute_code',
            bindKey: {win: 'Shift-Enter'},
            exec: function (editor) {
                console.log('Running code for ' + snippet_index);
                runcode(snippet_index);
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

    function handleoutput(output, snippet_index) {
        console.log('Handling output for ' + snippet_index);
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
                            //.insertAfter('#run_btn');
                            // insert just after the snippet which ran the code
                            .insertAfter($('#editor' + snippet_index));
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

    function getcode(snippet_index) {
        var resulting_code = '',
            code_snippet;
        for(var i=0; i<snippet_index; i++) {
            if (editor[i]) {
                code_snippet = editor[i].getValue();
            } else {
                code_snippet = encodedcode.eq(i).html();
                code_snippet = Base64.decode(code_snippet);
            }
            resulting_code = resulting_code + code_snippet;
            resulting_code = resulting_code + clear_images;
        }
        resulting_code = resulting_code + editor[snippet_index].getValue();
        return resulting_code;
    }

    function runcode(snippet_index) {
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
            $('#reload').hide();

            var code = getcode(snippet_index),
            // console.log(code);
                jcode = codetoJSON(code);
                // get editor-bg
                editor_color = $('#editor' + snippet_index).css('background-color');
                readonly_editor_color = '#F5F5F5';

            // disable editing when code is run
            editor[snippet_index].setReadOnly(true);
            $('#editor' + snippet_index).css('background-color', readonly_editor_color);

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
                    editor[snippet_index].setReadOnly(false);
                    $('#editor' + snippet_index).css('background-color', editor_color);

                    // remove animation, show Run
                    // TODO: Refactor to something like reset
                    $('#loading').hide();
                    $('#runcode').show();
                    $('#reload').show();
                    handleoutput(e, snippet_index);
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
                    editor[snippet_index].setReadOnly(false);
                    $('#editor' + snippet_index).css('background-color', editor_color);

                    // TODO: Refactor to something like reset
                    $('#loading').hide();
                    $('#runcode').show();
                    $('#reload').show();

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
        // replace all code snippets (in edit mode or otherwise) with default value
        console.log(backup.length);
        for(var i=0; i<backup.length; i++){
            if (editor[i]) {
                $('#editor' + i).replaceWith(backup[i]);
            }
        }
        
        $('div.highlight-python').each(function (index) {
            $(this).data('index', index);
        });

        $('div.highlight-python').unbind('click');
        // replacing messes up event handlers so need to bind again
        $('div.highlight-python').bind('click', function (){
            // fetch height of div which showed the code
            var code_height = $(this).height(),
                snippet_index = $(this).data('index'),
                snippet = encodedcode.eq(snippet_index).html();

            $(this).replaceWith(String.format('<div id="editor{0}"></editor>', snippet_index));

            snippet = Base64.decode(snippet);
            editcode(snippet, snippet_index, code_height);
        });
        // hide Run, only visible when code edited
        $('#runcode').hide();
        $('#loading').hide();
        $('.editcode').show();
        $('.all-output').hide();
    }

    // TODO: make the snippet selection code more modular
    // edit button fetches code from the URL
    $('.editcode').bind('click', function () {

        var edit_btn_index = $(this).data('index'),
            snippet_index = edit_btn_index,
            snippet = encodedcode.eq(snippet_index).html();

        $('div.highlight-python').each(function (){
            if ($(this).data('index') === snippet_index) {
                $(this).replaceWith(String.format('<div id="editor{0}"></editor>', snippet_index));
                snippet = Base64.decode(snippet);
                editcode(snippet, snippet_index);
            }
        });

    });

    // clicking on the snippet fetches only the snippet for editing
    $('div.highlight-python').bind('click', function (){
        // fetch height of div which showed the code
        var code_height = $(this).height(),
            snippet_index = $(this).data('index'),
            snippet = encodedcode.eq(snippet_index).html();

        $(this).replaceWith(String.format('<div id="editor{0}"></editor>', snippet_index));

        snippet = Base64.decode(snippet);
        editcode(snippet, snippet_index, code_height);
    });

    // TODO: Finalize what this button is for
    //$('#runcode').bind('click', runcode);

    // revert back to example inside div
    $('#reload').bind('click', reload);
});
