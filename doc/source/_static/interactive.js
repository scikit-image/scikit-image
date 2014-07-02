$(document).ready(function() {
	// create backup of div containing code
	var backup = $('.highlight-python');

	// edit button
	$('#editcode').bind('click', function(){
		// fetch code url
		code_url = $('a.download.internal:first').attr('href');
		// fetch height of div which showed the code
		var code_height = $('.highlight-python').height();
		// replace div with editor
		$('.highlight-python').replaceWith('<div id="editor"></div>');

		// fetch code and insert into editor
		$.get(code_url, function(data){
			var editor = ace.edit("editor");
			
			$('#editor').height(code_height);
			// place curson at end to prevent entire code being selected 
			// after using setValue (which is a feature)
			editor.setValue(data, 1);
    		
    		// editor.setTheme("ace/theme/monokai");
    		editor.getSession().setMode("ace/mode/python");
		});
	});

	// revert back to example inside div
	$('#reload').bind('click', function(){
		$('div#editor').replaceWith(backup);
	});
});