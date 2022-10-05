/*!
 * togglebutton.js
 *
 * Modified version of
 * https://github.com/python/python-docs-theme/blob/b862f1159388ed3134818e56b065e18b835d652c/python_docs_theme/static/copybutton.js *
 * The modifications utilize the tooltip included by sphinx_copybutton for consistence
 * and omit the CSS.
 *
 * Copyright 2019 PSF
 *
 * PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
 * --------------------------------------------
 *
 * 1. This LICENSE AGREEMENT is between the Python Software Foundation
 * ("PSF"), and the Individual or Organization ("Licensee") accessing and
 * otherwise using this software ("Python") in source or binary form and
 * its associated documentation.
 *
 * 2. Subject to the terms and conditions of this License Agreement, PSF hereby
 * grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
 * analyze, test, perform and/or display publicly, prepare derivative works,
 * distribute, and otherwise use Python alone or in any derivative version,
 * provided, however, that PSF's License Agreement and PSF's notice of copyright,
 * i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
 * 2011, 2012, 2013, 2014, 2015, 2016, 2017 Python Software Foundation; All Rights
 * Reserved" are retained in Python alone or in any derivative version prepared by
 * Licensee.
 *
 * 3. In the event Licensee prepares a derivative work that is based on
 * or incorporates Python or any part thereof, and wants to make
 * the derivative work available to others as provided herein, then
 * Licensee hereby agrees to include in any such work a brief summary of
 * the changes made to Python.
 *
 * 4. PSF is making Python available to Licensee on an "AS IS"
 * basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
 * IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
 * DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
 * FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
 * INFRINGE ANY THIRD PARTY RIGHTS.
 *
 * 5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
 * FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
 * A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
 * OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.
 *
 * 6. This License Agreement will automatically terminate upon a material
 * breach of its terms and conditions.
 *
 * 7. Nothing in this License Agreement shall be deemed to create any
 * relationship of agency, partnership, or joint venture between PSF and
 * Licensee.  This License Agreement does not grant permission to use PSF
 * trademarks or trade name in a trademark sense to endorse or promote
 * products or services of Licensee, or any third party.
 *
 * 8. By copying, installing or otherwise using Python, Licensee
 * agrees to be bound by the terms and conditions of this License
 * Agreement.
 */

$(document).ready(function() {
    /* Add a [>>>] button on the top-right corner of code samples to hide
     * the >>> and ... prompts and the output and thus make the code
     * copyable. */
    var div = $('.highlight-python .highlight,' +
                '.highlight-python3 .highlight,' +
                '.highlight-pycon .highlight,' +
                '.highlight-default .highlight');
    var pre = div.find('pre');

    // get the styles from the current theme
    pre.parent().parent().css('position', 'relative');
    var hide_text = 'Hide prompts and output';
    var show_text = 'Show prompts and output';

    // create and add the button to all the code blocks that contain >>>
    div.each(function(index) {
        var jthis = $(this);
        if (jthis.find('.gp').length > 0) {
            var button = $('<span class="togglebutton o-tooltip--left">&gt;&gt;&gt;</span>');
            button.attr('data-tooltip', hide_text);
            button.data('hidden', 'false');
            jthis.prepend(button);
        }
        // tracebacks (.gt) contain bare text elements that need to be
        // wrapped in a span to work with .nextUntil() (see later)
        jthis.find('pre:has(.gt)').contents().filter(function() {
            return ((this.nodeType == 3) && (this.data.trim().length > 0));
        }).wrap('<span>');
    });

    // define the behavior of the button when it's clicked
    $('.togglebutton').click(function(e){
        e.preventDefault();
        var button = $(this);
        if (button.data('hidden') === 'false') {
            // hide the code output
            button.parent().find('.go, .gp, .gt').hide();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'hidden');
            button.css('text-decoration', 'line-through');
            button.attr('data-tooltip', show_text);
            button.data('hidden', 'true');
        } else {
            // show the code output
            button.parent().find('.go, .gp, .gt').show();
            button.next('pre').find('.gt').nextUntil('.gp, .go').css('visibility', 'visible');
            button.css('text-decoration', 'none');
            button.attr('data-tooltip', hide_text);
            button.data('hidden', 'false');
        }
    });
});
