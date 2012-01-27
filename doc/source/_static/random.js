
   function insert_gallery() {
       var images = ['http://scikits-image.org/docs/dev/_images/plot_otsu.png', 'http://scikits-image.org/docs/dev/_images/plot_canny.png', 'http://scikits-image.org/docs/dev/_images/plot_medial_transform.png', 'http://scikits-image.org/docs/dev/_images/plot_skeleton.png', 'http://scikits-image.org/docs/dev/_images/plot_glcm.png', 'http://scikits-image.org/docs/dev/_images/plot_lena_tv_denoise.png', 'http://scikits-image.org/docs/dev/_images/plot_equalize.png', 'http://scikits-image.org/docs/dev/_images/plot_watershed.png', 'http://scikits-image.org/docs/dev/_images/plot_convex_hull.png', 'http://scikits-image.org/docs/dev/_images/plot_contours.png', 'http://scikits-image.org/docs/dev/_images/plot_hough_transform.png', 'http://scikits-image.org/docs/dev/_images/plot_radon_transform.png'];
       var links = ['http://scikits-image.org/docs/dev/auto_examples/plot_plot_otsu.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_canny.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_medial_transform.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_skeleton.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_glcm.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_lena_tv_denoise.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_equalize.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_watershed.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_convex_hull.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_contours.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_hough_transform.html', 'http://scikits-image.org/docs/dev/auto_examples/plot_plot_radon_transform.html'];

       ix = Math.floor(Math.random() * images.length);
       document.write(
'<div class="gallery_image">      <a href="URL"><img src="IMG"/></a></div>'.replace('IMG', images[ix]).replace('URL', links[ix])
       );

       console.log('<div class="gallery_image">      <a href="URL"><img src="IMG"/></a></div>'.replace('IMG', images[ix]).replace('URL', links[ix]));
   };

