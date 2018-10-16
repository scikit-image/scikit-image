Handling Video Files
--------------------

Sometimes it is necessary to read a sequence of images from a standard video
file, such as .avi and .mov files.

In a scientific context, it is usually better to avoid these formats in favor
of a simple directory of images or a multi-dimensional TIF. Video formats are
more difficult to read piecemeal, typically do not support random frame access
or research-minded meta data, and use lossy compression if not carefully
configured. But video files are in widespread use, and they are easy to share,
so it is convenient to be equipped to read and write them when necessary.

Tools for reading video files vary in their ease of installation and use, their
disk and memory usage, and their cross-platform compatibility.  This is a
practical guide.

A Workaround: Convert the Video to an Image Sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a one-off solution, the simplest, surest route is to convert the video to a
collection of sequentially-numbered image files, often called an image
sequence. Then the images files can be read into an `ImageCollection` by
`skimage.io.imread_collection`. Converting the video to frames can be done
easily in `ImageJ <http://imagej.nih.gov/ij/>`__, a cross-platform, GUI-based
program from the bio-imaging community, or `FFmpeg <https://www.ffmpeg.org/>`__, a
powerful command-line utility for manipulating video files.

In FFmpeg, the following command generates an image file from each frame in a
video. The files are numbered with five digits, padded on the left with zeros.

.. code-block:: bash

   ffmpeg -i "video.mov" -f image2 "video-frame%05d.png"

More information is available in an `FFmpeg tutorial on image sequences 
<http://en.wikibooks.org/wiki/FFMPEG_An_Intermediate_Guide/image_sequence#Making_an_Image_Sequence_from_a_video>`__.

Generating an image sequence has disadvantages: they can be large and unwieldy,
and generating them can take some time. It is generally preferrable to work
directly with the original video file. For a more direct solution, we need to
execute FFmpeg or LibAV from Python to read frames from the video.
FFmpeg and LibAV are two large open-source
projects that decode video from the sprawling variety of formats used in the
wild. There are several ways to use them from Python. Each, unfortunately,
has some disadvantages.


PyAV
^^^^

`PyAV <http://mikeboers.github.io/PyAV/>`__ uses FFmpeg's (or LibAV's) libraries
to read image data directly from the video file. It invokes them using Cython
bindings, so it is very fast.

.. code-block:: python

   import av
   v = av.open('path/to/video.mov')

PyAV's API reflects the way frames are stored in a video file.

.. code-block:: python

   for packet in container.demux():
       for frame in packet.decode():
           if frame.type == 'video':
               img = frame.to_image()  # PIL/Pillow image
               arr = np.asarray(img)  # numpy array
               # Do something!

Adding Random Access to PyAV
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `Video` class in `PIMS <https://github.com/soft-matter/pims>`__
invokes PyAV and adds additional functionality to solve a common
problem in scientific applications, accessing a video by frame
number. Video file formats are designed to be searched in an
approximate way, by time, and they do not support an efficient means
of seeking a specific frame number. PIMS adds this missing
functionality by decoding (but not reading) the entire video at and
producing an internal table of contents that supports indexing by
frame.

.. code-block:: python

   import pims
   v = pims.Video('path/to/video.mov')
   v[-1]  # a 2D numpy array representing the last frame

MoviePy
^^^^^^^

`Moviepy <http://zulko.github.io/moviepy>`__ invokes FFmpeg through a
subprocess, pipes the decoded video from FFmpeg
into RAM, and reads it out. This approach is straightforward, but it can be
brittle, and it's not workable for large videos that exceed available RAM.
It works on all platforms if FFmpeg is installed.

Since it does not link to FFmpeg's underlying libraries, it is easier to
install but about `half as fast <https://gist.github.com/mikeboers/6843684>`__.

.. code-block:: python

    from moviepy.editor import VideoFileClip
    myclip = VideoFileClip("some_video.avi")

Imageio
^^^^^^^^

`Imageio <http://imageio.github.io/>`_ takes the same approach as MoviePy. It
supports a wide range of other image file formats as well.

.. code-block:: python

    import imageio
    filename = '/tmp/file.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')

    for num, image in vid.iter_data():
        print(image.mean())

    metadata = vid.get_meta_data()

OpenCV
^^^^^^

Finally, another solution is the `VideoReader
<https://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-open>`__
class in OpenCV, which has bindings to FFmpeg. If you need OpenCV for other reasons,
then this may be the best approach.
