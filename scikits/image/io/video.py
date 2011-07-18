import cv
import numpy as np
import pygst
import os, time
pygst.require("0.10")
import gst, gobject
gobject.threads_init()
from gst.extend.discoverer import Discoverer

"""
@add_backends
class Video(object):
    def __init__(self):
        raise NotImplementedError("Please select a video backend.")
"""

class CvVideo(object):
    """
    Opencv-based video player.

    Parameters
    ----------
    source : str
        Media location.
    size: tuple, optional
        Size of returned array.
    """
    def __init__(self, source=None, size=None):
        self.source = source
        self.capture = cv.CreateFileCapture(self.source)
        self.size = size
        
    def get(self):
        """
        Retrieve a video frame as a numpy array.
        
        Returns
        -------
        output : array (image)
            Retrieved image.
        """
        img = cv.QueryFrame(self.capture)
        if not self.size:
            self.size = cv.GetSize(img)
        img_mat = np.empty((self.size[1], self.size[0], 3), dtype=np.uint8)
        if cv.GetSize(img) == self.size:
            cv.Copy(img, img_mat)
        else:
            cv.Resize(img, img_mat)
        return img_mat
    
    def seek_frame(self, frame_number):
        """
        Seek to specified frame in video.
        
        Parameters
        ----------
        frame_number : int
            Frame position
        """
        cv.SetCaptureProperty(self.capture, cv.CV_CAP_PROP_POS_FRAMES, frame_number)
        
    def seek_time(self, milliseconds):
        """
        Seek to specified time in video.
        
        Parameters
        ----------
        milliseconds : int
            Time position
        """
        cv.SetCaptureProperty(self.capture, cv.CV_CAP_PROP_POS_MSEC, milliseconds)
    
    def frame_count(self):
        """
        Returns frame count of video.

        Returns
        -------
        output : int
            Frame count.
        """
        return cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FRAME_COUNT)
    
    def duration(self):
        """
        Returns time length of video in milliseconds.

        Returns
        -------
        output : int
            Time length [ms].
        """
        return cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FPS) * \
            cv.GetCaptureProperty(self.capture, cv.CV_CAP_PROP_FRAME_COUNT)
    
    
        
class GstVideo(object):
    """
    GStreamer-based video player.

    Parameters
    ----------
    source : str
        Media location.
    size: tuple, optional
        Size of returned array.
    sync: bool, optional
        Frames are extracted per frame or per time basis.
    """
    def __init__(self, source=None, size=None, sync=False):
        self.source = source
        self.size = size
        self.video_length = 0
        self.video_rate = 0
        # extract video size
        if not size: 
            gobject.idle_add(self._discover_one)
            self.mainloop = gobject.MainLoop()
            self.mainloop.run()
        if not self.size:
            self.size = (640, 480)
        if os.path.exists(self.source):
            self.source = "file://" + self.source
        self._create_main_pipeline(self.source, self.size, sync)

    def _discover_one(self):
        """
        Callback to start media discovery process, used to retrieve video parameters.
        """
        discoverer = Discoverer(self.source)
        discoverer.connect('discovered', self._discovered)
        discoverer.discover()          
        return False

    def _discovered(self, d, is_media):
        """
        Callback to on media discovery result.
        """
        if is_media:
            self.size = (d.videowidth, d.videoheight)
            self.video_length = d.videolength / gst.MSECOND            
            self.video_rate = d.videorate.num
        self.mainloop.quit()
        return False
    
    def _create_main_pipeline(self, source, size, sync):
        """ 
        Create the frame extraction pipeline.
        """
        pipeline_string = "uridecodebin name=decoder uri=%s ! ffmpegcolorspace ! videoscale ! appsink name=play_sink" % self.source
        self.pipeline = gst.parse_launch(pipeline_string)
        caps = "video/x-raw-rgb, width=%d, height=%d, depth=24, bpp=24" % size
        self.decoder = self.pipeline.get_by_name("decoder")
        self.appsink = self.pipeline.get_by_name('play_sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.set_property('sync', sync)
        self.appsink.set_property('drop', True)
        self.appsink.set_property('max-buffers', 1)
        self.appsink.set_property('caps', gst.caps_from_string(caps))
        if self.pipeline.set_state(gst.STATE_PLAYING) == gst.STATE_CHANGE_FAILURE:
            raise NameError("Failed to load video source %s" % self.source)
	    buff = self.appsink.emit('pull-preroll')
      
    def get(self):
        """
        Retrieve a video frame as a numpy array.
        
        Returns
        -------
        output : array (image)
            Retrieved image.
        """
        buff = self.appsink.emit('pull-buffer')
        img_mat = np.ndarray(shape=(self.size[1], self.size[0], 3), dtype='uint8', buffer=buff.data)
        return img_mat

    def seek_frame(self, frame_number):
        """
        Seek to specified frame in video.
        
        Parameters
        ----------
        frame_number : int
            Frame position
        """
        self.pipeline.seek_simple(gst.FORMAT_DEFAULT, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT, frame_number)
        
    def seek_time(self, milliseconds):
        """
        Seek to specified time in video.
        
        Parameters
        ----------
        milliseconds : int
            Time position
        """
        self.pipeline.seek_simple(gst.FORMAT_TIME, gst.SEEK_FLAG_FLUSH | gst.SEEK_FLAG_KEY_UNIT, milliseconds/1000.0 * gst.SECOND)

    def frame_count(self):
        """
        Returns frame count of video.

        Returns
        -------
        output : int
            Frame count.
        """
        return self.video_length/1000*self.video_rate
    
    def duration(self):
        """
        Returns time length of video in milliseconds.

        Returns
        -------
        output : int
            Time length [ms].
        """
        return self.video_length


if __name__ == '__main__':
    cv.NamedWindow ('display', cv.CV_WINDOW_AUTOSIZE)
    cv.MoveWindow("display", 100, 100);
    camera = GstVideo(source="http://146.232.169.185/video.mjpg")
    #$camera = GstVideo(source="/home/tzhau/hacking/video/ing1.avi", sync=1)
#    camera = CvVideo(source="/home/tzhau/hacking/video/ing1.avi")
    #camera = CvVideo(source="/home/tzhau/Desktop/lego.flv")
    print camera.frame_count()
    print camera.duration()
    i = 0
    while 1:
        i += 1
        a = time.time()
        img = camera.get()

        b = time.time()
        print b-a
        cv.ShowImage('display', img)
        cv.WaitKey(100)    
