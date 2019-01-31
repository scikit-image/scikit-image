__all__ = ['register_affine']

from matplotlib import pyplot as plt
from matplotlib import animation
from ..transform import resize


def resized(images):
    """
    Resize all images to the largest width and height

    Parameters
    ----------
    images : array of (M, N) ndarrays
        Original images

    Returns
    -------
    modified_images : array of (M, N) ndarrays
        images in a modified uniform size
    """

    h = max([image.shape[0] for image in images])
    v = max([image.shape[1] for image in images])
    size = (h, v)

    modified_images = [resize(img, size) for img in images]
    return modified_images


def _animate(images, frames_per_second=2, file_name=None, delay=0,
             title='', xlabel='', ylabel=''):
    """
    Animate a gif out of an array of images

    Parameters
    ----------
    images : array of (M, N) ndarrays
        Input images.
    frames_per_second : float, optional
        The number of frames per second in the animations.
    file_name : str, optional
        The name of the file to which the animation
        will be saved. If no file_name is given, the
        file will not be saved.
    delay : int, optional
        A waiting period measured in ms. The delay occurs
        between the time the animation finishes and a new
        loop starts.
    title : str, optional
        The title of the plot
    xlabel : str, optional
        The label of the horisontal axis of the plot.
    ylabel : str, optional
        The label of the vertical axis of the plot.

    Returns
    -------
    anim : animatplot.Animation
        The animation created.
    """

    images = resized(images)

    fig = plt.figure()
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()

    ims = [[plt.imshow(image)] for image in images]
    frame_time = int(1000/frames_per_second)
    anim = animation.ArtistAnimation(fig, ims,
                                     interval=frame_time, repeat_delay=delay)

    if file_name:
        anim.save(file_name)

    plt.show()

    return anim
