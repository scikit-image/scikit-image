"""
:author: Yashawant Parab, Sachin Vaidya, Siddharth Bammani 2020
This module supports image augmentation
"""

import os
import random
from skimage import io
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage import img_as_ubyte, exposure, filters
from skimage.transform import rotate, AffineTransform, warp


class augmentation:
    def __init__(self, input_path, output_path, image_name):
        """Parameters
    ----------
    input_path :
        takes image from the path specified.
    output_path :
        saves images to the path specified.
    images_to_generate :
        generates the number of images requested.
    """
    def anti_clockwise(self):
        """AntiClockwise rotation of Image 
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
        rotated : n-dimensional array
            Rotated version of image.
        """
        angle= random.randint(0, 180)
        return rotate(self, angle)
    def rgb2gray(self):
        """rgb to gray conversion of Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
            
        Returns
        -------
            Convert image form rgb to gray.
        """
        return rgb2gray(self)
    def clockwise(self):
        """Clockwise rotation of Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
        rotated : n-dimensional array
            Rotated version of image.
        """
        angle= random.randint(0, 180)
        return rotate(self, -angle)
    def h_flip(self):
        """Horizontal flip of Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
        flip : n-dimensional array
            flipped version of image.
        """
        return (self[:, ::-1])
    def v_flip(self):
        """Vertical flip of Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
        flip : n-dimensional array
            flipped version of image.
        """
        return (self[::-1, :])
    def warp_shift(self):
        """warp image using the estimated transformation
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Image with Affine Transform.
        """
        transform = AffineTransform(translation=(0, 40))
        warp_image = warp(self, transform, mode="wrap")
        return warp_image
    def add_noise(self):
        """Add random noise to Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Image with random noise.
        """
        return random_noise(self)
    def blur_image(self):
        """Add blurring effect to Image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Blured Image.
        """
        return filters.gaussian(self, (2, 3), multichannel = True)
    def logcorr(self):
        """logarithmic exposure added to image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Image with logarathmic exposure.
        """
        return exposure.adjust_log(self)
    def sigmoid(self):
        """sigmoid exposure added to image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Image with sigmoid exposure.
        """
        return exposure.adjust_sigmoid(self)
    def rescale(self):
        """rescale image with random value.
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            rescaled image.
        """
        v_min = random.randint(0, 2)
        v_max = random.randint(95, 99)
        return exposure.rescale_intensity(self, in_range = (v_min, v_max))
    def gamma(self):
        """gamma exposure added to image
        Parameters
        ----------
        image : n-dimensional array
            Input image folder.
        Returns
        -------
            Image with gamma exposure.
        """
        gamma = round(random.uniform(0.4, 0.9), 2)
        gain = round(random.uniform(0.4, 0.9), 2)
        return exposure.adjust_gamma(self, gamma, gain)
    
    
def augment_img(input_path, output_path, images_to_generate):
    """Augment Image
    This colabrated function will do the following
    augmentation techiniques to the image:
    [1] Anti clockwise rotation
    [2] RGB to gray conversion
    [3] Clockwise rotation
    [4] Horizontal flip
    [5] Vertical flip
    [6] Warp shift
    [7] Add noise
    [8] Blur image
    [9] Logarithmic correction
    [10] Sigmoid correction
    [11] Rescale exposure intensity
    [12] Adjust gamma exposure
    Parameters
    ----------
    input_path :
        Add the path of the image.
    output_path :
        output path for augmented images to be saved.
    images_to_generate :
        Enter the number of image that should be
        generated after augmentation.
    Returns:
    This function will generate the specified number
    of augmented copies(images_to_generate) to a
    specified location(output_path).
    Example
    -------
    >>>from skimage.transform import augment-img
    >>>input_path = specified_path_of_the_image
    >>>output_path = path_to_save_the_image
    >>>images_to_generate = specify_the_number
    """
    images=[]
    for img in os.listdir(input_path):
        images.append(os.path.join(input_path,img))
    transform = {'anticlockwise_rotation': augmentation.anti_clockwise,
                 'clockwise_rotation': augmentation.clockwise,
                 'rgb2gray': augmentation.rgb2gray,
                 'horizontal_flip': augmentation.h_flip,
                 'vertical_flip': augmentation.v_flip,
                 'warp_shift': augmentation.warp_shift,
                 'adding_noise': augmentation.add_noise,
                 'blur_image': augmentation.blur_image,
                 'logarithmic': augmentation.logcorr,
                 'sigmoid': augmentation.sigmoid,
                 'rescale': augmentation.rescale,
                 'gamma': augmentation.gamma
                }
    images_to_generate= images_to_generate
    generate_image= 1
    while generate_image <= images_to_generate:
        image= random.choice(images)
        original_img = io.imread(image)
        transformed_image= None
        count = 0
        transform_count = len(transform)
        while count <= transform_count:
            key = random.choice(list(transform))
            transformed_image = transform[key](original_img)
            count = count + 1
        new_img_path= "%s/Image_aug%s.jpg" %(output_path, generate_image)
        transformed_image= img_as_ubyte(transformed_image)
        io.imsave(new_img_path, transformed_image)
        generate_image= generate_image+ 1
    print('Generation of ' + str(images_to_generate) + ' images completed.')