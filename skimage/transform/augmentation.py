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
        self.input_path = input_path
        self.output_path = output_path
        self.image_name = image_name
        
    def anti_clockwise(self):
        angle= random.randint(0,180)
        return rotate(self, angle)
    
    def rgb2gray(self): 
        return rgb2gray(self)

    def clockwise(self):
        angle= random.randint(0,180)
        return rotate(self, -angle)

    def h_flip(self):
        return (self[:, ::-1])

    def v_flip(self):
        return (self[::-1,:])
    
    def warp_shift(self): 
        transform = AffineTransform(translation=(0,40))  
        warp_image = warp(self, transform, mode="wrap")
        return warp_image

    def add_noise(self):
        return random_noise(self)

    def blur_image(self):
        return filters.gaussian(self, (2,3), multichannel = True)
      
    def logcorr(self):
        return exposure.adjust_log(self)
    
    def sigmoid(self):
        return exposure.adjust_sigmoid(self)
    
    def rescale(self):
        v_min = random.randint(0,2)
        v_max = random.randint(95,99)
        return exposure.rescale_intensity(self, in_range = (v_min, v_max))
    
    def gamma(self):
        gamma = round(random.uniform(0.4,0.9),2)
        gain = round(random.uniform(0.4,0.9),2)
        return exposure.adjust_gamma(self, gamma, gain)
        
def augment_img(input_path, output_path, images_to_generate):
    
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
                       'blur_image':augmentation.blur_image,
                       'logarithmic': augmentation.logcorr,
                       'sigmoid': augmentation.sigmoid,
                       'rescale': augmentation.rescale,
                       'gamma': augmentation.gamma
                      } 
    
    images_to_generate=images_to_generate  
    generate_image=1                        
    
    while generate_image <=images_to_generate:
        image=random.choice(images)
        original_img = io.imread(image)
        transformed_image=None
        
        count = 0       
        transform_count = len(transform)
        
        while count <= transform_count:
            key = random.choice(list(transform)) 
            transformed_image = transform[key](original_img)
            count = count + 1
            
        new_image_path= "%s/Image_augmentation%s.jpg" %(output_path, generate_image)
        transformed_image = img_as_ubyte(transformed_image)  
        
        io.imsave(new_image_path, transformed_image) 
        generate_image = generate_image+ 1
        
    print('Generation of ' + str(images_to_generate) + ' images completed.')