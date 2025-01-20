import glob
import matplotlib.pyplot as plt
import torch
import torchvision
import threading
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import PIL
import random
import copy
import logging
import math
from numpy import zeros, uint8, asarray
import os
from typing import List
import fnmatch
import tarfile
import os.path
import cv2 as cv2
import numpy as np
import random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



###################### Verify image #############################
def verify_image(image):
    if is_numpy_array(image):
        pass
    else:
        raise Exception("not a numpy array or list of numpy array")

###################### is numpy array #############################
def is_numpy_array(x):
    return isinstance(x, np.ndarray)


###################### HLS #############################
def hls(image,src='RGB'):
    verify_image(image)
    image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    return image_HLS

###################### RGB #############################
def rgb(image, src='BGR'):
    verify_image(image)
    image_RGB= eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    return image_RGB

###################### Image To Numpy #############################
def imageToNumpy(image):
    return asarray(image)

###################### Image From Numpy #############################
def imageFromNumpy(image):
    return Image.fromarray(image)


class Fault:
    def __init__(self, intensity, fault_type):
        self.fault_type = fault_type
        self.weight = 0.20
        self.intensity = intensity
    
    def inject(self, clean_image):
        pass
    
class NoFault(Fault):
    def __init__(self):
        Fault.__init__(self, intensity = None, fault_type = 'NoFault')
        self.weight = None
    
    def inject(self, clean_image):
        pass
    
    

class Brightness(Fault):
    def __init__(self, intensity = 'l'):
        Fault.__init__(self, intensity, fault_type = 'Brightness')
        self.weight = 0.1

    def inject(self, clean_image):
        faulty_image = clean_image
        try:
            enhancer = ImageEnhance.Brightness(clean_image)
            if self.intensity.lower()=='e':
                self.weight = (np.random.randint(60,91))/100 + 1
            elif self.intensity.lower()=='m':
                self.weight = (np.random.randint(40,60))/100 + 1
            else:
                self.weight = (np.random.randint(10,30))/100 + 1
            faulty_image = enhancer.enhance(self.weight)
            # print(self.weight)
            return faulty_image
        except Exception as error_log:
            pass
            # print(error_log, self.fault_type)
            
class Dark(Fault):
    def __init__(self, intensity = 'l'):
        Fault.__init__(self, intensity, fault_type = 'Dark')
        self.weight = 0.01
        
    def inject(self, clean_image):
        faulty_image = clean_image
        try:
            enhancer = ImageEnhance.Brightness(clean_image)
            
            if self.intensity.lower()=='e':
                self.weight = (np.random.randint(30,55))/100
            elif self.intensity.lower()=='m':
                self.weight = (np.random.randint(15,30))/100
            else:
                self.weight = (np.random.randint(60,90))/100
                
            faulty_image = enhancer.enhance(self.weight)
            return faulty_image
        except Exception as error_log:
            pass
            # print(error_log, self.fault_type)
        
    
class Blur(Fault):
    def __init__(self, intensity = 'l'):
        Fault.__init__(self, intensity, fault_type = 'Blur')
        self.weight = 1.0

    def inject(self, clean_image):
        faulty_image = clean_image
        
        if self.intensity.lower()=='m':
            self.weight = np.random.randint(7,9)
            ksize = (self.weight, self.weight)
        elif self.intensity.lower()=='e':
            self.weight = np.random.randint(12,14)
            ksize = (self.weight, self.weight)
        else:
            self.weight = np.random.randint(4,6)
            ksize = (self.weight, self.weight)
        try:
            clean_nparray = imageToNumpy(clean_image)
            faulty_image = cv2.blur(clean_nparray, ksize)
            return imageFromNumpy(faulty_image)
        except Exception as error_log:
            pass
            # print(error_log, self.fault_type)
        
    
class SpeckleNoise(Fault):
    def __init__(self, intensity = 'l'):
        Fault.__init__(self, intensity, fault_type = 'SpeckleNoise')
        self.weight = 0.028
        
    def inject(self, clean_image):
        faulty_image = clean_image
        try:
            if self.intensity.lower()=='e':
                self.weight = (np.random.randint(14,17))/10
            elif self.intensity.lower()=='m':
                self.weight = (np.random.randint(9,12))/10
            else:
                self.weight = (np.random.randint(4,7))/10
            clean_nparray = imageToNumpy(clean_image)
            gauss = np.random.normal(0, self.weight, clean_nparray.shape)
            gauss = gauss.reshape(clean_nparray.shape[0], clean_nparray.shape[1], clean_nparray.shape[2]).astype('uint8')
            faulty_image = clean_image + clean_image * gauss
            return imageFromNumpy(faulty_image)
        except Exception as error_log:
            pass
            # print(error_log, self.fault_type)
        
    
class Rain(Fault):
    def __init__(self, intensity = 'l'):
        Fault.__init__(self, intensity, fault_type = 'Rain')
        self.weight = 1 # factor to change the drop_length(5,7,10)
        # specify drops density, drops direction, drops length and width, and the drops color
        self.drop_length = 1
        self.drop_width = 1
        self.drop_color = (200,200,200)
        
    def inject(self, clean_image):
        rain_drops = []
        clean_image = imageToNumpy(clean_image)
        verify_image(clean_image)
        imshape = clean_image.shape
        area = imshape[0] * imshape[1]
        try:
            # specify drops density, drops direction, and drops length
            if self.intensity.lower()=='m':
                self.no_of_drops = np.random.randint(380,450) # self.area//int(imshape[1])  # 200 
                self.slant= np.random.randint(3,7) * random.choice((-1, 1))
                drop_length = self.drop_length * random.choice((3, 5))
            elif self.intensity.lower()=='e':
                self.no_of_drops = np.random.randint(550,620) # self.area // int(imshape[1]*0.5) # 100
                self.slant= np.random.randint(7,10) * random.choice((-1, 1))
                drop_length = self.drop_length * np.random.randint(4,8)
            else:
                self.no_of_drops = np.random.randint(250,320) # self.area//int(imshape[1]*2) # 400
                self.slant= np.random.randint(1,3) * random.choice((-1, 1))
                drop_length = self.drop_length * np.random.randint(2,4)
                
            for i in range(self.no_of_drops): ## If You want heavy rain, try increasing this
                if self.slant<0:
                    x= np.random.randint(self.slant, imshape[1])
                else:
                    x= np.random.randint(0,imshape[1]-self.slant)
                y= np.random.randint(0,imshape[0]-drop_length)
                rain_drops.append((x,y))
            image_t= clean_image.copy()
            for rain_drop in rain_drops:
                cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+self.slant,rain_drop[1]+drop_length),self.drop_color,self.drop_width)
            clean_image = cv2.blur(image_t,(1,4)) ## rainy view are blurry
            brightness_coefficient = 0.7 ## rainy days are usually shady 
            image_HLS = hls(clean_image) ## Conversion to HLS
            image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
            faulty_image = rgb(image_HLS,'hls') ## Conversion to RGB
            return imageFromNumpy(faulty_image)
        except Exception as error_log:
            pass
            # print(error_log, self.fault_type)