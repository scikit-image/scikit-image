from __future__ import division
import numpy as np
import cv

def rgb2grey(rgb):
    grey = np.empty_like(rgb[:,:,0])
    cv.CvtColor(rgb, grey, cv.CV_RGB2GRAY)
    return grey

def rgb2hsv(rgb):
    hsv = np.empty_like(rgb)
    cv.CvtColor(rgb, hsv, cv.CV_RGB2HSV)
    return hsv

def hsv2rgb(hsv):
    rgb = np.empty_like(hsv)
    cv.CvtColor(hsv, rgb, cv.CV_HSV2RGB)
    return rgb

def rgb2xyz(rgb):
    xyz = np.empty_like(rgb)
    cv.CvtColor(hsv, xyz, cv.CV_RGB2XYZ)
    return xyz

def xyz2rgb(xyz):
    rgb = np.empty_like(xyz)
    cv.CvtColor(xyz, rgb, cv.CV_XYZ2RGB)
    return xyz
