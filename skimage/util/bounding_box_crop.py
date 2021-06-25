# -*- coding: utf-8 -*-
"""
Bounding boxes cropping module for any image.

Cropped images are saved to a folder along with the name of image.
"""


import cv2
from pathlib import Path
import os


def download_path():
    """Download folder path along with folder creation if needed.

    Returns
    -------
    downloads_path : str
        Path to the folder in the Downloads

    """
    downloads_path = str(Path.home() / "Downloads" / "Cropped")
    if not os.path.exists(downloads_path):
        os.makedirs(downloads_path)
    return downloads_path


def bounding_box_crop(image_url):
    """Bounding boxes are found and cropped.

    Bounding boxes saved to the Cropped folder created in Downloads folder.

    Parameters
    ----------
    image_url : str
        URL of the image.

    Returns
    -------
    None. Cropped Folder in Downloads to be checked.

    """
    image_name = os.path.basename(image_url).split('.')[0]
    org = cv2.imread(image_url)
    r = 1000.0 / org.shape[1]
    dim = (1000, int(org.shape[0] * r))
    o_resized = cv2.resize(org, dim, interpolation=cv2.INTER_LINEAR)

    img_number = 0

    img = cv2.imread(image_url, 0)
    r = 1000.0 / img.shape[1]
    dim = (1000, int(img.shape[0] * r))
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

    ret, thresh = cv2.threshold(resized, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 5000:
            continue
        img = cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crp = o_resized[y:y + h, x:x + w]
        cv2.imwrite(download_path() + '//' + image_name + '{}.png'.format(
            img_number), crp)
        img_number += 1
