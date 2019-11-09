from GHM import GHM
from GHMHelperFuncs import *

import numpy as np
# import traceback
import matplotlib.pyplot as plt


def create_black(dimensions=(100,100)):
	return np.zeros(dimensions)

def create_white_img(dimensions=(100,100)):
	return np.ones(dimensions)*255

def save_img(img, filename, folder="."):
	filepath = folder + "/" + filename
	np.save(filepath, img)
	plt.imsave(filepath + ".jpg", img, cmap='gray')

def is_black(img):
	return np.all(img == 0)

def is_white(img):
	return np.all(img == 255)

def test_all_combinations(imgs, img_names, type="pdf", output_folder=""):
	for i in range(len(imgs)):
		for j in range(len(imgs)):
			try:
				imgA = imgs[i]
				imgA_name = img_names[i]
				imgB = imgs[j]
				imgB_name = img_names[j]
				if type=="pdf":
					matched_imgA = GHM(imgA, imgB)
				else:
					matched_imgA = cdfGHM(imgA, imgB)
				save_img(matched_imgA, imgA_name + "_" + imgB_name, "255_branch")
			except Exception as err:
				filepath = "255_branch/" + imgA_name + "_" + imgB_name + "ERROR"
				print(err)
				# print(traceback.format_exc())
				f = open(filepath + ".txt", "w")
				f.write(str(err))
				# f.write(traceback.format_exc())
				f.close()

def load_imgs():
	black = read_and_check_img("black.jpg")
	# black = create_black()
	save_img(black, "black")
	assert is_black(black), "black image is not actually black (i.e. not all pixel values are 0)"

	grayish = read_and_check_img("grayish.jpg")

	house = read_and_check_img("house.jpg")

	streetlights = read_and_check_img("streetlights.jpg")

	white = read_and_check_img("white.jpg")
	assert is_white(white), "white image is not actually white (i.e. not all pixel values are 255)"

	imgs = [black, grayish, house, streetlights, white]
	for i in range(len(imgs)):
		assert is_from_0_to_255(imgs[i]), img_names[i] + " not in 0-255 range"
	img_names = ['black', 'grayish', 'house', 'streetlights', 'white']
	return imgs, img_names

test_all_combinations(*load_imgs())
















